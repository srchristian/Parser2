"""
syllabus_extractor.py

Orchestration module for syllabus extraction, focusing on
extracting structured data like class_data using an OpenAI-based parser.
It now uses targeted text segments for more efficient extraction.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    from utils.parsers.openai_parser import OpenAIParser
    OPENAI_PARSER_AVAILABLE = True
except ImportError:
    OPENAI_PARSER_AVAILABLE = False
    print("Warning: OpenAIParser could not be imported for SyllabusExtractor.")

try:
    from utils.helpers import setup_logger # Assuming setup_logger is available
except ImportError:
    def setup_logger(log_dir="logs", log_name="DefaultLogger", log_level=logging.INFO, **kwargs): # Added **kwargs
        temp_logger = logging.getLogger(log_name)
        if not temp_logger.handlers:
            temp_logger.setLevel(log_level)
            ch = logging.StreamHandler(); ch.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter); temp_logger.addHandler(ch)
        return temp_logger
    print("Warning: setup_logger from utils.helpers not found for SyllabusExtractor, using basic console logger.")


class SyllabusExtractor:
    """
    Orchestrates syllabus extraction, primarily for 'class_data', using OpenAIParser.
    Constructs a targeted text input from syllabus segments for efficiency.
    """
    
    # Segments most likely to contain class_data information
    PRIMARY_SEGMENTS_FOR_CLASS_DATA = [
        "course_identification",
        "instructor_information",
        "course_description_prerequisites"
    ]
    SECONDARY_SEGMENTS_FOR_CLASS_DATA = [
        "course_policies", # Often contains school/term info
        "communication_student_support" # Might have more instructor details
    ]
    # Max characters for the initial snippet of full text if used
    INITIAL_SNIPPET_MAX_CHARS = 1000


    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None, 
                 openai_model: Optional[str] = None, 
                 logger_instance: Optional[logging.Logger] = None,
                 openai_parser_instance: Optional[OpenAIParser] = None):
        self.config = config or {}
        # Ensure 'logging' key exists in self.config before accessing its sub-keys
        log_config = self.config.get("logging", {})
        self.logger = logger_instance or setup_logger(log_name="SyllabusExtractorStandalone", **log_config)
        
        self.logger.info("Initializing SyllabusExtractor...")
        
        _model_to_use = openai_model or self.config.get("extraction", {}).get("openai_model", "gpt-4o")
        self.openai_model_preference = _model_to_use
        self.logger.info(f"SyllabusExtractor preferred OpenAI model: {self.openai_model_preference}")

        if openai_parser_instance:
            self.openai_parser = openai_parser_instance
            self.logger.info("Using provided OpenAIParser instance.")
        elif OPENAI_PARSER_AVAILABLE:
            self.logger.info(f"Creating new OpenAIParser instance with model: {self.openai_model_preference}")
            self.openai_parser = OpenAIParser(model=self.openai_model_preference, logger_instance=self.logger, config=self.config)
        else:
            self.openai_parser = None
            self.logger.error("OpenAIParser class not imported and no instance provided. SyllabusExtractor cannot function effectively.")

        if self.openai_parser: self.logger.info("SyllabusExtractor initialized successfully.")
        else: self.logger.error("SyllabusExtractor initialization failed: OpenAIParser not available.")
        
    def _construct_targeted_text(self, 
                                 full_syllabus_text: str, 
                                 segmented_data: Dict[str, Any]) -> str:
        """
        Constructs a targeted text block from relevant syllabus segments and an initial
        snippet of the full text to optimize LLM calls for class_data extraction.
        """
        text_parts: List[str] = []
        added_content_hashes = set() # To avoid adding identical segment content multiple times

        # 1. Add an initial snippet of the full text to catch headers/university info
        # Use initial_snippet_max_chars from config if available, else class default
        snippet_max_chars = self.config.get("syllabus_extractor", {}).get("initial_snippet_max_chars", self.INITIAL_SNIPPET_MAX_CHARS)
        initial_snippet = full_syllabus_text[:snippet_max_chars].strip()
        if initial_snippet:
            snippet_hash = hash(initial_snippet) # Simple hash for deduplication
            if snippet_hash not in added_content_hashes:
                text_parts.append(f"--- Initial Syllabus Snippet ---\n{initial_snippet}")
                added_content_hashes.add(snippet_hash)

        # 2. Add content from primary segments
        for key in self.PRIMARY_SEGMENTS_FOR_CLASS_DATA:
            segment_content = str(segmented_data.get(key, "")).strip()
            if segment_content:
                content_hash = hash(segment_content)
                if content_hash not in added_content_hashes:
                    text_parts.append(f"--- Segment: {key.replace('_', ' ').title()} ---\n{segment_content}")
                    added_content_hashes.add(content_hash)
        
        # 3. Add content from secondary segments
        for key in self.SECONDARY_SEGMENTS_FOR_CLASS_DATA:
            segment_content = str(segmented_data.get(key, "")).strip()
            if segment_content:
                content_hash = hash(segment_content)
                if content_hash not in added_content_hashes:
                    text_parts.append(f"--- Segment: {key.replace('_', ' ').title()} ---\n{segment_content}")
                    added_content_hashes.add(content_hash)
        
        # 4. Add unclassified content if it seems relevant
        unclassified_content_list = segmented_data.get("unclassified_content", [])
        if isinstance(unclassified_content_list, list):
            combined_unclassified = "\n".join(str(item).strip() for item in unclassified_content_list if str(item).strip()).strip()
            if combined_unclassified:
                if len(combined_unclassified) < 2000 : 
                    content_hash = hash(combined_unclassified)
                    if content_hash not in added_content_hashes:
                        text_parts.append(f"--- Segment: Unclassified Content ---\n{combined_unclassified}")
                        added_content_hashes.add(content_hash)


        if not text_parts: 
            self.logger.warning("No relevant segments found or all were empty. Using initial part of full text as fallback for OpenAIParser.")
            max_chars = self.config.get("openai_parser", {}).get("max_chars_for_metadata", 15000)
            return full_syllabus_text[:max_chars].strip()

        final_text = "\n\n".join(text_parts)
        self.logger.info(f"Constructed targeted text for OpenAIParser. Length: {len(final_text)} chars. Number of parts: {len(text_parts)}.")
        self.logger.debug(f"Targeted text preview (first 500 chars): {final_text[:500]}...")
        return final_text

    def extract_all(self, 
                    full_syllabus_text: str, 
                    segmented_data: Dict[str, Any], 
                    unique_id: str) -> Dict[str, Any]:
        """
        Extracts structured data using OpenAIParser on a targeted text block
        constructed from relevant syllabus segments and an initial snippet.
        
        Args:
            full_syllabus_text (str): The complete raw syllabus text.
            segmented_data (Dict[str, Any]): Segmented syllabus data from SyllabusParser.
            unique_id (str): Unique identifier for this processing job (content_hash).
            
        Returns:
            Dict[str, Any]: Dictionary with extracted information.
        """
        self.logger.info(f"SyllabusExtractor.extract_all started for Unique ID: {unique_id}")
        
        if not self.openai_parser:
            self.logger.error("OpenAIParser not available. Cannot extract data.")
            return self._create_empty_results(unique_id, full_syllabus_text, "OpenAIParser not available.")

        if not full_syllabus_text or not full_syllabus_text.strip():
            self.logger.warning(f"Full syllabus text for ID {unique_id} is empty. Returning empty results.")
            return self._create_empty_results(unique_id, full_syllabus_text, "Input full_syllabus_text was empty")

        text_for_openai = self._construct_targeted_text(full_syllabus_text, segmented_data)
        
        if not text_for_openai.strip():
            self.logger.warning(f"Constructed targeted text for ID {unique_id} is empty. Returning empty results from extractor.")
            return self._create_empty_results(unique_id, full_syllabus_text, "Constructed targeted text was empty")

        start_time = datetime.now()
        
        try:
            self.logger.info(f"Delegating extraction to OpenAIParser with targeted text (length: {len(text_for_openai)}).")
            results = self.openai_parser.extract(text_for_openai, unique_id) 
            
            results.setdefault("metadata", {}).setdefault("unique_id", unique_id)
            results["metadata"]["original_full_text_length"] = len(full_syllabus_text) 
            results["metadata"]["targeted_text_length_for_openai"] = len(text_for_openai) 
            results.setdefault("class_data", {})
            results.setdefault("event_data", [])
            results.setdefault("task_data", [])

            elapsed_time = (datetime.now() - start_time).total_seconds()
            results["metadata"]["process_time_extractor"] = elapsed_time 
            results["metadata"]["extraction_complete"] = results["metadata"].get("extraction_complete", True) 
            
            self.logger.info(f"OpenAIParser extraction (via SyllabusExtractor) completed in {elapsed_time:.2f}s for ID: {unique_id}")
            self.logger.info(f"Missing fields reported by parser: {results['metadata'].get('missing_fields_by_parser', 'N/A')}")
            
            return results
                
        except Exception as e:
            self.logger.error(f"SyllabusExtractor: Failed to extract syllabus data for ID {unique_id}: {e}", exc_info=True)
            return self._create_empty_results(unique_id, full_syllabus_text, str(e))
    
    def _create_empty_results(self, unique_id: str, syllabus_text: str = "", error_message: Optional[str] = None) -> Dict[str, Any]:
        metadata = {
            "extraction_time": datetime.now().isoformat(), "unique_id": unique_id,
            "text_length": len(syllabus_text) if syllabus_text else 0, "extraction_complete": False,
            "missing_fields_by_parser": [], 
            "original_file": "", "process_time_extractor": 0.0,
            "targeted_text_length_for_openai": 0
        }
        if error_message: metadata["error"] = error_message
        
        # Attempt to get CLASS_DATA_FIELDS from OpenAIParser if available, otherwise provide a default list
        class_data_field_keys = []
        if OPENAI_PARSER_AVAILABLE and hasattr(OpenAIParser, 'CLASS_DATA_FIELDS'):
            class_data_field_keys = OpenAIParser.CLASS_DATA_FIELDS
        else: # Fallback if OpenAIParser or its fields are not accessible
            class_data_field_keys = [
                "School Name", "Term", "Course Title", "Course Code", "Instructor Name",
                "Instructor Email", "Class Time", "Time Zone", "Days of Week",
                "Term Start Date", "Term End Date", "Class Start Date", "Class End Date",
                "Office Hours", "Telephone", "Class Location", "Additional"
            ]

        return {
            "metadata": metadata,
            "class_data": {key: "" for key in class_data_field_keys},
            "event_data": [], "task_data": []
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s', handlers=[logging.StreamHandler()])
    test_logger = logging.getLogger("SyllabusExtractorTest")

    mock_config = {
        "directories": {"logs": "temp_logs"},
        "extraction": {"openai_model": "gpt-3.5-turbo"}, 
        "openai_parser": {"max_chars_for_metadata": 10000}, 
        "syllabus_extractor": {"initial_snippet_max_chars": 500 }, # Test with a smaller snippet
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")} 
    }

    class MockOpenAIParser: # Defined within __main__ for test isolation
        CLASS_DATA_FIELDS = ["School Name", "Term", "Course Title", "Instructor Name"] 
        def __init__(self, model, logger_instance, config=None): 
            self.model = model; self.logger = logger_instance; self.config = config
            self.logger.info(f"MockOpenAIParser initialized with model: {self.model}")
        def extract(self, text: str, unique_id: str) -> Dict[str, Any]:
            self.logger.info(f"MockOpenAIParser.extract called for ID: {unique_id} with text len: {len(text)}. Preview: {text[:100]}...")
            return {
                "metadata": {"unique_id": unique_id, "missing_fields_by_parser": [], "extraction_complete": True, "model_used": self.model},
                "class_data": {"Course Title": "Mock Course from Targeted Text", "Instructor Name": "Dr. Mock Segment"},
                "event_data": [], "task_data": []}

    # Test with injected MockOpenAIParser
    # Ensure OPENAI_PARSER_AVAILABLE is True for the constructor to use the mock correctly if it were checking type
    # For this test, we directly inject, so the flag's state for this specific branch isn't critical.
    original_openai_parser_available_flag = OPENAI_PARSER_AVAILABLE
    OPENAI_PARSER_AVAILABLE = True # Temporarily set for testing init path if it relied on it.

    mock_parser_instance = MockOpenAIParser(model=mock_config["extraction"]["openai_model"], logger_instance=test_logger, config=mock_config)
    extractor_with_mock = SyllabusExtractor(config=mock_config, logger_instance=test_logger, openai_parser_instance=mock_parser_instance)
    
    OPENAI_PARSER_AVAILABLE = original_openai_parser_available_flag # Restore flag


    test_logger.info("\n--- Testing with injected MockOpenAIParser and segmented data ---")
    sample_full_text = "UNIVERSITY OF ADVANCED STUDIES. CS101 Intro to CS. Prof Minerva. Fall 2024. This is the main description. Policies: All assignments due Fridays. Schedule: Week 1 - Intro. Unclassified: Important note here."
    sample_segmented_data = {
        "course_identification": "UNIVERSITY OF ADVANCED STUDIES. CS101 Intro to CS. Fall 2024.",
        "instructor_information": "Prof Minerva.",
        "course_description_prerequisites": "This is the main description.",
        "course_policies": "Policies: All assignments due Fridays.",
        "course_schedule": "Schedule: Week 1 - Intro.", # Not used by _construct_targeted_text
        "unclassified_content": ["Important note here."] 
    }
    results_mock = extractor_with_mock.extract_all(sample_full_text, sample_segmented_data, "test-id-segmented")
    test_logger.info(f"Results (with mock parser & segments): {json.dumps(results_mock, indent=2)}")

    test_logger.info("\n--- Testing with empty segmented data (should use full text snippet from config) ---")
    results_empty_segments = extractor_with_mock.extract_all(sample_full_text, {}, "test-id-empty-segments")
    test_logger.info(f"Results (empty segments): {json.dumps(results_empty_segments, indent=2)}")
    
    test_logger.info("\n--- Testing with only unclassified content in segments ---")
    only_unclassified_segments = {"unclassified_content": ["This is some crucial unclassified info about the course."]}
    results_unclassified = extractor_with_mock.extract_all(sample_full_text, only_unclassified_segments, "test-id-unclassified")
    test_logger.info(f"Results (only unclassified): {json.dumps(results_unclassified, indent=2)}")


    test_logger.info("\n--- Standalone SyllabusExtractor tests finished ---")
