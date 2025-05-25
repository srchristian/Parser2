"""
openai_parser.py

Extraction of metadata (primarily class_data) from syllabi using OpenAI's API.
Revised to extract a more comprehensive set of fields as per the project plan.
"""

import os
import json
import re
import time
import logging # Standard library logging
import sys 
from typing import Dict, List, Any, Optional
import openai
from datetime import datetime

# Attempt to import httpx for client timeout configuration
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("Warning: httpx library not found. OpenAI client timeouts may not be configurable.")

# Module-level logger. Its configuration should be set by the importing application
# or by a basicConfig in __main__ if run standalone.
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Add a NullHandler if no handlers are configured
    logger.addHandler(logging.NullHandler())


class OpenAIParser:
    """
    Parser that uses OpenAI API to extract structured metadata (class_data) from syllabi.
    It focuses on accurately extracting predefined fields based on a detailed prompt.
    """
    
    # Define all class_data fields as per the Project Plan (Section II.B)
    # This list is used for initializing the output and guiding the extraction.
    CLASS_DATA_FIELDS = [
        "School Name", "Term", "Course Title", "Course Code", "Instructor Name",
        "Instructor Email", "Class Time", "Time Zone", "Days of Week",
        "Term Start Date", "Term End Date", "Class Start Date", "Class End Date",
        "Office Hours", "Telephone", "Class Location", "Additional"
    ]

    def __init__(self, 
                 model: str = "gpt-4o", 
                 logger_instance: Optional[logging.Logger] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenAI parser.

        Args:
            model (str, optional): OpenAI model to use. Defaults to "gpt-4o".
            logger_instance (logging.Logger, optional): An external logger instance. 
                                               If None, the module-level logger is used.
            config (dict, optional): Configuration dictionary. Defaults to an empty dict.
        """
        self.model = model
        self.logger = logger_instance or logger # Use provided logger or module-level one
        self.config = config or {}
        self.client = None # OpenAI API client instance

        self.logger.info(f"Initializing OpenAIParser with model: {self.model}")
        
        # Initialize OpenAI client
        # API key resolution: config (openai or extraction section) > environment variable
        api_key_config = self.config.get("openai", {}).get("api_key") or \
                         self.config.get("extraction", {}).get("openai_api_key")
        api_key = api_key_config or os.environ.get("OPENAI_API_KEY")
        
        if api_key:
            try:
                # Configure client timeout settings from config or use defaults
                client_timeout_config = self.config.get("openai", {}).get("client_timeout", 
                                                                      {"read": 60.0, "connect": 10.0}) # Default timeouts
                read_timeout = float(client_timeout_config.get("read", 60.0))
                connect_timeout = float(client_timeout_config.get("connect", 10.0))
                
                if HTTPX_AVAILABLE:
                    timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
                    self.client = openai.OpenAI(api_key=api_key, timeout=timeout)
                    self.logger.info(f"OpenAI client created for model {self.model}. Timeout: {read_timeout}s read, {connect_timeout}s connect.")
                else: # If httpx is not available, initialize client without explicit timeout
                    self.client = openai.OpenAI(api_key=api_key)
                    self.logger.warning("httpx library not available. OpenAI client initialized without explicit timeout configuration. Default timeouts will apply.")
                
            except Exception as e_client_init:
                self.logger.error(f"Error initializing OpenAI client: {e_client_init}", exc_info=True)
                self.client = None # Ensure client is None on error
        else:
            self.logger.error("No OpenAI API key found in config or environment variables. OpenAI client not initialized.")
            self.client = None
    
    def extract(self, syllabus_text: str, unique_id: str) -> Dict[str, Any]:
        """
        Extract structured data (class_data, and placeholders for event/task data)
        from the provided syllabus text.

        Args:
            syllabus_text (str): The full text of the syllabus.
            unique_id (str): A unique identifier for this extraction job, used for logging.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "metadata": Information about the extraction process.
                - "class_data": Extracted general course information.
                - "event_data": Placeholder (empty list).
                - "task_data": Placeholder (empty list).
        """
        self.logger.info(f"OpenAIParser: Starting data extraction for ID: {unique_id}")
        extraction_start_time = time.monotonic()

        if not self.client:
            self.logger.error(f"OpenAI client not initialized for ID {unique_id}. Cannot perform extraction.")
            results = self._create_empty_results(unique_id, error_message="OpenAI client not initialized.")
            results["metadata"]["process_time"] = round(time.monotonic() - extraction_start_time, 3)
            return results
        
        if not syllabus_text or not syllabus_text.strip():
            self.logger.warning(f"Input syllabus text for ID {unique_id} is empty or whitespace only. Returning empty results.")
            results = self._create_empty_results(unique_id, error_message="Input syllabus text was empty.")
            results["metadata"]["process_time"] = round(time.monotonic() - extraction_start_time, 3)
            return results

        try:
            # Configurable character limit for the text sent to OpenAI for metadata extraction
            max_chars_for_metadata = int(self.config.get("openai_parser", {}).get("max_chars_for_metadata", 30000)) # Default 30k
            
            processed_text = syllabus_text
            if len(syllabus_text) > max_chars_for_metadata:
                self.logger.warning(
                    f"Syllabus text for ID {unique_id} (length {len(syllabus_text)}) exceeds "
                    f"limit of {max_chars_for_metadata} chars for metadata extraction. Truncating text."
                )
                processed_text = syllabus_text[:max_chars_for_metadata] # Truncate if too long
            
            # Call the internal method to get class_data using OpenAI
            class_data = self._extract_class_metadata(processed_text, unique_id) 
            
            # Perform checks for missing critical fields based on config or defaults
            missing_fields = []
            # These fields are examples; "required_fields" from main config might be used by the pipeline orchestrator later.
            # This internal check is more for OpenAIParser's own assessment of its extraction quality.
            critical_fields_for_check = self.config.get("openai_parser", {}).get(
                "critical_fields_check_after_extraction", 
                ["Term", "Course Title", "Class Time", "Days of Week", "Class Start Date", "Instructor Name"] # Example critical fields
            )

            for field in critical_fields_for_check:
                field_value = str(class_data.get(field, "")).strip() # Ensure it's a string and stripped
                if not field_value: # If field is empty after extraction
                    missing_fields.append(field)
                # Specific heuristic checks (can be expanded)
                elif field == "Class Time" and not ("-" in field_value or "to" in field_value.lower() or re.search(r'\d\s*(?:AM|PM)', field_value, re.IGNORECASE)):
                    missing_fields.append(field)
                    self.logger.debug(f"ID {unique_id}: 'Class Time' ('{field_value}') might be incomplete. Marked as potentially missing.")
                elif field == "Course Title" and ":" not in field_value and not re.search(r'[A-Z]{2,4}\s*\d{3,4}', field_value):
                    missing_fields.append(field) # If no colon and no typical course code pattern
                    self.logger.debug(f"ID {unique_id}: 'Course Title' ('{field_value}') may be missing course code. Marked as potentially missing.")
            
            process_duration = time.monotonic() - extraction_start_time
            results = {
                "metadata": {
                    "extraction_time": datetime.now().isoformat(),
                    "unique_id": unique_id,
                    "text_length": len(syllabus_text), 
                    "processed_text_length": len(processed_text), # Length of text actually sent to API
                    "missing_fields_by_parser": list(set(missing_fields)), # Fields deemed missing by this parser's checks
                    "original_file": f"{unique_id}.txt", # Placeholder, actual filename managed by pipeline
                    "process_time": round(process_duration, 3),
                    "model_used": self.model,
                    "extraction_complete": True # Assume success if no major exception
                },
                "class_data": class_data,
                "event_data": [], # OpenAIParser primarily focuses on class_data
                "task_data": []   # Other parsers handle events/tasks
            }
            
            self.logger.info(f"Extraction for ID {unique_id} completed. Fields flagged by parser: {missing_fields}. Process time: {process_duration:.2f}s")
            return results

        except Exception as e_extract: # Catch-all for unexpected errors during extraction
            self.logger.error(f"Error during OpenAIParser.extract for ID {unique_id}: {e_extract}", exc_info=True)
            results = self._create_empty_results(unique_id, error_message=str(e_extract))
            results["metadata"]["process_time"] = round(time.monotonic() - extraction_start_time, 3)
            results["metadata"]["extraction_complete"] = False # Mark as incomplete on error
            return results
    
    def _extract_class_metadata(self, syllabus_text_segment: str, unique_id: str) -> Dict[str, str]:
        """
        Extracts class metadata from a syllabus text segment using the OpenAI API.
        The prompt is designed to fetch all fields defined in self.CLASS_DATA_FIELDS.
        Includes retry logic for API calls.

        Args:
            syllabus_text_segment (str): The (potentially truncated) syllabus text.
            unique_id (str): Unique ID for logging.

        Returns:
            Dict[str, str]: A dictionary where keys are from self.CLASS_DATA_FIELDS
                            and values are the extracted strings (or empty strings).
        """
        # Initialize with all expected keys, ensuring they are present in the output.
        class_data: Dict[str, str] = {key: "" for key in self.CLASS_DATA_FIELDS}
        
        # System prompt instructing the LLM on how to extract the defined fields.
        # This prompt is crucial for getting accurate and well-formatted data.
        system_prompt = f"""
You are an AI assistant specialized in parsing academic syllabi with extreme precision.
Your task is to extract ONLY explicitly stated information for the fields listed below from the provided syllabus text.
Output MUST be a single, valid JSON object.

CRITICAL RULES:
1.  STRICT ADHERENCE: DO NOT infer, guess, assume, calculate, or rephrase ANY information. Extract verbatim or as close as possible.
2.  EXPLICIT EXTRACTION: Extract ONLY what is explicitly written in the syllabus text provided.
3.  EMPTY STRINGS: If a field's information is not explicitly stated in the text, or if you are uncertain, its value MUST be an empty string (""). Do not omit the key.
4.  JSON FORMAT: The entire response must be a single JSON object. Do not add any text, comments, or markdown formatting before or after the JSON object.
5.  SPECIFIC FORMATS & GUIDANCE:
    - "School Name": Full official name of the school, college, or university (e.g., "University of Rhode Island", "Harvard University - Faculty of Arts and Sciences"). Check headers, footers, or policy sections.
    - "Term": Academic term (e.g., "Fall 2024", "Spring Semester 2025", "Summer Session I 2023").
    - "Course Title": The FULL title of the course, INCLUDING the course code if it's part of the title (e.g., "CS 101: Introduction to Computer Science", "PHY203: ELEMENTARY PHYSICS I").
    - "Course Code": The official course identifier (e.g., "CS101", "PHY203", "ENG-220A"). If part of "Course Title", extract it separately here. If not explicitly separate, derive from "Course Title" if a clear pattern like "DEPT ###" exists.
    - "Instructor Name": Full name of the primary instructor(s) (e.g., "Dr. Ada Lovelace", "Professor Charles Babbage, PhD").
    - "Instructor Email": Instructor's email address (e.g., "alovelace@example.edu").
    - "Class Time": Regular meeting time(s) of the class (e.g., "1:00 PM - 1:50 PM", "Tuesdays 6:00pm-8:50pm EST", "14:00-15:30").
    - "Time Zone": Time zone for class meetings if specified (e.g., "Eastern Time (ET)", "PST", "GMT+2"). If not specified, leave empty.
    - "Days of Week": Days the class meets, preferably full names (e.g., "Monday, Wednesday, Friday", "Tuesday, Thursday", "MWF").
    - "Term Start Date": Official start date of the academic term. Format: "Month Day, Year" (e.g., "September 1, 2024").
    - "Term End Date": Official end date of the academic term. Format: "Month Day, Year" (e.g., "December 15, 2024").
    - "Class Start Date": First actual meeting day of THIS class. Format: "Month Day, Year" (e.g., "September 4, 2024").
    - "Class End Date": Last actual meeting day of THIS class. Format: "Month Day, Year" (e.g., "December 11, 2024").
    - "Office Hours": Instructor's office hours and location/method (e.g., "Mondays 10-11 AM, Tech Hall Room 202 or by appointment", "Virtual: Tuesdays 2-3 PM via Zoom").
    - "Telephone": Instructor's contact telephone number, if provided.
    - "Class Location": Physical room or online platform for class meetings (e.g., "East Hall Auditorium", "Online via Brightspace", "Room 305, Science Building").
    - "Additional": Include a brief, direct quote of the course description if available. If not, list other relevant general course information like LMS details (e.g., "Course materials available on Brightspace."). If nothing suitable, leave empty.

FIELDS TO EXTRACT (use these exact key names in the JSON):
{json.dumps(self.CLASS_DATA_FIELDS, indent=2)}

EXAMPLE (Illustrative - your output should be based on the actual syllabus text):
Input Text Snippet: "URI - College of Engineering. ELE 201: Circuit Analysis. Fall 2024. Prof. Ohm. MWF 10:00 AM - 10:50 AM in Bliss Hall 101. Course starts Sep 4, 2024, ends Dec 10, 2024. Term: 9/1/2024 - 12/20/2024. Email: prof.ohm@uri.edu. This course introduces..."
Expected JSON:
{{
  "School Name": "URI - College of Engineering",
  "Term": "Fall 2024",
  "Course Title": "ELE 201: Circuit Analysis",
  "Course Code": "ELE201",
  "Instructor Name": "Prof. Ohm",
  "Instructor Email": "prof.ohm@uri.edu",
  "Class Time": "10:00 AM - 10:50 AM",
  "Time Zone": "",
  "Days of Week": "Monday, Wednesday, Friday",
  "Term Start Date": "September 1, 2024",
  "Term End Date": "December 20, 2024",
  "Class Start Date": "September 4, 2024",
  "Class End Date": "December 10, 2024",
  "Office Hours": "",
  "Telephone": "",
  "Class Location": "Bliss Hall 101",
  "Additional": "This course introduces..."
}}
"""
        
        # Max retries for this specific API call, configurable via openai_parser settings
        max_local_retries = int(self.config.get("openai_parser", {}).get("max_api_retries", 2)) # Default 2 retries

        for attempt in range(max_local_retries):
            try:
                self.logger.info(f"Calling OpenAI API for class metadata (ID: {unique_id}, attempt {attempt + 1}/{max_local_retries}). Model: {self.model}")
                
                response = self.client.chat.completions.create(
                    model=self.model, 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Extract key information from this academic syllabus segment:\n\n---\n{syllabus_text_segment}\n---"}
                    ],
                    temperature=0.0, # Low temperature for more deterministic, factual extraction
                    response_format={"type": "json_object"} # Request JSON object output if model supports
                )
                response_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message and response.choices[0].message.content else ""
                
                self.logger.info(f"OpenAI API response received for class_data (ID: {unique_id}, length: {len(response_text)} chars).")
                self.logger.debug(f"OpenAI Raw Response for class_data (ID: {unique_id}, first 500 chars): {response_text[:500]}...")
                
                # Attempt to parse the JSON from the response
                json_data = self._extract_json_from_response(response_text)
                if json_data and isinstance(json_data, dict):
                    # Populate class_data dictionary, ensuring all keys are present
                    for key in self.CLASS_DATA_FIELDS:
                        class_data[key] = str(json_data.get(key, "")).strip() # Get value or default to empty string, then strip
                    self.logger.info(f"Successfully extracted and parsed class metadata from API response for ID: {unique_id}.")
                    return class_data # Return the populated dictionary
                else: # JSON parsing failed or was not a dictionary
                    self.logger.warning(f"No valid JSON dictionary found in API response on attempt {attempt + 1} for ID: {unique_id}.")
                    if attempt < max_local_retries - 1: # If not the last attempt
                        time.sleep(1 + (2 ** attempt))  # Exponential backoff before retrying
                        self.logger.info(f"Retrying API call for class_data (ID: {unique_id}) after parsing failure.")
                    else: # Max retries reached after parsing failure
                        self.logger.error(f"Max local retries ({max_local_retries}) reached for ID {unique_id} after failing to parse JSON from API response for class_data.")
            
            # Handle specific OpenAI and httpx errors for retry logic
            except openai.RateLimitError as e_rate_limit:
                self.logger.warning(f"OpenAI Rate Limit Error (ID: {unique_id}, attempt {attempt+1}): {e_rate_limit}. Retrying after longer backoff...")
                if attempt < max_local_retries - 1: time.sleep( (2**attempt) * 5 + 1 ) # Longer backoff for rate limits
                else: self.logger.error(f"OpenAI Rate Limit Error persisted after {max_local_retries} attempts for ID: {unique_id}.", exc_info=True)
            except openai.APITimeoutError as e_api_timeout: # Includes httpx.ReadTimeout etc. via SDK wrapping
                self.logger.warning(f"OpenAI API Timeout Error (ID: {unique_id}, attempt {attempt+1}): {e_api_timeout}. Retrying...")
                if attempt < max_local_retries - 1: time.sleep( 1 + (2**attempt) )
                else: self.logger.error(f"OpenAI API Timeout Error persisted after {max_local_retries} attempts for ID: {unique_id}.", exc_info=True)
            except openai.APIConnectionError as e_api_conn: 
                self.logger.error(f"OpenAI API Connection Error (ID: {unique_id}, attempt {attempt+1}): {e_api_conn}. SDK might handle retries; local retry may not be effective. Breaking local retries.", exc_info=True)
                break # Break local retries for persistent connection errors
            except openai.APIStatusError as e_api_status: # Handles HTTP status errors from OpenAI (e.g., 500, 503)
                 self.logger.warning(f"OpenAI API Status Error {e_api_status.status_code} (ID: {unique_id}, attempt {attempt+1}): {getattr(e_api_status, 'message', str(e_api_status))}. Retrying for server-side errors...")
                 if 500 <= e_api_status.status_code < 600 and attempt < max_local_retries - 1 : # Retry only for server-side errors (5xx)
                     time.sleep( (2**attempt) * 2 + 1 ) # Exponential backoff
                 else: # Client-side errors (4xx) or max retries for server errors
                     self.logger.error(f"OpenAI API Status Error {e_api_status.status_code} for ID: {unique_id} not retried or max retries reached.", exc_info=True)
                     break # Do not retry client errors or if max retries hit for server errors
            except Exception as e_generic_api: # Catch any other unexpected errors during API call or processing
                self.logger.error(f"Unexpected error during OpenAI API call or processing for class_data (ID: {unique_id}, attempt {attempt+1}): {e_generic_api}", exc_info=True)
                if attempt < max_local_retries - 1:
                    time.sleep(1 + (2 ** attempt)) # Exponential backoff
                else: # Max retries reached after unexpected error
                    self.logger.error(f"Max retries ({max_local_retries}) reached for ID {unique_id} after unexpected error during class_data extraction.", exc_info=True)
        
        self.logger.error(f"Failed to extract class metadata for ID {unique_id} after all retries or due to a critical/non-retryable error.")
        return class_data # Return the initialized (likely empty) class_data dict
    
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Robustly extracts a JSON object from the OpenAI response string.
        It tries direct parsing first. If that fails, it looks for JSON enclosed
        in markdown code fences (e.g., ```json ... ```). As a final fallback,
        it attempts to find the first '{' and last '}' to delineate a JSON object.
        """
        if not response_text or not response_text.strip():
            self.logger.warning("JSON extraction: Received empty or whitespace-only response text.")
            return None
            
        try:
            # Attempt 1: Direct JSON parsing (assuming response is pure JSON)
            self.logger.debug("Attempting direct JSON parse of API response.")
            return json.loads(response_text)
        except json.JSONDecodeError:
            self.logger.debug("Direct JSON parse failed. Attempting to find JSON within markdown code fences.")
            # Attempt 2: Extract JSON from markdown code fences (e.g., ```json ... ``` or ``` ... ```)
            # This regex captures content between triple backticks, optionally preceded by 'json'.
            # It's non-greedy for the content (.*?) and handles potential leading/trailing whitespace.
            match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text, re.DOTALL)
            if match:
                json_str_from_fence = match.group(1).strip() # Extract the captured group (the JSON string)
                try:
                    self.logger.debug(f"Found potential JSON in code fence. Preview (first 200 chars): {json_str_from_fence[:200]}...")
                    return json.loads(json_str_from_fence)
                except json.JSONDecodeError as e_fence_decode:
                    self.logger.warning(f"JSON decode error for content found within code fence: {e_fence_decode}. Content preview: {json_str_from_fence[:200]}...")
                    # Fall through to next attempt if fence content is not valid JSON
            
            # Attempt 3: Fallback to find the first '{' and last '}' if no valid fenced JSON
            self.logger.debug("Code fence JSON not found or invalid. Attempting to find any general JSON block (first '{' to last '}').")
            first_brace_idx = response_text.find('{')
            last_brace_idx = response_text.rfind('}')
            if first_brace_idx != -1 and last_brace_idx > first_brace_idx:
                json_like_block_str = response_text[first_brace_idx : last_brace_idx + 1].strip()
                try:
                    self.logger.debug(f"Found general JSON-like block. Preview (first 200 chars): {json_like_block_str[:200]}...")
                    return json.loads(json_like_block_str)
                except json.JSONDecodeError as e_block_decode:
                    self.logger.warning(f"JSON decode error for general block identified by braces: {e_block_decode}. Block preview: {json_like_block_str[:200]}...")
        
        except Exception as e_outer_extract: # Catch any other unexpected error during these attempts
            self.logger.error(f"Unexpected error during _extract_json_from_response: {e_outer_extract}", exc_info=True)

        # If all attempts fail
        self.logger.warning("Could not extract a valid JSON object from the API response after multiple strategies.")
        self.logger.debug(f"Full API response text that failed all JSON extraction attempts (first 500 chars): {response_text[:500]}")
        return None
    
    def _create_empty_results(self, unique_id="unknown", error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Creates a standardized empty or error results structure, ensuring all
        CLASS_DATA_FIELDS are present in class_data with empty strings.
        """
        metadata = {
            "extraction_time": datetime.now().isoformat(),
            "unique_id": unique_id,
            "text_length": 0, 
            "processed_text_length": 0,
            # missing_fields_by_parser will be populated by the extract method's checks
            "missing_fields_by_parser": [], 
            "original_file": "", # Placeholder, managed by pipeline
            "process_time": 0.0, 
            "model_used": self.model, # Model that was attempted or configured
            "extraction_complete": False # False by default for empty/error results
        }
        if error_message:
            metadata["error"] = error_message # Include error message in metadata
        
        # Initialize class_data with all expected keys from CLASS_DATA_FIELDS
        empty_class_data = {key: "" for key in self.CLASS_DATA_FIELDS}
        
        return {
            "metadata": metadata,
            "class_data": empty_class_data,
            "event_data": [], # Placeholder
            "task_data": []   # Placeholder
        }

if __name__ == '__main__':
    # Basic logging setup for standalone testing of this module
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    main_logger = logging.getLogger(__name__) # Get the module logger, now configured

    # Test only if OPENAI_API_KEY is available
    if os.getenv("OPENAI_API_KEY"):
        # Example configuration for testing
        parser_test_config = {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"), # Use environment variable
                "client_timeout": {"read": 75.0, "connect": 15.0} # Custom timeouts for test
            },
            "openai_parser": { # Specific settings for this parser
                 "max_chars_for_metadata": 25000, # Slightly reduced for testing
                 "max_api_retries": 1, # Reduce retries for faster standalone tests
                 "critical_fields_check_after_extraction": ["Term", "Course Title", "Instructor Name"] # Example
            }
        }
        
        openai_parser_instance = OpenAIParser(
            model="gpt-4o", # Or test with "gpt-3.5-turbo"
            logger_instance=main_logger, # Pass the configured logger
            config=parser_test_config
        )
        
        sample_syllabus_text = """
        Global Tech University - Department of Computer Science
        CS501: Advanced Algorithms. Term: Spring 2025. Credits: 3.
        Instructor: Dr. Evelyn Reed (eva.reed@gtu.edu), Office: Turing Hall 303
        Class Meetings: Tuesdays and Thursdays, 2:30 PM - 3:45 PM PST (Pacific Standard Time)
        Location: Online via Zoom (Link available on Canvas LMS)
        Course Start Date: January 21, 2025. Last Day of Class: May 1, 2025. Term Ends: May 15, 2025.
        This course delves into complex algorithm design, analysis techniques, and advanced data structures.
        Office Hours: Wednesdays 10:00 AM - 12:00 PM (Virtual via Zoom). Tel: (555) 123-4567
        """
        test_unique_id_main = "adv_algorithms_test_001"
        
        main_logger.info(f"\n--- Testing OpenAIParser.extract() with sample data (ID: {test_unique_id_main}) ---")
        extracted_data_main = openai_parser_instance.extract(sample_syllabus_text, test_unique_id_main)
        main_logger.info(f"Extraction Results for {test_unique_id_main}:\n{json.dumps(extracted_data_main, indent=2)}")

        main_logger.info("\n--- Testing with empty text input ---")
        empty_text_results_test = openai_parser_instance.extract("", "empty_text_test_002")
        main_logger.info(f"Empty Text Input Results:\n{json.dumps(empty_text_results_test, indent=2)}")
        
        main_logger.info("\n--- Testing with OpenAI client not initialized (simulated by no API key) ---")
        # Create an instance with a config that effectively provides no API key
        faulty_parser_config = {"openai": {"api_key": None}} 
        faulty_openai_parser = OpenAIParser(model="gpt-4o", logger_instance=main_logger, config=faulty_parser_config)
        no_client_results_test = faulty_openai_parser.extract(sample_syllabus_text, "no_client_test_003")
        main_logger.info(f"No Client (No API Key) Results:\n{json.dumps(no_client_results_test, indent=2)}")

    else: # If API key is not set
        main_logger.warning("OPENAI_API_KEY environment variable not set. Skipping live API tests for OpenAIParser.")

    main_logger.info("\n--- OpenAIParser standalone tests finished ---")
