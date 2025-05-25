"""
lab_parser.py

Handles extraction and processing of laboratory session information from syllabi.
Uses an LLM for primary extraction from the relevant syllabus text segment.
"""
import re
import logging
import os
import json
from typing import Dict, List, Any, Optional

try:
    from utils.parsers.date_parser import DateParser as SyllabusDateParser
    DATE_PARSER_CLASS_AVAILABLE = True
except ImportError:
    DATE_PARSER_CLASS_AVAILABLE = False
    print("Warning: SyllabusDateParser not found for LabParser. Date parsing will be limited.")

try:
    from utils.helpers import extract_course_code
    EXTRACT_COURSE_CODE_AVAILABLE = True
except ImportError:
    EXTRACT_COURSE_CODE_AVAILABLE = False
    print("Warning: extract_course_code from utils.helpers not found for LabParser.")
    def _fallback_extract_course_code(course_title: Optional[str]) -> str:
        if not course_title or not isinstance(course_title, str): return "COURSE"
        match = re.search(r'([A-Z]{2,4}\s*\d{3,4}[A-Z]?)', course_title, re.IGNORECASE)
        if match: return "".join(match.group(1).split()).upper()
        first_word = course_title.split(' ')[0]
        if re.match(r'^[A-Za-z]{2,4}\d{3,4}[A-Za-z]?$', first_word): return first_word.upper()
        return "COURSE"

try:
    from utils.parsers.openai_parser import OpenAIParser # For LLM calls
    OPENAI_PARSER_CLASS_AVAILABLE = True
except ImportError:
    OPENAI_PARSER_CLASS_AVAILABLE = False
    print("Warning: OpenAIParser not found. LLM-based lab extraction in LabParser will not be available.")


class LabParser:
    """
    Parser dedicated to extracting laboratory session information using an LLM.
    """

    def __init__(self,
                 logger_instance: logging.Logger,
                 date_parser_instance: Optional[SyllabusDateParser] = None,
                 openai_parser_instance: Optional[OpenAIParser] = None, # Added
                 config: Optional[Dict[str, Any]] = None):
        self.logger = logger_instance
        if not self.logger:
            self.logger = logging.getLogger(__name__)
            if not self.logger.hasHandlers(): logging.basicConfig(level=logging.INFO)
            self.logger.critical("LabParser initialized with a default fallback logger.")

        self.config = config or {}
        self.openai_parser = openai_parser_instance

        if not self.openai_parser and OPENAI_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No OpenAIParser instance provided to LabParser. Attempting to create one.")
            model = self.config.get("extraction", {}).get("openai_model", "gpt-4o")
            self.openai_parser = OpenAIParser(model=model, logger_instance=self.logger, config=self.config)
        elif not self.openai_parser:
            self.logger.error("OpenAIParser not available/creatable. LLM-based lab extraction will fail.")

        if date_parser_instance:
            self.date_parser = date_parser_instance
        elif DATE_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No DateParser instance provided. Creating new one for LabParser.")
            self.date_parser = SyllabusDateParser(logger_instance=self.logger)
        else:
            self.date_parser = None
            self.logger.error("DateParser not available. Lab date normalization will be limited/fail.")
        
        self._extract_course_code = extract_course_code if EXTRACT_COURSE_CODE_AVAILABLE else _fallback_extract_course_code
        self.logger.info("LabParser initialized.")

    def _generate_lab_extraction_prompt(self, text_segment: str, course_code: str) -> str:
        """Generates the prompt for the LLM to extract lab sessions."""
        prompt = f"""
You are an expert academic assistant. Your task is to meticulously extract all distinct laboratory session details
from the provided syllabus text segment. The course code is {course_code}.

Input Text Segment (likely a 'Laboratory Schedule' or similar section):
---
{text_segment}
---

Extraction Instructions for Each Lab Session:
1. Identify every distinct lab session or experiment mentioned.
2. For each lab session, extract the following details:
   - "lab_title": (String) The specific title, name, or experiment number of the lab (e.g., "Experiment 1: Free Fall", "Lab 3: DNA Extraction", "Microscopy Lab").
   - "lab_date_str": (String) The date or date range for this lab session, as it appears in the text (e.g., "Sept. 9", "Week of 9/16-9/20", "Mondays"). If no specific date, use "Not Specified".
   - "lab_time_str": (String) The specific time of the lab session if mentioned (e.g., "1:00 PM - 3:50 PM", "2pm"). If no time is specified, use an empty string "".
   - "location_str": (String) The location of the lab (e.g., "Biology Lab room 101", "Chem Annex B"). If no location, use an empty string "".
   - "description_str": (String) Any relevant description, topics covered, pre-lab work, or specific instructions for this lab session.

Output Format:
Respond with a single JSON array, where each element is an object representing a lab session with the fields
"lab_title", "lab_date_str", "lab_time_str", "location_str", and "description_str".
Example of an array element:
{{
  "lab_title": "Experiment 1: Motion in 1D (Free Fall)",
  "lab_date_str": "9/16-20", 
  "lab_time_str": "1:00 PM - 3:50 PM",
  "location_str": "Physics Lab Room 205",
  "description_str": "Investigate the principles of motion under constant acceleration due to gravity."
}}
If no lab sessions are found, return an empty JSON array [].

CRITICAL:
- Extract information ONLY as it is explicitly stated. Do NOT infer or guess details not present.
- If a detail for a field is not present, use an empty string "" for string fields (except for lab_date_str, use "Not Specified" if truly no date).
- Ensure the output is a valid JSON array.
"""
        return prompt

    def extract_labs_from_text(self, syllabus_text_segment: str, class_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.logger.info(f"Attempting to extract labs using LLM from text segment of length {len(syllabus_text_segment)} chars.")
        if not syllabus_text_segment or not syllabus_text_segment.strip():
            self.logger.info("Input text segment for lab extraction is empty.")
            return []
        if not self.openai_parser:
            self.logger.error("OpenAIParser not available in LabParser. Cannot perform LLM-based lab extraction.")
            return []
        if not self.date_parser:
            self.logger.warning("DateParser not available. Extracted lab dates will not be normalized by this method.")

        current_class_data = class_data if isinstance(class_data, dict) else {}
        course_code = self._extract_course_code(current_class_data.get("Course Title", "COURSE"))

        prompt = self._generate_lab_extraction_prompt(syllabus_text_segment, course_code)
        
        llm_response_json_list: Optional[List[Dict[str, Any]]] = None
        try:
            if hasattr(self.openai_parser, 'get_json_list_from_prompt'):
                 llm_response_json_list = self.openai_parser.get_json_list_from_prompt(prompt, f"{course_code}_labs")
            else: # Fallback direct call
                if self.openai_parser.client:
                    response = self.openai_parser.client.chat.completions.create(
                        model=self.openai_parser.model,
                        messages=[{"role": "system", "content": "You are an AI assistant that returns JSON."},
                                  {"role": "user", "content": prompt}],
                        response_format={"type": "json_object"} 
                    )
                    raw_response_text = response.choices[0].message.content.strip()
                    parsed_outer_json = json.loads(raw_response_text)
                    if isinstance(parsed_outer_json, list):
                        llm_response_json_list = parsed_outer_json
                    elif isinstance(parsed_outer_json, dict) and "labs" in parsed_outer_json and isinstance(parsed_outer_json["labs"], list):
                        llm_response_json_list = parsed_outer_json["labs"]
                    else:
                        self.logger.error(f"LLM response for labs was not a list or expected object: {raw_response_text[:500]}")
                else:
                    self.logger.error("OpenAI client not available in LabParser's OpenAIParser instance.")
                    return []
        except Exception as e_llm:
            self.logger.error(f"LLM call for lab extraction failed: {e_llm}", exc_info=True)
            return []

        if not llm_response_json_list:
            self.logger.info("LLM did not return any lab items or response was invalid.")
            return []

        extracted_labs: List[Dict[str, Any]] = []
        for lab_item in llm_response_json_list:
            if not isinstance(lab_item, dict):
                self.logger.warning(f"Skipping non-dict item from LLM lab list: {lab_item}")
                continue

            title = str(lab_item.get("lab_title", "")).strip()
            date_str = str(lab_item.get("lab_date_str", "")).strip()
            time_str = str(lab_item.get("lab_time_str", "")).strip()
            location_str = str(lab_item.get("location_str", "")).strip()
            description_str = str(lab_item.get("description_str", "")).strip()

            if not title or not date_str or date_str.upper() == "NOT SPECIFIED":
                self.logger.warning(f"Skipping lab due to missing title or date_str: {lab_item}")
                continue
            
            # Date normalization is crucial. If lab_date_str is a range or pattern (e.g., "Week of 9/9-13", "Mondays"),
            # this simple normalization won't create individual lab events.
            # This part would need significant enhancement in a full implementation, possibly by
            # integrating with ScheduleParser's logic to expand weekly entries based on term dates.
            # For now, we normalize what DateParser can handle as a single date.
            normalized_date = self.date_parser.normalize_date(date_str) if self.date_parser else date_str
            if normalized_date.upper() in ["TBD", "NOT SPECIFIED"]: normalized_date = "TBD"

            final_title = f"{course_code}: {title}" if not title.lower().startswith(course_code.lower()) else title
            
            lab_entry = {
                "Lab Date": normalized_date, # This might be a string like "Week of 9/9-13" or a normalized single date
                "Lab Title": final_title,
                "Class Time": time_str, # Using "Class Time" to match event_data structure for potential merging
                "Location": location_str,
                "Description": description_str
            }
            extracted_labs.append(lab_entry)
            self.logger.info(f"LLM Extracted Lab: '{final_title}', Date: '{normalized_date}', Time: '{time_str}'")
        
        return extracted_labs

    def process_labs_from_structured_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.debug("Ensuring 'lab_data' key exists and is a list in structured data.")
        if "lab_data" not in data or not isinstance(data["lab_data"], list):
            if "lab_data" in data:
                 self.logger.warning(f"'lab_data' was present but not a list. Re-initializing.")
            data["lab_data"] = []
        
        # TODO: Future: If lab_data contains entries with date patterns (e.g., "Mondays 1-3 PM for first 5 weeks"),
        # this method could expand them into individual dated lab events, using term dates from class_data.
        # This would require robust parsing of recurrence patterns.

        return data

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s', handlers=[logging.StreamHandler()])
    main_logger = logging.getLogger("LabParserTest")

    mock_config = {
        "extraction": {"openai_model": "gpt-4o"},
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")}
    }

    class MockDateParser:
        def normalize_date(self, date_str: str) -> str:
            if "9/9-13" in date_str: return "Week of September 9-13, 2024" # Simulate range
            if "9/16-20" in date_str: return "Week of September 16-20, 2024"
            if "Sept. 9" in date_str: return "September 09, 2024"
            return date_str
    mock_dp = MockDateParser()

    class MockOpenAIParserForLabs:
        def __init__(self, model, logger_instance, config=None): self.model=model; self.logger=logger_instance; self.client=True
        def get_json_list_from_prompt(self, prompt: str, uid: str) -> List[Dict[str,Any]]:
            self.logger.info(f"MockOpenAIParserForLabs received prompt for {uid}")
            if "Tentative Laboratory Schedule" in prompt: # Simulate PHY203 lab schedule
                return [
                    {"lab_title": "Intro lab", "lab_date_str": "9/9-13", "lab_time_str": "", "location_str": "Room 112 (Assumed)", "description_str": "Introduction to laboratory procedures."},
                    {"lab_title": "1.Motion in 1D (free fall)", "lab_date_str": "9/16-20", "lab_time_str": "", "location_str": "Room 112 (Assumed)", "description_str": "Experiment on free fall motion."},
                    {"lab_title": "Make-up Lab (Pendulum)", "lab_date_str": "12/12 (Reading Day)", "lab_time_str": "", "location_str": "", "description_str": "Make-up for Pendulum lab."}
                ]
            return []
    mock_openai_parser_labs = MockOpenAIParserForLabs("gpt-4o", main_logger, mock_config)

    lab_parser_llm = LabParser(
        logger_instance=main_logger,
        date_parser_instance=mock_dp, #type: ignore
        openai_parser_instance=mock_openai_parser_labs, #type: ignore
        config=mock_config
    )
    
    main_logger.info("\n--- Testing LLM-Enhanced LabParser ---")
    phy203_lab_segment = """
    Tentative Laboratory Schedule
    You will be conducting seven experiments. See specific dates below:
    Experiment Week
    Intro lab 9/9-13
    1.Motion in 1D (free fall) 9/16-20
    1.Motion in one 1D (analysis) 9/23-27
    2.Motion in 2D (projectiles) 9/30-10/4
    Make-up Lab (Pendulum) 12/12 (Reading Day)
    """
    phy203_class_data = {"Course Title": "PHY203 Elementary Physics I"}
    extracted_labs = lab_parser_llm.extract_labs_from_text(phy203_lab_segment, phy203_class_data)
    main_logger.info(f"LLM Extracted Labs for PHY203:\n{json.dumps(extracted_labs, indent=2)}")

    main_logger.info("\n--- LabParser standalone tests finished ---")
