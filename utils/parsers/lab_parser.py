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
            try: # Ensure self.openai_parser is assigned after successful creation
                self.openai_parser = OpenAIParser(model=model, logger_instance=self.logger, config=self.config)
            except Exception as e_op_init_lp:
                self.logger.error(f"Failed to auto-initialize OpenAIParser in LabParser: {e_op_init_lp}", exc_info=True)
                self.openai_parser = None # Ensure it's None on failure
        elif not self.openai_parser:
            self.logger.error("OpenAIParser not available/creatable. LLM-based lab extraction will fail.")

        if date_parser_instance:
            self.date_parser = date_parser_instance
        elif DATE_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No DateParser instance provided. Creating new one for LabParser.")
            try: # Ensure self.date_parser is assigned
                self.date_parser = SyllabusDateParser(logger_instance=self.logger, config=self.config.get("date_parser"))
            except Exception as e_dp_init_lp:
                self.logger.error(f"Failed to auto-initialize DateParser in LabParser: {e_dp_init_lp}", exc_info=True)
                self.date_parser = None # Ensure it's None on failure
        else:
            self.date_parser = None
            self.logger.error("DateParser not available. Lab date normalization will be limited/fail.")
        
        self._extract_course_code = extract_course_code if EXTRACT_COURSE_CODE_AVAILABLE else _fallback_extract_course_code
        self.logger.info("LabParser initialized.")

    def _generate_lab_extraction_prompt(self, text_segment: str, course_code: str) -> str:
        """Generates the prompt for the LLM to extract lab sessions."""
        # It might be beneficial to instruct the LLM to use a specific key if json_object mode is anticipated.
        # For example: "Respond with a single JSON object containing a key 'lab_sessions' whose value is an array..."
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
If you are an AI model that must return a JSON object as the top-level structure,
return a JSON object with a single key "lab_sessions". The value of "lab_sessions"
must be an array of objects, where each object represents a lab session with the fields
"lab_title", "lab_date_str", "lab_time_str", "location_str", and "description_str".

Example:
{{
  "lab_sessions": [
    {{
      "lab_title": "Experiment 1: Motion in 1D (Free Fall)",
      "lab_date_str": "9/16-20", 
      "lab_time_str": "1:00 PM - 3:50 PM",
      "location_str": "Physics Lab Room 205",
      "description_str": "Investigate the principles of motion under constant acceleration due to gravity."
    }}
  ]
}}

If no lab sessions are found, the "lab_sessions" array should be empty: {{"lab_sessions": []}}.
If you are an AI model that can return a JSON array directly as the top-level structure, then return just the array.

CRITICAL:
- Extract information ONLY as it is explicitly stated. Do NOT infer or guess details not present.
- If a detail for a field is not present, use an empty string "" for string fields (except for lab_date_str, use "Not Specified" if truly no date).
- Ensure your entire output is a single, valid JSON structure as specified.
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
        if not self.date_parser: # Check if date_parser was successfully initialized
            self.logger.warning("DateParser not available in LabParser. Extracted lab dates will not be normalized by this method.")
            # Continue without date normalization if date_parser is None

        current_class_data = class_data if isinstance(class_data, dict) else {}
        course_code = self._extract_course_code(current_class_data.get("Course Title", "COURSE"))

        prompt = self._generate_lab_extraction_prompt(syllabus_text_segment, course_code)
        
        llm_response_json_list: Optional[List[Dict[str, Any]]] = None
        try:
            # Try using get_json_list_from_prompt if available (preferred)
            if hasattr(self.openai_parser, 'get_json_list_from_prompt') and callable(getattr(self.openai_parser, 'get_json_list_from_prompt')):
                 self.logger.debug("LabParser using OpenAIParser.get_json_list_from_prompt.")
                 llm_response_json_list = self.openai_parser.get_json_list_from_prompt(prompt, f"{course_code}_labs")
            # Fallback to direct client call if get_json_list_from_prompt is not available or fails implicitly
            elif hasattr(self.openai_parser, 'client') and self.openai_parser.client:
                self.logger.debug("LabParser using direct OpenAI client call (response_format: json_object).")
                response = self.openai_parser.client.chat.completions.create(
                    model=self.openai_parser.model,
                    messages=[{"role": "system", "content": "You are an AI assistant. Respond strictly with the JSON structure requested by the user."}, # System prompt refined
                              {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"} 
                )
                raw_response_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message and response.choices[0].message.content else ""
                if raw_response_text:
                    self.logger.debug(f"LabParser LLM Raw Response (direct call): {raw_response_text[:1000]}") # Log more of the response
                    parsed_outer_json = json.loads(raw_response_text)
                    if isinstance(parsed_outer_json, list): # Should not happen with json_object mode
                        llm_response_json_list = parsed_outer_json
                        self.logger.warning("LLM returned a direct list in json_object mode, which is unexpected but handled.")
                    elif isinstance(parsed_outer_json, dict):
                        if "lab_sessions" in parsed_outer_json and isinstance(parsed_outer_json["lab_sessions"], list):
                            llm_response_json_list = parsed_outer_json["lab_sessions"]
                        elif "labs" in parsed_outer_json and isinstance(parsed_outer_json["labs"], list): # Keep previous fallback
                            llm_response_json_list = parsed_outer_json["labs"]
                            self.logger.info("LLM response for labs was a dict with 'labs' key.")
                        else: # Try to find any list if keys don't match
                            found_list_in_dict = None
                            for key, value in parsed_outer_json.items():
                                if isinstance(value, list):
                                    found_list_in_dict = value
                                    self.logger.info(f"Found list under unexpected key '{key}' in LLM response for labs.")
                                    break
                            if found_list_in_dict is not None:
                                llm_response_json_list = found_list_in_dict
                            else:
                                self.logger.error(f"LLM response for labs was a dict but did not contain 'lab_sessions', 'labs', or any other parsable list: {raw_response_text[:500]}")
                    else:
                        self.logger.error(f"LLM response for labs after JSON parsing was not a list or dict: Type={type(parsed_outer_json)}, Content: {raw_response_text[:500]}")
                else:
                    self.logger.warning(f"LLM returned empty response for lab details (direct call).")
            else:
                self.logger.error("OpenAIParser client not available or method get_json_list_from_prompt missing and no client. Cannot extract labs.")
                return []
        except json.JSONDecodeError as e_json_decode:
            self.logger.error(f"JSONDecodeError for lab extraction. Raw text: '{raw_response_text if 'raw_response_text' in locals() else 'N/A'}' Error: {e_json_decode}", exc_info=True)
            return []
        except Exception as e_llm:
            self.logger.error(f"LLM call or processing for lab extraction failed: {e_llm}", exc_info=True)
            return []

        if not llm_response_json_list: # Check if it's still None or empty
            self.logger.info("LLM did not return any lab items or response was invalid after all parsing attempts.")
            return []
        if not isinstance(llm_response_json_list, list): # Final check
            self.logger.error(f"Processed LLM response for labs is not a list. Type: {type(llm_response_json_list)}. Skipping lab processing.")
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
                self.logger.debug(f"Skipping lab due to missing title or 'Not Specified' date_str: {lab_item}")
                continue
            
            normalized_date = date_str # Default if DateParser is not available or fails
            if self.date_parser: # Check if date_parser was successfully initialized
                # Pass term context if available from class_data
                term_context = current_class_data.get("Term")
                normalized_date = self.date_parser.normalize_date(date_str, term_year_str=term_context)
            
            if normalized_date.upper() in ["TBD", "NOT SPECIFIED"]: 
                normalized_date = "TBD" # Standardize

            final_title = f"{course_code}: {title}" if not title.lower().startswith(course_code.lower()) and course_code != "COURSE" else title
            
            lab_entry = {
                "Lab Date": normalized_date, 
                "Lab Title": final_title,
                "Class Time": time_str, 
                "Location": location_str,
                "Description": description_str,
                # Add a raw date string for reference if normalization changes it significantly
                "Raw Lab Date String": date_str if normalized_date != date_str else None 
            }
            # Remove Raw Lab Date String if it's None
            if lab_entry["Raw Lab Date String"] is None:
                del lab_entry["Raw Lab Date String"]

            extracted_labs.append(lab_entry)
            self.logger.info(f"LLM Extracted Lab: '{final_title}', Date: '{normalized_date}', Time: '{time_str}'")
        
        return extracted_labs

    def process_labs_from_structured_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.debug("Ensuring 'lab_data' key exists and is a list in structured data.")
        if "lab_data" not in data or not isinstance(data["lab_data"], list):
            if "lab_data" in data:
                 self.logger.warning(f"'lab_data' was present but not a list ({type(data['lab_data'])}). Re-initializing to empty list.")
            data["lab_data"] = []
        
        # TODO: Future: If lab_data contains entries with date patterns (e.g., "Mondays 1-3 PM for first 5 weeks"),
        # this method could expand them into individual dated lab events, using term dates from class_data.
        # This would require robust parsing of recurrence patterns and integration with ScheduleParser logic.

        return data

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s', handlers=[logging.StreamHandler()])
    main_logger = logging.getLogger("LabParserTest")

    mock_config_main = { # Renamed to avoid conflict
        "extraction": {"openai_model": "gpt-4o"},
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
        "date_parser": {"output_date_format": "%B %d, %Y"} # Config for DateParser
    }

    # Mock DateParser for testing LabParser independently
    mock_dp_instance = None
    if DATE_PARSER_CLASS_AVAILABLE:
        mock_dp_instance = SyllabusDateParser(logger_instance=main_logger, config=mock_config_main.get("date_parser"))
    else: # Fallback mock if SyllabusDateParser class itself is unavailable
        class MockDateParserForLabTest:
            def normalize_date(self, date_str: str, term_year_str: Optional[str] = None) -> str:
                main_logger.debug(f"MockDateParserForLabTest.normalize_date called with: '{date_str}', term: '{term_year_str}'")
                if "9/9-13" in date_str: return "Week of September 9-13, 2024" 
                if "9/16-20" in date_str: return "Week of September 16-20, 2024"
                if "Sept. 9" in date_str: return "September 09, 2024"
                if "12/12" in date_str: return "December 12, 2024"
                return date_str
        mock_dp_instance = MockDateParserForLabTest() #type: ignore

    # Mock OpenAIParser for testing LabParser
    mock_openai_parser_labs_instance = None
    if OPENAI_PARSER_CLASS_AVAILABLE:
        class MockOpenAIParserForLabs(OpenAIParser):
            def get_json_list_from_prompt(self, prompt: str, uid: str) -> Optional[List[Dict[str,Any]]]:
                self.logger.info(f"MockOpenAIParserForLabs.get_json_list_from_prompt received prompt for {uid}. Prompt starts: {prompt[:200]}")
                # Simulate the LLM returning the problematic dict structure
                if "Tentative Laboratory Schedule" in prompt:
                    self.logger.info(f"MockOpenAIParserForLabs: Matched PHY203 lab segment for {uid}. Returning dict.")
                    return {"lab_sessions": [ # This is the structure seen in the logs
                        {"lab_title": "Intro lab", "lab_date_str": "9/9-13", "lab_time_str": "", "location_str": "Room 112 (Assumed)", "description_str": "Introduction to laboratory procedures."},
                        {"lab_title": "1.Motion in 1D (free fall)", "lab_date_str": "9/16-20", "lab_time_str": "", "location_str": "Room 112 (Assumed)", "description_str": "Experiment on free fall motion."},
                        {"lab_title": "Make-up Lab (Pendulum)", "lab_date_str": "12/12 (Reading Day)", "lab_time_str": "", "location_str": "", "description_str": "Make-up for Pendulum lab."}
                    ]} # type: ignore 
                    # The above line is intentionally returning a dict to test the fix, 
                    # but get_json_list_from_prompt should return List. We are testing the calling code's robustness.
                    # In a real scenario, get_json_list_from_prompt should handle this unwrapping.
                return [] # Default empty list
        mock_openai_parser_labs_instance = MockOpenAIParserForLabs("gpt-4o", main_logger, mock_config_main) # type: ignore
    else:
        main_logger.error("OpenAIParser class not available. Cannot run effective LabParser LLM tests.")


    lab_parser_llm_instance = LabParser( # Renamed
        logger_instance=main_logger,
        date_parser_instance=mock_dp_instance, 
        openai_parser_instance=mock_openai_parser_labs_instance, 
        config=mock_config_main
    )
    
    main_logger.info("\n--- Testing LLM-Enhanced LabParser ---")
    phy203_lab_segment_text = """
    Tentative Laboratory Schedule
    You will be conducting seven experiments. See specific dates below:
    Experiment Week
    Intro lab 9/9-13
    1.Motion in 1D (free fall) 9/16-20
    1.Motion in one 1D (analysis) 9/23-27
    2.Motion in 2D (projectiles) 9/30-10/4
    Make-up Lab (Pendulum) 12/12 (Reading Day)
    """
    phy203_class_data_map = {"Course Title": "PHY203 Elementary Physics I", "Term": "Fall 2024"} # Added Term
    
    if mock_openai_parser_labs_instance: # Only run if mock is available
        extracted_labs_list = lab_parser_llm_instance.extract_labs_from_text(phy203_lab_segment_text, phy203_class_data_map)
        main_logger.info(f"LLM Extracted Labs for PHY203:\n{json.dumps(extracted_labs_list, indent=2)}")
        assert len(extracted_labs_list) == 3, f"Expected 3 labs, got {len(extracted_labs_list)}"
        assert extracted_labs_list[0]["Lab Title"] == "PHY203: Intro lab", "First lab title mismatch"
    else:
        main_logger.warning("Skipping LLM extraction test for labs as OpenAIParser mock is not fully available.")

    main_logger.info("\n--- LabParser standalone tests finished ---")

