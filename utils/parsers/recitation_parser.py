"""
recitation_parser.py

Handles extraction and processing of recitation or discussion session information.
Uses an LLM for primary extraction from the relevant syllabus text segment,
and then attempts to expand recurring or clearly defined schedules into specific events.
"""
import re
import os
import logging
import json # For parsing LLM response string
from datetime import datetime, date, timedelta # Ensure all are imported
from typing import Dict, List, Any, Optional

# --- Dependency Imports with Fallbacks ---
try:
    from utils.parsers.date_parser import DateParser as SyllabusDateParser
    DATE_PARSER_CLASS_AVAILABLE = True
    try:
        # Attempt to import ParserError directly from dateutil if date_parser was found
        from dateutil.parser import ParserError as DateutilParserError
    except ImportError:
        class DateutilParserError(ValueError): pass # Define a fallback
        if DATE_PARSER_CLASS_AVAILABLE: # Only print if dateutil was expected
             print("Warning: Could not import ParserError from dateutil.parser. Using fallback for DateutilParserError.")
except ImportError:
    DATE_PARSER_CLASS_AVAILABLE = False
    class DateutilParserError(ValueError): pass # Define a fallback if SyllabusDateParser itself is missing
    print("Warning: SyllabusDateParser from utils.parsers.date_parser not found for RecitationParser. Date parsing limited, DateutilParserError fallback defined.")

try:
    from utils.helpers import extract_course_code
    EXTRACT_COURSE_CODE_AVAILABLE = True
except ImportError:
    EXTRACT_COURSE_CODE_AVAILABLE = False
    print("Warning: extract_course_code from utils.helpers not found for RecitationParser. Using internal fallback.")
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
    print("Warning: OpenAIParser not found. LLM-based recitation extraction in RecitationParser will not be available.")


class RecitationParser:
    """
    Parser dedicated to extracting recitation/discussion session information using an LLM
    and processing it into specific event data where possible.
    """

    def __init__(self,
                 logger_instance: logging.Logger,
                 date_parser_instance: Optional[SyllabusDateParser] = None,
                 openai_parser_instance: Optional[OpenAIParser] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.logger = logger_instance
        if not self.logger or (hasattr(self.logger, 'handlers') and not self.logger.handlers): # Check if logger has handlers
            self.logger = logging.getLogger(__name__) # Get a new logger instance
            if not self.logger.handlers: # If still no handlers, configure a basic one
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger.warning("RecitationParser was provided with a logger that had no handlers or was None. Initialized with basic/default logger.")

        self.config = config or {}
        self.openai_parser = openai_parser_instance

        if not self.openai_parser and OPENAI_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No OpenAIParser instance provided to RecitationParser. Attempting to create one.")
            model = self.config.get("extraction", {}).get("openai_model",
                        self.config.get("openai", {}).get("model", "gpt-4o")) # Fallback model
            self.openai_parser = OpenAIParser(model=model, logger_instance=self.logger, config=self.config)
        elif not self.openai_parser:
            self.logger.error("OpenAIParser not available and could not be created. LLM-based recitation extraction will fail.")

        if date_parser_instance:
            self.date_parser = date_parser_instance
        elif DATE_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No DateParser instance provided. Creating new one for RecitationParser.")
            self.date_parser = SyllabusDateParser(logger_instance=self.logger)
        else:
            self.date_parser = None
            self.logger.error("DateParser class is not available and no instance was provided. RecitationParser date operations will be severely limited or fail.")
        
        self._extract_course_code = extract_course_code if EXTRACT_COURSE_CODE_AVAILABLE else _fallback_extract_course_code
        # self.logger.info("RecitationParser initialized.") # This might be redundant if app.py logs it

    def _generate_recitation_extraction_prompt(self, text_segment: str, course_code: str) -> str:
        """Generates the prompt for the LLM to extract recitation/discussion sessions."""
        prompt = f"""
You are an expert academic assistant. Your task is to meticulously extract all distinct recitation, discussion,
or tutorial session details from the provided syllabus text segment. The course code is {course_code}.

Input Text Segment (likely a 'Recitation Schedule', 'Discussion Sections', or similar section):
---
{text_segment}
---

Extraction Instructions for Each Session:
1. Identify every distinct recitation, discussion, or tutorial session mentioned. This might include specific section numbers (e.g., R01, D02), days/times, or topics.
2. For each session, extract the following details:
   - "session_title": (String) A descriptive title for the session (e.g., "Recitation R01: Problem Solving", "Discussion Section 1", "Tutorial on Chapter 3"). If no specific title, use a generic one like "Recitation Session" or "Discussion Group".
   - "session_date_str": (String) The date, date range, or recurring day pattern for this session, as it appears in the text (e.g., "Sept. 6", "Fridays", "Week of 9/9", "Mondays 3-4pm"). If no specific date/pattern, use "Not Specified".
   - "session_time_str": (String) The specific time of the session if mentioned (e.g., "10:00 AM - 10:50 AM", "3-4 PM"). If no time is specified, use an empty string "".
   - "location_str": (String) The location of the session (e.g., "Math Building Room 105", "Online via Zoom"). If no location, use an empty string "".
   - "description_str": (String) Any relevant description, topics covered, TA leading the session, activities, or specific instructions.

Output Format:
Respond with a single JSON object containing a key "sessions" (or "recitations"), whose value is an array of objects. Each object represents a session with the fields "session_title", "session_date_str", "session_time_str", "location_str", and "description_str".
Example:
{{
  "sessions": [
    {{
      "session_title": "Recitation R01: Problem Set 1 Review",
      "session_date_str": "Fridays", 
      "session_time_str": "10:00 AM - 10:50 AM",
      "location_str": "Math Building Room 105",
      "description_str": "Focus on problems from PS1. TA: John Doe."
    }}
  ]
}}
If no recitation/discussion sessions are found, the array should be empty, e.g., {{"sessions": []}}.

CRITICAL:
- Extract information ONLY as it is explicitly stated. Do NOT infer or guess details not present.
- If a detail for a field is not present, use an empty string "" for string fields (except for session_date_str, use "Not Specified" if truly no date/pattern).
- Ensure the output is a valid JSON object containing the specified array.
"""
        return prompt

    def extract_recitations_from_text(self, syllabus_text_segment: str, class_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.logger.info(f"Attempting to extract recitations using LLM from text segment of length {len(syllabus_text_segment)} chars.")
        if not syllabus_text_segment or not syllabus_text_segment.strip():
            self.logger.info("Input text segment for recitation extraction is empty.")
            return []
        if not self.openai_parser or not self.openai_parser.client: # Check client too
            self.logger.error("OpenAIParser or its client not available in RecitationParser. Cannot perform LLM-based extraction.")
            return []

        current_class_data = class_data if isinstance(class_data, dict) else {}
        course_code = self._extract_course_code(current_class_data.get("Course Title", "COURSE"))
        prompt = self._generate_recitation_extraction_prompt(syllabus_text_segment, course_code)
        
        llm_response_json_list: List[Dict[str, Any]] = [] # Initialize to empty list
        raw_response_text = "" # For logging in case of error
        try:
            response = self.openai_parser.client.chat.completions.create(
                model=self.openai_parser.model,
                messages=[{"role": "system", "content": "You are an AI assistant that returns structured JSON output."}, # Slightly more generic system role
                            {"role": "user", "content": prompt}],
                response_format={"type": "json_object"} 
            )
            raw_response_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message and response.choices[0].message.content else ""
            if not raw_response_text:
                self.logger.warning("LLM returned an empty response for recitations.")
                return []

            self.logger.debug(f"RecitationParser LLM Raw Response: {raw_response_text[:500]}")
            parsed_outer_json = json.loads(raw_response_text)

            if isinstance(parsed_outer_json, dict):
                if "recitations" in parsed_outer_json and isinstance(parsed_outer_json["recitations"], list):
                    llm_response_json_list = parsed_outer_json["recitations"]
                elif "sessions" in parsed_outer_json and isinstance(parsed_outer_json["sessions"], list):
                    llm_response_json_list = parsed_outer_json["sessions"]
                else:
                    self.logger.error(f"LLM response dict for recitations missing 'recitations' or 'sessions' list key. Keys: {list(parsed_outer_json.keys())}")
            elif isinstance(parsed_outer_json, list): # If LLM returns a direct list (less likely with json_object type)
                 llm_response_json_list = parsed_outer_json
            else:
                self.logger.error(f"LLM response for recitations was not a dict or list as expected: Type {type(parsed_outer_json)}")
        
        except json.JSONDecodeError as e_json:
            self.logger.error(f"Failed to decode JSON from LLM response for recitations: {e_json}. Response text: {raw_response_text[:500]}...", exc_info=True)
        except Exception as e_llm:
            self.logger.error(f"LLM call or processing for recitation extraction failed: {e_llm}", exc_info=True)

        if not llm_response_json_list:
            self.logger.info("No valid recitation items extracted from LLM response.")
            return []

        extracted_recitations: List[Dict[str, Any]] = []
        for item in llm_response_json_list:
            if not isinstance(item, dict):
                self.logger.warning(f"Skipping non-dict item from LLM recitation list: {item}")
                continue

            title = str(item.get("session_title", "")).strip()
            date_str = str(item.get("session_date_str", "")).strip()
            time_str = str(item.get("session_time_str", "")).strip()
            location_str = str(item.get("location_str", "")).strip()
            description_str = str(item.get("description_str", "")).strip()

            if not title or not date_str or date_str.upper() == "NOT SPECIFIED":
                self.logger.debug(f"Skipping recitation due to missing title or essential date_str: {item}")
                continue
            
            final_title = f"{course_code}: {title}" if not title.lower().startswith(course_code.lower()) and not any(k in title.lower() for k in ["recitation", "discussion", "tutorial", "section"]) else title
            if not title and course_code: # Default title generation
                final_title = f"{course_code}: Recitation/Discussion"
            
            rec_entry = {
                "Recitation Date String": date_str, # Store the raw string for later processing
                "Recitation Title": final_title,
                "Class Time": time_str, # Use "Class Time" for consistency with event_data structure
                "Location": location_str,
                "Description": description_str
            }
            extracted_recitations.append(rec_entry)
            self.logger.info(f"LLM Extracted Recitation (Raw): '{final_title}', Date Str: '{date_str}', Time: '{time_str}'")
        
        return extracted_recitations

    def process_recitations_from_structured_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Processing structured recitation data to expand into event_data for course: {data.get('class_data', {}).get('Course Code', 'N/A')}")
        if not self.date_parser:
            self.logger.error("DateParser not available. Cannot process structured recitation data for event expansion.")
            data.setdefault("recitation_data", []) # Ensure key exists
            return data

        raw_recitation_entries = data.get("recitation_data", [])
        if not isinstance(raw_recitation_entries, list):
            self.logger.warning(f"'recitation_data' is not a list (type: {type(raw_recitation_entries)}). Initializing as empty list.")
            data["recitation_data"] = []
            return data # Return early if no valid raw entries to process

        unresolved_recitation_entries = [] 
        newly_generated_recitation_events: List[Dict[str, Any]] = []

        class_data = data.get("class_data", {})
        class_start_str = class_data.get("Class Start Date")
        class_end_str = class_data.get("Class End Date")
        course_code = self._extract_course_code(class_data.get("Course Title", "COURSE"))
        
        default_year_dt: Optional[datetime] = None
        term_str = class_data.get("Term", "")
        if term_str:
            year_match = re.search(r'\b(20\d{2})\b', term_str) # Look for a 4-digit year
            if year_match:
                try: default_year_dt = datetime(int(year_match.group(1)), 1, 1) # Use Jan 1 of that year as default
                except ValueError: self.logger.warning(f"Could not parse year from term string: {term_str}")

        if not (class_start_str and class_end_str and 
                class_start_str.upper() not in ["TBD", "NOT SPECIFIED"] and 
                class_end_str.upper() not in ["TBD", "NOT SPECIFIED"]):
            self.logger.warning("Class Start Date or Class End Date not available/valid in class_data. Cannot reliably expand recurring recitation schedules. All raw recitation entries will be flagged for review.")
            data["recitation_data"] = raw_recitation_entries # Keep all for user review
            if raw_recitation_entries: # If there are entries to flag
                 data.setdefault("metadata", {}).setdefault("unresolved_recitation_schedules", 
                    [f"{r.get('Recitation Title', 'Unknown Recitation')} (Date String: {r.get('Recitation Date String', 'N/A')})" for r in raw_recitation_entries])
            return data

        for rec_entry in raw_recitation_entries:
            if not isinstance(rec_entry, dict): continue

            original_date_str = str(rec_entry.get("Recitation Date String", "")).strip()
            rec_title = str(rec_entry.get("Recitation Title", f"{course_code}: Recitation")).strip()
            rec_time = str(rec_entry.get("Class Time", "")).strip() # Using "Class Time" to match event structure
            rec_location = str(rec_entry.get("Location", "")).strip()
            rec_description = str(rec_entry.get("Description", "")).strip()

            if not original_date_str or original_date_str.upper() == "NOT SPECIFIED":
                self.logger.debug(f"Recitation '{rec_title}' has no specified date string. Adding to unresolved.")
                unresolved_recitation_entries.append(rec_entry)
                continue
            
            processed_as_event = False
            # Attempt 1: Parse as a single, specific date
            try:
                normalized_single_date_str = self.date_parser.normalize_date(original_date_str, default_datetime=default_year_dt)
                # Test if it's a valid, non-TBD date after normalization
                if normalized_single_date_str.upper() not in ["TBD", "NOT SPECIFIED"]:
                    parsed_dt_object = self.date_parser.date_parser_module.parse(normalized_single_date_str, ignoretz=True).date() # Throws error if not parsable
                    
                    new_event = {
                        "Event Date": parsed_dt_object.strftime(self.date_parser.output_date_format),
                        "Event Title": rec_title, "Class Time": rec_time, "Class Location": rec_location,
                        "reading": [], "assignment": [], "assignment_description": None, "test": [],
                        "special": [rec_description] if rec_description else [], "Type": "Recitation"
                    }
                    newly_generated_recitation_events.append(new_event)
                    self.logger.info(f"Scheduled specific recitation event: '{rec_title}' on {new_event['Event Date']}")
                    processed_as_event = True
            except (DateutilParserError, ValueError, TypeError, AttributeError) as e_single_date:
                self.logger.debug(f"'{original_date_str}' for recitation '{rec_title}' is not a simple single date (Error: {e_single_date}). Checking for recurring patterns.")
            
            if processed_as_event: continue

            # Attempt 2: Parse as a recurring day pattern (e.g., "Fridays", "Tuesdays 1-3pm")
            try:
                # DateParser.parse_weekdays_to_indices should handle strings like "Fridays", "Mon", "Tuesdays and Thursdays"
                weekday_indices = self.date_parser.parse_weekdays_to_indices(original_date_str)
                if weekday_indices: # If it's a recognizable day pattern
                    # DateParser.get_class_dates uses these indices and term start/end to list all actual dates
                    specific_rec_dates = self.date_parser.get_class_dates(class_start_str, class_end_str, original_date_str, default_datetime=default_year_dt)
                    if specific_rec_dates:
                        for specific_date_obj in specific_rec_dates:
                            new_event = {
                                "Event Date": specific_date_obj.strftime(self.date_parser.output_date_format),
                                "Event Title": rec_title, "Class Time": rec_time, "Class Location": rec_location,
                                "reading": [], "assignment": [], "assignment_description": None, "test": [],
                                "special": [rec_description] if rec_description else [], "Type": "Recitation"
                            }
                            newly_generated_recitation_events.append(new_event)
                        self.logger.info(f"Expanded recurring recitation '{rec_title}' (pattern: '{original_date_str}') into {len(specific_rec_dates)} events.")
                        processed_as_event = True
                    else: # Pattern recognized but couldn't expand (e.g., term dates issue, or pattern was complex like "First Monday")
                        self.logger.warning(f"Recognized day pattern '{original_date_str}' for recitation '{rec_title}' but could not expand into specific dates. Flagging for review.")
                # else: Not a simple recurring pattern DateParser can handle directly via parse_weekdays_to_indices
            except Exception as e_expand: # Catch errors during pattern expansion
                self.logger.error(f"Error attempting to expand recurring recitation pattern '{original_date_str}' for '{rec_title}': {e_expand}", exc_info=True)
            
            if not processed_as_event: # If not a specific date and not a simple recurring pattern
                self.logger.info(f"Recitation '{rec_title}' with date string '{original_date_str}' is a range or complex pattern requiring user clarification. Adding to unresolved.")
                unresolved_recitation_entries.append(rec_entry)

        # Update the main data structure
        data["recitation_data"] = unresolved_recitation_entries # Contains only recitations needing review
        # Add newly generated specific recitation events to the main event_data list
        data.setdefault("event_data", []).extend(newly_generated_recitation_events) 
        
        if unresolved_recitation_entries:
            data.setdefault("metadata", {}).setdefault("unresolved_recitation_schedules", 
                [f"{r.get('Recitation Title', 'Unknown Recitation')} (Date String: {r.get('Recitation Date String', 'N/A')})" for r in unresolved_recitation_entries])
        else: # Clear the flag if all were resolved
            if data.get("metadata"): data["metadata"].pop("unresolved_recitation_schedules", None)
            
        return data

# Standalone testing block
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s', handlers=[logging.StreamHandler()])
    main_logger = logging.getLogger("RecitationParserTest")

    # Mock or initialize DateParser and OpenAIParser for testing
    mock_config = { "extraction": {"openai_model": "gpt-4o"}, "openai": {"api_key": os.getenv("OPENAI_API_KEY")} }
    
    date_parser_for_test = None
    if DATE_PARSER_CLASS_AVAILABLE:
        date_parser_for_test = SyllabusDateParser(logger_instance=main_logger)
    else: # Minimal mock if real one isn't available
        class MockDateParser:
            output_date_format = "%B %d, %Y"
            def __init__(self, li): self.logger = li
            def normalize_date(self, ds, default_datetime=None): return ds if ds.upper() != "NOT SPECIFIED" else "TBD"
            def parse_weekdays_to_indices(self, ds): return [4] if "fri" in ds.lower() else [] # Mock Fridays
            def get_class_dates(self, s, e, dps, default_datetime=None): return [date(2024,9,6), date(2024,9,13)] if "fri" in dps.lower() else []
            class date_parser_module: # Mock the inner attribute for .parse().date()
                @staticmethod
                def parse(date_string, ignoretz=False, default=None): return datetime.strptime(date_string, "%B %d, %Y") if "," in date_string else datetime.strptime(date_string, "%m/%d/%Y")

        date_parser_for_test = MockDateParser(main_logger) # type: ignore

    openai_parser_for_test = None
    if OPENAI_PARSER_CLASS_AVAILABLE:
        openai_parser_for_test = OpenAIParser(model="gpt-4o", logger_instance=main_logger, config=mock_config)
    # If OpenAIParser is not available, tests requiring LLM calls will be skipped or use a more elaborate mock if defined.

    rec_parser_instance = RecitationParser(
        logger_instance=main_logger,
        date_parser_instance=date_parser_for_test, # type: ignore
        openai_parser_instance=openai_parser_for_test,
        config=mock_config
    )

    main_logger.info("\n--- Testing RecitationParser ---")
    sample_rec_text = """
    Discussion Section R01: Fridays, 10:00 AM - 10:50 AM, Room MTH 105. Focus on Problem Set 1.
    Tutorial for Chapter 3: Week of Sept 9th. Check Brightspace for exact day/time.
    No recitation on Sept 20th (Holiday).
    """
    sample_class_data = {"Course Title": "MATH101 Calculus", "Term": "Fall 2024", 
                         "Class Start Date": "September 4, 2024", "Class End Date": "December 11, 2024"}
    
    if rec_parser_instance.openai_parser: # Only run LLM dependent test if parser is available
        main_logger.info("--- Testing extract_recitations_from_text (LLM based) ---")
        # This test would ideally mock the self.openai_parser.client.chat.completions.create call
        # For now, it will try a live call if API key is set, or fail if not.
        extracted_raw_recitations = rec_parser_instance.extract_recitations_from_text(sample_rec_text, sample_class_data)
        main_logger.info(f"LLM Extracted Raw Recitations:\n{json.dumps(extracted_raw_recitations, indent=2)}")
        
        main_logger.info("\n--- Testing process_recitations_from_structured_data with LLM output ---")
        test_pipeline_data = {
            "class_data": sample_class_data,
            "recitation_data": extracted_raw_recitations, # Use output from LLM extraction
            "event_data": [],
            "metadata": {}
        }
        processed_results = rec_parser_instance.process_recitations_from_structured_data(test_pipeline_data)
        main_logger.info(f"Final event_data after processing recitations:\n{json.dumps(processed_results.get('event_data'), indent=2)}")
        main_logger.info(f"Remaining recitation_data (unresolved):\n{json.dumps(processed_results.get('recitation_data'), indent=2)}")
        main_logger.info(f"Metadata flags:\n{json.dumps(processed_results.get('metadata'), indent=2)}")
    else:
        main_logger.warning("Skipping LLM-dependent tests for RecitationParser as OpenAIParser is not available/configured.")

    main_logger.info("\n--- RecitationParser standalone tests finished ---")
