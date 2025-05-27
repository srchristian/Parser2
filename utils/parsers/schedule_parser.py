"""
schedule_parser.py

Handles class schedule processing. Generates base class sessions, populates ISO datetimes
and unique IDs, and then uses an LLM to extract detailed topics, readings, 
and activities from the course schedule text segment, merging this information 
into the event_data.
"""

import re
import os
import logging
import sys
import json
import uuid 
from datetime import date, timedelta, datetime, time 
from typing import Dict, List, Any, Optional

try:
    from utils.parsers.date_parser import DateParser as SyllabusDateParser
    DATE_PARSER_CLASS_AVAILABLE = True
except ImportError:
    DATE_PARSER_CLASS_AVAILABLE = False
    # This print is okay for module-level feedback during startup
    print("Warning (schedule_parser.py): SyllabusDateParser not found. ScheduleParser functionality will be limited.")

try:
    from utils.parsers.openai_parser import OpenAIParser 
    OPENAI_PARSER_CLASS_AVAILABLE = True
except ImportError:
    OPENAI_PARSER_CLASS_AVAILABLE = False
    print("Warning (schedule_parser.py): OpenAIParser not found. LLM-based schedule detail extraction will not be available.")

class ScheduleParser:
    """
    Generates a detailed class schedule (event_data) by first creating base class sessions
    with ISO datetimes and unique IDs, and then using an LLM to parse the 
    "course_schedule" text segment for topics, readings, and in-class activities, 
    merging these details.
    """
    
    def __init__(self, 
                 logger_instance: logging.Logger, 
                 date_parser_instance: Optional[SyllabusDateParser] = None, 
                 openai_parser_instance: Optional[OpenAIParser] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.logger = logger_instance
        if not self.logger: # Should ideally not happen if app.py provides one
            self.logger = logging.getLogger(__name__) 
            if not self.logger.hasHandlers(): 
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s')
            self.logger.critical("ScheduleParser initialized with a default fallback logger as no logger_instance was provided.")
            
        self.config = config or {}
        self.openai_parser = openai_parser_instance

        if not self.openai_parser and OPENAI_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No OpenAIParser instance provided to ScheduleParser. Attempting to create one if configured.")
            model = self.config.get("extraction", {}).get("openai_model", "gpt-4o") # Default model
            try:
                self.openai_parser = OpenAIParser(model=model, logger_instance=self.logger, config=self.config)
            except Exception as e_op_init:
                self.logger.error(f"Failed to auto-initialize OpenAIParser in ScheduleParser: {e_op_init}", exc_info=True)
                self.openai_parser = None # Ensure it's None on failure
        elif not self.openai_parser: 
            self.logger.error("OpenAIParser not available/creatable. LLM-based schedule detail extraction will be impaired or fail.")

        if date_parser_instance:
            self.date_parser = date_parser_instance
        elif DATE_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No DateParser instance provided. Creating new one for ScheduleParser.")
            try:
                self.date_parser = SyllabusDateParser(logger_instance=self.logger, config=self.config.get("date_parser"))
            except Exception as e_dp_init:
                self.logger.error(f"Failed to auto-initialize DateParser in ScheduleParser: {e_dp_init}", exc_info=True)
                self.date_parser = None # Ensure it's None on failure
        else:
            self.date_parser = None 
            self.logger.error("DateParser class not available. ScheduleParser functionality severely limited.")
        
        self.default_term_weeks = int(self.config.get("schedule_parser", {}).get("default_term_weeks", 15))
        self.output_date_format = self.date_parser.output_date_format if self.date_parser and hasattr(self.date_parser, 'output_date_format') else "%B %d, %Y"
        self.logger.info(f"ScheduleParser initialized. Default term weeks: {self.default_term_weeks}. Output date format: {self.output_date_format}")
    
    def _generate_schedule_detail_extraction_prompt(self, schedule_text_segment: str, class_data: Dict[str, Any]) -> str:
        course_code = class_data.get("Course Code", "COURSE")
        class_start_date_str = class_data.get("Class Start Date", "Not Specified")
        class_end_date_str = class_data.get("Class End Date", "Not Specified")
        days_of_week_str = class_data.get("Days of Week", "Not Specified")
        
        prompt = f"""
You are an expert academic assistant specializing in parsing course schedules.
Your task is to extract detailed information for each chronological entry (e.g., week, day, session)
from the provided "Course Schedule" text segment.

Course Context:
- Course Code: {course_code}
- Class Start Date: {class_start_date_str}
- Class End Date: {class_end_date_str}
- Regular Meeting Days: {days_of_week_str}

"Course Schedule" Text Segment to Analyze:
---
{schedule_text_segment}
---

Extraction Instructions for Each Schedule Entry:
1. Identify distinct chronological entries.
2. For each entry, extract the following details:
   - "date_reference": (String) The primary specific date or the start date of a range for this entry.
     - If a specific date is given (e.g., "Sept. 11", "Mon, 9/11", "09/11/2024"), extract it in a simple, parsable format like "MM/DD" or "Month Day" or "MM/DD/YYYY" (e.g., "09/11", "Sept 11", "09/11/2024").
     - If a date range is given (e.g., "Week 3 (9/16-9/20)", "Sept 16-20"), extract the START DATE of that range in a simple, parsable format (e.g., "09/16", "Sept 16").
     - If only a relative reference is given (e.g., "Week 1", "First day", "Monday of Week 2"), extract that textual reference as is (e.g., "Week 1").
     - If no date information can be clearly associated with the entry, return an empty string "".
   - "topics": (List of Strings) The main topic(s) or lecture titles for this entry/session(s).
   - "readings": (List of Strings) Required or recommended reading materials (e.g., ["Chapter 1", "Smith et al. (2020) pp. 25-30"]). If none, use an empty list [].
   - "activities_notes": (List of Strings) Brief descriptions of in-class activities, discussions, minor quizzes (not major exams), or other important notes for the session(s). If none, use an empty list [].
   - "is_holiday_or_no_class": (Boolean) True if this entry explicitly states it's a holiday, break, or no class, otherwise false.

Output Format:
Respond with a single JSON array, where each element is an object representing a schedule entry.
Example of an array element:
{{
  "date_reference": "Sept. 11",
  "topics": ["Motion in 1D", "Problem Solving Session"],
  "readings": ["Chapter 1", "Sections 3.1-3.5"],
  "activities_notes": ["HW #1 due", "Review of kinematics"],
  "is_holiday_or_no_class": false
}}
If the schedule segment is empty or no structured entries can be found, return an empty JSON array [].

CRITICAL:
- Focus on information directly related to what happens during or is assigned for specific class sessions or weeks.
- Do NOT re-extract major assignments or exams if they are only mentioned as due dates (e.g., "HW #1 due").
- Ensure the output is a valid JSON array.
"""
        return prompt

    def _extract_specific_date_from_reference(self, date_ref_str: str) -> Optional[str]:
        """
        Attempts to extract a more specific, parsable date part from a date_reference string.
        e.g., "09/09" from "Week 1 (09/09-13)" or "Sept. 11" from "Wed., Sept. 11"
        """
        if not date_ref_str or not isinstance(date_ref_str, str): 
            return None

        # Pattern 1: MM/DD or MM-DD (possibly with year) inside parentheses or as part of week
        # e.g., (9/16-9/20) -> 9/16, Week 3 9/16 -> 9/16, Week 3 (Sept 16 - Sept 20) -> Sept 16
        # Looks for a date pattern that might be the start of a range or a specific day.
        match = re.search(
            r'(?:\bWk\s*\d+\s*\(?|\(|Meeting\s*\d+\s*\(?)\s*' # Optional "Wk X (" or "(" or "Meeting X ("
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{2,4})?|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)' # Date part
            # Optional range part that we don't capture for the return value: (?:[^\S\r\n]*[-–—to][^\S\r\n]*.*)?
            # Optional closing parenthesis: \)?
            , date_ref_str, re.IGNORECASE
        )
        if match and match.group(1):
            extracted = match.group(1).strip()
            self.logger.debug(f"Date regex pattern 1 matched '{extracted}' from '{date_ref_str}'")
            return extracted

        # Pattern 2: Month Day (possibly with year), could be standalone or after a day name
        # e.g., "Sept. 11", "September 11, 2024", "Wed, Sept 11"
        match = re.search(r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{2,4})?)\b', date_ref_str, re.IGNORECASE)
        if match and match.group(1):
            extracted = match.group(1).strip()
            self.logger.debug(f"Date regex pattern 2 matched '{extracted}' from '{date_ref_str}'")
            return extracted
        
        # Pattern 3: Simpler MM/DD or MM-DD-YYYY if it's the main part of the string or clearly separated
        # This is more restrictive to avoid grabbing random numbers.
        match = re.search(r'\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b', date_ref_str)
        if match and match.group(1):
            extracted = match.group(1).strip()
            # Basic validation: check if it looks like a plausible date part
            if len(re.findall(r'\d+', extracted)) >= 2: # At least two numbers (month, day)
                self.logger.debug(f"Date regex pattern 3 matched '{extracted}' from '{date_ref_str}'")
                return extracted

        self.logger.debug(f"No specific date part extracted by regex from date_reference: '{date_ref_str}'. Will attempt to parse original.")
        return date_ref_str # Return original if no specific part is confidently extracted

    def _populate_event_details_with_llm(self, 
                                         event_data_base: List[Dict[str, Any]], 
                                         schedule_text_segment: str, 
                                         class_data: Dict[str, Any],
                                         unique_id: str) -> List[Dict[str, Any]]:
        if not self.date_parser:
            self.logger.error("DateParser not available in _populate_event_details_with_llm. Skipping LLM detail extraction.")
            return event_data_base
        if not schedule_text_segment.strip() or not self.openai_parser:
            self.logger.warning("Schedule text segment empty or OpenAIParser unavailable. Skipping LLM detail extraction.")
            return event_data_base

        prompt = self._generate_schedule_detail_extraction_prompt(schedule_text_segment, class_data)
        llm_schedule_items: Optional[List[Dict[str, Any]]] = None
        try:
            if hasattr(self.openai_parser, 'get_json_list_from_prompt') and callable(getattr(self.openai_parser, 'get_json_list_from_prompt')):
                 llm_schedule_items = self.openai_parser.get_json_list_from_prompt(prompt, f"{class_data.get('Course Code', 'COURSE')}_schedule_details_{unique_id}")
            elif self.openai_parser.client:
                self.logger.debug("Using fallback manual OpenAI call for schedule details in _populate_event_details_with_llm.")
                response = self.openai_parser.client.chat.completions.create(
                    model=self.openai_parser.model,
                    messages=[{"role": "system", "content": "You are an AI assistant that returns JSON formatted data as requested."},
                              {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"} 
                )
                raw_response_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message and response.choices[0].message.content else ""
                if raw_response_text:
                    parsed_outer_json = json.loads(raw_response_text) 
                    if isinstance(parsed_outer_json, list): llm_schedule_items = parsed_outer_json
                    elif isinstance(parsed_outer_json, dict): 
                        found_list = None
                        # Prefer a key like "schedule_entries" or "events" if LLM uses one
                        preferred_keys = ["schedule_entries", "events", "sessions", "items"]
                        for pk in preferred_keys:
                            if pk in parsed_outer_json and isinstance(parsed_outer_json[pk], list):
                                found_list = parsed_outer_json[pk]
                                break
                        if not found_list: # Fallback: find first list value
                            for val in parsed_outer_json.values():
                                if isinstance(val, list): found_list = val; break
                        if found_list is not None: llm_schedule_items = found_list
                        else: self.logger.error(f"LLM response (json_object mode) dict did not contain a clear list of schedule items: {str(parsed_outer_json)[:500]}")
                    else: self.logger.error(f"LLM response for schedule details was not a list or expected dict: {raw_response_text[:500]}")
                else: self.logger.warning("LLM returned empty response for schedule details.")
            else: self.logger.error("OpenAI client not available or get_json_list_from_prompt missing for schedule details.")
        except Exception as e_llm:
            self.logger.error(f"LLM call for schedule detail extraction failed for ID {unique_id}: {e_llm}", exc_info=True)
        
        if not llm_schedule_items:
            self.logger.info(f"LLM did not return schedule items or response invalid for ID {unique_id}.")
            return event_data_base

        event_map_by_date: Dict[str, List[Dict[str, Any]]] = {}
        for event in event_data_base:
            if event.get("Event Date") and event.get("event_id"):
                event_map_by_date.setdefault(event["Event Date"], []).append(event)
        
        course_code = class_data.get("Course Code", "COURSE")
        current_term_context = class_data.get("Term")

        for item in llm_schedule_items:
            if not isinstance(item, dict): continue
            date_ref_str_from_llm = str(item.get("date_reference", "")).strip()
            topics = item.get("topics", [])
            readings = item.get("readings", [])
            activities_notes = item.get("activities_notes", [])
            is_holiday_no_class = item.get("is_holiday_or_no_class", False)

            if not date_ref_str_from_llm: continue
            
            # Attempt to get a more parsable date part from the LLM's reference
            specific_date_to_parse = self._extract_specific_date_from_reference(date_ref_str_from_llm)
            
            target_event_date_str_normalized = None
            parsed_date_obj_for_item = None

            if specific_date_to_parse: # If we extracted something, try to parse it
                try:
                    normalized_specific_date = self.date_parser.normalize_date(specific_date_to_parse, term_year_str=current_term_context)
                    if normalized_specific_date.upper() not in ["TBD", "TBA", "NOT SPECIFIED", ""]:
                        parsed_date_obj_for_item = self.date_parser.parse(normalized_specific_date, ignoretz=True).date()
                        target_event_date_str_normalized = parsed_date_obj_for_item.strftime(self.output_date_format)
                        self.logger.debug(f"Successfully parsed '{specific_date_to_parse}' from '{date_ref_str_from_llm}' to '{target_event_date_str_normalized}'.")
                    else:
                        self.logger.debug(f"Specific date part '{specific_date_to_parse}' normalized to TBD for '{date_ref_str_from_llm}'.")
                except Exception as e_parse_specific:
                    self.logger.warning(f"Could not parse specific date part '{specific_date_to_parse}' (from '{date_ref_str_from_llm}'). Error: {e_parse_specific}")
            
            events_to_update_this_iteration: List[Dict[str, Any]] = []
            if target_event_date_str_normalized and target_event_date_str_normalized in event_map_by_date:
                events_to_update_this_iteration.extend(event_map_by_date[target_event_date_str_normalized])
            else:
                # If direct date match fails, this is where more complex logic for "Week X" or "Mondays" would go.
                # For now, if no specific date is matched, we log and the item might not be applied.
                self.logger.info(f"Date reference '{date_ref_str_from_llm}' (parsed as '{target_event_date_str_normalized}') did not directly map to a base event date. Further logic needed for relative dates.")

            if not events_to_update_this_iteration:
                self.logger.debug(f"No base event session found for LLM item with date_reference: '{date_ref_str_from_llm}'. Details: {item}")
                continue

            for event_to_update in events_to_update_this_iteration:
                current_title = event_to_update.get("Event Title", f"{course_code}: Class")
                if topics:
                    new_topic_str = ", ".join(topics)
                    if current_title == f"{course_code}: Class" or not current_title.startswith(course_code):
                        event_to_update["Event Title"] = f"{course_code}: {new_topic_str}"
                    elif new_topic_str not in current_title: # Avoid duplicate topic appending
                        event_to_update["Event Title"] += f" - {new_topic_str}" 
                
                if readings: event_to_update.setdefault("reading", []).extend(r for r in readings if r not in event_to_update.get("reading",[]))
                if activities_notes: event_to_update.setdefault("special", []).extend(an for an in activities_notes if an not in event_to_update.get("special",[]))
                
                if is_holiday_no_class and parsed_date_obj_for_item: 
                    event_to_update["Event Title"] = f"{course_code}: No Class - {topics[0] if topics else 'Holiday/Break'}"
                    event_to_update.setdefault("special", []).append(topics[0] if topics else "Holiday/Break")
                    event_to_update["Class Time"] = "" 
                    event_to_update["start_datetime_iso"] = parsed_date_obj_for_item.isoformat() 
                    event_to_update["end_datetime_iso"] = (parsed_date_obj_for_item + timedelta(days=1)).isoformat() 
                
                self.logger.debug(f"Updated event on {event_to_update['Event Date']} (ID: {event_to_update.get('event_id')}) with LLM details. New Title: {event_to_update['Event Title']}")
        
        return event_data_base 

    def process_schedule(self, data: Dict[str, Any], unique_id: str) -> Dict[str, Any]:
        self.logger.info(f"Processing class schedule for ID {unique_id} to generate/update event_data...")
        try:
            if not self.date_parser:
                self.logger.error("DateParser unavailable. Cannot process schedule.")
                data.setdefault("event_data", [])
                return data

            class_data = data.get("class_data")
            if not isinstance(class_data, dict):
                self.logger.error("'class_data' missing or invalid. Cannot process schedule.")
                data.setdefault("event_data", [])
                return data

            class_data = self._ensure_class_start_end_dates(class_data) 
            data["class_data"] = class_data 

            final_class_start_str = class_data.get("Class Start Date")
            final_class_end_str = class_data.get("Class End Date")
            days_of_week_str = class_data.get("Days of Week", "")
            course_timezone_str = class_data.get("Time Zone") 
            term_for_context = class_data.get("Term")

            if not course_timezone_str:
                self.logger.warning(f"Time Zone missing in class_data for {unique_id}. ISO datetimes for events will be naive or omitted.")
            
            base_event_data: List[Dict[str, Any]] = []
            if final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED", ""] and \
               final_class_end_str and final_class_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED", ""] and \
               days_of_week_str:
                weekday_indices = self.date_parser.parse_weekdays_to_indices(days_of_week_str)
                if not weekday_indices:
                    self.logger.warning(f"No valid weekdays from '{days_of_week_str}'. Cannot generate base schedule for {unique_id}.")
                else:
                    course_code_from_data = class_data.get("Course Code", "").strip()
                    if not course_code_from_data: # If Course Code is empty, try to extract from Title
                        course_code_from_data = self._extract_course_code(class_data.get("Course Title", "Course"))
                    
                    class_time_str = class_data.get("Class Time", "") 
                    class_location = class_data.get("Class Location", "")
                    base_event_data = self._generate_class_sessions(
                        final_class_start_str, final_class_end_str, weekday_indices, 
                        course_code_from_data, class_time_str, class_location, course_timezone_str,
                        term_year_str=term_for_context 
                    )
            else:
                self.logger.warning(f"Missing critical info (start/end date, days) for base schedule generation for {unique_id}.")

            existing_events = data.get("event_data", [])
            if not isinstance(existing_events, list): existing_events = []
            
            merged_event_data_before_llm = self._merge_schedules(base_event_data, existing_events, term_year_str=term_for_context) 

            schedule_text_segment = data.get("segmented_syllabus", {}).get("course_schedule", "")
            if schedule_text_segment.strip() and self.openai_parser:
                self.logger.info(f"Populating event details using LLM for ID {unique_id} from 'course_schedule' segment (length: {len(schedule_text_segment)}).")
                final_event_data = self._populate_event_details_with_llm(
                    merged_event_data_before_llm, schedule_text_segment, class_data, unique_id
                )
            else:
                self.logger.info(f"Skipping LLM detail population (no schedule text or no OpenAIParser) for ID {unique_id}.")
                final_event_data = merged_event_data_before_llm
            
            if self.date_parser:
                try:
                    final_event_data.sort(key=lambda x: self.date_parser.parse(
                                                self.date_parser.normalize_date(x["Event Date"], term_year_str=term_for_context), 
                                                ignoretz=True).date() 
                                            if x.get("Event Date") and x["Event Date"] != "TBD" 
                                            else date.max)
                except Exception as e_sort:
                    self.logger.error(f"Error sorting final event_data for ID {unique_id}: {e_sort}", exc_info=True)

            data["event_data"] = final_event_data
            self.logger.info(f"Schedule processing complete for ID {unique_id}. Total events: {len(final_event_data)}.")
            return data
        except Exception as e_proc_schedule:
            self.logger.error(f"Error processing schedule for ID {unique_id}: {e_proc_schedule}", exc_info=True)
            data.setdefault("event_data", [])
            return data

    def _ensure_class_start_end_dates(self, class_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.debug("Ensuring 'Class Start Date' and 'Class End Date' are determined and refined.")
        if not self.date_parser: 
            self.logger.error("DateParser not available in _ensure_class_start_end_dates.")
            return class_data

        current_term = class_data.get("Term") 

        term_start_str = self.date_parser.normalize_date(class_data.get("Term Start Date", ""), term_year_str=current_term)
        term_end_str = self.date_parser.normalize_date(class_data.get("Term End Date", ""), term_year_str=current_term)
        class_start_str = self.date_parser.normalize_date(class_data.get("Class Start Date", ""), term_year_str=current_term)
        class_end_str = self.date_parser.normalize_date(class_data.get("Class End Date", ""), term_year_str=current_term)
        
        days_of_week_str = class_data.get("Days of Week", "")

        if class_data.get("Term Start Date", "") != term_start_str : class_data["Term Start Date"] = term_start_str
        if class_data.get("Term End Date", "") != term_end_str : class_data["Term End Date"] = term_end_str
        
        final_class_start_str = class_start_str
        final_class_end_str = class_end_str

        if not final_class_start_str or final_class_start_str.upper() in ["TBD", "TBA", "NOT SPECIFIED", ""]:
            if term_start_str and term_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED", ""]: 
                final_class_start_str = term_start_str
                self.logger.info(f"Using Term Start Date '{term_start_str}' as Class Start Date.")
            else: 
                self.logger.warning("Cannot determine class start date from Term Start Date or existing Class Start Date.")
        
        if not final_class_end_str or final_class_end_str.upper() in ["TBD", "TBA", "NOT SPECIFIED", ""]:
            if term_end_str and term_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED", ""]: 
                final_class_end_str = term_end_str
                self.logger.info(f"Using Term End Date '{term_end_str}' as Class End Date.")
            elif final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED", ""]:
                try:
                    default_dt_for_parse = None
                    if current_term:
                        year_match_term = re.search(r'\b(20\d{2})\b', current_term)
                        if year_match_term: default_dt_for_parse = datetime(int(year_match_term.group(1)),1,1)
                    
                    start_date_obj = self.date_parser.parse(final_class_start_str, default=default_dt_for_parse, ignoretz=True).date()
                    estimated_end_obj = start_date_obj + timedelta(weeks=self.default_term_weeks) - timedelta(days=1) 
                    final_class_end_str = estimated_end_obj.strftime(self.output_date_format)
                    self.logger.info(f"Estimated Class End Date: '{final_class_end_str}' using default term weeks ({self.default_term_weeks}).")
                except Exception as e_est: 
                    self.logger.warning(f"Could not estimate Class End Date: {e_est}", exc_info=True)
            else: 
                self.logger.warning("Cannot determine class end date from Term End Date or by estimation.")

        if final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED", ""] and \
           final_class_end_str and final_class_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED", ""] and \
           days_of_week_str:
            try:
                weekday_indices = self.date_parser.parse_weekdays_to_indices(days_of_week_str)
                if weekday_indices:
                    default_dt_for_refine = None
                    if current_term:
                        year_match_refine = re.search(r'\b(20\d{2})\b', current_term)
                        if year_match_refine: default_dt_for_refine = datetime(int(year_match_refine.group(1)),1,1)

                    start_obj_curr = self.date_parser.parse(final_class_start_str, default=default_dt_for_refine, ignoretz=True).date()
                    end_obj_curr = self.date_parser.parse(final_class_end_str, default=default_dt_for_refine, ignoretz=True).date()
                    
                    if start_obj_curr > end_obj_curr: 
                        self.logger.warning(f"Class Start Date {start_obj_curr} is after Class End Date {end_obj_curr} before refinement. Using original end date as start for refinement logic.")
                        start_obj_curr = end_obj_curr 
                    
                    refined_start = False; temp_date_start = start_obj_curr
                    for _ in range(7): 
                        if temp_date_start.weekday() in weekday_indices: 
                            final_class_start_str = temp_date_start.strftime(self.output_date_format)
                            refined_start = True; break
                        if temp_date_start > end_obj_curr : break 
                        temp_date_start += timedelta(days=1)
                    if refined_start: self.logger.debug(f"Refined Class Start Date to actual first class day: {final_class_start_str}")

                    refined_end = False; temp_date_end = end_obj_curr
                    for _ in range(7): 
                        if temp_date_end.weekday() in weekday_indices: 
                            final_class_end_str = temp_date_end.strftime(self.output_date_format)
                            refined_end = True; break
                        if temp_date_end < start_obj_curr : break 
                        temp_date_end -= timedelta(days=1)
                    if refined_end: self.logger.debug(f"Refined Class End Date to actual last class day: {final_class_end_str}")
            except Exception as e_ref: 
                self.logger.error(f"Error refining class dates based on weekdays: {e_ref}", exc_info=True)

        class_data["Class Start Date"] = final_class_start_str if final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED",""] else ""
        class_data["Class End Date"] = final_class_end_str if final_class_end_str and final_class_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED",""] else ""
        return class_data

    def _generate_class_sessions(self, class_start_date_str: str, class_end_date_str: str, 
                                 weekday_indices: List[int], course_code: str, 
                                 global_class_time_str: str, global_class_location: str,
                                 course_timezone_str: Optional[str], term_year_str: Optional[str] = None) -> List[Dict[str, Any]]:
        self.logger.info(f"Generating regular class sessions for '{course_code}' from {class_start_date_str} to {class_end_date_str}.")
        if not self.date_parser: 
            self.logger.error("DateParser not available in _generate_class_sessions.")
            return []
            
        generated_events: List[Dict[str, Any]] = []
        try:
            default_dt_for_parsing = None
            if term_year_str:
                year_match = re.search(r'\b(20\d{2})\b', term_year_str)
                if year_match:
                    try: default_dt_for_parsing = datetime(int(year_match.group(1)), 1, 1)
                    except ValueError: self.logger.warning(f"Could not parse year from term '{term_year_str}' for session generation default.")
            
            start_obj_date = self.date_parser.parse(class_start_date_str, default=default_dt_for_parsing, ignoretz=True).date()
            end_obj_date = self.date_parser.parse(class_end_date_str, default=default_dt_for_parsing, ignoretz=True).date()
            
            if start_obj_date > end_obj_date:
                self.logger.warning(f"Class start date {start_obj_date} is after end date {end_obj_date} in _generate_class_sessions. No sessions will be generated.")
                return []

            start_time_obj: Optional[time] = None
            end_time_obj: Optional[time] = None
            if global_class_time_str:
                start_time_obj, end_time_obj = self.date_parser.parse_time_string_to_objects(global_class_time_str)

            current_date = start_obj_date
            while current_date <= end_obj_date:
                if current_date.weekday() in weekday_indices:
                    event_id = str(uuid.uuid4())
                    start_datetime_iso = None
                    end_datetime_iso = None

                    if start_time_obj and course_timezone_str:
                        start_datetime_iso = self.date_parser.get_iso_datetime_str(current_date, start_time_obj, course_timezone_str)
                        if end_time_obj:
                            end_datetime_iso = self.date_parser.get_iso_datetime_str(current_date, end_time_obj, course_timezone_str)
                        elif start_datetime_iso: 
                            try:
                                # Attempt to create datetime from ISO, add duration, then format back
                                # This requires pytz for proper timezone handling if start_datetime_iso has offset
                                if self.date_parser.PYTZ_AVAILABLE and isinstance(start_datetime_iso, str):
                                    temp_start_dt_aware = datetime.fromisoformat(start_datetime_iso)
                                    temp_end_dt_aware = temp_start_dt_aware + timedelta(hours=1) # Default 1 hour duration
                                    end_datetime_iso = temp_end_dt_aware.isoformat()
                                else: # Fallback if no pytz or complex ISO string
                                     self.logger.debug(f"Cannot reliably calculate end_datetime_iso without pytz or from '{start_datetime_iso}'.")
                            except ValueError as e_iso_calc:
                                self.logger.warning(f"Could not form default end_datetime_iso from start {start_datetime_iso}: {e_iso_calc}")
                    elif not start_time_obj and course_timezone_str: # All-day event if no time but timezone exists
                        start_datetime_iso = current_date.isoformat()
                        end_datetime_iso = (current_date + timedelta(days=1)).isoformat()


                    generated_events.append({
                        "event_id": event_id,
                        "Event Date": current_date.strftime(self.output_date_format), 
                        "Class Time": global_class_time_str, 
                        "start_datetime_iso": start_datetime_iso, 
                        "end_datetime_iso": end_datetime_iso,   
                        "time_zone": course_timezone_str,       
                        "Event Title": f"{course_code}: Class", 
                        "Class Location": global_class_location,
                        "reading": [], "assignment": [], "assignment_description": None, 
                        "test": [], "special": [], "is_holiday_or_no_class": False 
                    })
                current_date += timedelta(days=1)
            self.logger.info(f"Generated {len(generated_events)} regular class sessions for '{course_code}'.")
        except Exception as e_gen: self.logger.error(f"Error generating class sessions for '{course_code}': {e_gen}", exc_info=True)
        return generated_events
    
    def _merge_schedules(self, generated_sessions: List[Dict[str, Any]], existing_events: List[Dict[str, Any]], term_year_str: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.date_parser: 
            self.logger.error("DateParser unavailable in _merge_schedules.")
            return existing_events + generated_sessions 
        self.logger.debug(f"Merging {len(generated_sessions)} generated sessions with {len(existing_events)} existing events. Context term: {term_year_str}")
        merged_event_map: Dict[str, Dict[str, Any]] = {}

        for session in generated_sessions:
            key = session.get("event_id", session.get("Event Date")) 
            if key: merged_event_map[key] = session

        for event in existing_events:
            if isinstance(event, dict) and event.get("Event Date"):
                norm_date_key = self.date_parser.normalize_date(str(event["Event Date"]), term_year_str=term_year_str)
                
                matched_session_key = None
                for gen_key, gen_session in merged_event_map.items():
                    gen_session_norm_date = self.date_parser.normalize_date(gen_session.get("Event Date",""), term_year_str=term_year_str)
                    if gen_session_norm_date == norm_date_key:
                        matched_session_key = gen_key
                        break
                
                if matched_session_key: 
                    target_session = merged_event_map[matched_session_key]
                    if event.get("Event Title") and (target_session.get("Event Title", "").endswith(": Class") or not target_session.get("Event Title")):
                        target_session["Event Title"] = event["Event Title"]
                    for list_key in ["assignment", "test", "special", "reading"]:
                        target_session.setdefault(list_key, [])
                        if isinstance(event.get(list_key), list):
                            for item_to_add in event[list_key]:
                                if item_to_add not in target_session[list_key]:
                                    target_session[list_key].append(item_to_add)
                    if event.get("assignment_description"):
                        target_session["assignment_description"] = (target_session.get("assignment_description", "") + "; " + event["assignment_description"]).strip("; ")
                    for iso_key in ["start_datetime_iso", "end_datetime_iso", "time_zone", "event_id"]:
                        target_session.setdefault(iso_key, event.get(iso_key)) 

                else: 
                    event.setdefault("event_id", str(uuid.uuid4()))
                    if not event.get("start_datetime_iso") and self.date_parser and event.get("Class Time") and event.get("time_zone") and event.get("Event Date") != "TBD":
                        try:
                            start_t, end_t = self.date_parser.parse_time_string_to_objects(event["Class Time"])
                            # Pass term_year_str to get_iso_datetime_str indirectly via normalize_date inside it
                            event["start_datetime_iso"] = self.date_parser.get_iso_datetime_str(event["Event Date"], start_t, event["time_zone"])
                            event["end_datetime_iso"] = self.date_parser.get_iso_datetime_str(event["Event Date"], end_t, event["time_zone"])
                        except Exception as e_iso_add:
                            self.logger.warning(f"Could not add ISO datetime for existing event '{event.get('Event Title')} on {event['Event Date']}': {e_iso_add}")
                    
                    key_for_existing = event.get("event_id") 
                    merged_event_map[key_for_existing] = event
        
        return list(merged_event_map.values())

    def _extract_course_code(self, course_title: Optional[str]) -> str:
        if not course_title or not isinstance(course_title, str): return "COURSE"
        # Attempt to find common course code patterns (e.g., DEPT 123, DEPT123A)
        match = re.search(r'\b([A-Z]{2,5}\s*\d{2,4}[A-Z]?)\b', course_title, re.IGNORECASE)
        if match: 
            # Normalize: remove spaces and uppercase
            return "".join(match.group(1).split()).upper() 
        
        # Fallback: if no clear pattern, take the first "word" if it looks like a code
        first_word = course_title.split(' ')[0]
        if re.fullmatch(r'[A-Za-z]{2,5}\d{2,4}[A-Za-z]?', first_word): # Stricter match for first word
            return first_word.upper()
        
        # Even simpler fallback: if first word is short, alphanumeric, and not just numbers
        if len(first_word) <= 7 and first_word.isalnum() and not first_word.isdigit(): 
            return first_word.upper()
            
        return "COURSE" # Ultimate fallback

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s', 
                        handlers=[logging.StreamHandler(sys.stdout)])
    main_logger = logging.getLogger(__name__)

    if not DATE_PARSER_CLASS_AVAILABLE or not OPENAI_PARSER_CLASS_AVAILABLE:
        main_logger.critical("DateParser or OpenAIParser class not available. ScheduleParser tests cannot run effectively.")
        sys.exit(1)

    mock_config_main = { # Renamed variable
        "extraction": {"openai_model": "gpt-4o"},
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
        "schedule_parser": {"default_term_weeks": 15},
        "openai_parser": {"max_api_retries": 1, "client_timeout": {"read": 60.0, "connect": 10.0}},
        "date_parser": {"output_date_format": "%B %d, %Y"}
    }
    
    test_date_parser_instance = SyllabusDateParser(logger_instance=main_logger, config=mock_config_main.get("date_parser")) # Renamed
    
    class MockOpenAIParserForScheduleTest(OpenAIParser):
         def get_json_list_from_prompt(self, prompt: str, unique_id: str) -> Optional[List[Dict[str, Any]]]:
            self.logger.info(f"MockOpenAIParserForScheduleTest.get_json_list_from_prompt called for {unique_id}.")
            # Simulate LLM returning more parsable date_references
            if "PHY203" in prompt and "Course Schedule" in prompt: 
                return [
                    {"date_reference": "Sept. 11", "topics": ["Kinematics Review", "Intro to Vectors"], "readings": ["Chapter 1"], "activities_notes": ["HW #1 due today"], "is_holiday_or_no_class": False},
                    {"date_reference": "09/27", "topics": ["Exam #1"], "readings": [], "activities_notes": ["Covers Chaps. 1,3,4."], "is_holiday_or_no_class": False},
                    {"date_reference": "Week 3 (Oct 2 - Oct 6)", "topics": ["Newton's Laws"], "readings": ["Chapter 5"], "activities_notes": [], "is_holiday_or_no_class": False}
                ]
            return []

    test_openai_parser_instance = MockOpenAIParserForScheduleTest(model="gpt-4o", logger_instance=main_logger, config=mock_config_main) # Renamed
    if not os.getenv("OPENAI_API_KEY"): main_logger.warning("OPENAI_API_KEY not set for real OpenAIParser test fallback.")

    schedule_parser_main_instance = ScheduleParser( # Renamed
        logger_instance=main_logger,
        date_parser_instance=test_date_parser_instance, # Use renamed var
        openai_parser_instance=test_openai_parser_instance, # Use renamed var
        config=mock_config_main
    )

    main_logger.info("\n--- Testing ScheduleParser with PHY203 Example (Revised Date Handling) ---")
    phy203_class_data_map = { # Renamed
        "Course Title": "PHY203: ELEMENTARY PHYSICS I", "Course Code": "PHY203",
        "Class Time": "1:00 PM - 1:50 PM", "Time Zone": "America/New_York", 
        "Days of Week": "Monday, Wednesday, Friday",
        "Class Start Date": "September 04, 2024", 
        "Class End Date": "December 11, 2024",   
        "Term": "Fall 2024" 
    }
    phy203_segmented_syllabus_map = { # Renamed
        "course_schedule": "Week 1: Intro. Wed., Sept. 11: Kinematics. Fri., 09/27: Exam #1. Week 3 (Oct 2 - Oct 6): Newton's Laws."
    }
    test_pipeline_input_map = { "class_data": phy203_class_data_map, "segmented_syllabus": phy203_segmented_syllabus_map, "event_data": [] } # Renamed
    
    final_processed_data_map = schedule_parser_main_instance.process_schedule(test_pipeline_input_map, "phy203_revised_date_test") # Renamed
    
    main_logger.info(f"\n--- Final Class Data (Schedule - Revised Date Test) ---\n{json.dumps(final_processed_data_map.get('class_data'), indent=2)}")
    main_logger.info(f"\n--- Final Event Data (Schedule - Revised Date Test) ---\n{json.dumps(final_processed_data_map.get('event_data'), indent=2)}")

    if final_processed_data_map.get('event_data'):
        found_sept_11 = any(event.get("Event Date") == "September 11, 2024" and "Kinematics Review" in event.get("Event Title", "") for event in final_processed_data_map['event_data'])
        found_sept_27 = any(event.get("Event Date") == "September 27, 2024" and "Exam #1" in event.get("Event Title", "") for event in final_processed_data_map['event_data'])
        # Note: "Week 3 (Oct 2 - Oct 6)" might not directly map to a single event if _extract_specific_date_from_reference only gets "Oct 2"
        # and if base events don't perfectly align. This part of the test might need adjustment based on how relative weeks are handled.
        # For now, we check if the specific dates were processed.
        assert found_sept_11, "Event for Sept 11 with Kinematics not found or not updated correctly."
        assert found_sept_27, "Event for Sept 27 with Exam #1 not found or not updated correctly."
        main_logger.info("Basic checks for specific date events passed.")

        for event in final_processed_data_map['event_data']:
            assert "event_id" in event, f"event_id missing in event: {event.get('Event Title')}"
            # Other assertions from previous test can remain
    else:
        main_logger.warning("No event_data generated in revised date handling test to assert fields.")
    main_logger.info("\n--- ScheduleParser revised date handling standalone test finished ---")

