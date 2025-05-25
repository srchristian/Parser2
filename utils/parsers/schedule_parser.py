"""
schedule_parser.py

Handles class schedule processing. Generates base class sessions and then uses an LLM
to extract detailed topics, readings, and activities from the course schedule text segment,
merging this information into the event_data.
"""

import re
import os
import logging
import sys
import json
from datetime import date, timedelta, datetime # Added datetime for completeness if needed
from typing import Dict, List, Any, Optional

try:
    # Assuming SyllabusDateParser is the actual class name in date_parser.py
    from utils.parsers.date_parser import DateParser as SyllabusDateParser
    DATE_PARSER_CLASS_AVAILABLE = True
except ImportError:
    DATE_PARSER_CLASS_AVAILABLE = False
    # This print is okay for module-level feedback during development/setup
    print("Warning (schedule_parser.py): SyllabusDateParser not found. ScheduleParser functionality will be limited.")

try:
    from utils.parsers.openai_parser import OpenAIParser # For LLM calls
    OPENAI_PARSER_CLASS_AVAILABLE = True
except ImportError:
    OPENAI_PARSER_CLASS_AVAILABLE = False
    print("Warning (schedule_parser.py): OpenAIParser not found. LLM-based schedule detail extraction will not be available.")

class ScheduleParser:
    """
    Generates a detailed class schedule (event_data) by first creating base class sessions
    and then using an LLM to parse the "course_schedule" text segment for topics,
    readings, and in-class activities, merging these details.
    """
    
    def __init__(self, 
                 logger_instance: logging.Logger, 
                 date_parser_instance: Optional[SyllabusDateParser] = None, 
                 openai_parser_instance: Optional[OpenAIParser] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.logger = logger_instance
        if not self.logger: # Should ideally not happen if app.py provides it
            self.logger = logging.getLogger(__name__) # Fallback logger
            if not self.logger.hasHandlers(): # Configure if no handlers exist
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s')
            self.logger.critical("ScheduleParser initialized with a default fallback logger. This is not recommended for production.")
            
        self.config = config or {}
        self.openai_parser = openai_parser_instance

        if not self.openai_parser and OPENAI_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No OpenAIParser instance provided to ScheduleParser. Attempting to create one if configured.")
            # Ensure OpenAIParser can be initialized this way or adjust as needed
            model = self.config.get("extraction", {}).get("openai_model", "gpt-4o") # Default model
            # OpenAIParser might need more config, ensure its __init__ matches
            try:
                self.openai_parser = OpenAIParser(model=model, logger_instance=self.logger, config=self.config)
            except Exception as e_op_init:
                self.logger.error(f"Failed to auto-initialize OpenAIParser in ScheduleParser: {e_op_init}", exc_info=True)
                self.openai_parser = None # Ensure it's None if init fails
        elif not self.openai_parser: # If still None after attempt or if OPENAI_PARSER_CLASS_AVAILABLE is False
            self.logger.error("OpenAIParser not available/creatable. LLM-based schedule detail extraction will be impaired or fail.")

        if date_parser_instance:
            self.date_parser = date_parser_instance
        elif DATE_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No DateParser instance provided. Creating new one for ScheduleParser.")
            try:
                self.date_parser = SyllabusDateParser(logger_instance=self.logger) # Assuming simple init
            except Exception as e_dp_init:
                self.logger.error(f"Failed to auto-initialize DateParser in ScheduleParser: {e_dp_init}", exc_info=True)
                self.date_parser = None
        else:
            self.date_parser = None # Explicitly None if not available
            self.logger.error("DateParser not available. ScheduleParser functionality severely limited.")
        
        self.default_term_weeks = int(self.config.get("schedule_parser", {}).get("default_term_weeks", 15))
        # Ensure self.date_parser exists before accessing attributes
        self.output_date_format = self.date_parser.output_date_format if self.date_parser and hasattr(self.date_parser, 'output_date_format') else "%B %d, %Y"
        self.logger.info(f"ScheduleParser initialized. Default term weeks: {self.default_term_weeks}.")
    
    def _generate_schedule_detail_extraction_prompt(self, schedule_text_segment: str, class_data: Dict[str, Any]) -> str:
        """Generates the prompt for the LLM to extract detailed schedule items."""
        
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
1. Identify distinct chronological entries (these might be per week, per class session, or topic-based).
2. For each entry, extract the following details:
   - "date_reference": (String) The date or date range this entry refers to, as written in the schedule (e.g., "Week 1", "Sept. 11", "Mon, Wed", "Week 3 (9/16-9/20)"). If an entry spans multiple days (like a week), provide the range.
   - "topics": (List of Strings) The main topic(s) or lecture titles for this entry/session(s).
   - "readings": (List of Strings) Required or recommended reading materials (e.g., ["Chapter 1", "Smith et al. (2020) pp. 25-30"]). If none, use an empty list [].
   - "activities_notes": (List of Strings) Brief descriptions of in-class activities, discussions, minor quizzes (not major exams), or other important notes for the session(s). If none, use an empty list [].
   - "is_holiday_or_no_class": (Boolean) True if this entry explicitly states it's a holiday, break, or no class, otherwise false.

Output Format:
Respond with a single JSON array, where each element is an object representing a schedule entry with the fields "date_reference", "topics", "readings", "activities_notes", and "is_holiday_or_no_class".
Example of an array element:
{{
  "date_reference": "Wed., Sept. 11",
  "topics": ["Motion in 1D", "Problem Solving Session"],
  "readings": ["Chapter 1", "Sections 3.1-3.5"],
  "activities_notes": ["HW #1 due", "Review of kinematics"],
  "is_holiday_or_no_class": false
}}
If the schedule segment is empty or no structured entries can be found, return an empty JSON array [].

CRITICAL:
- Focus on information directly related to what happens during or is assigned for specific class sessions or weeks.
- Do NOT re-extract major assignments or exams if they are only mentioned as due dates (e.g., "HW #1 due"). Another part of the system handles primary task extraction. Focus on topics, readings, and in-class events. If an exam is *held during class time* as per the schedule, you can note it in "topics" or "activities_notes".
- If a reading is tied to a due date (e.g., "HW #1 due Reading: Chapter 1"), associate "Chapter 1" with the class sessions leading up to or on that due date.
- Ensure the output is a valid JSON array.
"""
        return prompt

    def _populate_event_details_with_llm(self, 
                                         event_data_base: List[Dict[str, Any]], 
                                         schedule_text_segment: str, 
                                         class_data: Dict[str, Any],
                                         unique_id: str) -> List[Dict[str, Any]]:
        if not self.date_parser:
            self.logger.error("DateParser not available in _populate_event_details_with_llm. Skipping LLM detail extraction.")
            return event_data_base
        if not schedule_text_segment.strip() or not self.openai_parser:
            self.logger.warning("Schedule text segment is empty or OpenAIParser is not available. Skipping LLM detail extraction.")
            return event_data_base

        prompt = self._generate_schedule_detail_extraction_prompt(schedule_text_segment, class_data)
        
        llm_schedule_items: Optional[List[Dict[str, Any]]] = None
        try:
            if hasattr(self.openai_parser, 'get_json_list_from_prompt') and callable(getattr(self.openai_parser, 'get_json_list_from_prompt')):
                 llm_schedule_items = self.openai_parser.get_json_list_from_prompt(prompt, f"{class_data.get('Course Code', 'COURSE')}_schedule_details_{unique_id}")
            elif self.openai_parser.client: # Fallback manual call
                self.logger.debug("Using fallback manual OpenAI call for schedule details.")
                response = self.openai_parser.client.chat.completions.create(
                    model=self.openai_parser.model,
                    messages=[{"role": "system", "content": "You are an AI assistant that returns JSON formatted data as requested."}, # More generic system prompt
                              {"role": "user", "content": prompt}],
                    # Ensure response_format is compatible with how you parse.
                    # If prompt asks for an array, but API forces an object, parsing needs to handle it.
                    # Forcing json_object here, means the LLM should output a JSON object that might contain the array.
                    response_format={"type": "json_object"} 
                )
                raw_response_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
                if raw_response_text:
                    parsed_outer_json = json.loads(raw_response_text) 
                    if isinstance(parsed_outer_json, list):
                        llm_schedule_items = parsed_outer_json
                    # Check if the prompt asks for an array and the LLM wrapped it in a key, e.g., "schedule_entries"
                    elif isinstance(parsed_outer_json, dict):
                        # Try to find a list if the LLM wrapped the array
                        found_list = None
                        for val in parsed_outer_json.values():
                            if isinstance(val, list):
                                found_list = val
                                break
                        if found_list is not None:
                             llm_schedule_items = found_list
                        else:
                            self.logger.error(f"LLM response (json_object mode) was a dict but did not contain a clear list of schedule items: {str(parsed_outer_json)[:500]}")
                    else: 
                        self.logger.error(f"LLM response for schedule details was not a list or expected dict structure: {raw_response_text[:500]}")
                else:
                    self.logger.warning("LLM returned empty response for schedule details.")
            else: 
                self.logger.error("OpenAI client not available or get_json_list_from_prompt missing for schedule detail extraction.")
        except Exception as e_llm:
            self.logger.error(f"LLM call for schedule detail extraction failed for ID {unique_id}: {e_llm}", exc_info=True)
        
        if not llm_schedule_items:
            self.logger.info(f"LLM did not return any schedule items or response was invalid for ID {unique_id}.")
            return event_data_base

        event_map: Dict[str, Dict[str, Any]] = {
            event["Event Date"]: event for event in event_data_base if event.get("Event Date")
        }
        course_code = class_data.get("Course Code", "COURSE")

        for item in llm_schedule_items:
            if not isinstance(item, dict): continue
            date_ref_str = str(item.get("date_reference", "")).strip()
            topics = item.get("topics", [])
            readings = item.get("readings", [])
            activities_notes = item.get("activities_notes", [])
            is_holiday_no_class = item.get("is_holiday_or_no_class", False)

            if not date_ref_str: continue
            
            try:
                # Assuming DateParser.normalize_date returns a standardized string like "Month Day, Year"
                # And DateParser.parse can handle this standardized string.
                normalized_date_str = self.date_parser.normalize_date(date_ref_str, term_year_str=class_data.get("Term")) # Pass term for context
                if not normalized_date_str or normalized_date_str.upper() in ["TBD", "TBA", "NOT SPECIFIED"]: 
                    self.logger.debug(f"Skipping LLM item due to unparsable/TBD date_reference: '{date_ref_str}' -> '{normalized_date_str}'")
                    continue
                
                # REVISED: Use self.date_parser.parse (assuming it exists and returns a datetime-like object)
                parsed_date_obj = self.date_parser.parse(normalized_date_str, ignoretz=True).date()
                target_event_date_str = parsed_date_obj.strftime(self.output_date_format)

                if target_event_date_str in event_map:
                    event_to_update = event_map[target_event_date_str]
                    current_title = event_to_update.get("Event Title", f"{course_code}: Class")

                    if topics:
                        if current_title == f"{course_code}: Class" or not current_title.startswith(course_code):
                            event_to_update["Event Title"] = f"{course_code}: {', '.join(topics)}"
                        else: 
                            event_to_update["Event Title"] += f" - {', '.join(topics)}"
                    
                    if readings: event_to_update.setdefault("reading", []).extend(r for r in readings if r not in event_to_update.get("reading",[]))
                    if activities_notes: event_to_update.setdefault("special", []).extend(an for an in activities_notes if an not in event_to_update.get("special",[]))

                    if is_holiday_no_class:
                        event_to_update["Event Title"] = f"{course_code}: No Class - {topics[0] if topics else 'Holiday/Break'}"
                        event_to_update.setdefault("special", []).append(topics[0] if topics else "Holiday/Break")
                        event_to_update["Class Time"] = "" 
                    
                    self.logger.debug(f"Updated event on {target_event_date_str} with LLM details. Title: {event_to_update['Event Title']}")
            except Exception as e_date_resolve:
                self.logger.warning(f"Could not resolve or apply LLM schedule item with date_reference '{date_ref_str}': {e_date_resolve}", exc_info=True)
        return list(event_map.values())

    def process_schedule(self, data: Dict[str, Any], unique_id: str) -> Dict[str, Any]:
        self.logger.info(f"Processing class schedule for ID {unique_id} to generate/update event_data...")
        try:
            if not self.date_parser:
                self.logger.error("DateParser not available. Cannot process schedule.")
                data.setdefault("event_data", [])
                return data

            class_data = data.get("class_data")
            if not isinstance(class_data, dict):
                self.logger.error("'class_data' missing or invalid. Cannot process schedule.")
                data.setdefault("event_data", [])
                return data

            class_data = self._ensure_class_start_end_dates(class_data)
            data["class_data"] = class_data # Persist any changes from _ensure_class_start_end_dates

            final_class_start_str = class_data.get("Class Start Date")
            final_class_end_str = class_data.get("Class End Date")
            days_of_week_str = class_data.get("Days of Week", "")
            
            base_event_data: List[Dict[str, Any]] = []
            if final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] and \
               final_class_end_str and final_class_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] and \
               days_of_week_str:
                weekday_indices = self.date_parser.parse_weekdays_to_indices(days_of_week_str)
                if not weekday_indices:
                    self.logger.warning(f"No valid weekdays from '{days_of_week_str}'. Cannot generate base schedule for {unique_id}.")
                else:
                    course_code = self._extract_course_code(class_data.get("Course Title", "Course"))
                    class_time = class_data.get("Class Time", "")
                    class_location = class_data.get("Class Location", "")
                    base_event_data = self._generate_class_sessions(
                        final_class_start_str, final_class_end_str, weekday_indices, 
                        course_code, class_time, class_location
                    )
            else:
                self.logger.warning(f"Missing critical info (start/end date, days) for base schedule generation for {unique_id}.")

            existing_events = data.get("event_data", [])
            if not isinstance(existing_events, list): existing_events = []
            
            merged_event_data_before_llm = self._merge_schedules(base_event_data, existing_events)

            schedule_text_segment = data.get("segmented_syllabus", {}).get("course_schedule", "")
            if schedule_text_segment.strip() and self.openai_parser: # Check if openai_parser is not None
                self.logger.info(f"Populating event details using LLM for ID {unique_id} from 'course_schedule' segment.")
                final_event_data = self._populate_event_details_with_llm(
                    merged_event_data_before_llm, 
                    schedule_text_segment, 
                    class_data, 
                    unique_id
                )
            else:
                self.logger.info(f"Skipping LLM detail population for schedule (no schedule text or no OpenAIParser) for ID {unique_id}.")
                final_event_data = merged_event_data_before_llm
            
            if self.date_parser: # Ensure date_parser is available for sorting
                try:
                    # REVISED: Use self.date_parser.parse
                    final_event_data.sort(key=lambda x: self.date_parser.parse(x["Event Date"], ignoretz=True).date() if x.get("Event Date") else date.max)
                except Exception as e_sort:
                    self.logger.error(f"Error sorting final event_data for ID {unique_id}: {e_sort}", exc_info=True)

            data["event_data"] = final_event_data
            self.logger.info(f"Schedule processing complete for ID {unique_id}. Total events: {len(final_event_data)}.")
            return data
        except Exception as e_proc_schedule:
            self.logger.error(f"Error processing schedule for ID {unique_id}: {e_proc_schedule}", exc_info=True)
            data.setdefault("event_data", []) # Ensure event_data key exists even on error
            return data

    def _ensure_class_start_end_dates(self, class_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.debug("Ensuring 'Class Start Date' and 'Class End Date' are determined and refined.")
        if not self.date_parser:
            self.logger.error("DateParser not available in _ensure_class_start_end_dates.")
            return class_data

        # Normalize dates first using DateParser's method
        term_start_str = self.date_parser.normalize_date(class_data.get("Term Start Date", ""), term_year_str=class_data.get("Term"))
        term_end_str = self.date_parser.normalize_date(class_data.get("Term End Date", ""), term_year_str=class_data.get("Term"))
        class_start_str = self.date_parser.normalize_date(class_data.get("Class Start Date", ""), term_year_str=class_data.get("Term"))
        class_end_str = self.date_parser.normalize_date(class_data.get("Class End Date", ""), term_year_str=class_data.get("Term"))
        days_of_week_str = class_data.get("Days of Week", "")
        
        # Update class_data with normalized dates if they changed and were not empty initially
        if class_data.get("Term Start Date") and term_start_str != class_data.get("Term Start Date"): class_data["Term Start Date"] = term_start_str
        if class_data.get("Term End Date") and term_end_str != class_data.get("Term End Date"): class_data["Term End Date"] = term_end_str
        # Class Start/End dates will be set at the end after all logic

        final_class_start_str = class_start_str
        final_class_end_str = class_end_str

        if not final_class_start_str or final_class_start_str.upper() in ["TBD", "TBA", "NOT SPECIFIED"]:
            if term_start_str and term_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"]:
                final_class_start_str = term_start_str
                self.logger.info(f"Using Term Start Date '{term_start_str}' as Class Start Date.")
            else: self.logger.warning("Cannot determine class start date from class or term start dates.")
        
        if not final_class_end_str or final_class_end_str.upper() in ["TBD", "TBA", "NOT SPECIFIED"]:
            if term_end_str and term_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"]:
                final_class_end_str = term_end_str
                self.logger.info(f"Using Term End Date '{term_end_str}' as Class End Date.")
            elif final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"]:
                try:
                    # REVISED: Use self.date_parser.parse
                    start_date_obj = self.date_parser.parse(final_class_start_str, ignoretz=True).date()
                    estimated_end_obj = start_date_obj + timedelta(weeks=self.default_term_weeks) - timedelta(days=1) # Common term length
                    final_class_end_str = estimated_end_obj.strftime(self.output_date_format)
                    self.logger.info(f"Estimated Class End Date: '{final_class_end_str}' based on start date and default term weeks.")
                except Exception as e_est: self.logger.warning(f"Could not estimate Class End Date: {e_est}", exc_info=True)
            else: self.logger.warning("Cannot determine class end date from class/term end dates or estimate from start.")

        if final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] and \
           final_class_end_str and final_class_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] and \
           days_of_week_str:
            try:
                weekday_indices = self.date_parser.parse_weekdays_to_indices(days_of_week_str)
                if weekday_indices:
                    # REVISED: Use self.date_parser.parse
                    start_obj_curr = self.date_parser.parse(final_class_start_str, ignoretz=True).date()
                    end_obj_curr = self.date_parser.parse(final_class_end_str, ignoretz=True).date()
                    
                    refined_start = False
                    temp_date = start_obj_curr
                    for _ in range(7): # Check current and next 6 days
                        if temp_date.weekday() in weekday_indices:
                            final_class_start_str = temp_date.strftime(self.output_date_format)
                            refined_start = True
                            break
                        temp_date += timedelta(days=1)
                    if refined_start: self.logger.debug(f"Refined Class Start Date to first meeting day: {final_class_start_str}")

                    refined_end = False
                    temp_date = end_obj_curr
                    for _ in range(7): # Check current and previous 6 days
                        if temp_date.weekday() in weekday_indices:
                            final_class_end_str = temp_date.strftime(self.output_date_format)
                            refined_end = True
                            break
                        temp_date -= timedelta(days=1)
                    if refined_end: self.logger.debug(f"Refined Class End Date to last meeting day: {final_class_end_str}")
            except Exception as e_ref: self.logger.error(f"Error refining class start/end dates to meeting days: {e_ref}", exc_info=True)
        
        class_data["Class Start Date"] = final_class_start_str if final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] else ""
        class_data["Class End Date"] = final_class_end_str if final_class_end_str and final_class_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] else ""
        return class_data

    def _generate_class_sessions(self, class_start_date_str: str, class_end_date_str: str, weekday_indices: List[int], course_code: str, global_class_time: str, global_class_location: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Generating regular class sessions for '{course_code}' from {class_start_date_str} to {class_end_date_str}.")
        if not self.date_parser: 
            self.logger.error("DateParser not available in _generate_class_sessions.")
            return []
            
        generated_events: List[Dict[str, Any]] = []
        try:
            # REVISED: Use self.date_parser.parse
            start_obj = self.date_parser.parse(class_start_date_str, ignoretz=True).date()
            end_obj = self.date_parser.parse(class_end_date_str, ignoretz=True).date()
            current_date = start_obj
            while current_date <= end_obj:
                if current_date.weekday() in weekday_indices:
                    generated_events.append({
                        "Event Date": current_date.strftime(self.output_date_format),
                        "Event Title": f"{course_code}: Class", 
                        "Class Time": global_class_time, 
                        "Class Location": global_class_location,
                        "reading": [], 
                        "assignment": [], 
                        "assignment_description": None, 
                        "test": [], 
                        "special": [] 
                    })
                current_date += timedelta(days=1)
            self.logger.info(f"Generated {len(generated_events)} regular class sessions for '{course_code}'.")
        except Exception as e_gen: self.logger.error(f"Error generating class sessions for '{course_code}': {e_gen}", exc_info=True)
        return generated_events
    
    def _merge_schedules(self, generated_sessions: List[Dict[str, Any]], existing_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.date_parser:
            self.logger.error("DateParser not available in _merge_schedules. Basic concatenation fallback.")
            return existing_events + generated_sessions 

        self.logger.debug(f"Merging {len(generated_sessions)} generated sessions with {len(existing_events)} existing events.")
        merged_event_map: Dict[str, Dict[str, Any]] = {}

        for event in existing_events:
            if isinstance(event, dict) and event.get("Event Date"):
                try:
                    # Normalize date using DateParser for consistent keying
                    # Assuming normalize_date returns a string in self.output_date_format or similar parsable format
                    norm_date_key = self.date_parser.normalize_date(str(event["Event Date"]), term_year_str=None) # Term context might not be needed if date is absolute
                    if not norm_date_key or norm_date_key.upper() in ["TBD", "TBA", "NOT SPECIFIED"]:
                        self.logger.warning(f"Skipping existing event with un-normalizable date: {event.get('Event Date')}")
                        continue

                    # Ensure all fields are present from the target schema
                    event.setdefault("Event Title", "") # Ensure title exists
                    event.setdefault("Class Time", "")
                    event.setdefault("Class Location", "")
                    event.setdefault("reading", [])
                    event.setdefault("assignment", [])
                    event.setdefault("assignment_description", None)
                    event.setdefault("test", [])
                    event.setdefault("special", [])
                    merged_event_map[norm_date_key] = event
                except Exception as e_norm: self.logger.warning(f"Could not normalize/process existing event date: {event.get('Event Date')}, error: {e_norm}", exc_info=True)
        
        for session in generated_sessions:
            # session["Event Date"] should already be in self.output_date_format from _generate_class_sessions
            session_date_key = session["Event Date"] 
            if session_date_key in merged_event_map:
                existing = merged_event_map[session_date_key]
                # Prioritize existing more specific titles or details
                if not existing.get("Event Title") or existing.get("Event Title") == f"{self._extract_course_code(existing.get('Course Title', 'COURSE'))}: Class":
                    existing["Event Title"] = session.get("Event Title", existing.get("Event Title"))
                existing["Class Time"] = existing.get("Class Time") or session.get("Class Time")
                existing["Class Location"] = existing.get("Class Location") or session.get("Class Location")
                # Ensure lists are initialized (already done above for existing, do for session if it was minimal)
                for key_list in ["reading", "assignment", "test", "special"]:
                    existing.setdefault(key_list, [])
                    # If session has items for these lists, they would be added by LLM population later
                existing.setdefault("assignment_description", None)
            else:
                merged_event_map[session_date_key] = session
        
        final_list = list(merged_event_map.values())
        return final_list
    
    def _extract_course_code(self, course_title: Optional[str]) -> str:
        if not course_title or not isinstance(course_title, str): return "COURSE"
        # Try to find common course code patterns (e.g., DEPT123, DEPT 123A)
        match = re.search(r'([A-Z]{2,5}\s*\d{2,4}[A-Z]?)', course_title, re.IGNORECASE)
        if match: 
            # Remove spaces from the matched group and uppercase
            return "".join(match.group(1).split()).upper() 
        
        # Fallback: if the first word looks like a course code
        first_word = course_title.split(' ')[0]
        if re.fullmatch(r'[A-Za-z]{2,4}\d{2,4}[A-Za-z]?', first_word): # More flexible regex for first word
            return first_word.upper()
        
        # Further fallback: take first word if it's short and alphanumeric
        if len(first_word) <= 7 and first_word.isalnum() and not first_word.isdigit():
            return first_word.upper()
            
        return "COURSE" # Default if no clear code found

if __name__ == '__main__':
    # BasicConfig for standalone testing, real app should configure logger.
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s', 
                        handlers=[logging.StreamHandler(sys.stdout)]) # Ensure output to console
    main_logger = logging.getLogger(__name__) # Get logger after basicConfig

    if not DATE_PARSER_CLASS_AVAILABLE: # Removed check for OPENAI_PARSER_CLASS_AVAILABLE for this basic test
        main_logger.critical("DateParser class (SyllabusDateParser) not available. ScheduleParser standalone tests cannot run effectively.")
        sys.exit(1)

    # --- Mock Config & Parsers for Standalone Test ---
    mock_config = {
        "extraction": {"openai_model": "gpt-4-turbo-preview"}, # Example model
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")}, # Needs API key for real LLM test
        "schedule_parser": {"default_term_weeks": 15},
        "openai_parser": {"max_api_retries": 1, "client_timeout": {"read": 60.0, "connect": 10.0}},
        "date_parser": {"output_date_format": "%B %d, %Y"} # Ensure DateParser uses this
    }
    
    # Use the real DateParser
    try:
        test_date_parser = SyllabusDateParser(logger_instance=main_logger, config=mock_config.get("date_parser"))
    except Exception as e_dp:
        main_logger.critical(f"Failed to initialize real DateParser for test: {e_dp}")
        sys.exit(1)

    # Mock OpenAIParser for schedule detail extraction if real one is problematic for testing
    class MockOpenAIParserForScheduleTest:
        def __init__(self, model, logger_instance, config=None):
            self.model = model; self.logger = logger_instance; self.config = config or {}
            self.client = True # Simulate client being ready
            self.logger.info(f"MockOpenAIParserForScheduleTest initialized with model: {self.model}")

        def get_json_list_from_prompt(self, prompt: str, unique_id: str) -> Optional[List[Dict[str, Any]]]:
            self.logger.info(f"MockOpenAIParserForScheduleTest.get_json_list_from_prompt called for {unique_id}.")
            # Simulate a response based on the PHY203 example
            if "PHY203" in prompt and "Exam and Homework Schedule" in prompt:
                return [
                    {"date_reference": "Wed., Sept. 11", "topics": ["Kinematics Review", "Intro to Vectors"], "readings": ["Chapter 1", "Handout A"], "activities_notes": ["HW #1 due today", "Clicker Quiz #1"], "is_holiday_or_no_class": False},
                    {"date_reference": "Fri., Sept. 27", "topics": ["Exam #1"], "readings": [], "activities_notes": ["Covers Chapters 1, 3, 4. Bring calculator."], "is_holiday_or_no_class": False},
                    {"date_reference": "Wed., Nov. 27", "topics": ["Thanksgiving Break"], "readings": [], "activities_notes": [], "is_holiday_or_no_class": True}
                ]
            self.logger.warning(f"MockOpenAIParserForScheduleTest: No matching mock response for prompt containing: {prompt[:200]}...")
            return []

    # Use the mock or real OpenAIParser
    test_openai_parser = None
    if OPENAI_PARSER_CLASS_AVAILABLE and os.getenv("OPENAI_API_KEY"): # Try real if key is available
        try:
            test_openai_parser = OpenAIParser(model=mock_config["extraction"]["openai_model"], 
                                            logger_instance=main_logger, 
                                            config=mock_config)
            main_logger.info("Using REAL OpenAIParser for test.")
        except Exception as e_op_real:
            main_logger.warning(f"Failed to init REAL OpenAIParser ({e_op_real}), falling back to MOCK for schedule details.")
            test_openai_parser = MockOpenAIParserForScheduleTest(model="mock-gpt", logger_instance=main_logger, config=mock_config)
    else:
        main_logger.info("Using MOCK OpenAIParser for schedule details in test (OpenAI key missing or class unavailable).")
        test_openai_parser = MockOpenAIParserForScheduleTest(model="mock-gpt", logger_instance=main_logger, config=mock_config)


    # Initialize ScheduleParser with the (potentially mock) parsers
    schedule_parser_instance = ScheduleParser(
        logger_instance=main_logger,
        date_parser_instance=test_date_parser,
        openai_parser_instance=test_openai_parser, # type: ignore # Allow mock
        config=mock_config
    )

    main_logger.info("\n--- Testing ScheduleParser with PHY203 Example ---")
    
    phy203_class_data = {
        "School Name": "University of Rhode Island", "Term": "Fall 2024", "Course Title": "PHY203: ELEMENTARY PHYSICS I",
        "Course Code": "PHY203", "Instructor Name": "Miquel Dorca", "Instructor Email": "miquel@uri.edu",
        "Class Time": "1:00 PM - 1:50 PM", "Time Zone": "America/New_York", # Using a standard TZ
        "Days of Week": "Monday, Wednesday, Friday",
        "Term Start Date": "September 4, 2024", "Term End Date": "December 13, 2024",
        "Class Start Date": "September 4, 2024", # Explicitly set
        "Class End Date": "December 11, 2024",   # Explicitly set
        "Office Hours": "MWF 12:30pm-1:00pm (East Hall Auditorium)",
        "Class Location": "East Hall Auditorium (section 1)"
    }
    phy203_segmented_syllabus = { # This is what SyllabusParser would provide
        "course_schedule": """
Week 1 (Sept 4, 6): Introduction to Physics, Units & Dimensions. Reading: Ch 1. HW #1 assigned.
Wed., Sept. 11: Kinematics Review. Intro to Vectors. Reading: Chapter 1. HW #1 due today. Clicker Quiz #1.
Fri., Sept. 13: Vector Addition. Projectile Motion. Reading: Ch 3.
...
Fri., Sept. 27: Exam #1. Covers Chapters 1, 3, 4. Bring calculator.
...
Wed., Nov. 27: Thanksgiving Break - No Class.
...
Wed., Dec. 11: Final Review Session. HW #12 due.
Final Exam: TBA (Covers Chaps. 1-11,15)
"""
    }
    phy203_initial_event_data = [ # Events that might exist from AssignmentParser linking tasks
        {"Event Date": "September 11, 2024", "Event Title": "PHY203: HW #1 Due", "Task Type": "Homework", "assignment": ["PHY203: HW #1"], "Class Time": "10:00 PM"}, # Task due
        {"Event Date": "September 27, 2024", "Event Title": "PHY203: Exam #1", "Task Type": "Exam", "test": ["PHY203: Exam #1"]}, # Exam
        {"Event Date": "December 11, 2024", "Event Title": "PHY203: HW #12 Due", "Task Type": "Homework", "assignment": ["PHY203: HW #12"], "Class Time": "10:00 PM"}
    ]

    test_pipeline_input = {
        "class_data": phy203_class_data.copy(), # Use a copy
        "segmented_syllabus": phy203_segmented_syllabus,
        "task_data": [], # Assuming task_data is processed separately or not relevant for this isolated test
        "event_data": phy203_initial_event_data # Start with some pre-linked events
    }

    final_processed_data = schedule_parser_instance.process_schedule(test_pipeline_input, "phy203_standalone_test")
    
    main_logger.info(f"\n--- Final Class Data after Schedule Processing ---\n{json.dumps(final_processed_data.get('class_data'), indent=2)}")
    main_logger.info(f"\n--- Final Event Data after Schedule Processing ---\n{json.dumps(final_processed_data.get('event_data'), indent=2)}")

    main_logger.info("\n--- ScheduleParser standalone test finished ---")
