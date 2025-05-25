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
import uuid # Added for unique event IDs
from datetime import date, timedelta, datetime, time # Added time
from typing import Dict, List, Any, Optional

try:
    # DateParser should have get_iso_datetime_str, parse_time_string_to_objects
    from utils.parsers.date_parser import DateParser as SyllabusDateParser
    DATE_PARSER_CLASS_AVAILABLE = True
except ImportError:
    DATE_PARSER_CLASS_AVAILABLE = False
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
        if not self.logger: 
            self.logger = logging.getLogger(__name__) 
            if not self.logger.hasHandlers(): 
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s')
            self.logger.critical("ScheduleParser initialized with a default fallback logger.")
            
        self.config = config or {}
        self.openai_parser = openai_parser_instance

        if not self.openai_parser and OPENAI_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No OpenAIParser instance provided to ScheduleParser. Attempting to create one if configured.")
            model = self.config.get("extraction", {}).get("openai_model", "gpt-4o")
            try:
                self.openai_parser = OpenAIParser(model=model, logger_instance=self.logger, config=self.config)
            except Exception as e_op_init:
                self.logger.error(f"Failed to auto-initialize OpenAIParser in ScheduleParser: {e_op_init}", exc_info=True)
                self.openai_parser = None
        elif not self.openai_parser: 
            self.logger.error("OpenAIParser not available/creatable. LLM-based schedule detail extraction will be impaired or fail.")

        if date_parser_instance:
            self.date_parser = date_parser_instance
        elif DATE_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No DateParser instance provided. Creating new one for ScheduleParser.")
            try:
                # DateParser might need config if output_date_format is to be configured
                self.date_parser = SyllabusDateParser(logger_instance=self.logger, config=self.config.get("date_parser"))
            except Exception as e_dp_init:
                self.logger.error(f"Failed to auto-initialize DateParser in ScheduleParser: {e_dp_init}", exc_info=True)
                self.date_parser = None
        else:
            self.date_parser = None 
            self.logger.error("DateParser not available. ScheduleParser functionality severely limited.")
        
        self.default_term_weeks = int(self.config.get("schedule_parser", {}).get("default_term_weeks", 15))
        self.output_date_format = self.date_parser.output_date_format if self.date_parser and hasattr(self.date_parser, 'output_date_format') else "%B %d, %Y"
        self.logger.info(f"ScheduleParser initialized. Default term weeks: {self.default_term_weeks}.")
    
    def _generate_schedule_detail_extraction_prompt(self, schedule_text_segment: str, class_data: Dict[str, Any]) -> str:
        # ... (prompt definition remains the same as schedule_parser_revised_v1) ...
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
        # ... (LLM call logic remains the same as schedule_parser_revised_v1) ...
        # This method now enriches events that should already have IDs and ISO datetimes.
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
                self.logger.debug("Using fallback manual OpenAI call for schedule details.")
                response = self.openai_parser.client.chat.completions.create(
                    model=self.openai_parser.model,
                    messages=[{"role": "system", "content": "You are an AI assistant that returns JSON formatted data as requested."},
                              {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"} 
                )
                raw_response_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
                if raw_response_text:
                    parsed_outer_json = json.loads(raw_response_text) 
                    if isinstance(parsed_outer_json, list): llm_schedule_items = parsed_outer_json
                    elif isinstance(parsed_outer_json, dict):
                        found_list = None
                        for val in parsed_outer_json.values():
                            if isinstance(val, list): found_list = val; break
                        if found_list is not None: llm_schedule_items = found_list
                        else: self.logger.error(f"LLM response (json_object mode) dict did not contain a clear list of schedule items: {str(parsed_outer_json)[:500]}")
                    else: self.logger.error(f"LLM response for schedule details was not a list or expected dict: {raw_response_text[:500]}")
                else: self.logger.warning("LLM returned empty response for schedule details.")
            else: self.logger.error("OpenAI client not available or get_json_list_from_prompt missing.")
        except Exception as e_llm:
            self.logger.error(f"LLM call for schedule detail extraction failed for ID {unique_id}: {e_llm}", exc_info=True)
        
        if not llm_schedule_items:
            self.logger.info(f"LLM did not return schedule items or response invalid for ID {unique_id}.")
            return event_data_base

        event_map: Dict[str, Dict[str, Any]] = {
            event["Event Date"]: event for event in event_data_base if event.get("Event Date") and event.get("event_id") # Ensure base events have IDs
        }
        course_code = class_data.get("Course Code", "COURSE")
        course_timezone = class_data.get("Time Zone") # For context if needed, though ISOs should be set

        for item in llm_schedule_items:
            # ... (logic for extracting topics, readings, etc. from item remains same) ...
            if not isinstance(item, dict): continue
            date_ref_str = str(item.get("date_reference", "")).strip()
            topics = item.get("topics", [])
            readings = item.get("readings", [])
            activities_notes = item.get("activities_notes", [])
            is_holiday_no_class = item.get("is_holiday_or_no_class", False)
            if not date_ref_str: continue
            
            try:
                normalized_date_str = self.date_parser.normalize_date(date_ref_str, term_year_str=class_data.get("Term"))
                if not normalized_date_str or normalized_date_str.upper() in ["TBD", "TBA", "NOT SPECIFIED"]: 
                    self.logger.debug(f"Skipping LLM item due to TBD date_reference: '{date_ref_str}' -> '{normalized_date_str}'")
                    continue
                
                parsed_date_obj = self.date_parser.parse(normalized_date_str, ignoretz=True).date()
                target_event_date_str = parsed_date_obj.strftime(self.output_date_format)

                if target_event_date_str in event_map:
                    event_to_update = event_map[target_event_date_str]
                    current_title = event_to_update.get("Event Title", f"{course_code}: Class")
                    if topics:
                        if current_title == f"{course_code}: Class" or not current_title.startswith(course_code):
                            event_to_update["Event Title"] = f"{course_code}: {', '.join(topics)}"
                        else: event_to_update["Event Title"] += f" - {', '.join(topics)}"
                    if readings: event_to_update.setdefault("reading", []).extend(r for r in readings if r not in event_to_update.get("reading",[]))
                    if activities_notes: event_to_update.setdefault("special", []).extend(an for an in activities_notes if an not in event_to_update.get("special",[]))
                    if is_holiday_no_class:
                        event_to_update["Event Title"] = f"{course_code}: No Class - {topics[0] if topics else 'Holiday/Break'}"
                        event_to_update.setdefault("special", []).append(topics[0] if topics else "Holiday/Break")
                        event_to_update["Class Time"] = "" 
                        # For holidays, ISO times might become just date or be cleared
                        event_to_update["start_datetime_iso"] = parsed_date_obj.isoformat() # All-day event
                        event_to_update["end_datetime_iso"] = (parsed_date_obj + timedelta(days=1)).isoformat() # All-day event convention
                    self.logger.debug(f"Updated event on {target_event_date_str} with LLM details. Title: {event_to_update['Event Title']}")
            except Exception as e_date_resolve:
                self.logger.warning(f"Could not resolve/apply LLM schedule item with date_reference '{date_ref_str}': {e_date_resolve}", exc_info=True)
        return list(event_map.values())

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
            course_timezone_str = class_data.get("Time Zone") # Get timezone for ISO generation

            if not course_timezone_str:
                self.logger.warning(f"Time Zone missing in class_data for {unique_id}. ISO datetimes for events will be naive or omitted.")
            
            base_event_data: List[Dict[str, Any]] = []
            if final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] and \
               final_class_end_str and final_class_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] and \
               days_of_week_str:
                weekday_indices = self.date_parser.parse_weekdays_to_indices(days_of_week_str)
                if not weekday_indices:
                    self.logger.warning(f"No valid weekdays from '{days_of_week_str}'. Cannot generate base schedule for {unique_id}.")
                else:
                    course_code = self._extract_course_code(class_data.get("Course Title", "Course"))
                    class_time_str = class_data.get("Class Time", "") # Human-readable
                    class_location = class_data.get("Class Location", "")
                    base_event_data = self._generate_class_sessions(
                        final_class_start_str, final_class_end_str, weekday_indices, 
                        course_code, class_time_str, class_location, course_timezone_str # Pass timezone
                    )
            else:
                self.logger.warning(f"Missing critical info (start/end date, days) for base schedule generation for {unique_id}.")

            existing_events = data.get("event_data", [])
            if not isinstance(existing_events, list): existing_events = []
            
            merged_event_data_before_llm = self._merge_schedules(base_event_data, existing_events)

            schedule_text_segment = data.get("segmented_syllabus", {}).get("course_schedule", "")
            if schedule_text_segment.strip() and self.openai_parser:
                self.logger.info(f"Populating event details using LLM for ID {unique_id} from 'course_schedule' segment.")
                final_event_data = self._populate_event_details_with_llm(
                    merged_event_data_before_llm, schedule_text_segment, class_data, unique_id
                )
            else:
                self.logger.info(f"Skipping LLM detail population (no text or no OpenAIParser) for ID {unique_id}.")
                final_event_data = merged_event_data_before_llm
            
            if self.date_parser:
                try:
                    final_event_data.sort(key=lambda x: self.date_parser.parse(x["Event Date"], ignoretz=True).date() if x.get("Event Date") and x["Event Date"] != "TBD" else date.max)
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
        # ... (implementation remains same as schedule_parser_revised_v1) ...
        self.logger.debug("Ensuring 'Class Start Date' and 'Class End Date' are determined and refined.")
        if not self.date_parser: self.logger.error("DateParser not available in _ensure_class_start_end_dates."); return class_data
        term_start_str = self.date_parser.normalize_date(class_data.get("Term Start Date", ""), term_year_str=class_data.get("Term"))
        term_end_str = self.date_parser.normalize_date(class_data.get("Term End Date", ""), term_year_str=class_data.get("Term"))
        class_start_str = self.date_parser.normalize_date(class_data.get("Class Start Date", ""), term_year_str=class_data.get("Term"))
        class_end_str = self.date_parser.normalize_date(class_data.get("Class End Date", ""), term_year_str=class_data.get("Term"))
        days_of_week_str = class_data.get("Days of Week", "")
        if class_data.get("Term Start Date") and term_start_str != class_data.get("Term Start Date"): class_data["Term Start Date"] = term_start_str
        if class_data.get("Term End Date") and term_end_str != class_data.get("Term End Date"): class_data["Term End Date"] = term_end_str
        final_class_start_str = class_start_str; final_class_end_str = class_end_str
        if not final_class_start_str or final_class_start_str.upper() in ["TBD", "TBA", "NOT SPECIFIED"]:
            if term_start_str and term_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"]: final_class_start_str = term_start_str; self.logger.info(f"Using Term Start Date '{term_start_str}' as Class Start Date.")
            else: self.logger.warning("Cannot determine class start date.")
        if not final_class_end_str or final_class_end_str.upper() in ["TBD", "TBA", "NOT SPECIFIED"]:
            if term_end_str and term_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"]: final_class_end_str = term_end_str; self.logger.info(f"Using Term End Date '{term_end_str}' as Class End Date.")
            elif final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"]:
                try:
                    start_date_obj = self.date_parser.parse(final_class_start_str, ignoretz=True).date()
                    estimated_end_obj = start_date_obj + timedelta(weeks=self.default_term_weeks) - timedelta(days=1)
                    final_class_end_str = estimated_end_obj.strftime(self.output_date_format)
                    self.logger.info(f"Estimated Class End Date: '{final_class_end_str}'.")
                except Exception as e_est: self.logger.warning(f"Could not estimate Class End Date: {e_est}", exc_info=True)
            else: self.logger.warning("Cannot determine class end date.")
        if final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] and \
           final_class_end_str and final_class_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] and \
           days_of_week_str:
            try:
                weekday_indices = self.date_parser.parse_weekdays_to_indices(days_of_week_str)
                if weekday_indices:
                    start_obj_curr = self.date_parser.parse(final_class_start_str, ignoretz=True).date()
                    end_obj_curr = self.date_parser.parse(final_class_end_str, ignoretz=True).date()
                    refined_start = False; temp_date = start_obj_curr
                    for _ in range(7): 
                        if temp_date.weekday() in weekday_indices: final_class_start_str = temp_date.strftime(self.output_date_format); refined_start = True; break
                        temp_date += timedelta(days=1)
                    if refined_start: self.logger.debug(f"Refined Class Start Date to: {final_class_start_str}")
                    refined_end = False; temp_date = end_obj_curr
                    for _ in range(7):
                        if temp_date.weekday() in weekday_indices: final_class_end_str = temp_date.strftime(self.output_date_format); refined_end = True; break
                        temp_date -= timedelta(days=1)
                    if refined_end: self.logger.debug(f"Refined Class End Date to: {final_class_end_str}")
            except Exception as e_ref: self.logger.error(f"Error refining class dates: {e_ref}", exc_info=True)
        class_data["Class Start Date"] = final_class_start_str if final_class_start_str and final_class_start_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] else ""
        class_data["Class End Date"] = final_class_end_str if final_class_end_str and final_class_end_str.upper() not in ["TBD", "TBA", "NOT SPECIFIED"] else ""
        return class_data

    def _generate_class_sessions(self, class_start_date_str: str, class_end_date_str: str, 
                                 weekday_indices: List[int], course_code: str, 
                                 global_class_time_str: str, global_class_location: str,
                                 course_timezone_str: Optional[str]) -> List[Dict[str, Any]]: # Added course_timezone_str
        self.logger.info(f"Generating regular class sessions for '{course_code}' from {class_start_date_str} to {class_end_date_str}.")
        if not self.date_parser: 
            self.logger.error("DateParser not available in _generate_class_sessions.")
            return []
            
        generated_events: List[Dict[str, Any]] = []
        try:
            start_obj_date = self.date_parser.parse(class_start_date_str, ignoretz=True).date()
            end_obj_date = self.date_parser.parse(class_end_date_str, ignoretz=True).date()
            
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
                        elif start_datetime_iso: # If only start time, assume 1 hour duration for end ISO for now
                            try:
                                temp_start_dt = datetime.fromisoformat(start_datetime_iso)
                                temp_end_dt = temp_start_dt + timedelta(hours=1)
                                end_datetime_iso = temp_end_dt.isoformat()
                            except ValueError:
                                self.logger.warning(f"Could not form default end_datetime_iso from start {start_datetime_iso}")
                    
                    generated_events.append({
                        "event_id": event_id,
                        "Event Date": current_date.strftime(self.output_date_format), # Human-readable
                        "Class Time": global_class_time_str, # Human-readable
                        "start_datetime_iso": start_datetime_iso, # Machine-readable
                        "end_datetime_iso": end_datetime_iso,   # Machine-readable
                        "time_zone": course_timezone_str,       # IANA timezone
                        "Event Title": f"{course_code}: Class", 
                        "Class Location": global_class_location,
                        "reading": [], "assignment": [], "assignment_description": None, 
                        "test": [], "special": [], "is_holiday_or_no_class": False 
                    })
                current_date += timedelta(days=1)
            self.logger.info(f"Generated {len(generated_events)} regular class sessions for '{course_code}'.")
        except Exception as e_gen: self.logger.error(f"Error generating class sessions for '{course_code}': {e_gen}", exc_info=True)
        return generated_events
    
    def _merge_schedules(self, generated_sessions: List[Dict[str, Any]], existing_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # ... (implementation remains same as schedule_parser_revised_v1, but ensure IDs/ISO fields are handled) ...
        # This method should now be careful about event_id uniqueness and merging ISO fields if applicable.
        # For simplicity, if an existing event (from AssignmentParser linking) matches a generated session by date,
        # we might prioritize the generated session's ID and ISO times, but merge descriptive content.
        if not self.date_parser: self.logger.error("DateParser unavailable in _merge_schedules."); return existing_events + generated_sessions 
        self.logger.debug(f"Merging {len(generated_sessions)} generated sessions with {len(existing_events)} existing events.")
        merged_event_map: Dict[str, Dict[str, Any]] = {}

        # First, add generated sessions to the map using their event_id if present, or human-readable date as fallback key
        for session in generated_sessions:
            key = session.get("event_id", session.get("Event Date")) # Prefer event_id if it exists
            if key: merged_event_map[key] = session

        # Now, merge or add existing events
        for event in existing_events:
            if isinstance(event, dict) and event.get("Event Date"):
                # Try to match by human-readable date if no event_id on existing event
                norm_date_key = self.date_parser.normalize_date(str(event["Event Date"]), term_year_str=None)
                
                # Check if any generated session already exists for this date to merge into
                matched_session_key = None
                for gen_key, gen_session in merged_event_map.items():
                    if gen_session.get("Event Date") == norm_date_key:
                        matched_session_key = gen_key
                        break
                
                if matched_session_key: # Merge into existing generated session for that date
                    target_session = merged_event_map[matched_session_key]
                    # Prioritize LLM-populated or more specific titles from existing event
                    if event.get("Event Title") and (target_session.get("Event Title", "").endswith(": Class") or not target_session.get("Event Title")):
                        target_session["Event Title"] = event["Event Title"]
                    # Merge lists like assignment, test, special, reading
                    for list_key in ["assignment", "test", "special", "reading"]:
                        target_session.setdefault(list_key, [])
                        if isinstance(event.get(list_key), list):
                            for item in event[list_key]:
                                if item not in target_session[list_key]:
                                    target_session[list_key].append(item)
                    if event.get("assignment_description"):
                        target_session["assignment_description"] = (target_session.get("assignment_description", "") + "; " + event["assignment_description"]).strip("; ")
                    # Ensure ISO fields from generated session are kept if existing event lacks them
                    for iso_key in ["start_datetime_iso", "end_datetime_iso", "time_zone", "event_id"]:
                        target_session.setdefault(iso_key, event.get(iso_key)) # Keep original if it had it

                else: # No generated session for this date, add existing event (ensure it has an ID)
                    event.setdefault("event_id", str(uuid.uuid4()))
                    # Attempt to add ISO fields if missing and possible
                    if not event.get("start_datetime_iso") and self.date_parser and event.get("Class Time") and event.get("time_zone"):
                        start_t, end_t = self.date_parser.parse_time_string_to_objects(event["Class Time"])
                        event["start_datetime_iso"] = self.date_parser.get_iso_datetime_str(event["Event Date"], start_t, event["time_zone"])
                        event["end_datetime_iso"] = self.date_parser.get_iso_datetime_str(event["Event Date"], end_t, event["time_zone"])

                    key_for_existing = event.get("event_id") # Use its new ID as key
                    merged_event_map[key_for_existing] = event
        
        return list(merged_event_map.values())

    def _extract_course_code(self, course_title: Optional[str]) -> str:
        # ... (implementation remains same as schedule_parser_revised_v1) ...
        if not course_title or not isinstance(course_title, str): return "COURSE"
        match = re.search(r'([A-Z]{2,5}\s*\d{2,4}[A-Z]?)', course_title, re.IGNORECASE)
        if match: return "".join(match.group(1).split()).upper() 
        first_word = course_title.split(' ')[0]
        if re.fullmatch(r'[A-Za-z]{2,4}\d{2,4}[A-Za-z]?', first_word): return first_word.upper()
        if len(first_word) <= 7 and first_word.isalnum() and not first_word.isdigit(): return first_word.upper()
        return "COURSE"

if __name__ == '__main__':
    # ... (standalone test block from schedule_parser_revised_v1, updated to check for new fields) ...
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s', 
                        handlers=[logging.StreamHandler(sys.stdout)])
    main_logger = logging.getLogger(__name__)

    if not DATE_PARSER_CLASS_AVAILABLE or not OPENAI_PARSER_CLASS_AVAILABLE:
        main_logger.critical("DateParser or OpenAIParser class not available. ScheduleParser tests cannot run effectively.")
        sys.exit(1)

    mock_config = {
        "extraction": {"openai_model": "gpt-4o"},
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
        "schedule_parser": {"default_term_weeks": 15},
        "openai_parser": {"max_api_retries": 1, "client_timeout": {"read": 60.0, "connect": 10.0}},
        "date_parser": {"output_date_format": "%B %d, %Y"}
    }
    
    test_date_parser = SyllabusDateParser(logger_instance=main_logger, config=mock_config.get("date_parser"))
    
    class MockOpenAIParserForScheduleTest(OpenAIParser):
         def get_json_list_from_prompt(self, prompt: str, unique_id: str) -> Optional[List[Dict[str, Any]]]:
            self.logger.info(f"MockOpenAIParserForScheduleTest.get_json_list_from_prompt called for {unique_id}.")
            if "PHY203" in prompt and "Course Schedule" in prompt: # Adjusted condition
                return [
                    {"date_reference": "Wed., Sept. 11", "topics": ["Kinematics Review", "Intro to Vectors"], "readings": ["Chapter 1"], "activities_notes": ["HW #1 due today"], "is_holiday_or_no_class": False},
                    {"date_reference": "Fri., Sept. 27", "topics": ["Exam #1"], "readings": [], "activities_notes": ["Covers Chaps. 1,3,4."], "is_holiday_or_no_class": False}
                ]
            return []

    test_openai_parser = MockOpenAIParserForScheduleTest(model="gpt-4o", logger_instance=main_logger, config=mock_config)
    if not os.getenv("OPENAI_API_KEY"): main_logger.warning("OPENAI_API_KEY not set for real OpenAIParser test fallback.")

    schedule_parser_instance = ScheduleParser(
        logger_instance=main_logger,
        date_parser_instance=test_date_parser,
        openai_parser_instance=test_openai_parser, 
        config=mock_config
    )

    main_logger.info("\n--- Testing ScheduleParser with PHY203 Example (ISO Datetimes) ---")
    phy203_class_data = {
        "Course Title": "PHY203: ELEMENTARY PHYSICS I", "Course Code": "PHY203",
        "Class Time": "1:00 PM - 1:50 PM", "Time Zone": "America/New_York", 
        "Days of Week": "Monday, Wednesday, Friday",
        "Class Start Date": "September 04, 2024", "Class End Date": "December 11, 2024",
        "Term": "Fall 2024" # For DateParser context
    }
    phy203_segmented_syllabus = {
        "course_schedule": "Week 1: Intro. Wed., Sept. 11: Kinematics. Fri., Sept. 27: Exam #1."
    }
    test_pipeline_input = { "class_data": phy203_class_data, "segmented_syllabus": phy203_segmented_syllabus, "event_data": [] }
    final_processed_data = schedule_parser_instance.process_schedule(test_pipeline_input, "phy203_iso_test")
    
    main_logger.info(f"\n--- Final Class Data (Schedule - ISO Test) ---\n{json.dumps(final_processed_data.get('class_data'), indent=2)}")
    main_logger.info(f"\n--- Final Event Data (Schedule - ISO Test) ---\n{json.dumps(final_processed_data.get('event_data'), indent=2)}")

    if final_processed_data.get('event_data'):
        for event in final_processed_data['event_data']:
            assert "event_id" in event, f"event_id missing in event: {event.get('Event Title')}"
            assert "start_datetime_iso" in event, f"start_datetime_iso missing in event: {event.get('Event Title')}"
            assert "end_datetime_iso" in event, f"end_datetime_iso missing in event: {event.get('Event Title')}"
            assert "time_zone" in event, f"time_zone missing in event: {event.get('Event Title')}"
            if event.get("start_datetime_iso"):
                assert phy203_class_data["Time Zone"] in event["start_datetime_iso"] or phy203_class_data["Time Zone"] == event.get("time_zone"), \
                    f"Timezone mismatch or not in ISO for event: {event.get('Event Title')}, ISO: {event['start_datetime_iso']}"
        main_logger.info("Basic ISO field assertion passed for generated events.")
    else:
        main_logger.warning("No event_data generated in ISO test to assert fields.")
    main_logger.info("\n--- ScheduleParser ISO standalone test finished ---")
