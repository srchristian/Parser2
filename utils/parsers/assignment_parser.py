"""
assignment_parser.py

Handles extraction, processing, and validation of assignments and exams (tasks) from syllabi.
It now primarily uses an LLM call for task extraction from relevant text segments
and aims to populate 'task_data' with both human-readable and ISO 8601 datetime
fields, unique IDs, and timezone information.
"""

import re
import os
import logging
import sys
import json
import uuid 
from typing import Dict, List, Any, Set, Optional

try:
    from utils.parsers.date_parser import DateParser as SyllabusDateParser
    DATE_PARSER_CLASS_AVAILABLE = True
except ImportError:
    DATE_PARSER_CLASS_AVAILABLE = False
    print("Warning (assignment_parser.py): SyllabusDateParser from utils.parsers.date_parser not found. AssignmentParser date operations will be limited.")

try:
    from utils.helpers import extract_course_code
    EXTRACT_COURSE_CODE_AVAILABLE = True
except ImportError:
    EXTRACT_COURSE_CODE_AVAILABLE = False
    print("Warning (assignment_parser.py): extract_course_code from utils.helpers not found for AssignmentParser. Using internal fallback.")
    def _fallback_extract_course_code(course_title: Optional[str]) -> str:
        if not course_title or not isinstance(course_title, str): return "COURSE"
        match = re.search(r'([A-Z]{2,4}\s*\d{3,4}[A-Z]?)', course_title, re.IGNORECASE)
        if match: return "".join(match.group(1).split()).upper()
        first_word = course_title.split(' ')[0]
        if re.match(r'^[A-Za-z]{2,4}\d{3,4}[A-Za-z]?$', first_word): return first_word.upper()
        return "COURSE"

try:
    from utils.parsers.openai_parser import OpenAIParser 
    OPENAI_PARSER_CLASS_AVAILABLE = True
except ImportError:
    OPENAI_PARSER_CLASS_AVAILABLE = False
    print("Warning (assignment_parser.py): OpenAIParser not found. LLM-based task extraction in AssignmentParser will not be available.")


class AssignmentParser:
    """
    Parser dedicated to extracting and processing assignment and exam information (tasks).
    """
    
    TASK_TYPES_KEYWORDS = {
        "Homework": ["homework", "hw", "problem set", "ps"],
        "Exam": ["exam", "midterm", "final exam", "final"],
        "Quiz": ["quiz"],
        "Project": ["project", "term project"],
        "Paper": ["paper", "essay", "research paper", "report"],
        "Presentation": ["presentation", "oral report"],
        "Lab Report": ["lab report", "laboratory report"],
        "Reading Assignment": ["reading assignment", "read chapter"],
        "Other": ["task", "assignment", "deliverable"] 
    }

    def __init__(self, 
                 logger_instance: logging.Logger, 
                 date_parser_instance: Optional[SyllabusDateParser] = None,
                 openai_parser_instance: Optional[OpenAIParser] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.logger = logger_instance
        if not self.logger: 
            self.logger = logging.getLogger(__name__)
            if not self.logger.hasHandlers():
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s')
            self.logger.critical("AssignmentParser initialized with a default fallback logger.")
            
        self.config = config or {}
        self.openai_parser = openai_parser_instance

        if not self.openai_parser and OPENAI_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No OpenAIParser instance provided to AssignmentParser. Attempting to create one.")
            model = self.config.get("extraction", {}).get("openai_model", "gpt-4o")
            try:
                self.openai_parser = OpenAIParser(model=model, logger_instance=self.logger, config=self.config)
            except Exception as e_op_init:
                self.logger.error(f"Failed to auto-initialize OpenAIParser in AssignmentParser: {e_op_init}", exc_info=True)
                self.openai_parser = None
        elif not self.openai_parser:
            self.logger.error("OpenAIParser not available/creatable. LLM-based task extraction will fail.")

        if date_parser_instance:
            self.date_parser = date_parser_instance
        elif DATE_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No DateParser instance provided. Creating a new one for AssignmentParser.")
            try:
                self.date_parser = SyllabusDateParser(logger_instance=self.logger, config=self.config.get("date_parser"))
            except Exception as e_dp_init:
                self.logger.error(f"Failed to auto-initialize DateParser for AssignmentParser: {e_dp_init}", exc_info=True)
                self.date_parser = None
        else:
            self.date_parser = None
            self.logger.error("DateParser class not available. Due date normalization and ISO generation will be limited/fail.")
        
        self._extract_course_code = extract_course_code if EXTRACT_COURSE_CODE_AVAILABLE else _fallback_extract_course_code
        self.logger.info("AssignmentParser initialized.")
    
    def _generate_task_extraction_prompt(self, text_segment: str, course_code: str) -> str:
        task_types_list_str = ", ".join(self.TASK_TYPES_KEYWORDS.keys())
        prompt = f"""
You are an expert academic assistant. Your task is to meticulously extract all assignments, exams, quizzes, projects, papers, presentations, and other graded or scheduled tasks from the provided syllabus text segment.
The course code for this syllabus is: {course_code}.

Input Text Segment:
---
{text_segment}
---

Extraction Instructions:
1. Identify every distinct task mentioned.
2. For each task, extract the following details:
   - "title": (String) A concise but descriptive title for the task (e.g., "Homework 1: Chapter Problems", "Midterm Exam 1", "Research Paper Draft"). If it's a numbered item (e.g., "HW #1"), include the number.
   - "due_date_str": (String) The due date exactly as it appears in the text. If a date is implied by a week (e.g., "due Friday of Week 3"), extract that phrase. If no date, use "TBD".
   - "due_time_str": (String) The specific due time if mentioned (e.g., "10 p.m.", "11:59 PM", "by start of class"). If no time is specified, use an empty string "".
   - "description": (String) A comprehensive description of the task, including topics covered, format, submission instructions, or any other relevant details provided. If minimal, extract what's there.
   - "task_type": (String) Classify the task. Choose one from: {task_types_list_str}. Be as specific as possible based on the title and description.

Output Format:
If you are an AI model that must return a JSON object as the top-level structure (e.g. when `response_format` is `json_object`),
return a JSON object with a single key "tasks". The value of "tasks"
must be an array of objects, where each object represents a task.
Example: {{"tasks": [{{"title": "HW #1", ...}}, {{...}}]}}
If only one task is found, "tasks" should be an array containing that single task object.
If no tasks are found, the "tasks" array should be empty: {{"tasks": []}}.

If you are an AI model that can return a JSON array directly as the top-level structure, then return just the array of task objects.
If no tasks are found, return an empty JSON array [].

Example of a single task object:
   {{
     "title": "HW #1: Chapter 1 Problems",
     "due_date_str": "Sept. 11",
     "due_time_str": "10 p.m.",
     "description": "Complete problems 1.1, 1.5, and 1.12 from Chapter 1. Submit via WebAssign.",
     "task_type": "Homework"
   }}

CRITICAL:
- Extract information ONLY as it is explicitly stated or very clearly implied by direct context. Do NOT infer or guess.
- If a detail for a field (like description or due_time_str) is not present, use an empty string "". For due_date_str, use "TBD" if truly no date information.
- Ensure your entire output is a single, valid JSON structure as specified.
"""
        return prompt

    def extract_tasks_from_text(self, syllabus_text_segment: str, class_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.logger.info(f"Attempting to extract tasks using LLM from text segment of length {len(syllabus_text_segment)} chars.")
        if not syllabus_text_segment or not syllabus_text_segment.strip():
            self.logger.info("Input text segment for task extraction is empty.")
            return []
        if not self.openai_parser:
            self.logger.error("OpenAIParser not available in AssignmentParser. Cannot perform LLM-based task extraction.")
            return []
        if not self.date_parser: 
            self.logger.error("DateParser not available in AssignmentParser. Due date normalization and ISO generation will fail.")
            return []

        current_class_data = class_data if isinstance(class_data, dict) else {}
        course_code_prefix = self._extract_course_code(current_class_data.get("Course Title", "Course"))
        course_timezone_str = current_class_data.get("Time Zone")
        if not course_timezone_str:
            self.logger.warning(f"Course Time Zone is missing in class_data for {course_code_prefix}. ISO datetimes for tasks may be incorrect or missing.")

        prompt = self._generate_task_extraction_prompt(syllabus_text_segment, course_code_prefix)
        
        llm_response_json_list: Optional[List[Dict[str, Any]]] = None
        raw_response_text_for_debug = "" # For logging in case of error
        try:
            if hasattr(self.openai_parser, 'get_json_list_from_prompt') and callable(getattr(self.openai_parser, 'get_json_list_from_prompt')):
                 self.logger.debug("AssignmentParser using OpenAIParser.get_json_list_from_prompt for tasks.")
                 llm_response_json_list = self.openai_parser.get_json_list_from_prompt(prompt, f"{course_code_prefix}_tasks")
            elif self.openai_parser.client: # Fallback direct client call
                self.logger.debug("AssignmentParser using direct OpenAI client call (response_format: json_object) for tasks.")
                response = self.openai_parser.client.chat.completions.create(
                    model=self.openai_parser.model,
                    messages=[{"role": "system", "content": "You are an AI assistant. Respond strictly with the JSON structure requested by the user."},
                              {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"} 
                )
                raw_response_text_for_debug = response.choices[0].message.content.strip() if response.choices and response.choices[0].message and response.choices[0].message.content else ""
                if raw_response_text_for_debug:
                    self.logger.debug(f"AssignmentParser LLM Raw Response (direct call): {raw_response_text_for_debug[:1000]}")
                    parsed_outer_json = json.loads(raw_response_text_for_debug) 
                    
                    if isinstance(parsed_outer_json, list): # Should ideally not happen with json_object
                        llm_response_json_list = parsed_outer_json
                        self.logger.warning("LLM returned a direct list for tasks in json_object mode, which is unexpected but handled.")
                    elif isinstance(parsed_outer_json, dict):
                        # Try to extract list from common keys
                        preferred_keys = ["tasks", "assignments", "task_list", "exams"]
                        found_list = None
                        for pk in preferred_keys:
                            if pk in parsed_outer_json and isinstance(parsed_outer_json[pk], list):
                                found_list = parsed_outer_json[pk]
                                self.logger.info(f"Found task list under key '{pk}' in LLM response.")
                                break
                        if found_list is not None:
                            llm_response_json_list = found_list
                        else: 
                            # Check if the dictionary itself IS a single task object (as seen in logs)
                            # A simple check: does it have a 'title' and 'due_date_str'?
                            if "title" in parsed_outer_json and "due_date_str" in parsed_outer_json:
                                self.logger.info("LLM returned a single task object directly. Wrapping it in a list.")
                                llm_response_json_list = [parsed_outer_json]
                            else: # Fallback: try to find any list value if no specific key matches
                                for val in parsed_outer_json.values():
                                    if isinstance(val, list): 
                                        llm_response_json_list = val
                                        self.logger.info(f"Found task list under an unexpected key in LLM response dict.")
                                        break
                                if not llm_response_json_list: # Still not found
                                    self.logger.error(f"LLM task response (json_object mode) was a dict but did not contain a recognized list of tasks nor was it a single task object: {str(parsed_outer_json)[:500]}")
                    else: 
                        self.logger.error(f"LLM response for tasks after JSON parsing was not a list or dict: Type={type(parsed_outer_json)}, Content: {raw_response_text_for_debug[:500]}")
                else: 
                    self.logger.warning("LLM returned empty response for tasks (direct call).")
            else:
                self.logger.error("OpenAIParser client not available or method get_json_list_from_prompt missing and no client. Cannot extract tasks.")
                return []
        except json.JSONDecodeError as e_json_decode:
            self.logger.error(f"JSONDecodeError for task extraction. Raw text: '{raw_response_text_for_debug}' Error: {e_json_decode}", exc_info=True)
            return []
        except Exception as e_llm:
            self.logger.error(f"LLM call or processing for task extraction failed: {e_llm}", exc_info=True)
            return []
        
        if not llm_response_json_list:
            self.logger.info("LLM did not return any tasks or response was invalid after all parsing attempts.")
            return []
        if not isinstance(llm_response_json_list, list):
             self.logger.error(f"Processed LLM response for tasks is not a list. Type: {type(llm_response_json_list)}. Skipping task processing.")
             return []


        extracted_tasks: List[Dict[str, Any]] = []
        for task_item in llm_response_json_list:
            if not isinstance(task_item, dict):
                self.logger.warning(f"Skipping non-dict item from LLM task list: {task_item}")
                continue

            title = str(task_item.get("title", "")).strip()
            due_date_str_raw = str(task_item.get("due_date_str", "")).strip() 
            due_time_str_raw = str(task_item.get("due_time_str", "")).strip() 
            description = str(task_item.get("description", "")).strip()
            task_type = str(task_item.get("task_type", "")).strip()

            if not task_type: # Try to infer task_type if LLM missed it
                for type_name, keywords in self.TASK_TYPES_KEYWORDS.items():
                    if any(keyword in title.lower() for keyword in keywords) or \
                       any(keyword in description.lower() for keyword in keywords):
                        task_type = type_name
                        self.logger.debug(f"Inferred task_type '{task_type}' for '{title}' based on keywords.")
                        break
                if not task_type: task_type = "Other"


            if not title or not due_date_str_raw or due_date_str_raw.upper() == "NOT SPECIFIED": # Allow "TBD"
                self.logger.debug(f"Skipping task due to missing title or 'Not Specified' raw due_date_str: {task_item}")
                continue
            
            normalized_due_date_hr = "TBD"
            if self.date_parser: # Check if date_parser was successfully initialized
                normalized_due_date_hr = self.date_parser.normalize_date(due_date_str_raw, term_year_str=current_class_data.get("Term"))
                if normalized_due_date_hr.upper() in ["TBD", "NOT SPECIFIED"]: # Standardize to TBD
                    normalized_due_date_hr = "TBD"
            else: # DateParser unavailable
                normalized_due_date_hr = due_date_str_raw # Use raw if no DateParser

            due_datetime_iso_str = None
            if self.date_parser and normalized_due_date_hr != "TBD" and course_timezone_str: 
                try:
                    due_datetime_iso_str = self.date_parser.get_iso_datetime_str(
                        normalized_due_date_hr, 
                        due_time_str_raw, 
                        course_timezone_str,
                        is_end_time_for_due_date=True 
                    )
                except Exception as e_iso:
                    self.logger.warning(f"Could not generate ISO for task '{title}' due date '{normalized_due_date_hr}' time '{due_time_str_raw}': {e_iso}")
            elif not course_timezone_str and normalized_due_date_hr != "TBD":
                self.logger.warning(f"Cannot generate ISO due date for task '{title}' because course timezone is missing (Date: {normalized_due_date_hr}).")


            final_title = f"{course_code_prefix}: {title}" if course_code_prefix != "COURSE" and not title.lower().startswith(course_code_prefix.lower()) else title
            task_id = str(uuid.uuid4()) 

            task_entry = {
                "task_id": task_id,
                "Task Title": final_title,
                "Due Date": normalized_due_date_hr, 
                "Due Time": due_time_str_raw,       
                "due_datetime_iso": due_datetime_iso_str, 
                "time_zone": course_timezone_str if course_timezone_str else None, # Store even if None
                "Task Description": description,
                "Task Type": task_type,
                "Raw Due Date String": due_date_str_raw if normalized_due_date_hr != due_date_str_raw else None # Store raw if different
            }
            if task_entry["Raw Due Date String"] is None:
                del task_entry["Raw Due Date String"]

            extracted_tasks.append(task_entry)
            self.logger.info(f"LLM Extracted Task: ID='{task_id}', Title='{final_title}', DueHR='{normalized_due_date_hr} {due_time_str_raw}', DueISO='{due_datetime_iso_str}'")
        
        return extracted_tasks

    def process_tasks_from_structured_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Processing tasks from structured data (task_data) for validation and linking.")
        try:
            if "task_data" not in data or not isinstance(data["task_data"], list):
                self.logger.info("No valid task_data found to process. Initializing as empty list.")
                data["task_data"] = [] 
            
            class_data = data.get("class_data", {})
            course_code_from_data = class_data.get("Course Code", "").strip()
            if not course_code_from_data:
                 course_code_from_data = self._extract_course_code(class_data.get("Course Title", "Course"))
            
            for task in data.get("task_data", []): # Iterate over a copy or ensure it's safe
                if isinstance(task, dict):
                    task.setdefault("Due Time", "") 
                    task.setdefault("due_datetime_iso", None) 
                    task.setdefault("time_zone", class_data.get("Time Zone")) # Ensure TZ is present
                    task.setdefault("task_id", str(uuid.uuid4())) 
                    # Re-normalize Due Date if it's just "TBD" but other date info might exist
                    if task.get("Due Date", "").upper() == "TBD" and self.date_parser and task.get("Raw Due Date String"):
                        new_norm_date = self.date_parser.normalize_date(
                            task["Raw Due Date String"], 
                            term_year_str=class_data.get("Term")
                        )
                        if new_norm_date.upper() not in ["TBD", "NOT SPECIFIED"]:
                            task["Due Date"] = new_norm_date
                            # Try to re-gen ISO
                            if task.get("time_zone"):
                                task["due_datetime_iso"] = self.date_parser.get_iso_datetime_str(
                                    new_norm_date, task.get("Due Time",""), task["time_zone"], is_end_time_for_due_date=True
                                )


            data = self._validate_assignment_sequence(data, course_code_from_data) 
            
            if self.date_parser:
                data = self._link_assignments_to_events(data)
            else:
                self.logger.warning("DateParser not available. Skipping linking of assignments/tasks to events.")
                
            return data
        except Exception as e_process_tasks:
            self.logger.error(f"Error processing tasks from structured data: {e_process_tasks}", exc_info=True)
            data.setdefault("task_data", []) 
            return data

    def _validate_assignment_sequence(self, data: Dict[str, Any], course_code: str) -> Dict[str, Any]:
        self.logger.info(f"Validating assignment sequence for course '{course_code}'.")
        task_data = data.get("task_data")
        if not task_data or not isinstance(task_data, list):
            self.logger.debug("No task_data or task_data is not a list. Skipping assignment sequence validation.")
            return data
        numbered_homework_tasks: List[Dict[str, Any]] = [] 
        for task in task_data:
            if not isinstance(task, dict): continue
            task_title_lower = task.get("Task Title", "").lower()
            task_type_lower = task.get("Task Type", "").lower()
            if "homework" in task_type_lower or any(keyword in task_title_lower for keyword in self.TASK_TYPES_KEYWORDS["Homework"]):
                num_match = re.search(r'(?:#|\b(?:no\.\s*)?)(\d+)\b', task.get("Task Title", ""), re.IGNORECASE) # More flexible number matching
                if num_match:
                    numbered_homework_tasks.append({"task_dict_ref": task, "extracted_number": int(num_match.group(1))})
        if not numbered_homework_tasks:
            self.logger.info(f"No numbered homework/assignments found for '{course_code}' to validate sequence.")
            return data
        numbered_homework_tasks.sort(key=lambda x: x["extracted_number"])
        seen_numbers: Set[int] = set()
        expected_next = 1
        for hw_item in numbered_homework_tasks:
            num = hw_item["extracted_number"]
            if num != expected_next and expected_next not in seen_numbers : # Only warn if it's an unexpected skip
                 self.logger.warning(f"Potential missing assignment number for '{course_code}'. Expected ~{expected_next}, found #{num}. Task: '{hw_item['task_dict_ref'].get('Task Title')}'.")
            if num in seen_numbers: 
                self.logger.warning(f"Duplicate assignment number #{num} detected for '{course_code}'. Task: '{hw_item['task_dict_ref'].get('Task Title')}'.")
            seen_numbers.add(num)
            if num >= expected_next: # Allow for jumps, but reset expectation
                expected_next = num + 1
        return data
    
    def _link_assignments_to_events(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Linking tasks to schedule events based on due dates.")
        if not self.date_parser: 
            self.logger.error("DateParser not available in _link_assignments_to_events. Cannot link tasks."); 
            return data
            
        task_data_list = data.get("task_data"); 
        event_data_list = data.get("event_data")
        
        if not isinstance(task_data_list, list) or not task_data_list: 
            self.logger.debug("No task_data to link.")
            return data
        if not isinstance(event_data_list, list) or not event_data_list: 
            self.logger.debug("No event_data to link tasks to.")
            return data

        class_data = data.get("class_data", {})
        term_for_context = class_data.get("Term")

        event_map_by_date: Dict[str, Dict[str, Any]] = {}
        for event in event_data_list:
            if isinstance(event, dict) and event.get("Event Date"):
                # Use term_for_context when normalizing event dates for matching
                normalized_event_date = self.date_parser.normalize_date(str(event["Event Date"]), term_year_str=term_for_context)
                if normalized_event_date and normalized_event_date.upper() not in ["TBD", "TBA", "NOT SPECIFIED"]:
                    event_map_by_date.setdefault(normalized_event_date, event) # Link to the first event on that date
        
        for task in task_data_list:
            if not isinstance(task, dict): continue
            task_title = str(task.get("Task Title", "")).strip()
            due_date_hr = str(task.get("Due Date", "")).strip() # This should already be normalized by extract_tasks or process_tasks
            task_type = str(task.get("Task Type", "")).strip().lower()

            if task_title and due_date_hr and due_date_hr.upper() not in ["TBD", "TBA", "NOT SPECIFIED"]:
                # Due Date in task is already normalized, so direct lookup should work if formats match
                if due_date_hr in event_map_by_date:
                    target_event = event_map_by_date[due_date_hr]
                    is_exam_type = "exam" in task_type or any(k in task_title.lower() for k in ["exam", "midterm", "final"])
                    is_quiz_type = "quiz" in task_type
                    link_key = "test" if is_exam_type or is_quiz_type else "assignment"
                    
                    target_event.setdefault(link_key, [])
                    if task_title not in target_event[link_key]: 
                        target_event[link_key].append(task_title)
                        self.logger.info(f"Linked task '{task_title}' to event on '{due_date_hr}' under key '{link_key}'.")
                    
                    task_desc = str(task.get("Task Description", "")).strip()
                    if link_key == "assignment" and task_desc: 
                        current_event_desc = target_event.get("assignment_description", "")
                        if not current_event_desc: 
                            target_event["assignment_description"] = task_desc
                        elif task_desc not in current_event_desc: 
                            target_event["assignment_description"] = f"{current_event_desc}; {task_desc}"
                else:
                    self.logger.debug(f"No matching event found in schedule for task '{task_title}' due on '{due_date_hr}'.")
        return data

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s', handlers=[logging.StreamHandler()])
    main_logger = logging.getLogger(__name__)

    mock_config_main = {
        "extraction": {"openai_model": "gpt-4o"},
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")}, 
        "openai_parser": {"max_api_retries": 1},
        "date_parser": {"output_date_format": "%B %d, %Y"} 
    }
    
    if not DATE_PARSER_CLASS_AVAILABLE or not OPENAI_PARSER_CLASS_AVAILABLE:
        main_logger.critical("DateParser or OpenAIParser class not available. AssignmentParser tests cannot run effectively.")
        sys.exit(1)
        
    test_date_parser_instance = SyllabusDateParser(logger_instance=main_logger, config=mock_config_main.get("date_parser")) # Renamed
    
    class MockOpenAIParserForTasksTest(OpenAIParser): 
        def get_json_list_from_prompt(self, prompt: str, unique_id: str) -> Optional[List[Dict[str, Any]]]:
            self.logger.info(f"MockOpenAIParserForTasksTest.get_json_list_from_prompt called for {unique_id}. Prompt starts: {prompt[:150]}...")
            # Simulate a case where the LLM returns a single object instead of a list initially
            if "PHY203_tasks" in unique_id and "Homework assignments" in prompt : # Check based on actual text for more reliability
                 self.logger.info("MockOpenAIParserForTasksTest: Simulating single task object return.")
                 return {"title": "Exam Unit #1", "due_date_str": "Fri. 09/27", "due_time_str": "Class Time", "description": "Exam on Chapters 1, 3, 4 covering Unit #1.", "task_type": "Exam"} # type: ignore
            elif "CS101_tasks" in unique_id: # For the other test case
                self.logger.info("MockOpenAIParserForTasksTest: Simulating list of tasks return for CS101.")
                return [
                    {"title": "HW #1", "due_date_str": "Sept. 11", "due_time_str": "10 p.m.", "description": "Problems from Chapter 1.", "task_type": "Homework"},
                    {"title": "Midterm Exam", "due_date_str": "October 15th", "due_time_str": "during class time", "description": "Covers first half of course.", "task_type": "Exam"},
                ]
            self.logger.warning("MockOpenAIParserForTasksTest: No specific mock response triggered, returning empty list.")
            return []

    test_openai_parser_instance = MockOpenAIParserForTasksTest(model="gpt-4o", logger_instance=main_logger, config=mock_config_main) # Renamed
    if not os.getenv("OPENAI_API_KEY") and test_openai_parser_instance.client is None: # Check if client would be None
        main_logger.warning("OPENAI_API_KEY not set and mock client not initialized. Direct calls in parser might fail if not mocked.")
    
    assignment_parser_main_instance = AssignmentParser( # Renamed
        logger_instance=main_logger,
        date_parser_instance=test_date_parser_instance,
        openai_parser_instance=test_openai_parser_instance,
        config=mock_config_main
    )

    main_logger.info("\n--- Testing AssignmentParser: extract_tasks_from_text (Simulating single object return from LLM) ---")
    # This text should trigger the single object return from the mock
    single_task_text_segment = "Final Exam: December 15th, 2 PM. Comprehensive. This is the PHY203_tasks trigger." 
    single_task_class_data = {
        "Course Title": "PHY203 Advanced Topics", "Course Code": "PHY203",
        "Term": "Spring 2025", "Time Zone": "America/New_York"
    }
    extracted_single_task = assignment_parser_main_instance.extract_tasks_from_text(single_task_text_segment, single_task_class_data)
    main_logger.info(f"Extracted Tasks (single object test):\n{json.dumps(extracted_single_task, indent=2)}")
    assert len(extracted_single_task) == 1, f"Expected 1 task from single object simulation, got {len(extracted_single_task)}"
    assert extracted_single_task[0]["Task Title"] == "PHY203: Exam Unit #1", "Task title mismatch for single object test"


    main_logger.info("\n--- Testing AssignmentParser: extract_tasks_from_text (Standard list return) ---")
    sample_text_segment_tasks = """
    Homework assignments are due weekly on Wednesdays by 10 p.m. unless otherwise noted.
    HW #1: Introduction to Python. Due: Sept. 11.
    Midterm Exam: October 15th during class time.
    Final Project: Full report due December 10, 5:00 PM.
    Quiz 1: Friday of Week 2, covers Chapter 1.
    """
    sample_class_data_map = { # Renamed
        "Course Title": "CS101 Intro to Programming", "Course Code": "CS101",
        "Term": "Fall 2024", "Time Zone": "America/New_York" 
    }
    
    extracted_tasks_list = assignment_parser_main_instance.extract_tasks_from_text(sample_text_segment_tasks, sample_class_data_map) # Renamed
    main_logger.info(f"Extracted Tasks (list test):\n{json.dumps(extracted_tasks_list, indent=2)}")

    if extracted_tasks_list: # Ensure there's something to check
        assert len(extracted_tasks_list) > 0, "No tasks were extracted by LLM mock for list test."
        first_task_in_list = extracted_tasks_list[0] # Renamed
        assert "task_id" in first_task_in_list, "task_id missing from extracted task."
        assert "due_datetime_iso" in first_task_in_list, "due_datetime_iso missing from extracted task."
        assert "time_zone" in first_task_in_list, "time_zone missing from extracted task."
        if first_task_in_list.get("Due Date") != "TBD" and sample_class_data_map["Time Zone"]:
             assert first_task_in_list.get("due_datetime_iso") is not None, f"due_datetime_iso should be populated for '{first_task_in_list.get('Task Title')}'"
             assert sample_class_data_map["Time Zone"] in first_task_in_list.get("due_datetime_iso", ""), "Timezone offset missing or wrong in ISO string."
    else:
        main_logger.warning("No tasks extracted in the list test scenario.")


    main_logger.info("\n--- Testing process_tasks_from_structured_data (with already extracted tasks) ---")
    data_for_processing_map = { # Renamed
        "class_data": sample_class_data_map,
        "task_data": extracted_tasks_list, # Use the tasks we just extracted
        "event_data": [ 
            {"Event Date": test_date_parser_instance.normalize_date("September 11, 2024", term_year_str="Fall 2024"), "Event Title": "CS101: Class", "event_id": "evt1"},
            {"Event Date": test_date_parser_instance.normalize_date("October 15, 2024", term_year_str="Fall 2024"), "Event Title": "CS101: Class", "event_id": "evt2"}
        ]
    }
    processed_data_final_map = assignment_parser_main_instance.process_tasks_from_structured_data(data_for_processing_map) # Renamed
    main_logger.info(f"Final Task Data after processing:\n{json.dumps(processed_data_final_map.get('task_data'), indent=2)}")
    main_logger.info(f"Event Data after linking:\n{json.dumps(processed_data_final_map.get('event_data'), indent=2)}")
    # Check if linking worked
    linked_event_sept11 = next((e for e in processed_data_final_map.get('event_data', []) if e.get("Event Date") == "September 11, 2024"), None)
    assert linked_event_sept11 and "HW #1" in linked_event_sept11.get("assignment", []), "HW #1 not linked to Sept 11 event."


    main_logger.info("\n--- AssignmentParser standalone tests finished ---")
