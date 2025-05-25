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
import uuid # Added for unique task IDs
from typing import Dict, List, Any, Set, Optional

try:
    # Assuming SyllabusDateParser is the actual class name in date_parser.py
    # This parser should now have get_iso_datetime_str and parse_time_string_to_objects methods
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
    from utils.parsers.openai_parser import OpenAIParser # For LLM calls
    OPENAI_PARSER_CLASS_AVAILABLE = True
except ImportError:
    OPENAI_PARSER_CLASS_AVAILABLE = False
    print("Warning (assignment_parser.py): OpenAIParser not found. LLM-based task extraction in AssignmentParser will not be available.")


class AssignmentParser:
    """
    Parser dedicated to extracting and processing assignment and exam information (tasks).
    Uses an LLM for primary extraction from text, normalizes due dates, generates ISO datetimes,
    assigns unique IDs, and links tasks to events.
    """
    
    TASK_TYPES_KEYWORDS = {
        "Homework": ["homework", "hw", "problem set", "ps"],
        "Exam": ["exam", "midterm", "final exam"],
        "Quiz": ["quiz"],
        "Project": ["project", "term project"],
        "Paper": ["paper", "essay", "research paper", "report"],
        "Presentation": ["presentation", "oral report"],
        "Lab Report": ["lab report", "laboratory report"],
        "Reading Assignment": ["reading assignment", "read chapter"],
        "Other": ["task", "assignment"] 
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
                # DateParser might need config if output_date_format is to be configured
                self.date_parser = SyllabusDateParser(logger_instance=self.logger, config=self.config.get("date_parser"))
            except Exception as e_dp_init:
                self.logger.error(f"Failed to auto-initialize DateParser for AssignmentParser: {e_dp_init}", exc_info=True)
                self.date_parser = None
        else:
            self.date_parser = None
            self.logger.error("DateParser not available. Due date normalization and ISO generation will be limited/fail.")
        
        self._extract_course_code = extract_course_code if EXTRACT_COURSE_CODE_AVAILABLE else _fallback_extract_course_code
        self.logger.info("AssignmentParser initialized.")
    
    def _generate_task_extraction_prompt(self, text_segment: str, course_code: str) -> str:
        # ... (prompt definition remains the same as in your provided file) ...
        task_schema_description = """
        For each identified task (assignment, exam, quiz, project, paper, presentation, etc.), provide:
        - "title": A concise but descriptive title for the task (e.g., "Homework 1: Chapter Problems", "Midterm Exam 1", "Research Paper Draft").
        - "due_date_str": The due date as it appears in the text (e.g., "Sept. 11", "Friday, October 25th", "Week 3 Friday", "TBA").
        - "due_time_str": The specific due time if mentioned (e.g., "10 p.m.", "11:59 PM", "start of class"). If no time is specified, use an empty string "".
        - "description": A comprehensive description of the task, including any relevant details like topics covered, submission instructions, or specific requirements.
        - "task_type": The type of task. Choose from: Homework, Exam, Quiz, Project, Paper, Presentation, Lab Report, Reading Assignment, Other.
        """
        prompt = f"""
You are an expert academic assistant. Your task is to meticulously extract all assignments, exams, quizzes, projects, papers, and other graded or scheduled tasks from the provided syllabus text segment.
The course code for this syllabus is: {course_code}. Please use this to help clarify task titles if needed, but primarily extract titles as they appear.

Input Text Segment:
---
{text_segment}
---

Extraction Instructions:
1. Identify every distinct task mentioned.
2. For each task, extract the following details:
   - "title": (String) A concise but descriptive title for the task. If it's a numbered item (e.g., "HW #1"), include the number.
   - "due_date_str": (String) The due date exactly as it appears in the text. If a date is implied by a week (e.g., "due Friday of Week 3"), extract that phrase. If no date, use "TBD" or "Not Specified".
   - "due_time_str": (String) The specific due time if mentioned (e.g., "10 p.m.", "11:59 PM", "by start of class"). If no time is specified, use an empty string "".
   - "description": (String) A comprehensive description of the task, including topics covered, format, submission instructions, or any other relevant details provided.
   - "task_type": (String) Classify the task. Choose one from: Homework, Exam, Quiz, Project, Paper, Presentation, Lab Report, Reading Assignment, Other. Be as specific as possible.

3. Output Format: Respond with a single JSON array, where each element is an object representing a task with the fields "title", "due_date_str", "due_time_str", "description", and "task_type".
   Example of an array element:
   {{
     "title": "HW #1: Chapter 1 Problems",
     "due_date_str": "Sept. 11",
     "due_time_str": "10 p.m.",
     "description": "Complete problems 1.1, 1.5, and 1.12 from Chapter 1. Submit via WebAssign.",
     "task_type": "Homework"
   }}
   If no tasks are found, return an empty JSON array [].

CRITICAL:
- Extract information ONLY as it is explicitly stated or very clearly implied by direct context. Do NOT infer or guess.
- If a detail for a field is not present, use an empty string "" for string fields (except for due_date_str, use "TBD" or "Not Specified" if truly no date).
- Ensure the output is a valid JSON array.
"""
        return prompt

    def extract_tasks_from_text(self, syllabus_text_segment: str, class_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.logger.info(f"Attempting to extract tasks using LLM from text segment of length {len(syllabus_text_segment)} chars.")
        if not syllabus_text_segment or not syllabus_text_segment.strip():
            self.logger.info("Input text segment for task extraction is empty.")
            return []
        if not self.openai_parser:
            self.logger.error("OpenAIParser not available. Cannot perform LLM-based task extraction.")
            return []
        if not self.date_parser: # DateParser is crucial for new fields
            self.logger.error("DateParser not available. Due date normalization and ISO generation will fail.")
            return []

        current_class_data = class_data if isinstance(class_data, dict) else {}
        course_code_prefix = self._extract_course_code(current_class_data.get("Course Title", "Course"))
        # Timezone is crucial for generating correct ISO datetime strings
        course_timezone_str = current_class_data.get("Time Zone")
        if not course_timezone_str:
            self.logger.warning(f"Course Time Zone is missing in class_data for {course_code_prefix}. ISO datetimes for tasks may be incorrect or missing.")
            # Consider a default timezone from config or skip ISO generation if critical
            # For now, we'll let get_iso_datetime_str handle a None timezone_str if it occurs

        prompt = self._generate_task_extraction_prompt(syllabus_text_segment, course_code_prefix)
        
        llm_response_json_list: Optional[List[Dict[str, Any]]] = None
        try:
            if hasattr(self.openai_parser, 'get_json_list_from_prompt') and callable(getattr(self.openai_parser, 'get_json_list_from_prompt')):
                 llm_response_json_list = self.openai_parser.get_json_list_from_prompt(prompt, f"{course_code_prefix}_tasks")
            elif self.openai_parser.client:
                self.logger.debug("Using fallback manual OpenAI call for task extraction.")
                response = self.openai_parser.client.chat.completions.create(
                    model=self.openai_parser.model,
                    messages=[{"role": "system", "content": "You are an AI assistant that returns JSON formatted data as requested."}, 
                              {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"} 
                )
                raw_response_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
                if raw_response_text:
                    parsed_outer_json = json.loads(raw_response_text) 
                    if isinstance(parsed_outer_json, list):
                        llm_response_json_list = parsed_outer_json
                    elif isinstance(parsed_outer_json, dict):
                        # Common pattern: LLM wraps array in a key, e.g., "tasks"
                        found_list = None
                        for key_in_response in ["tasks", "assignments", "deliverables", "task_list"]: # Try common keys
                            if key_in_response in parsed_outer_json and isinstance(parsed_outer_json[key_in_response], list):
                                found_list = parsed_outer_json[key_in_response]
                                break
                        if found_list is not None:
                            llm_response_json_list = found_list
                        else: # Fallback: try to find any list value if no specific key matches
                            for val in parsed_outer_json.values():
                                if isinstance(val, list): llm_response_json_list = val; break
                            if not llm_response_json_list:
                                self.logger.error(f"LLM task response (json_object mode) was a dict but did not contain a clear list of tasks: {str(parsed_outer_json)[:500]}")
                    else: 
                        self.logger.error(f"LLM response for tasks was not a list or expected dict structure: {raw_response_text[:500]}")
                else: self.logger.warning("LLM returned empty response for tasks.")
            else: self.logger.error("OpenAI client not available or get_json_list_from_prompt missing.")
        except Exception as e_llm:
            self.logger.error(f"LLM call for task extraction failed: {e_llm}", exc_info=True)
        
        if not llm_response_json_list:
            self.logger.info("LLM did not return any tasks or response was invalid.")
            return []

        extracted_tasks: List[Dict[str, Any]] = []
        for task_item in llm_response_json_list:
            if not isinstance(task_item, dict):
                self.logger.warning(f"Skipping non-dict item from LLM task list: {task_item}")
                continue

            title = str(task_item.get("title", "")).strip()
            due_date_str_raw = str(task_item.get("due_date_str", "")).strip() # Raw from LLM
            due_time_str_raw = str(task_item.get("due_time_str", "")).strip() # Raw from LLM
            description = str(task_item.get("description", "")).strip()
            task_type = str(task_item.get("task_type", "Other")).strip()

            if not title or not due_date_str_raw:
                self.logger.warning(f"Skipping task due to missing title or raw due_date_str: {task_item}")
                continue
            
            # Human-readable normalized date
            normalized_due_date_hr = self.date_parser.normalize_date(due_date_str_raw, term_year_str=current_class_data.get("Term"))
            if normalized_due_date_hr.upper() in ["TBD", "NOT SPECIFIED"]:
                normalized_due_date_hr = "TBD"
            
            # Generate ISO 8601 datetime string
            due_datetime_iso_str = None
            if normalized_due_date_hr != "TBD" and course_timezone_str: # Only if date is valid and timezone is known
                # Pass the human-readable normalized date, original time string, and course timezone
                # is_end_time_for_due_date=True suggests defaulting to end of day if time is ambiguous/missing
                due_datetime_iso_str = self.date_parser.get_iso_datetime_str(
                    normalized_due_date_hr, # Use already normalized date for consistency
                    due_time_str_raw, 
                    course_timezone_str,
                    is_end_time_for_due_date=True 
                )
            elif not course_timezone_str:
                self.logger.warning(f"Cannot generate ISO due date for task '{title}' because course timezone is missing.")


            final_title = f"{course_code_prefix}: {title}" if not title.lower().startswith(course_code_prefix.lower()) else title
            task_id = str(uuid.uuid4()) # Generate unique ID

            task_entry = {
                "task_id": task_id,
                "Task Title": final_title,
                "Due Date": normalized_due_date_hr, # Human-readable
                "Due Time": due_time_str_raw,       # Human-readable (original string)
                "due_datetime_iso": due_datetime_iso_str, # Machine-readable
                "time_zone": course_timezone_str,         # Store the timezone used
                "Task Description": description,
                "Task Type": task_type
            }
            extracted_tasks.append(task_entry)
            self.logger.info(f"LLM Extracted Task: ID='{task_id}', Title='{final_title}', DueHR='{normalized_due_date_hr} {due_time_str_raw}', DueISO='{due_datetime_iso_str}'")
        
        return extracted_tasks

    def process_tasks_from_structured_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # This method now primarily ensures task_data exists and then calls validation/linking.
        # The main transformation (adding ISO dates, IDs) should ideally happen in extract_tasks_from_text
        # or a dedicated transformation step if tasks can come from multiple sources.
        # For now, assuming extract_tasks_from_text is the main populator.
        self.logger.info("Processing tasks from structured data (task_data) for validation and linking.")
        try:
            if "task_data" not in data or not isinstance(data["task_data"], list):
                self.logger.info("No valid task_data found to process. Initializing as empty list.")
                data["task_data"] = [] # Ensure it exists
            
            class_data = data.get("class_data", {})
            course_code = self._extract_course_code(class_data.get("Course Title", "Course"))
            
            # Post-validation/cleanup if needed on the task_data (already populated by extract_tasks_from_text)
            for task in data["task_data"]:
                if isinstance(task, dict):
                    task.setdefault("Due Time", "") # Ensure field exists for human display
                    task.setdefault("due_datetime_iso", None) # Ensure field exists
                    task.setdefault("time_zone", class_data.get("Time Zone"))
                    task.setdefault("task_id", str(uuid.uuid4())) # Ensure ID if somehow missed

            data = self._validate_assignment_sequence(data, course_code) 
            
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
        # ... (implementation remains the same as your provided file) ...
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
            if "homework" in task_type_lower or any(keyword in task_title_lower for keyword in ["homework", "hw", "problem set", "ps"]):
                num_match = re.search(r'(?:#|\b)(\d+)\b', task.get("Task Title", ""))
                if num_match:
                    numbered_homework_tasks.append({"task_dict_ref": task, "extracted_number": int(num_match.group(1))})
        if not numbered_homework_tasks:
            self.logger.info(f"No numbered homework/assignments found for '{course_code}' to validate sequence.")
            return data
        numbered_homework_tasks.sort(key=lambda x: x["extracted_number"])
        seen_numbers: Set[int] = set()
        for hw_item in numbered_homework_tasks:
            num = hw_item["extracted_number"]
            if num in seen_numbers: self.logger.warning(f"Duplicate assignment number #{num} detected for '{course_code}'. Task: '{hw_item['task_dict_ref'].get('Task Title')}'.")
            seen_numbers.add(num)
        return data
    
    def _link_assignments_to_events(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # ... (implementation remains the same as your provided file, uses human-readable Due Date) ...
        self.logger.info("Linking tasks to schedule events based on due dates.")
        if not self.date_parser: self.logger.error("DateParser not available. Cannot link tasks."); return data
        task_data_list = data.get("task_data"); event_data_list = data.get("event_data")
        if not task_data_list or not isinstance(task_data_list, list): return data
        if not event_data_list or not isinstance(event_data_list, list): return data
        event_map_by_date: Dict[str, Dict[str, Any]] = {}
        for event in event_data_list:
            if isinstance(event, dict) and event.get("Event Date"):
                normalized_event_date = self.date_parser.normalize_date(str(event["Event Date"]), term_year_str=data.get("class_data",{}).get("Term"))
                if normalized_event_date and normalized_event_date.upper() not in ["TBD", "TBA"]:
                    event_map_by_date.setdefault(normalized_event_date, event)
        for task in task_data_list:
            if not isinstance(task, dict): continue
            task_title = str(task.get("Task Title", "")).strip()
            due_date_hr = str(task.get("Due Date", "")).strip() # Use the human-readable normalized date
            task_type = str(task.get("Task Type", "")).strip().lower()
            if task_title and due_date_hr and due_date_hr.upper() not in ["TBD", "TBA"]:
                if due_date_hr in event_map_by_date:
                    target_event = event_map_by_date[due_date_hr]
                    is_exam_type = "exam" in task_type or any(k in task_title.lower() for k in ["exam", "midterm", "final"])
                    is_quiz_type = "quiz" in task_type
                    link_key = "test" if is_exam_type or is_quiz_type else "assignment"
                    target_event.setdefault(link_key, [])
                    if task_title not in target_event[link_key]: target_event[link_key].append(task_title)
                    self.logger.info(f"Linked task '{task_title}' to event on '{due_date_hr}'.")
                    task_desc = str(task.get("Task Description", "")).strip()
                    if link_key == "assignment" and task_desc: # Add description for assignments
                        if not target_event.get("assignment_description"): target_event["assignment_description"] = task_desc
                        elif task_desc not in target_event["assignment_description"]: target_event["assignment_description"] += f"; {task_desc}"
        return data

if __name__ == '__main__':
    # ... (standalone test block remains largely the same as your provided file,
    #      but ensure it tests the new fields: task_id, due_datetime_iso, time_zone) ...
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s', handlers=[logging.StreamHandler()])
    main_logger = logging.getLogger(__name__)

    mock_config = {
        "extraction": {"openai_model": "gpt-4o"},
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")}, 
        "openai_parser": {"max_api_retries": 1},
        "date_parser": {"output_date_format": "%B %d, %Y"} # For DateParser internal config
    }
    
    if not DATE_PARSER_CLASS_AVAILABLE or not OPENAI_PARSER_CLASS_AVAILABLE:
        main_logger.critical("DateParser or OpenAIParser class not available. AssignmentParser tests cannot run effectively.")
        sys.exit(1)
        
    test_date_parser = SyllabusDateParser(logger_instance=main_logger, config=mock_config.get("date_parser"))
    
    # Mock OpenAIParser specifically for tasks
    class MockOpenAIParserForTasksTest(OpenAIParser): # Inherit to get client setup if real API key is used
        def get_json_list_from_prompt(self, prompt: str, unique_id: str) -> Optional[List[Dict[str, Any]]]:
            self.logger.info(f"MockOpenAIParserForTasksTest.get_json_list_from_prompt called for {unique_id}. Prompt starts: {prompt[:150]}...")
            if "PHY203" in prompt and "assignments, exams, quizzes" in prompt :
                return [
                    {"title": "HW #1", "due_date_str": "Sept. 11", "due_time_str": "10 p.m.", "description": "Problems from Chapter 1.", "task_type": "Homework"},
                    {"title": "Exam #1", "due_date_str": "Fri., Sept. 27", "due_time_str": "", "description": "Covers Chaps. 1,3,4.", "task_type": "Exam"},
                    {"title": "Final Project Report", "due_date_str": "December 10", "due_time_str": "5:00 PM", "description": "Submit final project PDF.", "task_type": "Project"}
                ]
            self.logger.warning("MockOpenAIParserForTasksTest: No specific mock response triggered.")
            return []

    test_openai_parser = MockOpenAIParserForTasksTest(model="gpt-4o", logger_instance=main_logger, config=mock_config)
    if not os.getenv("OPENAI_API_KEY"):
        main_logger.warning("OPENAI_API_KEY not set. MockOpenAIParserForTasksTest might not fully simulate client if it relies on it.")
    
    assignment_parser_instance = AssignmentParser(
        logger_instance=main_logger,
        date_parser_instance=test_date_parser,
        openai_parser_instance=test_openai_parser,
        config=mock_config
    )

    main_logger.info("\n--- Testing AssignmentParser: extract_tasks_from_text ---")
    sample_text_segment = """
    Homework assignments are due weekly on Wednesdays by 10 p.m. unless otherwise noted.
    HW #1: Introduction to Python. Due: Sept. 11.
    Midterm Exam: October 15th during class time.
    Final Project: Full report due December 10, 5:00 PM.
    Quiz 1: Friday of Week 2, covers Chapter 1.
    """
    sample_class_data = {
        "Course Title": "CS101 Intro to Programming",
        "Course Code": "CS101", # Used by _extract_course_code
        "Term": "Fall 2024",     # Used by DateParser for year context if needed
        "Time Zone": "America/New_York" # Crucial for ISO datetime
    }
    
    extracted_tasks = assignment_parser_instance.extract_tasks_from_text(sample_text_segment, sample_class_data)
    main_logger.info(f"Extracted Tasks (with ISO and IDs):\n{json.dumps(extracted_tasks, indent=2)}")

    assert len(extracted_tasks) > 0, "No tasks were extracted by LLM mock."
    first_task = extracted_tasks[0]
    assert "task_id" in first_task, "task_id missing from extracted task."
    assert "due_datetime_iso" in first_task, "due_datetime_iso missing from extracted task."
    assert "time_zone" in first_task, "time_zone missing from extracted task."
    if first_task.get("Due Date") != "TBD" and sample_class_data["Time Zone"]:
         assert first_task.get("due_datetime_iso") is not None, f"due_datetime_iso should be populated for '{first_task.get('Task Title')}'"
         assert sample_class_data["Time Zone"] in first_task.get("due_datetime_iso", ""), "Timezone offset missing or wrong in ISO string."


    main_logger.info("\n--- Testing process_tasks_from_structured_data (with already extracted tasks) ---")
    data_for_processing = {
        "class_data": sample_class_data,
        "task_data": extracted_tasks, # Use the tasks we just extracted
        "event_data": [ # Dummy event data for linking test
            {"Event Date": test_date_parser.normalize_date("September 11, 2024", term_year_str="Fall 2024"), "Event Title": "CS101: Class"},
            {"Event Date": test_date_parser.normalize_date("October 15, 2024", term_year_str="Fall 2024"), "Event Title": "CS101: Class"}
        ]
    }
    processed_data_final = assignment_parser_instance.process_tasks_from_structured_data(data_for_processing)
    main_logger.info(f"Final Task Data after processing:\n{json.dumps(processed_data_final.get('task_data'), indent=2)}")
    main_logger.info(f"Event Data after linking:\n{json.dumps(processed_data_final.get('event_data'), indent=2)}")

    main_logger.info("\n--- AssignmentParser standalone tests finished ---")
