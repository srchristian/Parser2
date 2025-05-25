"""
assignment_parser.py

Handles extraction, processing, and validation of assignments and exams (tasks) from syllabi.
It now primarily uses an LLM call for task extraction from relevant text segments
and aims to populate 'task_data' and link tasks to 'event_data'.
"""

import re
import os
import logging
import json
from typing import Dict, List, Any, Set, Optional

try:
    from utils.parsers.date_parser import DateParser as SyllabusDateParser
    DATE_PARSER_CLASS_AVAILABLE = True
except ImportError:
    DATE_PARSER_CLASS_AVAILABLE = False
    print("Warning: SyllabusDateParser from utils.parsers.date_parser not found. AssignmentParser date operations will be limited.")

try:
    from utils.helpers import extract_course_code
    EXTRACT_COURSE_CODE_AVAILABLE = True
except ImportError:
    EXTRACT_COURSE_CODE_AVAILABLE = False
    print("Warning: extract_course_code from utils.helpers not found for AssignmentParser. Using internal fallback.")
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
    print("Warning: OpenAIParser not found. LLM-based task extraction in AssignmentParser will not be available.")


class AssignmentParser:
    """
    Parser dedicated to extracting and processing assignment and exam information (tasks).
    Uses an LLM for primary extraction from text, normalizes due dates, and links tasks to events.
    """
    
    TASK_TYPES_KEYWORDS = {
        "Homework": ["homework", "hw", "problem set", "ps"],
        "Exam": ["exam", "midterm", "final exam"], # "final" alone might be too broad
        "Quiz": ["quiz"],
        "Project": ["project", "term project"],
        "Paper": ["paper", "essay", "research paper", "report"],
        "Presentation": ["presentation", "oral report"],
        "Lab Report": ["lab report", "laboratory report"], # If labs are part of main tasks
        "Reading Assignment": ["reading assignment", "read chapter"],
        "Other": ["task", "assignment"] # Generic fallback
    }


    def __init__(self, 
                 logger_instance: logging.Logger, 
                 date_parser_instance: Optional[SyllabusDateParser] = None,
                 openai_parser_instance: Optional[OpenAIParser] = None, # Added
                 config: Optional[Dict[str, Any]] = None):
        self.logger = logger_instance
        if not self.logger: # Fallback if logger_instance is None
            self.logger = logging.getLogger(__name__)
            if not self.logger.hasHandlers():
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s')
            self.logger.critical("AssignmentParser initialized with a default fallback logger.")
            
        self.config = config or {}
        self.openai_parser = openai_parser_instance

        if not self.openai_parser and OPENAI_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No OpenAIParser instance provided to AssignmentParser. Attempting to create one.")
            # Ensure model and config are passed to OpenAIParser if created internally
            model = self.config.get("extraction", {}).get("openai_model", "gpt-4o")
            self.openai_parser = OpenAIParser(model=model, logger_instance=self.logger, config=self.config)
        elif not self.openai_parser:
            self.logger.error("OpenAIParser not available and could not be created. LLM-based task extraction will fail.")

        if date_parser_instance:
            self.date_parser = date_parser_instance
        elif DATE_PARSER_CLASS_AVAILABLE:
            self.logger.warning("No DateParser instance provided. Creating a new one for AssignmentParser.")
            self.date_parser = SyllabusDateParser(logger_instance=self.logger)
        else:
            self.date_parser = None
            self.logger.error("DateParser not available. Due date normalization will be limited/fail.")
        
        self._extract_course_code = extract_course_code if EXTRACT_COURSE_CODE_AVAILABLE else _fallback_extract_course_code
        self.logger.info("AssignmentParser initialized.")
    
    def _generate_task_extraction_prompt(self, text_segment: str, course_code: str) -> str:
        """Generates the prompt for the LLM to extract tasks."""
        # Define the desired JSON structure for each task item
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
            self.logger.error("OpenAIParser not available in AssignmentParser. Cannot perform LLM-based task extraction.")
            return [] # Fallback to empty list or could call a regex method here
        if not self.date_parser:
            self.logger.warning("DateParser not available. Extracted due dates will not be normalized by this method.")

        current_class_data = class_data if isinstance(class_data, dict) else {}
        course_code_prefix = self._extract_course_code(current_class_data.get("Course Title", "Course"))

        prompt = self._generate_task_extraction_prompt(syllabus_text_segment, course_code_prefix)
        
        # Use OpenAIParser's generic call_llm or a more specific method if available
        # Assuming OpenAIParser has a method like `call_llm_for_json_list` or similar
        # For now, using a conceptual `self.openai_parser.call_llm_for_structured_output`
        # This would internally handle the API call and JSON parsing.
        # Let's assume it returns a list of dicts or None on failure.

        # This part needs to align with how OpenAIParser is designed to return structured lists.
        # For now, let's assume a method that directly returns a list of task dicts based on a prompt.
        # If OpenAIParser.extract is too specific for class_data, we need a more generic method in it.
        # Let's simulate a direct call for now, assuming OpenAIParser can handle this prompt.
        
        # This is a placeholder for how the LLM call would be made.
        # In a real scenario, OpenAIParser would have a method that takes a prompt
        # and returns the parsed JSON list.
        llm_response_json_list: Optional[List[Dict[str, Any]]] = None
        try:
            # Conceptual: self.openai_parser.call_llm_for_json_list(prompt, unique_id_for_log)
            # For now, using the existing extract method and adapting its use.
            # This is not ideal as extract() is tailored for class_data.
            # A better OpenAIParser would have a more generic method.
            # Let's assume we add a method to OpenAIParser: get_json_response(prompt, unique_id)
            
            # Simplified: Assume a method in OpenAIParser like `get_json_list_from_prompt`
            if hasattr(self.openai_parser, 'get_json_list_from_prompt'):
                 llm_response_json_list = self.openai_parser.get_json_list_from_prompt(prompt, f"{course_code_prefix}_tasks")
            else:
                # Fallback: Manually construct the call if OpenAIParser doesn't have a generic list method
                # This is a simplified representation of what OpenAIParser might do
                if self.openai_parser.client:
                    response = self.openai_parser.client.chat.completions.create(
                        model=self.openai_parser.model, # Use model from openai_parser
                        messages=[{"role": "system", "content": "You are an AI assistant that returns JSON."}, 
                                  {"role": "user", "content": prompt}],
                        response_format={"type": "json_object"} # Expecting a root JSON object that contains the list
                    )
                    raw_response_text = response.choices[0].message.content.strip()
                    # The prompt asks for a JSON array. If response_format={"type": "json_object"}
                    # the LLM might wrap the array in an object like {"tasks": [...]}.
                    # Adjust parsing accordingly.
                    parsed_outer_json = json.loads(raw_response_text)
                    if isinstance(parsed_outer_json, list):
                        llm_response_json_list = parsed_outer_json
                    elif isinstance(parsed_outer_json, dict) and "tasks" in parsed_outer_json and isinstance(parsed_outer_json["tasks"], list):
                        llm_response_json_list = parsed_outer_json["tasks"]
                    else:
                        self.logger.error(f"LLM response for tasks was not a list or expected object: {raw_response_text[:500]}")
                        llm_response_json_list = []

                else:
                    self.logger.error("OpenAI client not available in AssignmentParser's OpenAIParser instance.")
                    return []


        except Exception as e_llm:
            self.logger.error(f"LLM call for task extraction failed: {e_llm}", exc_info=True)
            return [] # Return empty on LLM error

        if not llm_response_json_list:
            self.logger.info("LLM did not return any tasks or response was invalid.")
            return []

        extracted_tasks: List[Dict[str, Any]] = []
        for task_item in llm_response_json_list:
            if not isinstance(task_item, dict):
                self.logger.warning(f"Skipping non-dict item from LLM task list: {task_item}")
                continue

            title = str(task_item.get("title", "")).strip()
            due_date_str = str(task_item.get("due_date_str", "")).strip()
            due_time_str = str(task_item.get("due_time_str", "")).strip()
            description = str(task_item.get("description", "")).strip()
            task_type = str(task_item.get("task_type", "Other")).strip()

            if not title or not due_date_str: # Title and due date are essential
                self.logger.warning(f"Skipping task due to missing title or due_date_str: {task_item}")
                continue

            normalized_due_date = self.date_parser.normalize_date(due_date_str) if self.date_parser else due_date_str
            if normalized_due_date.upper() in ["TBD", "NOT SPECIFIED"]: # Standardize TBD
                normalized_due_date = "TBD"
            
            final_title = f"{course_code_prefix}: {title}" if not title.lower().startswith(course_code_prefix.lower()) else title

            task_entry = {
                "Task Title": final_title,
                "Due Date": normalized_due_date,
                "Due Time": due_time_str, # Store as extracted
                "Task Description": description,
                "Task Type": task_type
            }
            extracted_tasks.append(task_entry)
            self.logger.info(f"LLM Extracted Task: '{final_title}', Due: '{normalized_due_date}' {due_time_str}, Type: {task_type}")
        
        return extracted_tasks

    def process_tasks_from_structured_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Processing tasks from structured data (task_data) for validation and linking.")
        try:
            task_data_list = data.get("task_data")
            if not task_data_list or not isinstance(task_data_list, list):
                self.logger.info("No valid task_data found to process.")
                data.setdefault("task_data", [])
                return data
            
            class_data = data.get("class_data", {})
            course_code = self._extract_course_code(class_data.get("Course Title", "Course"))
            
            # Ensure all tasks have a "Due Time" field, defaulting to "" if missing
            for task in task_data_list:
                if isinstance(task, dict):
                    task.setdefault("Due Time", "")
            
            data = self._validate_assignment_sequence(data, course_code) 
            
            if self.date_parser:
                data = self._link_assignments_to_events(data)
            else:
                self.logger.warning("DateParser not available. Skipping linking of assignments/tasks to events.")
                
            return data
        except Exception as e_process_tasks:
            self.logger.error(f"Error processing tasks from structured data: {e_process_tasks}", exc_info=True)
            return data

    def _validate_assignment_sequence(self, data: Dict[str, Any], course_code: str) -> Dict[str, Any]:
        # (Logic for this method remains largely the same as in the original file)
        # It operates on the task_data list, which is now populated by the LLM.
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
                    numbered_homework_tasks.append({
                        "task_dict_ref": task,
                        "extracted_number": int(num_match.group(1))
                    })
        
        if not numbered_homework_tasks:
            self.logger.info(f"No numbered homework/assignments found for '{course_code}' to validate sequence.")
            return data
            
        numbered_homework_tasks.sort(key=lambda x: x["extracted_number"])
        seen_numbers: Set[int] = set()
        for hw_item in numbered_homework_tasks:
            num = hw_item["extracted_number"]
            if num in seen_numbers:
                self.logger.warning(f"Duplicate assignment number #{num} detected for '{course_code}'. Task: '{hw_item['task_dict_ref'].get('Task Title')}'.")
            seen_numbers.add(num)
        # (Gap checking logic can be added here if needed)
        return data
    
    def _link_assignments_to_events(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # (Logic for this method remains largely the same)
        # It uses normalized "Due Date" from task_data (now LLM-extracted and DateParser-normalized)
        # and links to event_data.
        self.logger.info("Linking tasks to schedule events based on due dates.")
        if not self.date_parser:
            self.logger.error("DateParser not available. Cannot link tasks to events.")
            return data
            
        task_data_list = data.get("task_data")
        event_data_list = data.get("event_data")

        if not task_data_list or not isinstance(task_data_list, list): return data
        if not event_data_list or not isinstance(event_data_list, list): return data
            
        event_map_by_date: Dict[str, Dict[str, Any]] = {}
        for event in event_data_list:
            if isinstance(event, dict) and event.get("Event Date"):
                normalized_event_date = self.date_parser.normalize_date(str(event["Event Date"]))
                if normalized_event_date and normalized_event_date.upper() not in ["TBD", "TBA"]:
                    event_map_by_date.setdefault(normalized_event_date, event) # Store first event for a date

        for task in task_data_list:
            if not isinstance(task, dict): continue
            task_title = str(task.get("Task Title", "")).strip()
            due_date_raw = str(task.get("Due Date", "")).strip()
            task_type = str(task.get("Task Type", "")).strip().lower()

            if task_title and due_date_raw:
                normalized_task_due_date = self.date_parser.normalize_date(due_date_raw)
                if normalized_task_due_date and normalized_task_due_date.upper() not in ["TBD", "TBA"]:
                    if normalized_task_due_date in event_map_by_date:
                        target_event = event_map_by_date[normalized_task_due_date]
                        
                        is_exam_type = "exam" in task_type or any(k in task_title.lower() for k in ["exam", "midterm", "final"])
                        is_quiz_type = "quiz" in task_type

                        if is_exam_type or is_quiz_type:
                            target_event.setdefault("test", [])
                            if task_title not in target_event["test"]: target_event["test"].append(task_title)
                            self.logger.info(f"Linked test/quiz '{task_title}' to event on '{normalized_task_due_date}'.")
                        else: # Default to assignment
                            target_event.setdefault("assignment", [])
                            if task_title not in target_event["assignment"]: target_event["assignment"].append(task_title)
                            # Link description
                            task_desc = str(task.get("Task Description", "")).strip()
                            if task_desc:
                                if not target_event.get("assignment_description"):
                                    target_event["assignment_description"] = task_desc
                                elif task_desc not in target_event["assignment_description"]: # Append if different
                                    target_event["assignment_description"] += f"; {task_desc}"
                            self.logger.info(f"Linked assignment '{task_title}' to event on '{normalized_task_due_date}'.")
        return data

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s', handlers=[logging.StreamHandler()])
    main_logger = logging.getLogger(__name__)

    mock_config = {
        "extraction": {"openai_model": "gpt-4o"},
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")}, # Ensure API key is available for test
        "openai_parser": {"max_api_retries": 1} # For OpenAIParser instance if created
    }
    
    # Mock DateParser
    class MockDateParser:
        def normalize_date(self, date_str: str) -> str:
            if not date_str or date_str.upper() in ["TBD", "NOT SPECIFIED"]: return "TBD"
            try: # Simplified parsing for test
                from dateutil import parser as dateutil_parser
                return dateutil_parser.parse(date_str).strftime("%B %d, %Y")
            except: return date_str # Fallback
        def parse_weekdays_to_indices(self, days_str: str) -> List[int]: return [0,2,4]

    mock_dp = MockDateParser()
    
    # Mock OpenAIParser (essential for this test)
    class MockOpenAIParserForTasks:
        def __init__(self, model, logger_instance, config=None):
            self.model = model; self.logger = logger_instance; self.config = config
            self.client = True # Simulate client being ready
            self.logger.info(f"MockOpenAIParserForTasks initialized with model: {self.model}")

        def get_json_list_from_prompt(self, prompt: str, unique_id: str) -> Optional[List[Dict[str, Any]]]: # Conceptual method
            self.logger.info(f"MockOpenAIParserForTasks.get_json_list_from_prompt called for {unique_id}. Prompt starts: {prompt[:100]}...")
            if "Exam and Homework Schedule" in prompt: # Simulate response for PHY203
                return [
                    {"title": "HW #1", "due_date_str": "Sept. 11", "due_time_str": "10 p.m.", "description": "Problems from Chapter 1.", "task_type": "Homework"},
                    {"title": "Exam #1", "due_date_str": "Fri., Sept. 27", "due_time_str": "", "description": "Covers Chaps. 1,3,4.", "task_type": "Exam"},
                    {"title": "HW #12", "due_date_str": "Wed., Dec. 11", "due_time_str": "10 p.m.", "description": "Problems from Sections 15.1,15.2,15.4.", "task_type": "Homework"},
                    {"title": "Exam #3", "due_date_str": "Wed., Dec. 11", "due_time_str": "", "description": "Covers Chaps. 9-11,15.", "task_type": "Exam"},
                    {"title": "Final Exam", "due_date_str": "TBA", "due_time_str": "", "description": "Covers Chaps. 1-11,15.", "task_type": "Exam"}
                ]
            return []
    
    mock_openai_parser_instance = MockOpenAIParserForTasks(model="gpt-4o", logger_instance=main_logger, config=mock_config)
    
    # Test AssignmentParser with LLM
    assignment_parser_llm = AssignmentParser(
        logger_instance=main_logger,
        date_parser_instance=mock_dp, # type: ignore
        openai_parser_instance=mock_openai_parser_instance, # type: ignore
        config=mock_config
    )

    main_logger.info("\n--- Testing LLM-Enhanced AssignmentParser: Extract Tasks from Text ---")
    # Using the course_schedule segment from phy203_target_output_v1 as input text
    phy203_schedule_segment = """Exam and Homework Schedule
Homework must be submitted to WebAssign by 10 p.m. of the date due.
Wed., Sept. 11 HW #1 due Reading: Chapter 1
Wed., Sept. 18 HW #2 due Reading: Sections 3.1-3.5
Wed., Sept. 25 HW #3 due Reading: Sections 2.1-2.3, 4.1-4.3
Fri., Sept. 27 Exam #1 (Chaps. 1,3,4)
Fri., Oct. 4 HW #4 due Reading: Chap. 5
Fri., Oct. 11 HW #5 due Reading: Sections 4.4, 6.1-6.3
Fri., Oct. 18 HW #6 due Reading: 1st part of Sec. 2.4, Chap. 7
Fri., Oct. 25 HW #7 due Reading: Chap. 8
Fri., Oct. 25 Exam #2 (Chaps. 5-8)
Wed., Oct. 30 No Homework due Reading: Sections 9.1-9.6
Wed., Nov. 6 HW #8 due Reading: Sections 10.1-10.6
Wed., Nov. 13 No Homework due Reading: 2nd part of Sec. 2.4, Secs. 11-1-11.3
Wed., Nov. 20 HW #9 due
Wed., Nov. 27 HW #10 due Reading: Sections 13.1-13.5
Wed., Dec. 4 HW #11 due Reading: Sections 15.1,15.2,15.4
Wed., Dec. 11 HW #12 due Reading: Sections 15.1,15.2,15.4
Wed., Dec. 11 Exam #3 (Chaps. 9-11,15)
Final Exam TBA Chaps. 1-11,15"""
    
    sample_class_data_for_phy203 = {"Course Title": "PHY203: ELEMENTARY PHYSICS I"}
    extracted_tasks_llm = assignment_parser_llm.extract_tasks_from_text(phy203_schedule_segment, sample_class_data_for_phy203)
    main_logger.info(f"LLM Extracted Tasks:\n{json.dumps(extracted_tasks_llm, indent=2)}")

    main_logger.info("\n--- Testing Linking with LLM Extracted Tasks ---")
    test_data_for_linking = {
        "class_data": sample_class_data_for_phy203,
        "task_data": extracted_tasks_llm, # Use tasks from LLM
        "event_data": [ # Sample events for linking (assuming ScheduleParser ran partially)
            {"Event Date": "September 11, 2024", "Event Title": "PHY203: Class", "reading": [], "assignment": [], "test": [], "special": []},
            {"Event Date": "September 27, 2024", "Event Title": "PHY203: Class", "reading": [], "assignment": [], "test": [], "special": []},
            {"Event Date": "December 11, 2024", "Event Title": "PHY203: Class", "reading": [], "assignment": [], "test": [], "special": []},
        ]
    }
    processed_data_llm = assignment_parser_llm.process_tasks_from_structured_data(test_data_for_linking)
    main_logger.info(f"Event Data after linking LLM tasks:\n{json.dumps(processed_data_llm.get('event_data'), indent=2)}")
    main_logger.info(f"Task Data after processing LLM tasks:\n{json.dumps(processed_data_llm.get('task_data'), indent=2)}")

    main_logger.info("\n--- LLM-Enhanced AssignmentParser standalone tests finished ---")
