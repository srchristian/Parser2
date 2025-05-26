"""
syllabus_parser.py

Text segmentation module that extracts and segments syllabus text into logical sections
based on a predefined schema. It focuses on identifying structure and removing
irrelevant content while preserving all schedule, assignment, and course information.
It now includes specific segmentation for Laboratory and Recitation/Discussion sections.
Optionally uses config.yaml if provided.
This version expects pre-converted plain text files (.txt) as input.
"""

import os
import sys
import re
import json
import uuid
import time
import logging # Standard library logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Attempt to load .env, though typically app.py or system env is used
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # This print statement is okay for initial setup feedback
    print("Info: python-dotenv not installed. Will use existing environment variables if set.")

# Attempt to import openai and httpx (for client timeout)
try:
    import openai
    import httpx # Used for client timeout configuration in OpenAI SDK v1.0.0+
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI SDK or httpx not installed. LLM-based functionality will be limited.")
    OPENAI_AVAILABLE = False

# --- Module-level Logger ---
# This logger will be used by default if no specific logger is passed to the class.
# Its configuration (handlers, level) should be set by the importing application (like app.py)
# or by the __main__ block if run standalone.
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Add a default null handler to prevent "No handler found" warnings
    logger.addHandler(logging.NullHandler())

# --- Project Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent # Assuming utils/parsers/ is two levels down from project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    # Use print for this initial setup message as logger might not be fully configured by app yet.
    print(f"Info (syllabus_parser.py): Added {PROJECT_ROOT} to sys.path for module imports.")


# --- Helper Function Imports (with Fallbacks) ---
# These fallbacks are crucial if the module is used in an environment where utils.helpers isn't available.
try:
    from utils.helpers import read_json, write_json
    HELPERS_JSON_IMPORTED = True
    # Logging of successful import will be done by the class instance logger.
except ImportError as e:
    HELPERS_JSON_IMPORTED = False
    print(f"Warning (syllabus_parser.py): utils.helpers import for JSON functions failed ({e}). Using fallback JSON functions. Ensure __init__.py files exist in 'utils' and 'utils/parsers'.")
    # Fallback functions if utils.helpers cannot be imported
    def read_json(file_path: Path, fallback_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        effective_fallback = fallback_data if fallback_data is not None else {}
        try:
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as exc_fallback_read:
            print(f"ERROR (Fallback read_json): Error reading JSON from {file_path}: {exc_fallback_read}")
            return effective_fallback

    def write_json(file_path: Path, data: Dict[str, Any], create_dirs: bool = True, backup_existing: bool = False, indent: int = 4) -> bool:
        try:
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            if backup_existing and file_path.exists():
                backup_path = file_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{file_path.suffix}")
                # In a simple fallback, might not have shutil, so just overwrite or skip backup.
                # For robustness, let's assume simple overwrite if shutil is not available.
                print(f"INFO (Fallback write_json): Backup requested but not implemented in simple fallback. Overwriting {file_path} if it exists.")
            with file_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            return True
        except Exception as exc_fallback_write:
            print(f"ERROR (Fallback write_json): Error writing JSON to {file_path}: {exc_fallback_write}")
            return False

class SyllabusParser:
    """
    Segments pre-converted syllabus text (from a .txt file) into predefined logical sections.
    Prioritizes LLM-based segmentation if available and configured,
    otherwise uses a rule-based fallback method.
    Includes specific handling for lab and recitation sections.
    """
    # Define schema keys for segmentation.
    # Core modules are expected to be present in some form.
    CORE_MODULES = [
        "course_identification", "instructor_information", "course_description_prerequisites",
        "learning_objectives", "required_texts_materials", "course_requirements_grading_policy",
        "assignments_exams", "course_schedule", "course_policies",
        "communication_student_support", "university_wide_policies_resources"
    ]
    # Optional modules are included if relevant content is found.
    OPTIONAL_MODULES = [
        "separate_laboratory_sections",
        "recitation_discussion_sections",
        "practicum_clinical_experiences", "field_trips_off_campus_activities",
        "service_learning_community_engagement_projects", "studio_workshop_sessions",
        "performance_recital_production_requirements", "safety_protocols_lab_attire_expanded",
        "internship_fieldwork_documentation", "emergency_health_safety_protocols_course_specific",
        "ethics_controversial_content_statements"
    ]
    UNCLASSIFIED_KEY = "unclassified_content" # For text not fitting other categories.
    ALL_POSSIBLE_KEYS = CORE_MODULES + OPTIONAL_MODULES + [UNCLASSIFIED_KEY]

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 base_dir: Optional[str] = None,
                 model: str = "gpt-4o", # Default model if not specified elsewhere
                 log_dir: Optional[str] = None, 
                 openai_api_key: Optional[str] = None,
                 logger_instance: Optional[logging.Logger] = None):

        self.logger = logger_instance if logger_instance else logging.getLogger(__name__)
        if not self.logger.handlers and logger_instance is None:
            self.logger.addHandler(logging.StreamHandler(sys.stderr))
            self.logger.setLevel(logging.WARNING)
            self.logger.warning(f"SyllabusParser's logger '{__name__}' was not configured. Using basic stderr handler.")

        self.config = config or {}
        # Prioritize model from config['extraction'], then config['openai'], then constructor arg 'model'
        self.model_preference = self.config.get("extraction", {}).get("openai_model", 
                                 self.config.get("openai", {}).get("model", model))
        self.client = None

        if HELPERS_JSON_IMPORTED:
             self.logger.info("SyllabusParser will use imported JSON helpers.")
        else:
             self.logger.warning("SyllabusParser using internal fallback JSON functions.")

        if base_dir: self.base_dir = Path(base_dir)
        elif "directories" in self.config and "parsed_syllabus_dir" in self.config["directories"]:
            self.base_dir = Path(self.config["directories"]["parsed_syllabus_dir"])
        else:
            self.base_dir = CURRENT_DIR / "syllabus_repository" / "parsed_syllabus_output" # Default path
        
        self._create_directories()

        self.openai_status = "not_initialized"
        if OPENAI_AVAILABLE:
            resolved_api_key = openai_api_key or os.getenv("OPENAI_API_KEY") or \
                               self.config.get("extraction", {}).get("openai_api_key") or \
                               self.config.get("openai", {}).get("api_key")

            if resolved_api_key:
                try:
                    client_timeout_config = self.config.get("openai_parser", {}).get("client_timeout",
                                             self.config.get("openai", {}).get("client_timeout", {"read": 90.0, "connect": 15.0}))
                    read_timeout = float(client_timeout_config.get("read", 90.0))
                    connect_timeout = float(client_timeout_config.get("connect", 15.0))
                    default_timeout = httpx.Timeout(read_timeout, connect=connect_timeout)

                    self.client = openai.OpenAI(api_key=resolved_api_key, timeout=default_timeout)
                    self.openai_status = "ready"
                    self.logger.info(f"OpenAI client initialized for SyllabusParser. Model preference: {self.model_preference}. Timeout: {read_timeout}s read, {connect_timeout}s connect.")
                except Exception as e_openai_init:
                    self.logger.error(f"Error initializing OpenAI client for SyllabusParser: {e_openai_init}", exc_info=True)
                    self.openai_status = "error_initialization"
            else:
                self.logger.warning("OpenAI API key not found. OpenAI client for SyllabusParser not initialized.")
                self.openai_status = "no_api_key"
        else:
            self.logger.warning("OpenAI SDK (openai package) or httpx not installed. LLM-based segmentation for SyllabusParser unavailable.")
            self.openai_status = "sdk_not_available"
        
        self.logger.info(f"SyllabusParser initialized. Output dir: {self.base_dir}. Expects pre-converted .txt files for parsing.")

    def _create_directories(self):
        """Creates necessary directories, primarily self.base_dir for output."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            # self.logger.info(f"SyllabusParser output directory ensured: {self.base_dir}") # Already logged at end of __init__
        except OSError as e_mkdir:
            self.logger.error(f"Error creating base directory {self.base_dir} for SyllabusParser: {e_mkdir}", exc_info=True)

    def _extract_text_from_file(self, file_path: Path) -> Optional[str]:
        """
        Extracts text from a pre-converted plain text file (.txt).
        """
        self.logger.debug(f"Attempting to extract text from pre-converted file: {file_path}")
        if file_path.suffix.lower() == ".txt":
            try:
                encodings_to_try = ['utf-8', 'latin-1', 'windows-1252']
                for enc in encodings_to_try:
                    try: 
                        return file_path.read_text(encoding=enc).strip()
                    except UnicodeDecodeError: 
                        self.logger.debug(f"Read {file_path.name} failed with {enc}.")
                self.logger.error(f"Could not decode {file_path.name} with common encodings.")
                return None
            except Exception as e_txt: 
                self.logger.error(f"Error reading {file_path.name}: {e_txt}", exc_info=True)
                return None
        else: 
            self.logger.error(f"Unsupported file: '{file_path.suffix}'. Expects '.txt'.")
            return None

    def parse_syllabus(self, input_text_file_path: str, unique_id: Optional[str] = None) -> Dict[str, Any]:
        input_file = Path(input_text_file_path)
        self.logger.info(f"Starting syllabus segmentation for: {input_file} (ID: {unique_id or 'N/A'})")
        start_time = time.monotonic()
        job_id = unique_id or str(uuid.uuid4())

        if not input_file.is_file(): 
            self.logger.error(f"Input file not found: {input_file}")
            return {"error": f"Input file not found: {input_file}", "status": "error_file_not_found", "full_text_content": None, "unique_id": job_id}
        if input_file.suffix.lower() != ".txt": 
            self.logger.error(f"Invalid file type: {input_file.suffix}. Expected '.txt'.")
            return {"error": f"Invalid file type: {input_file.suffix}. Expected '.txt'.", "status": "error_invalid_file_type", "full_text_content": None, "unique_id": job_id}

        syllabus_text = self._extract_text_from_file(input_file)
        if syllabus_text is None: 
            self.logger.error(f"Could not extract text from {input_file}")
            return {"error": f"Could not extract text from {input_file}", "status": "error_text_extraction_failed", "full_text_content": None, "unique_id": job_id}
        if not syllabus_text.strip(): 
            self.logger.warning(f"Extracted text from {input_file} is empty.")

        self.logger.info(f"Extracted {len(syllabus_text)} chars from {input_file.name} for segmentation (ID: {job_id}).")

        parsed_sections: Optional[Dict[str, Any]] = None
        parsing_method = "llm"
        if self.openai_status == "ready" and self.client:
            if len(syllabus_text.strip()) < 150: 
                self.logger.warning(f"Syllabus text for {job_id} too short ({len(syllabus_text.strip())} chars). Using fallback segmentation.")
            else:
                prompt = self._generate_segmentation_prompt(syllabus_text)
                response_text = self._call_openai_api(prompt, job_id)
                if response_text:
                    parsed_sections = self._parse_llm_response(response_text, job_id) # Pass job_id for logging
                    if not parsed_sections: 
                        self.logger.warning(f"Failed to parse LLM response for {job_id}. Using fallback.")
                else: 
                    self.logger.warning(f"No valid response from OpenAI API for {job_id}. Using fallback.")
        else: 
            self.logger.warning(f"OpenAI client not ready ({self.openai_status}). Using fallback segmentation for {job_id}.")

        if not parsed_sections:
            self.logger.info(f"Using fallback rule-based segmentation for {job_id}.")
            parsed_sections = self._fallback_segmentation(syllabus_text)
            parsing_method = "fallback"
            if not any(parsed_sections.get(key) for key in self.CORE_MODULES if key != self.UNCLASSIFIED_KEY):
                self.logger.warning(f"Fallback segmentation for {job_id} resulted in mostly empty core sections.")
        
        # Post-process sections, whether from LLM or fallback
        if parsed_sections: # Ensure parsed_sections is not None
            parsed_sections = self._post_process_sections(parsed_sections, syllabus_text)
        else: # Should ideally not happen if fallback always returns a dict
            parsed_sections = {key: "" for key in self.CORE_MODULES}
            parsed_sections[self.UNCLASSIFIED_KEY] = [syllabus_text] if syllabus_text.strip() else []
            self.logger.error(f"Catastrophic failure in segmentation for {job_id}, parsed_sections was None before post-processing.")


        parsed_sections["UUID"] = job_id 
        parsed_sections["parsing_method_used"] = parsing_method
        parsed_sections["segmentation_timestamp"] = datetime.now().isoformat()
        
        output_file_name = f"{job_id}_segmented_syllabus.json"
        parsed_output_file = self.base_dir / output_file_name
        save_success = write_json(parsed_output_file, parsed_sections)
        process_time_seconds = time.monotonic() - start_time
        
        result_payload = {
            "status": "success", "message": "Syllabus segmentation complete.",
            "parsed_output_file": str(parsed_output_file) if save_success else None,
            "parsed_data": parsed_sections, "process_time": round(process_time_seconds, 3),
            "full_text_content": syllabus_text, "unique_id": job_id 
        }
        if not save_success:
            result_payload["status"] = "warning_save_failed"
            result_payload["message"] = f"Segmentation OK, but failed to save output to {parsed_output_file}."
            result_payload["error"] = f"Failed to save to {parsed_output_file}"
        self.logger.info(f"Segmentation for {job_id} done in {result_payload['process_time']:.2f}s. Method: {parsing_method}. Output: {result_payload['parsed_output_file'] or 'Not Saved'}")
        return result_payload

    def get_module_description(self, key: str) -> str:
        # (Content from user's uploaded file - already very detailed)
        descriptions = {
            "course_identification": "Course number/code (e.g., CS 101, MATH 200), full title, term/semester (Fall 2024, Spring 2025), credit hours, meeting times, modality (in-person/online/hybrid), section numbers",
            "instructor_information": "Instructor/professor name(s), titles (Dr., Prof.), email addresses, phone numbers, office location, office hours (days/times), teaching assistants (TAs) information, preferred contact methods",
            "course_description_prerequisites": "Official catalog description, course overview/summary, learning approach, prerequisites (required prior courses), corequisites (concurrent requirements), recommended preparation, placement test requirements",
            "learning_objectives": "Learning outcomes, course goals, competencies, skills to be developed, 'By the end of this course, students will...', measurable objectives using action verbs (analyze, create, evaluate)",
            "required_texts_materials": "Textbook titles, authors, editions, ISBN numbers, publishers, required vs optional materials, software requirements, hardware needs, online access codes, lab equipment, calculators, art supplies",
            "course_requirements_grading_policy": "Grade breakdown/weights (e.g., Exams 40%, Homework 30%), grading scale (A: 90-100, B: 80-89), assignment types and percentages, attendance/participation points, extra credit policies, grade calculation methods",
            "assignments_exams": "Specific assignments with due dates, exam dates and times, project deadlines, paper due dates, presentation schedules, quiz information, homework submission details, major deliverables list",
            "course_schedule": "Weekly/daily schedule, class-by-class topics, reading assignments by date, topic progression, exam dates within schedule, assignment due dates in context, holiday/break notations, 'Week 1:', 'Session 1:', date-based entries",
            "course_policies": "Attendance requirements, late work policy, make-up exam policy, academic integrity/plagiarism statement, classroom behavior expectations, technology use policies, absence procedures, disability accommodations process",
            "communication_student_support": "Email response times, preferred communication channels, LMS/Canvas/Blackboard info, discussion forum guidelines, tutoring center info, writing center resources, library support, study group information",
            "university_wide_policies_resources": "FERPA privacy statement, Title IX information, non-discrimination policy, ADA/disability services, honor code, campus safety resources, counseling services, emergency procedures, withdrawal policies",
            "separate_laboratory_sections": "Lab section numbers/times, lab location/room numbers, lab instructor/TA names, lab requirements, safety protocols, lab manual info, experiment schedules, lab report due dates, lab attendance policies",
            "recitation_discussion_sections": "Recitation/discussion section times, section numbers, room locations, TA names for sections, participation requirements, discussion topics, problem-solving sessions, review session schedules",
            "practicum_clinical_experiences": "Clinical rotation schedules, practicum requirements, field placement information, supervisor contacts, hour requirements, documentation needs",
            "field_trips_off_campus_activities": "Field trip dates/locations, transportation arrangements, costs, permission forms, off-campus meeting points, required preparations",
            "service_learning_community_engagement_projects": "Community partner information, service hour requirements, project descriptions, reflection assignments, community engagement goals",
            "studio_workshop_sessions": "Studio times/locations, workshop schedules, critique dates, portfolio requirements, exhibition information, materials for studios",
            "performance_recital_production_requirements": "Performance dates, rehearsal schedules, audition information, costume/equipment needs, ticket information, venue details",
            "safety_protocols_lab_attire_expanded": "Required safety equipment, PPE requirements, lab dress code, chemical handling procedures, emergency protocols, safety training dates",
            "internship_fieldwork_documentation": "Internship requirements, fieldwork hours, supervisor evaluations, journal/log requirements, portfolio submissions, placement procedures",
            "emergency_health_safety_protocols_course_specific": "Course-specific safety procedures, health requirements, immunization needs, emergency contacts, evacuation procedures, first aid information",
            "ethics_controversial_content_statements": "Content warnings, sensitive topic notifications, ethical guidelines, professional conduct expectations, trigger warnings, alternative assignment options"
        }
        return descriptions.get(key, "Extract all relevant information for this section, including all specific details, dates, and requirements.")

    def _generate_segmentation_prompt(self, syllabus_text: str) -> str: # Note: Renamed from your copy
        # (Content from your uploaded file - this is the prompt generation logic)
        module_details = "\n".join([f"- **{key}**: {self.get_module_description(key)}" for key in self.CORE_MODULES])
        optional_module_details = "\n".join([f"- **{key}**: (Include if relevant content found) {self.get_module_description(key)}" for key in self.OPTIONAL_MODULES])

        prompt = f"""
You are an expert academic assistant specialized in segmenting university course syllabi into a structured JSON format.
Analyze the provided syllabus text and categorize its content into the predefined sections listed below.

**Output Requirements:**
- Your entire response MUST be a single, valid JSON object.
- The JSON object must contain keys for ALL Core Modules. If no content is found for a Core Module, use an empty string "" as its value.
- Optional Modules should only be included as keys if relevant content is found for them. If no content, omit the key or use an empty string "".
- The "unclassified_content" key MUST be present, and its value must be a list of strings. Each string in the list should be a distinct block of text that could not be confidently assigned to any other module. Use an empty list `[]` if all text is classified.
- Preserve original line breaks within the extracted text for each module value by representing newlines as `\\n`.
- Extract text verbatim. Do not summarize or rephrase.

**Modules for Segmentation:**

**Core Modules (Always include these keys):**
{module_details}

**Optional Modules (Include key if content is found):**
{optional_module_details}

**Unclassified Content Module (Always include this key with a list value):**
- **unclassified_content**: List of text blocks not fitting elsewhere.

**Example of expected JSON structure (partial):**
```json
{{
  "course_identification": "CS 101: Introduction to Computer Science\\nFall 2024\\n3 Credits",
  "instructor_information": "Professor Ada Lovelace\\nEmail: ada@example.edu\\nOffice Hours: MWF 10-11 AM, Tech Hall 123",
  "course_schedule": "Week 1 (Sept 2-6): Introduction to CS. Reading: Chapter 1.\\nWeek 2 (Sept 9-13): Algorithms. Reading: Chapter 2. HW1 Due Sept 13.",
  "separate_laboratory_sections": "Lab L01: Mondays 2-4 PM, STEM Bldg Rm 105. TA: Charles Babbage.",
  "unclassified_content": ["Additional university resources list...", "Note on classroom conduct..."]
}}
```

---
**Full Syllabus Text to Parse:**
```text
{syllabus_text}
```

---
**Your JSON Output (ensure it is a single, valid JSON object):**
```json
"""
        return prompt.strip()

    def _post_process_sections(self, sections: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Post-process extracted sections to ensure quality and completeness."""
        
        critical_sections = ["course_identification", "course_schedule", "assignments_exams"]
        unclassified_key = self.UNCLASSIFIED_KEY # Use class constant
        
        for section_name in critical_sections:
            if not sections.get(section_name, "").strip():
                self.logger.warning(f"Critical section '{section_name}' is empty. Attempting recovery from unclassified content.")
                
                if isinstance(sections.get(unclassified_key), list):
                    recovered_content = []
                    remaining_unclassified = []
                    keywords_to_look_for = []

                    if section_name == "course_schedule":
                        keywords_to_look_for = ["week", "session", "chapter", "topic", "reading", "date", "schedul"] # 'schedul' for schedule/scheduled
                    elif section_name == "assignments_exams":
                        keywords_to_look_for = ["assignment", "exam", "quiz", "due", "submission", "project", "homework", "test"]
                    elif section_name == "course_identification": # Could be harder to recover with simple keywords
                         keywords_to_look_for = ["course code", "course id", "department", r"[A-Z]{2,4}\s*\d{3,4}"] # Simple regex for course code

                    if keywords_to_look_for:
                        for i, content_block in enumerate(sections[unclassified_key]):
                            if isinstance(content_block, str) and any(re.search(keyword, content_block, re.IGNORECASE) for keyword in keywords_to_look_for):
                                recovered_content.append(content_block)
                                self.logger.info(f"Found potential content for '{section_name}' in unclassified block {i}.")
                            else:
                                remaining_unclassified.append(content_block)
                        
                        if recovered_content:
                            sections[section_name] = "\n\n".join(recovered_content)
                            sections[unclassified_key] = remaining_unclassified
                            self.logger.info(f"Recovered and moved content for '{section_name}' from unclassified. {len(remaining_unclassified)} unclassified blocks remain.")
                            break # Assuming one recovery pass is enough, or first found block
        
        if not isinstance(sections.get(unclassified_key), list):
            self.logger.warning(f"Unclassified content was not a list (type: {type(sections.get(unclassified_key))}), resetting to empty list.")
            sections[unclassified_key] = []
        
        if isinstance(sections[unclassified_key], list):
            sections[unclassified_key] = [
                content for content in sections[unclassified_key] 
                if isinstance(content, str) and len(content.strip()) > 25 # Increased threshold for meaningful unclassified content
            ]
        return sections

    def _call_openai_api(self, prompt: str, unique_id: str, max_retries: int = -1) -> str:
        """Call OpenAI API with robust error handling and retry logic."""
        # ... (Implementation from user's uploaded file, already reviewed as good) ...
        if not self.client or self.openai_status != "ready": self.logger.error(f"OpenAI client not ready (Status: {self.openai_status}) for ID {unique_id}."); return ""
        debug_dir = self.base_dir / "segmentation_debug"; 
        try: debug_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e_mkdir_debug: self.logger.warning(f"Could not create debug dir {debug_dir}: {e_mkdir_debug}."); debug_dir = self.base_dir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        prompt_file = debug_dir / f"prompt_segment_{unique_id}_{timestamp}.txt"
        try: prompt_file.write_text(prompt, encoding="utf-8"); self.logger.debug(f"Saved API segmentation prompt to {prompt_file}")
        except Exception as e_prompt_save: self.logger.warning(f"Could not save prompt debug file {prompt_file}: {e_prompt_save}")
        api_model_to_use = self.model_preference
        max_r = self.config.get("openai_parser", {}).get("max_api_retries", 3) if max_retries == -1 else max_retries
        for attempt in range(max_r):
            try:
                self.logger.info(f"Calling OpenAI API for segmentation (Attempt {attempt + 1}/{max_r}) with model {api_model_to_use} for ID {unique_id}...")
                messages = [{"role": "system", "content": "You are an expert at parsing academic syllabi. Return only valid JSON with no additional text."}, {"role": "user", "content": prompt}]
                response = self.client.chat.completions.create(model=api_model_to_use, messages=messages, temperature=0.1, response_format={"type": "json_object"})
                response_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message and response.choices[0].message.content else ""
                response_file = debug_dir / f"response_segment_{unique_id}_{timestamp}_attempt{attempt+1}.txt"
                try: response_file.write_text(response_text, encoding="utf-8")
                except Exception as e_resp_save: self.logger.warning(f"Could not save response debug file {response_file}: {e_resp_save}")
                if not response_text or len(response_text) < 100: 
                    self.logger.warning(f"Short/empty response (len {len(response_text)}) from API (Attempt {attempt+1}) for ID {unique_id}.")
                    if attempt < max_r - 1: self.logger.info("Retrying..."); time.sleep(2 ** attempt); continue
                    else: self.logger.error(f"Max retries ({max_r}) reached with short/empty API response for ID {unique_id}."); return ""
                self.logger.info(f"Received OpenAI response for segmentation (len {len(response_text)}) (Attempt {attempt+1}) for ID {unique_id}.")
                return response_text
            except (httpx.ReadTimeout, openai.APITimeoutError) as e_timeout: self.logger.warning(f"Timeout Error (Attempt {attempt+1}/{max_r}) for ID {unique_id}: {e_timeout}. Retrying...")
            except openai.RateLimitError as e_rate: self.logger.warning(f"Rate Limit Error (Attempt {attempt+1}/{max_r}) for ID {unique_id}: {e_rate}. Retrying after longer backoff..."); time.sleep((2 ** attempt) * 5)
            except openai.APIConnectionError as e_conn: self.logger.error(f"API Connection Error (Attempt {attempt+1}/{max_r}) for ID {unique_id}: {e_conn}. Retrying..."); time.sleep(5 + (2**attempt))
            except openai.APIStatusError as e_status:
                self.logger.warning(f"API Status Error {e_status.status_code} (Attempt {attempt+1}/{max_r}) for ID {unique_id}: {e_status.response}.")
                if not (500 <= e_status.status_code < 600): self.logger.error(f"Client-side API Status Error {e_status.status_code}. Not retrying. Msg: {e_status.message}"); return ""
            except openai.APIError as e_api: self.logger.warning(f"General API Error (Attempt {attempt+1}/{max_r}) for ID {unique_id}: {e_api}. Retrying...")
            except Exception as e_generic: self.logger.error(f"Unexpected error in OpenAI call (Attempt {attempt+1}/{max_r}) for ID {unique_id}: {e_generic}", exc_info=True)
            if attempt < max_r - 1: time.sleep(2 ** attempt)
            else: self.logger.error(f"Exhausted all {max_r} retries for OpenAI API call for ID {unique_id}."); return ""
        return ""

    def _parse_llm_response(self, response_text: str, job_id: str) -> Optional[Dict[str, Any]]: # Added job_id for logging
        # ... (Implementation from user's uploaded file, already reviewed as good, just passing job_id) ...
        if not response_text or not response_text.strip(): self.logger.error(f"LLM response for {job_id} is empty."); return None
        content_to_parse: Optional[str] = None
        json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text, re.DOTALL)
        if json_match: content_to_parse = json_match.group(1).strip(); self.logger.debug(f"Extracted JSON from markdown for {job_id}.")
        else:
            first_brace = response_text.find('{'); last_brace = response_text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace: content_to_parse = response_text[first_brace : last_brace + 1].strip(); self.logger.debug(f"Extracted JSON by braces for {job_id}.")
            else: self.logger.error(f"No JSON structure found in LLM response for {job_id}. Response: {response_text[:500]}..."); return None
        if not content_to_parse: self.logger.error(f"Content to parse is empty for {job_id}."); return None
        try:
            parsed_json = json.loads(content_to_parse)
            if not isinstance(parsed_json, dict): self.logger.error(f"Parsed LLM content for {job_id} not a dict: {type(parsed_json)}"); return None
        except json.JSONDecodeError as e_decode:
            self.logger.error(f"JSON decode error for {job_id}: {e_decode.msg} at pos {e_decode.pos}") # Use pos from error
            try:
                repaired_content = re.sub(r',\s*([\}\]])', r'\1', content_to_parse)
                if repaired_content != content_to_parse:
                    parsed_json = json.loads(repaired_content)
                    if not isinstance(parsed_json, dict): self.logger.error(f"Repaired LLM content for {job_id} still not a dict."); return None
                    self.logger.info(f"Successfully parsed LLM response for {job_id} after repairing trailing comma.")
                else: return None 
            except json.JSONDecodeError as e_repair: self.logger.error(f"JSON decode error persisted after repair for {job_id}: {e_repair.msg}"); return None
        validated_data: Dict[str, Any] = {}; missing_core_keys = []
        for key in self.CORE_MODULES:
            if key in parsed_json and isinstance(parsed_json[key], str): validated_data[key] = parsed_json[key]
            else:
                validated_data[key] = ""
                if key not in parsed_json: missing_core_keys.append(key)
                elif not isinstance(parsed_json[key], str): self.logger.warning(f"LLM value for core key '{key}' in {job_id} not a string. Defaulted empty."); validated_data[key] = str(parsed_json[key]) if parsed_json[key] is not None else ""
        if missing_core_keys: self.logger.warning(f"LLM response for {job_id} missing core keys: {missing_core_keys}. Added with empty strings.")
        for key in self.OPTIONAL_MODULES:
            if key in parsed_json and isinstance(parsed_json[key], str): validated_data[key] = parsed_json[key]
            elif key in parsed_json: self.logger.warning(f"LLM value for optional key '{key}' in {job_id} not a string. Omitting."); validated_data[key] = str(parsed_json[key]) if parsed_json[key] is not None else ""
        unclassified = parsed_json.get(self.UNCLASSIFIED_KEY)
        if isinstance(unclassified, list): validated_data[self.UNCLASSIFIED_KEY] = [str(item) for item in unclassified if isinstance(item, (str, int, float, bool)) and str(item).strip()]
        elif isinstance(unclassified, str) and unclassified.strip(): validated_data[self.UNCLASSIFIED_KEY] = [unclassified.strip()]
        else:
            validated_data[self.UNCLASSIFIED_KEY] = []
            if unclassified is not None: self.logger.warning(f"'{self.UNCLASSIFIED_KEY}' from LLM for {job_id} not list/string. Defaulted empty. Value: {str(unclassified)[:100]}")
        unexpected_keys = [k for k in parsed_json if k not in self.ALL_POSSIBLE_KEYS]
        if unexpected_keys: self.logger.warning(f"LLM response for {job_id} included unexpected keys (ignored): {unexpected_keys}")
        self.logger.info(f"LLM response for {job_id} parsed and validated successfully.")
        return validated_data

    def _fallback_segmentation(self, syllabus_text: str) -> Dict[str, Any]:
        # ... (implementation remains the same as the user's enhanced version) ...
        self.logger.info("Executing enhanced fallback rule-based segmentation.")
        sections: Dict[str, Any] = {key: "" for key in self.CORE_MODULES}
        sections[self.UNCLASSIFIED_KEY] = []
        header_patterns_map = { # From user's enhanced version
            "course_schedule": r"^\s*(Course\s*Schedule|Class\s*Schedule|Weekly\s*(?:Schedule|Plan|Outline)|Calendar|Topic(?:al)?\s*(?:Schedule|Outline)|Schedule\s*of\s*(?:Topics|Classes)|Tentative\s*Schedule|Course\s*Calendar|Class\s*Calendar|Week(?:ly)?\s*Topics?|Session\s*Topics?|Daily\s*Schedule|Meeting\s*Schedule|Lecture\s*Schedule|Course\s*Outline|Class\s*Meetings?|Important\s*Dates|Key\s*Dates|Course\s*Timeline)\s*:?\s*$",
            "assignments_exams": r"^\s*(Assignments?(?:\s*(?:and|&)\s*Exams?)?|Exams?(?:\s*(?:and|&)\s*Assignments?)?|Midterms?|Final\s*Exams?|Projects?|Papers?|Essays?|Quizzes?|Tests?|Major\s*Assessments?|Assessment\s*Schedule|Deliverables?|Homework|Evaluation\s*Schedule|Due\s*Dates?|Important\s*Deadlines?|Submission\s*Schedule|Graded\s*Work)\s*:?\s*$",
            "course_requirements_grading_policy": r"^\s*(Grading(?:\s*(?:Policy|Scale|Scheme|System|Criteria|Breakdown))?|Course\s*Requirements?|Assessment\s*Methods?|Evaluation(?:\s*Methods?)?|Grade\s*(?:Distribution|Breakdown|Weights?)|Marking\s*Scheme|Point\s*Distribution|How\s*(?:You\s*Will\s*Be\s*)?(?:Graded|Evaluated)|Performance\s*Evaluation|Credit\s*Requirements?|Pass(?:ing)?\s*Requirements?)\s*:?\s*$",
            "required_texts_materials": r"^\s*((?:Required|Recommended)\s*(?:Texts?|Textbooks?|Books?|Materials?|Readings?|Resources?)|Course\s*(?:Materials?|Resources?)|Reading\s*List|Bibliography|Software\s*Requirements?|Equipment\s*(?:Required|Needed)|Supplies|What\s*(?:You|Students?)\s*Need|Materials?\s*List|Course\s*Texts?|Books?\s*(?:and|&)\s*Materials?)\s*:?\s*$",
            "learning_objectives": r"^\s*((?:Learning|Course|Student)\s*(?:Objectives?|Outcomes?|Goals?)|What\s*(?:You|Students?)\s*Will\s*Learn|By\s*the\s*End\s*of\s*(?:This\s*)?Course|Course\s*Aims?|Educational\s*(?:Objectives?|Goals?)|Competenc(?:ies|y)|Skills\s*(?:to\s*be\s*)?Developed|Program\s*Outcomes?|Expected\s*Outcomes?)\s*:?\s*$",
            "course_description_prerequisites": r"^\s*(Course\s*Descriptions?|Course\s*Overview|About\s*(?:This\s*Course|the\s*Course)|Prerequisites?|Pre-?reqs?|Co-?requisites?|Prior\s*Knowledge|Background\s*(?:Required|Needed)|Course\s*Summary|What\s*is\s*This\s*Course|Introduction\s*to\s*(?:the\s*)?Course|Course\s*Information|General\s*Information)\s*:?\s*$",
            "instructor_information": r"^\s*(Instructor(?:\s*Information)?|Professor(?:\s*Information)?|Faculty(?:\s*Information)?|Teaching\s*(?:Staff|Team)|Your\s*Instructor|Course\s*Instructor|Contact\s*Information|Office\s*Hours?|How\s*to\s*(?:Contact|Reach)\s*(?:Me|Instructor)|Instructor\s*Details?|Faculty\s*Contact|Teaching\s*Assistants?|TAs?(?:\s*Information)?)\s*:?\s*$",
            "course_identification": r"^\s*(?:([A-Z]{2,5}\s*\d{2,4}[A-Z]?\s*[:\-–—]?\s*[^\n]+)|(?:Course\s*(?:Number|Code|ID):\s*[^\n]+))", # Kept from user's version
            "course_policies": r"^\s*(Course\s*Polic(?:y|ies)|(?:Class(?:room)?|Course)\s*(?:Rules|Expectations)|Attendance\s*(?:Policy|Requirements?)|Late\s*(?:Work|Submission)\s*Policy|Academic\s*(?:Integrity|Honesty)|Plagiarism\s*Policy|Make-?up\s*(?:Policy|Work)|Disability\s*(?:Statement|Accommodations?)|Special\s*Accommodations?|Classroom\s*(?:Behavior|Conduct)|Technology\s*Policy|Electronics?\s*Policy|Course\s*(?:Rules|Guidelines)|Important\s*Policies)\s*:?\s*$",
            "communication_student_support": r"^\s*(Communications?|Student\s*Support|Office\s*Hours?|How\s*to\s*Succeed|Academic\s*(?:Support|Resources?)|Getting\s*Help|Tutoring|Writing\s*Center|Library\s*(?:Resources?|Support)|Study\s*(?:Groups?|Resources?)|Additional\s*(?:Help|Resources?)|Where\s*to\s*Get\s*Help|Support\s*Services?|Student\s*Resources?|Help\s*Resources?)\s*:?\s*$",
            "university_wide_policies_resources": r"^\s*(University(?:-?wide)?\s*Polic(?:y|ies)|Institutional\s*Polic(?:y|ies)|Campus\s*Polic(?:y|ies)|General\s*(?:University\s*)?Polic(?:y|ies)|FERPA|Title\s*IX|Non-?discrimination|Equal\s*Opportunity|ADA\s*(?:Statement|Compliance)|Americans?\s*with\s*Disabilities|Honor\s*Code|Academic\s*Calendar|Student\s*Conduct|Code\s*of\s*Conduct|Campus\s*(?:Safety|Security)|Emergency\s*Procedures?|Withdrawal\s*Policy|Drop\s*(?:Policy|Dates)|Academic\s*(?:Calendar|Dates))\s*:?\s*$",
            "separate_laboratory_sections": r"^\s*(Laboratory?(?:\s*(?:Sections?|Schedule|Information|Details?))?|Labs?(?:\s*(?:Sections?|Schedule|Information|Details?))?|Experiment(?:al)?\s*(?:Schedule|Sessions?)|Lab(?:oratory)?\s*(?:Meeting\s*)?Times?|Lab(?:oratory)?\s*Requirements?|Lab(?:oratory)?\s*Safety|Lab(?:oratory)?\s*Manual|Practical\s*Sessions?|Hands-?on\s*(?:Sessions?|Work))\s*:?\s*$",
            "recitation_discussion_sections": r"^\s*(Recitations?(?:\s*Sections?)?|Discussion\s*Sections?|Tutorial\s*Sessions?|Problem(?:-?solving)?\s*Sessions?|Review\s*Sessions?|Small\s*Group\s*(?:Meetings?|Sessions?)|Breakout\s*Sessions?|Section\s*Meetings?|TA(?:-?led)?\s*Sessions?|Supplemental\s*Instruction|SI\s*Sessions?|Study\s*Sessions?)\s*:?\s*$",
        }
        initial_content_offset = 0; course_id_pattern = header_patterns_map.pop("course_identification")
        course_id_match = re.match(course_id_pattern, syllabus_text, re.IGNORECASE | re.MULTILINE) # Use MULTILINE for ^
        if course_id_match: sections["course_identification"] = course_id_match.group(0).strip(); initial_content_offset = course_id_match.end(); self.logger.info(f"Fallback: Found 'course_identification' ({len(sections['course_identification'])} chars).")
        all_matches: List[Dict[str, Any]] = []
        for key, pattern_str in header_patterns_map.items():
            try:
                for match in re.finditer(pattern_str, syllabus_text[initial_content_offset:], re.IGNORECASE | re.MULTILINE): all_matches.append({"key": key, "start": match.start() + initial_content_offset, "header_text": match.group(0).strip()})
            except re.error as e_re_fallback: self.logger.error(f"Regex error for key '{key}': {e_re_fallback}")
        all_matches.sort(key=lambda m: m["start"])
        current_position = initial_content_offset; last_assigned_key = "course_identification" if sections["course_identification"] else None
        for i, match_info in enumerate(all_matches):
            section_key, section_start, header_text = match_info["key"], match_info["start"], match_info["header_text"]
            text_before = syllabus_text[current_position:section_start].strip()
            if text_before and len(text_before) > 50:
                if last_assigned_key and not sections.get(last_assigned_key): sections[last_assigned_key] = text_before; self.logger.info(f"Fallback: Assigned to previous key '{last_assigned_key}' ({len(text_before)} chars).")
                else: sections[self.UNCLASSIFIED_KEY].append(text_before); self.logger.debug(f"Fallback: Added to unclassified (before '{section_key}', {len(text_before)} chars).")
            content_end = all_matches[i+1]["start"] if (i + 1) < len(all_matches) else len(syllabus_text)
            section_content_body = syllabus_text[section_start + len(header_text) : content_end].strip()
            full_section_content = (header_text + "\n" + section_content_body).strip()
            if sections.get(section_key): sections[section_key] += "\n\n" + full_section_content
            else: sections[section_key] = full_section_content
            self.logger.info(f"Fallback: Assigned to '{section_key}' ({len(full_section_content)} chars).")
            current_position = content_end; last_assigned_key = section_key
        if current_position < len(syllabus_text):
            remaining_text = syllabus_text[current_position:].strip()
            if remaining_text and len(remaining_text) > 50:
                if last_assigned_key and sections.get(last_assigned_key) and len(sections[last_assigned_key]) < 200: sections[last_assigned_key] += "\n" + remaining_text; self.logger.info(f"Fallback: Appended remaining to '{last_assigned_key}'.")
                else: sections[self.UNCLASSIFIED_KEY].append(remaining_text); self.logger.info(f"Fallback: Added remaining to unclassified ({len(remaining_text)} chars).")
        for opt_key in self.OPTIONAL_MODULES: sections.setdefault(opt_key, "")
        self.logger.info(f"Fallback segmentation done. Unclassified blocks: {len(sections[self.UNCLASSIFIED_KEY])}.")
        return sections


if __name__ == "__main__":
    # ... (Standalone test block from user's uploaded file) ...
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s', handlers=[logging.FileHandler("syllabus_parser_standalone.log", mode='a', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    main_logger = logging.getLogger(__name__) 
    import argparse # Moved import here
    arg_parser_main = argparse.ArgumentParser(description="Parse syllabus text file into sections.")
    arg_parser_main.add_argument("input_file", help="Path to pre-converted syllabus text file (.txt).")
    arg_parser_main.add_argument("--output-dir", help="Base directory for output JSONs.")
    arg_parser_main.add_argument("--model", default="gpt-4.1-nano", help="OpenAI model for segmentation.")
    arg_parser_main.add_argument("--config", help="Path to YAML config file.")
    main_args = arg_parser_main.parse_args()
    main_app_config = {}
    if main_args.config:
        try:
            import yaml
            main_config_path = Path(main_args.config)
            if main_config_path.is_file():
                with main_config_path.open('r', encoding='utf-8') as f_yaml: main_app_config = yaml.safe_load(f_yaml)
                if main_app_config: main_logger.info(f"Loaded config from: {main_args.config}")
                else: main_logger.warning(f"Config file {main_args.config} empty/invalid.")
            else: main_logger.error(f"Config file not found: {main_args.config}.")
        except ImportError: main_logger.warning("PyYAML not installed. Cannot load YAML config.")
        except Exception as e_cfg: main_logger.error(f"Error loading config {main_args.config}: {e_cfg}", exc_info=True)
    if main_args.output_dir: main_app_config.setdefault("directories", {})["parsed_syllabus_dir"] = main_args.output_dir
    if main_args.model: main_app_config.setdefault("extraction", {})["openai_model"] = main_args.model
    parser_main_instance = SyllabusParser(config=main_app_config, model=main_args.model, logger_instance=main_logger)
    main_result = parser_main_instance.parse_syllabus(main_args.input_file)
    status_main = main_result.get("status", "error_unknown")
    if "error" in status_main: main_logger.error(f"Syllabus parsing FAILED. Status: {status_main}, Msg: {main_result.get('error', main_result.get('message', 'N/A'))}")
    elif "warning" in status_main: main_logger.warning(f"Syllabus parsing completed WITH WARNINGS. Status: {status_main}, Msg: {main_result.get('message', 'N/A')}")
    else: main_logger.info("Syllabus parsing successful.")
    if main_result.get('parsed_data'): main_logger.info(f"Parsed data keys found: {list(main_result['parsed_data'].keys())}")
    if main_result.get('parsed_output_file'): main_logger.info(f"Output saved to: {main_result['parsed_output_file']}")
    else: main_logger.warning("Output file path not available in results.")
    sys.exit(0 if "error" not in status_main else 1)
