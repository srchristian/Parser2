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
from typing import Dict, Any, List, Optional, Tuple

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
                 model: str = "gpt-4o",
                 log_dir: Optional[str] = None, # Maintained for compatibility, less used if logger_instance is primary
                 openai_api_key: Optional[str] = None,
                 logger_instance: Optional[logging.Logger] = None):

        self.logger = logger_instance if logger_instance else logging.getLogger(__name__)
        if not self.logger.hasHandlers() and logger_instance is None:
            self.logger.addHandler(logging.StreamHandler(sys.stderr))
            self.logger.setLevel(logging.WARNING)
            self.logger.warning(f"SyllabusParser's logger '{__name__}' was not configured by the application. Using basic stderr handler with WARNING level.")

        self.config = config or {}
        self.model_preference = self.config.get("extraction", {}).get("openai_model", model)
        self.client = None

        if HELPERS_JSON_IMPORTED:
             self.logger.info("SyllabusParser will use imported helper functions (read_json, write_json) from utils.helpers.")
        else:
             self.logger.warning("SyllabusParser is using internal fallback JSON read/write functions because utils.helpers import failed.")

        if base_dir: self.base_dir = Path(base_dir)
        elif "directories" in self.config and "parsed_syllabus_dir" in self.config["directories"]:
            self.base_dir = Path(self.config["directories"]["parsed_syllabus_dir"])
        else:
            self.base_dir = CURRENT_DIR / "syllabus_repository" / "parsed_syllabus_output"
        
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
                self.logger.warning("OpenAI API key not found (checked args, env, config). OpenAI client for SyllabusParser not initialized.")
                self.openai_status = "no_api_key"
        else:
            self.logger.warning("OpenAI SDK (openai package) or httpx not installed. LLM-based segmentation for SyllabusParser unavailable.")
            self.openai_status = "sdk_not_available"
        
        self.logger.info("SyllabusParser initialized. Expects pre-converted .txt files for parsing.")

    def _create_directories(self):
        """Creates necessary directories, primarily self.base_dir for output."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"SyllabusParser output directory ensured: {self.base_dir}")
        except OSError as e_mkdir:
            self.logger.error(f"Error creating base directory {self.base_dir} for SyllabusParser: {e_mkdir}", exc_info=True)

    def _extract_text_from_file(self, file_path: Path) -> Optional[str]:
        """
        Extracts text from a pre-converted plain text file (.txt).
        It is expected that the input file is already in plain text format.
        """
        self.logger.info(f"Attempting to extract text from pre-converted file: {file_path}")
        extension = file_path.suffix.lower()

        if extension == ".txt":
            try:
                # Attempt common encodings for text files for robustness
                encodings_to_try = ['utf-8', 'latin-1', 'windows-1252']
                for enc in encodings_to_try:
                    try:
                        return file_path.read_text(encoding=enc).strip()
                    except UnicodeDecodeError:
                        self.logger.debug(f"Failed to read {file_path.name} with encoding {enc}.")
                self.logger.error(f"Could not decode text file {file_path.name} with any common encodings.")
                return None # Return None if all attempts fail
            except Exception as e_txt:
                self.logger.error(f"Error reading text file {file_path.name}: {e_txt}", exc_info=True)
                return None
        else:
            self.logger.error(f"Unsupported file type: '{extension}'. SyllabusParser expects a pre-converted '.txt' file. Cannot extract text from {file_path.name}.")
            return None


    def parse_syllabus(self, input_text_file_path: str, unique_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Parses the pre-converted syllabus text file to segment its content.
        Returns a dictionary containing the status of the operation,
        any messages, the segmented data, and the full extracted text content.

        Args:
            input_text_file_path (str): The path to the pre-converted syllabus text file (.txt).
            unique_id (Optional[str]): A unique identifier for this parsing job.
                                       If None, a new UUID will be generated.

        Returns:
            Dict[str, Any]: A result dictionary with keys:
                "status" (str): "success", "warning_...", "error_..."
                "message" (Optional[str]): User-friendly message.
                "parsed_output_file" (Optional[str]): Path to the saved segmented JSON.
                "parsed_data" (Optional[Dict[str, Any]]): The segmented syllabus content.
                "process_time" (Optional[float]): Duration of parsing in seconds.
                "full_text_content" (Optional[str]): The complete text extracted from the syllabus.
                "error" (Optional[str]): Error message if status is an error type.
        """
        input_file = Path(input_text_file_path)
        self.logger.info(f"Starting syllabus segmentation for pre-converted text file: {input_file} (ID: {unique_id or 'N/A'})")
        start_time = time.monotonic()

        if not input_file.is_file():
            self.logger.error(f"Input text file not found: {input_file}")
            return {"error": f"Input text file not found: {input_file}", "status": "error_file_not_found", "full_text_content": None}
        
        if input_file.suffix.lower() != ".txt":
            self.logger.error(f"Invalid file type: {input_file.suffix}. SyllabusParser expects a pre-converted '.txt' file.")
            return {"error": f"Invalid file type: {input_file.suffix}. Expected '.txt'.", "status": "error_invalid_file_type", "full_text_content": None}

        if not unique_id: unique_id = str(uuid.uuid4())
        self.logger.info(f"Using unique ID for this segmentation run: {unique_id}")

        syllabus_text = self._extract_text_from_file(input_file)

        if syllabus_text is None:
            self.logger.error(f"Could not extract text from {input_file}. Segmentation cannot proceed.")
            return {"error": f"Could not extract text from {input_file}", "status": "error_text_extraction_failed", "full_text_content": None}
        elif not syllabus_text.strip():
             self.logger.warning(f"Extracted text from {input_file} is empty or whitespace only. Segmentation will likely result in empty sections.")

        self.logger.info(f"Extracted {len(syllabus_text)} characters from file {input_file.name} for segmentation.")

        parsed_sections: Optional[Dict[str, Any]] = None
        parsing_method = "llm" # Default assumption

        if self.openai_status == "ready" and self.client:
            if len(syllabus_text.strip()) < 100: # Arbitrary threshold for "substantial"
                self.logger.warning(f"Syllabus text for {unique_id} is very short (len: {len(syllabus_text.strip())}). Skipping LLM call, will use fallback segmentation.")
            else:
                prompt = self._generate_segmentation_prompt(syllabus_text)
                response_text = self._call_openai_api(prompt, unique_id=unique_id)
                if response_text:
                    parsed_sections = self._parse_llm_response(response_text)
                    if not parsed_sections:
                        self.logger.warning(f"Failed to parse LLM response into valid JSON for {unique_id}. Attempting fallback segmentation.")
                else:
                    self.logger.warning(f"No valid response from OpenAI API for {unique_id}. Attempting fallback segmentation.")
        else:
            self.logger.warning(f"OpenAI client not ready (status: {self.openai_status}). Attempting fallback segmentation for {unique_id}.")

        if not parsed_sections:
            self.logger.info(f"Using fallback rule-based segmentation method for {unique_id}.")
            parsed_sections = self._fallback_segmentation(syllabus_text)
            parsing_method = "fallback"

        parsed_sections["UUID"] = unique_id
        parsed_sections["parsing_method_used"] = parsing_method
        parsed_sections["segmentation_timestamp"] = datetime.now().isoformat()
        
        parsed_output_file = self.base_dir / f"{unique_id}_segmented_syllabus.json"

        save_success = write_json(
            file_path=parsed_output_file,
            data=parsed_sections,
            backup_existing=True, 
            indent=self.config.get("json_output", {}).get("indent", 2)
        )

        process_time_seconds = time.monotonic() - start_time
        
        result_payload = {
            "status": "success",
            "message": "Syllabus segmentation complete.",
            "parsed_output_file": str(parsed_output_file) if save_success else None,
            "parsed_data": parsed_sections,
            "process_time": round(process_time_seconds, 3),
            "full_text_content": syllabus_text
        }

        if not save_success:
            self.logger.error(f"Could not save segmented syllabus to {parsed_output_file} for {unique_id}.")
            result_payload["status"] = "warning_save_failed"
            result_payload["message"] = f"Segmentation complete, but could not save output to {parsed_output_file}."
            result_payload["error"] = f"Failed to save to {parsed_output_file}"


        self.logger.info(f"Syllabus segmentation for {unique_id} completed in {result_payload['process_time']:.2f}s. Method: {parsing_method}. Output: {result_payload['parsed_output_file'] or 'Not Saved'}")
        return result_payload

    def _generate_segmentation_prompt(self, syllabus_text: str) -> str:
        """
        Generates the prompt for the LLM to segment the syllabus text.
        Updated to include "separate_laboratory_sections" and "recitation_discussion_sections".
        """
        module_descriptions = {
            "course_identification": "Extract: Course Number & Title (e.g., ENGL 101: Introduction to Academic Writing), Term/Semester & Credit Hours (e.g., Spring 2025, 3 credits), Meeting Days, Times & Location (e.g., MWF 10:00-10:50 AM, Room 101 or Online), Modality (In-person, Hybrid, Online Synchronous/Asynchronous).",
            "instructor_information": "Extract: Instructor’s Full Name & Academic Title (e.g., Dr. Jane Doe, Associate Professor), Contact Information (institutional email, expected response time), Office Location & Hours (physical room or virtual link, scheduled weekly hours), Teaching Assistants (Names, contact, office hours/location).",
            "course_description_prerequisites": "Extract: Official Course Description (from university catalog), Overview of Topics (narrative summary of core themes, key questions, subject matter), Prerequisites/Corequisites (required prior coursework, placement scores, concurrent courses).",
            "learning_objectives": "Extract: Clearly stated goals describing what students should know or do upon course completion (Specific, Measurable, Attainable, Relevant, Time-bound - SMART). Use observable action verbs (e.g., analyze, compare, design, evaluate).",
            "required_texts_materials": "Extract: Full bibliographic citations (Author, Title, Edition, Publisher, ISBN) for all required textbooks and readings. Distinguish required vs. recommended. Information on accessing readings (bookstore, library reserve, online links). Specifications for necessary software, hardware, lab kits, art supplies, or online platform access codes. Access to supplementary materials (lecture slides, articles via LMS).",
            "course_requirements_grading_policy": "Extract: Comprehensive list of all graded components (e.g., homework, quizzes, midterms, final exam, projects, papers, participation) with their percentage weight towards the final grade. Clear definition of the grading scale (e.g., A = 90-100%). Policies for assignment submission methods, deadlines (including time zone if online), penalties for late submissions, conditions for make-up work. Statement referencing institutional honor code or academic integrity policy.",
            "assignments_exams": "Extract: Descriptions of significant assignments (papers, projects, presentations) including format, length, due dates, and alignment with learning objectives. Details on major exams (midterm, final) including format (e.g., multiple-choice, essay, open/closed-book), duration, specific dates, and types of questions. Information on preparation resources (study guides, practice problems, review sessions).",
            "course_schedule": "Extract: A week-by-week or session-by-session schedule, often in a table. List dates, corresponding topics, required readings, assignment due dates, and exam dates. Note any holidays or breaks when class does not meet.",
            "course_policies": "Extract: Instructor's policy on class attendance (required/recommended, excused/unexcused), tardiness, and participation (in-class or online). Detailed late/missed work policy. Links to academic support (tutoring, writing centers). Universal Design & Accommodations statement (Disability Services contact and process).",
            "communication_student_support": "Extract: Instructor's preferred communication methods (email, LMS messages, forum), expected response timeframe, communication etiquette. Specific links/info for Tutoring, Writing Center, Math Lab, Library Support, Counseling & Well-Being services. Campus emergency procedures contact.",
            "university_wide_policies_resources": "Extract: Links/summaries of university-wide policies: FERPA (student privacy), Non-Discrimination/Title IX, Disability Accommodations (ADA/Section 504), Academic Integrity/Honor Code, Emergency Procedures/Campus Safety, IT Acceptable Use/Copyright, Grade Appeal/Grievance, Student Conduct Code, Health/Counseling/Wellness resources, Academic Calendar (add/drop deadlines).",
            "separate_laboratory_sections": "(Optional) Extract: Specific schedule (days, times, dates), location for laboratory sessions if detailed separately from main class. Lab Instructor/TA details if provided. Pre-lab requirements. Lab attendance policies and specific safety (PPE) rules for labs.",
            "recitation_discussion_sections": "(Optional) Extract: Schedule (days, times, dates) and locations for recitation, discussion, or tutorial sections if detailed separately. TA or section leader details. Typical activities, topics covered, or participation policy for these sections.",
            "unclassified_content": "This MUST be a list of strings. Place any significant text blocks here that do not clearly fit into any of the other defined modules. If all content is classified, use an empty list `[]`."
        }
        core_module_list_str = "\n".join(f"- `{key}`: {module_descriptions.get(key, 'Extract relevant information for this section.')}" for key in self.CORE_MODULES)
        optional_module_list_str = "\n".join(f"- `{key}`: {module_descriptions.get(key, 'Extract relevant information if present.')}" for key in self.OPTIONAL_MODULES if key in module_descriptions)

        few_shot_example_1_text = """
COURSE: BIOL 202L Human Anatomy Lab (Fall 2024)
Instructor: Dr. Lab Tech, Office: Bio Hall 101
This is the lab component for BIOL 202.
Lab Schedule: All labs meet in Bio Hall Room 105.
Section L01: Mondays 1:00 PM - 3:50 PM. First Lab: Aug 26. Topic: Safety & Intro.
Section L02: Tuesdays 9:00 AM - 11:50 AM. First Lab: Aug 27. Topic: Safety & Intro.
Required: Lab Coat & Goggles.
"""
        few_shot_example_1_json = """
{
  "course_identification": "COURSE: BIOL 202L Human Anatomy Lab (Fall 2024)",
  "instructor_information": "Instructor: Dr. Lab Tech, Office: Bio Hall 101",
  "course_description_prerequisites": "This is the lab component for BIOL 202.",
  "learning_objectives": "",
  "required_texts_materials": "Required: Lab Coat & Goggles.",
  "course_requirements_grading_policy": "",
  "assignments_exams": "",
  "course_schedule": "", 
  "separate_laboratory_sections": "Lab Schedule: All labs meet in Bio Hall Room 105.\\nSection L01: Mondays 1:00 PM - 3:50 PM. First Lab: Aug 26. Topic: Safety & Intro.\\nSection L02: Tuesdays 9:00 AM - 11:50 AM. First Lab: Aug 27. Topic: Safety & Intro.",
  "course_policies": "",
  "communication_student_support": "",
  "university_wide_policies_resources": "",
  "unclassified_content": []
}
"""
        prompt = f"""
You are an expert academic assistant. Your task is to segment the following course syllabus text into a structured JSON format.
Adhere strictly to the defined modules and instructions.

### Core Modules (Always include these keys in the JSON. If no content is found for a key, use an empty string ""):
{core_module_list_str}

### Optional Modules (Include these keys ONLY if relevant content is found. If no content, omit the key or use an empty string ""):
{optional_module_list_str}

### Unclassified Content Module:
- `unclassified_content`: This MUST be a list of strings. Place any significant text blocks here that do not clearly fit into any of the defined Core or Optional modules. If all content is classified, use an empty list `[]`.

### Key Instructions:
1.  **Complete and Accurate Extraction:** Preserve all specific details: dates, times, names, locations, policies, ISBNs, links, etc.
2.  **Comprehensive Coverage:** Extract all relevant text for each module. If content seems to fit multiple modules, choose the most specific one or the one where it's most emphasized. Avoid duplicating large chunks of text across modules unless absolutely necessary for context.
3.  **Strict JSON Output:** Your entire response MUST be a single, valid JSON object. Do not include any explanatory text, comments, or markdown formatting outside the JSON structure. Ensure correct JSON syntax, especially for strings (e.g., escape internal quotes if necessary, use double quotes for keys and string values).
4.  **Text Formatting:** Preserve original line breaks within the extracted text for each module where it aids readability (e.g., in schedules, lists of policies). Represent newlines as `\\n` in the JSON strings.

### Example:
**Input Syllabus Text Snippet:**
```text
{few_shot_example_1_text}
```
**Expected JSON Output Snippet (Illustrative of structure and newlines):**
```json
{few_shot_example_1_json}
```
---
### Full Syllabus Text to Parse:

```text
{syllabus_text}
```

### Your JSON Output (ensure it is a single, valid JSON object):
```json
"""
        return prompt.strip()

    def _call_openai_api(self, prompt: str, unique_id: str, max_retries: int = 2) -> str:
        if not self.client or self.openai_status != "ready":
            self.logger.error(f"OpenAI client not ready (Status: {self.openai_status}). Cannot call API for ID {unique_id}.")
            return ""

        debug_dir = self.base_dir / "segmentation_debug"
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e_mkdir_debug:
            self.logger.warning(f"Could not create debug directory {debug_dir}: {e_mkdir_debug}. Prompts/responses may not be saved.")
            debug_dir = self.base_dir


        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        prompt_file = debug_dir / f"prompt_segment_{unique_id}_{timestamp}.txt"
        try:
            prompt_file.write_text(prompt, encoding="utf-8")
            self.logger.debug(f"Saved API segmentation prompt to {prompt_file}")
        except Exception as e_prompt_save:
            self.logger.warning(f"Could not save prompt debug file {prompt_file}: {e_prompt_save}")

        api_model_to_use = self.model_preference
        max_retries_from_config = self.config.get("openai_parser", {}).get("max_api_retries", max_retries)


        for attempt in range(max_retries_from_config):
            try:
                self.logger.info(f"Calling OpenAI API for segmentation (Attempt {attempt + 1}/{max_retries_from_config}) with model {api_model_to_use} for ID {unique_id}...")
                response = self.client.chat.completions.create(
                    model=api_model_to_use,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message and response.choices[0].message.content else ""
                
                response_file = debug_dir / f"response_segment_{unique_id}_{timestamp}_attempt{attempt+1}.txt"
                try:
                    response_file.write_text(response_text, encoding="utf-8")
                except Exception as e_resp_save:
                    self.logger.warning(f"Could not save response debug file {response_file}: {e_resp_save}")

                if not response_text or len(response_text) < 50:
                    self.logger.warning(f"Short or empty response (length {len(response_text)}) from API on attempt {attempt+1} for ID {unique_id}.")
                    if attempt < max_retries_from_config - 1:
                        self.logger.info("Retrying due to short/empty response...")
                        time.sleep(1 + (2 ** attempt))
                        continue
                    else:
                        self.logger.error(f"Max retries ({max_retries_from_config}) reached with short/empty response from API for ID {unique_id}.")
                        return ""
                self.logger.info(f"Received OpenAI response for segmentation (length {len(response_text)} chars) on attempt {attempt+1} for ID {unique_id}.")
                return response_text

            except httpx.ReadTimeout as e_timeout:
                self.logger.warning(f"HTTPX Read Timeout (Attempt {attempt+1}/{max_retries_from_config}) for ID {unique_id}: {e_timeout}. Retrying...")
            except openai.RateLimitError as e_rate:
                self.logger.warning(f"OpenAI Rate Limit Error (Attempt {attempt+1}/{max_retries_from_config}) for ID {unique_id}: {e_rate}. Retrying after longer backoff...")
                time.sleep((2 ** attempt) * 5)
            except openai.APITimeoutError as e_api_timeout:
                self.logger.warning(f"OpenAI API Timeout Error (Attempt {attempt+1}/{max_retries_from_config}) for ID {unique_id}: {e_api_timeout}. Retrying...")
            except openai.APIConnectionError as e_conn:
                self.logger.error(f"OpenAI API Connection Error (Attempt {attempt+1}/{max_retries_from_config}) for ID {unique_id}: {e_conn}. Retrying...")
                time.sleep(5 + (2**attempt))
            except openai.APIStatusError as e_status:
                self.logger.warning(f"OpenAI API Status Error {e_status.status_code} (Attempt {attempt+1}/{max_retries_from_config}) for ID {unique_id}: {e_status.response}. Retrying for server-side errors...")
                if 500 <= e_status.status_code < 600:
                    pass
                else:
                    self.logger.error(f"OpenAI API Client-side Status Error {e_status.status_code} for ID {unique_id}. Not retrying this error. Error: {e_status.message}")
                    return ""
            except openai.APIError as e_api:
                self.logger.warning(f"General OpenAI API Error (Attempt {attempt+1}/{max_retries_from_config}) for ID {unique_id}: {e_api}. Retrying...")
            except Exception as e_generic:
                self.logger.error(f"Unexpected error during OpenAI API call (Attempt {attempt+1}/{max_retries_from_config}) for ID {unique_id}: {e_generic}", exc_info=True)

            if attempt < max_retries_from_config - 1:
                time.sleep(1 + (2 ** attempt))
            else:
                self.logger.error(f"Exhausted all {max_retries_from_config} retries for OpenAI API call for ID {unique_id}. No successful response obtained.")
                return ""
        return ""

    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        if not response_text or not response_text.strip():
            self.logger.error("Empty or whitespace-only response text received from LLM, cannot parse.")
            return None

        json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text, re.DOTALL)
        content_to_parse: Optional[str] = None
        if json_match:
            content_to_parse = json_match.group(1).strip()
            self.logger.debug("Extracted JSON content from markdown code block.")
        else:
            first_brace_index = response_text.find('{')
            last_brace_index = response_text.rfind('}')
            if first_brace_index != -1 and last_brace_index != -1 and last_brace_index > first_brace_index:
                content_to_parse = response_text[first_brace_index : last_brace_index + 1].strip()
                self.logger.debug("Extracted JSON content by finding first and last braces (no markdown block).")
            else:
                self.logger.error("No JSON structure (markdown block or direct object) found in LLM response.")
                self.logger.debug(f"LLM Response (first 500 chars for context): {response_text[:500]}...")
                return None
        
        if not content_to_parse:
            self.logger.error("Content to parse is empty after extraction attempts.")
            return None

        try:
            parsed_json = json.loads(content_to_parse)
            if not isinstance(parsed_json, dict):
                self.logger.error(f"Parsed content from LLM is not a JSON object (dictionary). Actual type: {type(parsed_json)}")
                return None
        except json.JSONDecodeError as e_decode:
            self.logger.error(f"JSON decode error from LLM response: {e_decode.msg} at line {e_decode.lineno} col {e_decode.colno} (pos {e_decode.pos})", exc_info=False)
            self.logger.debug(f"Content failing JSON parse (first 500 chars): {content_to_parse[:500]}...")
            try:
                repaired_content = re.sub(r',\s*([\}\]])', r'\1', content_to_parse)
                if repaired_content != content_to_parse:
                    self.logger.info("Attempted repair of trailing commas in JSON from LLM response.")
                    parsed_json = json.loads(repaired_content)
                    if not isinstance(parsed_json, dict):
                        self.logger.error("Repaired content from LLM is still not a JSON object after trailing comma removal.")
                        return None
                else:
                    return None
            except json.JSONDecodeError as e_repair:
                self.logger.error(f"JSON decode error persisted even after repair attempt: {e_repair.msg}", exc_info=False)
                return None
        
        validated_data: Dict[str, Any] = {}
        missing_core_keys_log = []

        for key in self.CORE_MODULES:
            if key in parsed_json and isinstance(parsed_json[key], str):
                validated_data[key] = parsed_json[key]
            else:
                validated_data[key] = ""
                if key not in parsed_json:
                    missing_core_keys_log.append(key)
                elif not isinstance(parsed_json[key], str):
                    self.logger.warning(f"LLM response for core key '{key}' was not a string (type: {type(parsed_json.get(key))}). Defaulted to empty string.")

        if missing_core_keys_log:
            self.logger.warning(f"LLM response was missing the following core keys, which have been added with empty strings: {missing_core_keys_log}")

        for key in self.OPTIONAL_MODULES:
            if key in parsed_json and isinstance(parsed_json[key], str):
                validated_data[key] = parsed_json[key]
            elif key in parsed_json:
                 self.logger.warning(f"LLM response for optional key '{key}' was not a string (type: {type(parsed_json[key])}). Omitting this key from final output.")
        
        unclassified_raw_content = parsed_json.get(self.UNCLASSIFIED_KEY)
        if isinstance(unclassified_raw_content, list):
            validated_data[self.UNCLASSIFIED_KEY] = [str(item) for item in unclassified_raw_content if isinstance(item, (str, int, float, bool))]
        elif isinstance(unclassified_raw_content, str) and unclassified_raw_content.strip():
            validated_data[self.UNCLASSIFIED_KEY] = [unclassified_raw_content.strip()]
        else:
            validated_data[self.UNCLASSIFIED_KEY] = []
            if unclassified_raw_content is not None:
                 self.logger.warning(f"'{self.UNCLASSIFIED_KEY}' from LLM was not a list or usable string (type: {type(unclassified_raw_content)}). Defaulted to empty list. Value (first 100 chars): {str(unclassified_raw_content)[:100]}")

        unexpected_keys_found = [k for k in parsed_json if k not in self.ALL_POSSIBLE_KEYS]
        if unexpected_keys_found:
            self.logger.warning(f"LLM response included unexpected keys which were ignored: {unexpected_keys_found}")

        self.logger.info("LLM response parsed, validated, and structured successfully.")
        return validated_data

    def _fallback_segmentation(self, syllabus_text: str) -> Dict[str, Any]:
        self.logger.info("Executing improved fallback rule-based segmentation.")
        sections: Dict[str, Any] = {key: "" for key in self.CORE_MODULES}
        sections[self.UNCLASSIFIED_KEY] = []

        header_patterns_map = {
            "course_schedule": r"^\s*(Course\s*Schedule|Class\s*Schedule|Weekly\s*(?:Schedule|Plan|Outline)|Calendar\s*(?:of\s*Topics)?|Topical\s*Outline|Course\s*Calendar|Detailed\s*Schedule|Exam\s*and\s*Homework\s*Schedule)\s*[:\n]",
            "assignments_exams": r"^\s*(Assignments?|Exams?|Midterm|Final\s*Exam|Projects?|Papers?|Major\s*Assessments?|Assessment\s*Overview|Key\s*Deliverables)\s*[:\n]",
            "course_requirements_grading_policy": r"^\s*(Grading\s*(?:Policy|Scale|Breakdown|Criteria)?|Course\s*Requirements?|Assessment\s*Methods?|Evaluation|Weighting\s*of\s*Grades)\s*[:\n]",
            "required_texts_materials": r"^\s*(Required\s*(?:Texts?|Materials|Readings?)|Textbooks?|Course\s*Materials|Reading\s*List|Resources)\s*[:\n]",
            "learning_objectives": r"^\s*(Learning\s*(?:Objectives?|Outcomes?)|Course\s*Goals|Student\s*Outcomes?|Aims\s*of\s*the\s*Course)\s*[:\n]",
            "course_description_prerequisites": r"^\s*(Course\s*Description|Course\s*Overview|About\s*this\s*Course|Prerequisites|Co-?requisites|Background)\s*[:\n]",
            "instructor_information": r"^\s*(Instructor(?:s)?(?:\s*Information)?|Professor|Faculty|Teaching\s*(?:Staff|Team|Assistants?))\s*[:\n]",
            "course_identification": r"^\s*((?:[A-Z]{2,5}\s*\d{2,4}[A-Z]?\s*[:\-–—\s]+.*?)(?:\n.*?){0,4})",
            "course_policies": r"^\s*(Course\s*Polic(?:y|ies)|General\s*Policies|Classroom\s*Policies|Attendance(?:\s*Policy)?|Academic\s*Integrity|Disability\s*(?:Services|Statement)|Accommodations?|Late\s*Work)\s*[:\n]",
            "communication_student_support": r"^\s*(Communication|Student\s*Support|Office\s*Hours|Academic\s*(?:Enhancement|Support)|Tutoring|Writing\s*Center|Counseling|Getting\s*Help)\s*[:\n]",
            "university_wide_policies_resources": r"^\s*(University\s*(?:-?Wide|Level)?\s*Polic(?:y|ies)|Institutional\s*Policies|General\s*Education\s*Areas|FERPA|Title\s*IX|Non-?Discrimination|ADA\s*Statement|Honor\s*Code|Campus\s*Safety|Student\s*Conduct)\s*[:\n]",
            "separate_laboratory_sections": r"^\s*(Laboratory\s*(?:Sections?|Schedule|Policies|Information)?|Lab\s*(?:Information|Schedule|Details)?|Experiment\s*Schedule)\s*[:\n]",
            "recitation_discussion_sections": r"^\s*(Recitation(?:\s*Sections?)?|Discussion\s*Section|Tutorial\s*Sessions?|Problem\s*Solving\s*Session)\s*[:\n]",
        }

        course_id_pattern_str = header_patterns_map.pop("course_identification")
        initial_content_offset = 0
        course_id_match = re.match(course_id_pattern_str, syllabus_text, re.IGNORECASE)
        if course_id_match:
            sections["course_identification"] = course_id_match.group(1).strip()
            initial_content_offset = course_id_match.end()
            self.logger.info(f"Fallback: Found 'course_identification' (approx. {len(sections['course_identification'])} chars).")

        all_matches: List[Dict[str, Any]] = []
        for key, pattern_str in header_patterns_map.items():
            try:
                for match in re.finditer(pattern_str, syllabus_text, re.IGNORECASE | re.MULTILINE):
                    if match.start() >= initial_content_offset:
                        all_matches.append({"key": key, "start": match.start(), "header_text": match.group(0).strip()})
            except re.error as e_re_fallback:
                self.logger.error(f"Regex error in fallback segmentation for key '{key}', pattern '{pattern_str}': {e_re_fallback}")

        all_matches.sort(key=lambda m: m["start"])

        current_position = initial_content_offset
        for i, match_info in enumerate(all_matches):
            section_key = match_info["key"]
            section_start = match_info["start"]
            header_text_content = match_info["header_text"]

            text_before_this_header = syllabus_text[current_position:section_start].strip()
            if text_before_this_header and len(text_before_this_header) > 30:
                sections[self.UNCLASSIFIED_KEY].append(text_before_this_header)
                self.logger.debug(f"Fallback: Added text block to unclassified (before '{section_key}', approx. {len(text_before_this_header)} chars).")

            content_end = all_matches[i+1]["start"] if (i + 1) < len(all_matches) else len(syllabus_text)
            
            section_content = header_text_content + "\n" + syllabus_text[section_start + len(header_text_content) : content_end].strip()
            section_content = section_content.strip()


            if section_content:
                if sections.get(section_key):
                    sections[section_key] += "\n\n" + section_content
                else:
                    sections[section_key] = section_content
                self.logger.info(f"Fallback: Assigned content to '{section_key}' (approx. {len(section_content)} chars).")
            
            current_position = content_end

        if current_position < len(syllabus_text):
            remaining_text_at_end = syllabus_text[current_position:].strip()
            if remaining_text_at_end and len(remaining_text_at_end) > 30:
                sections[self.UNCLASSIFIED_KEY].append(remaining_text_at_end)
                self.logger.info(f"Fallback: Added remaining text at end to unclassified (approx. {len(remaining_text_at_end)} chars).")

        for opt_key in self.OPTIONAL_MODULES:
            sections.setdefault(opt_key, "")

        self.logger.info(f"Fallback segmentation completed. Number of unclassified content blocks: {len(sections[self.UNCLASSIFIED_KEY])}.")
        return sections


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s',
        handlers=[
            logging.FileHandler("syllabus_parser_standalone.log", mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__) # Re-get after basicConfig


    import argparse
    arg_parser = argparse.ArgumentParser(description="Parse and segment a pre-converted syllabus text file (.txt) into logical sections.")
    arg_parser.add_argument("input_file", help="Path to the pre-converted syllabus text file (.txt).") # Updated help text
    arg_parser.add_argument("--output-dir", help="Base directory for saving output JSON files.")
    arg_parser.add_argument("--log-dir", help="Directory for log files.")
    arg_parser.add_argument("--model", default="gpt-4o", help="OpenAI model preference for API calls.")
    arg_parser.add_argument("--config", help="Path to a YAML configuration file to load settings.")
    args = arg_parser.parse_args()

    app_config = {}
    if args.config:
        try:
            import yaml
            config_path_arg = Path(args.config)
            if config_path_arg.is_file():
                with config_path_arg.open('r', encoding='utf-8') as f_cfg_yaml:
                    app_config = yaml.safe_load(f_cfg_yaml)
                if app_config: logger.info(f"Successfully loaded configuration from: {args.config}")
                else: logger.warning(f"Configuration file {args.config} is empty or invalid.")
            else: logger.error(f"Configuration file not found: {args.config}.")
        except ImportError: logger.warning("PyYAML library not installed. Cannot load YAML config file.")
        except Exception as e_cfg_load: logger.error(f"Error loading config file {args.config}: {e_cfg_load}", exc_info=True)

    if args.output_dir:
        if "directories" not in app_config: app_config["directories"] = {}
        app_config["directories"]["parsed_syllabus_dir"] = args.output_dir
    if args.log_dir:
        if "directories" not in app_config: app_config["directories"] = {}
        app_config["directories"]["logs"] = args.log_dir
    if args.model:
        if "extraction" not in app_config: app_config["extraction"] = {}
        app_config["extraction"]["openai_model"] = args.model

    parser_instance = SyllabusParser(
        config=app_config,
        base_dir=args.output_dir if args.output_dir else app_config.get("directories",{}).get("parsed_syllabus_dir"),
        log_dir=args.log_dir if args.log_dir else app_config.get("directories",{}).get("logs"),
        model=args.model,
        logger_instance=logger
    )
    
    # Crucially, pass the input_file argument from args to parse_syllabus
    result = parser_instance.parse_syllabus(args.input_file)


    status = result.get("status", "error_unknown_status")
    if "error" in status:
        logger.error(f"Syllabus parsing failed. Status: {status}, Message: {result.get('error', result.get('message', 'N/A'))}")
        sys.exit(1)
    elif "warning" in status:
        logger.warning(f"Syllabus parsing completed with warnings. Status: {status}, Message: {result.get('message', 'N/A')}")
    else:
        logger.info("Syllabus parsing successful.")

    if result.get('parsed_data'):
        logger.info(f"Parsed data keys: {list(result.get('parsed_data', {}).keys())}")
        logger.info(f"Full text content length: {len(result.get('full_text_content', ''))} characters.")

    if result.get('parsed_output_file'):
        logger.info(f"Segmented syllabus output saved to: {result.get('parsed_output_file')}")
    else:
        logger.warning("Segmented syllabus output file path not available in results.")

    sys.exit(0)
