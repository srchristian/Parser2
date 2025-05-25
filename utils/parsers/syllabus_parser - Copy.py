"""
syllabus_parser.py

Text segmentation module that extracts and segments syllabus text into logical sections.
It focuses on identifying structure and removing irrelevant content while preserving all
schedule, assignment, and course information. Optionally uses config.yaml if provided.
"""

import os
import sys
import re
import json
import uuid
import time
import traceback
from datetime import datetime
from pathlib import Path

# Attempt to load .env, though typically app.py or system env is used
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Will use existing environment variables.")

# Attempt to import openai
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI package not installed. Limited functionality for segmentation.")
    OPENAI_AVAILABLE = False

# Add the project root to sys.path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Attempt to import helper functions from helpers.py
try:
    from utils.helpers import read_json, write_json
except ImportError:
    print("Warning: utils.helpers not found, using fallback implementations.")
    def read_json(file_path):
        """Fallback: read JSON from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON from {file_path}: {e}")
            return None
    
    def write_json(file_path, data, create_dirs=True, backup_existing=False):
        """Fallback: write JSON to a file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error writing JSON to {file_path}: {e}")
            return False

class SyllabusParser:
    """
    A class to parse and segment course syllabi into logical sections,
    removing irrelevant content while preserving important information.
    """

    def __init__(self, config=None, base_dir=None, model="gpt-4o", log_dir=None):
        """
        Initializes the SyllabusParser. If a config is provided, it can override
        the base_dir and log_dir. Otherwise, the old approach is used.

        Args:
            config (dict): Optional dictionary loaded from config.yaml (helpers.load_configuration).
            base_dir (str): The base directory path for saving parsed syllabi (used if config not provided).
            model (str): The OpenAI model to use for parsing.
            log_dir (str): Directory to save logs (used if config not provided).
        """
        self.config = config or {}

        # If config is provided, fetch base_dir and log_dir from config
        if config and "directories" in config:
            # Syllabus repository base
            self.base_dir = config["directories"].get("parsed_syllabus_dir") or os.path.join(current_dir, "syllabus_repository", "parsed_syllabus")
            self.log_dir = config["directories"].get("logs") or os.path.join(project_root, "logs")
        else:
            # Fallback to older approach
            self.base_dir = base_dir if base_dir else os.path.join(current_dir, "syllabus_repository", "parsed_syllabus")
            self.log_dir = log_dir if log_dir else os.path.join(project_root, "logs")

        self.model = model
        self.create_directories()
        
        # Initialize OpenAI client
        self.openai_status = "not_initialized"
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                try:
                    openai.api_key = api_key
                    self.openai_status = "ready"
                    print(f"OpenAI initialized with model: {self.model}")
                except Exception as e:
                    print(f"Error initializing OpenAI: {e}")
                    self.openai_status = "error"
            else:
                print("Warning: OPENAI_API_KEY not found in environment variables.")
                self.openai_status = "no_api_key"

    def create_directories(self):
        """Ensure base_dir and log_dir exist."""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"SyllabusParser directories ensured:\n- base_dir: {self.base_dir}\n- log_dir: {self.log_dir}")

    def parse_syllabus(self, input_file, unique_id=None):
        """
        Orchestrates the syllabus segmentation process.

        Args:
            input_file (str): Path to the syllabus text file.
            unique_id (str, optional): Unique identifier for the syllabus.

        Returns:
            dict: Results dictionary with status and data.
        """
        print(f"Starting syllabus parsing for: {input_file}")
        start_time = time.time()

        # Verify the input file
        if not os.path.isfile(input_file):
            error_msg = f"Error: Input file not found: {input_file}"
            print(error_msg)
            return {"error": error_msg, "status": "error"}

        if not unique_id:
            unique_id = str(uuid.uuid4())
        print(f"Using unique ID: {unique_id}")

        # Read the syllabus text
        syllabus_text = self._read_syllabus(input_file)
        if not syllabus_text:
            error_msg = f"Error: Could not read syllabus from {input_file}"
            print(error_msg)
            return {"error": error_msg, "status": "error"}

        print(f"Read {len(syllabus_text)} characters from syllabus file.")

        # Generate prompt and call OpenAI
        prompt = self._generate_segmentation_prompt(syllabus_text)
        response_text = self._call_openai_api(prompt)

        if not response_text:
            # Fallback
            warning_msg = "Warning: Failed to get a valid response from OpenAI. Using fallback segmentation."
            print(warning_msg)
            parsed_sections = self._fallback_segmentation(syllabus_text)
            parsed_sections["UUID"] = unique_id
            # Save fallback
            parsed_output_file = os.path.join(self.base_dir, f"{unique_id}.json")
            write_json(parsed_output_file, parsed_sections)
            return {
                "message": warning_msg,
                "status": "warning",
                "parsed_output_file": parsed_output_file,
                "parsed_data": parsed_sections
            }

        # Attempt to parse the response
        parsed_sections = self._parse_response(response_text)
        if not parsed_sections:
            warning_msg = "Warning: Failed to parse API response. Using fallback segmentation."
            print(warning_msg)
            parsed_sections = self._fallback_segmentation(syllabus_text)
            parsed_sections["UUID"] = unique_id
            parsed_output_file = os.path.join(self.base_dir, f"{unique_id}.json")
            write_json(parsed_output_file, parsed_sections)
            return {
                "message": warning_msg,
                "status": "warning",
                "parsed_output_file": parsed_output_file,
                "parsed_data": parsed_sections
            }

        # If successful, record the parsed data
        parsed_sections["UUID"] = unique_id
        parsed_output_file = os.path.join(self.base_dir, f"{unique_id}.json")
        save_success = write_json(parsed_output_file, parsed_sections)
        if not save_success:
            warning_msg = f"Warning: Could not save parsed syllabus to {parsed_output_file}"
            print(warning_msg)
            return {"error": warning_msg, "status": "warning", "parsed_data": parsed_sections}

        process_time = time.time() - start_time
        print(f"Syllabus parsing completed in {process_time:.2f} seconds")
        return {
            "message": "Syllabus parsing complete",
            "status": "success",
            "parsed_output_file": parsed_output_file,
            "parsed_data": parsed_sections,
            "process_time": process_time
        }

    # ----------------------------------------------------------------
    # Internal / Private-like Methods
    # ----------------------------------------------------------------

    def _read_syllabus(self, file_path):
        """Reads plain text from a file, tries UTF-8 or falls back to latin-1."""
        if not os.path.isfile(file_path):
            print(f"Error: File does not exist: {file_path}")
            return ""
        try:
            # Try UTF-8
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            # Fallback to latin-1
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
                print(f"Warning: {file_path} was not UTF-8. Used latin-1 fallback.")
                return content
            except Exception as e:
                print(f"Error reading file {file_path} with latin-1: {e}")
                return ""

    def _generate_segmentation_prompt(self, syllabus_text):
        """Builds a prompt for the OpenAI model to segment the syllabus text."""
        sections = [
            "course_info",      # Basic course info (title, instructor, location)
            "schedule",         # Class schedule, meeting times
            "assignments",      # Homework, projects, deadlines
            "exams",            # Exam and assessment info
            "readings",         # Required/recommended reading materials
            "special_sessions"  # Special classes, guest lectures, etc.
        ]
        sections_list = "\n".join(f"- {s}: Segment for {s.replace('_', ' ')}" for s in sections)

        prompt = (
            "You are an academic assistant tasked with segmenting a course syllabus into logical sections. "
            "Extract and categorize the content into the following distinct sections:\n"
            f"{sections_list}\n\n"
            "For each section, extract all relevant content, preserving:\n"
            "- All dates and times\n"
            "- All locations/room numbers\n"
            "- All assignment descriptions and due dates\n"
            "- All exam information\n"
            "- Course meeting patterns (MWF, TR, TTh)\n"
            "- Instructor contact details\n\n"
            "IMPORTANT: If content belongs in multiple sections, include it in each relevant section. "
            "Exclude irrelevant content like general academic policies.\n\n"
            "Return the segmented syllabus as valid JSON with these keys:\n"
            "`course_info, schedule, assignments, exams, readings, special_sessions`.\n\n"
            "### Output Format ###\n"
            "```json\n"
            "{\n"
            "  \"course_info\": \"...\",\n"
            "  \"schedule\": \"...\",\n"
            "  \"assignments\": \"...\",\n"
            "  \"exams\": \"...\",\n"
            "  \"readings\": \"...\",\n"
            "  \"special_sessions\": \"...\"\n"
            "}\n"
            "```\n\n"
            "Original Syllabus Text:\n"
            f"{syllabus_text}\n"
        )
        return prompt

    def _call_openai_api(self, prompt, max_retries=3):
        """Calls the OpenAI API with a prompt and basic retry logic."""
        if not OPENAI_AVAILABLE:
            print("Error: OpenAI package not installed. Segmentation by AI unavailable.")
            return ""
        if self.openai_status != "ready":
            print(f"Error: OpenAI not ready. Status: {self.openai_status}")
            return ""

        debug_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save prompt for debugging if needed
        debug_dir = os.path.join(self.base_dir, "debug_prompts")
        os.makedirs(debug_dir, exist_ok=True)
        prompt_file = os.path.join(debug_dir, f"prompt_{debug_id}.txt")
        try:
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception as e:
            print(f"Warning: could not save prompt debug file: {e}")

        for attempt in range(max_retries):
            try:
                print(f"Calling OpenAI (attempt {attempt+1}/{max_retries})...")
                # Updated to use the new OpenAI API format without temperature parameter
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content.strip()
                # Save response
                resp_file = os.path.join(debug_dir, f"response_{debug_id}_attempt{attempt+1}.txt")
                try:
                    with open(resp_file, "w", encoding="utf-8") as f:
                        f.write(response_text)
                except Exception as e:
                    print(f"Warning: could not save response debug file: {e}")

                if not response_text or len(response_text) < 50:
                    print(f"Warning: short response ({len(response_text)} chars). Retrying...")
                    time.sleep(2)
                    continue

                print(f"Received OpenAI response ({len(response_text)} chars)")
                return response_text
            except Exception as e:
                print(f"Error calling OpenAI (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        return ""

    def _parse_response(self, response_text):
        """Extract JSON from the model's response if possible, otherwise return None."""
        if not response_text.strip():
            print("Error: Empty response from OpenAI.")
            return None

        # Attempt to find JSON in code fences
        codeblock_json = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if codeblock_json:
            content = codeblock_json.group(1)
        else:
            # Or look for bare JSON
            any_json = re.search(r"(\{[\s\S]*\})", response_text)
            content = any_json.group(1) if any_json else response_text

        # Attempt to parse JSON
        try:
            parsed = json.loads(content)
            # Ensure all expected sections exist
            for key in ["course_info", "schedule", "assignments", "exams", "readings", "special_sessions"]:
                if key not in parsed:
                    parsed[key] = ""
            return parsed
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None

    def _fallback_segmentation(self, syllabus_text):
        """
        Extremely basic fallback method if AI call fails.
        Scans for common headings and lumps them together.
        """
        sections = {
            "course_info": "",
            "schedule": "",
            "assignments": "",
            "exams": "",
            "readings": "",
            "special_sessions": ""
        }

        # Minimal approach: place entire text in course_info
        # or attempt regex-based extraction. This is just a placeholder.
        # For now, we just store full text in 'course_info'.
        sections["course_info"] = syllabus_text
        return sections

# If run directly:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse and segment a syllabus file.")
    parser.add_argument("input_file", help="Path to the syllabus text file.")
    parser.add_argument("--output-dir", help="Base directory for output files.")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use.")
    args = parser.parse_args()

    # In a real pipeline, we'd load config.yaml via helpers.load_configuration,
    # but for direct script usage, we skip it or do minimal logic:
    parser_instance = SyllabusParser(base_dir=args.output_dir, model=args.model)
    result = parser_instance.parse_syllabus(args.input_file)

    if result["status"] == "error":
        print(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print(f"Success! Parsed syllabus saved to: {result.get('parsed_output_file')}")
        sys.exit(0)