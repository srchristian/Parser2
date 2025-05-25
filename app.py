import os
import sys
import uuid
import json
from datetime import datetime
from pathlib import Path
import logging # Standard library logging

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename # For secure file operations
from typing import Union, Optional, Dict, Any, List # For type hinting

# --------------------------------------------------
# 1. Setup Project Root
# --------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --------------------------------------------------
# 2. Import Custom Helper Functions & Key Classes (with Fallbacks)
# --------------------------------------------------
HELPERS_AVAILABLE = False
ALL_POSSIBLE_CLASS_DATA_FIELDS = [] # For class_data structure

def _fallback_setup_logger(log_dir="logs", log_name="FallbackLogger", log_level_val=logging.INFO, config_format=None, config_datefmt=None):
    fb_logger = logging.getLogger(log_name)
    if not fb_logger.handlers: # Configure only once
        log_format_str = config_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format_str = config_datefmt or '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(log_format_str, datefmt=date_format_str)
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(formatter)
        fb_logger.addHandler(ch)
        fb_logger.setLevel(log_level_val)
        try:
            os.makedirs(log_dir, exist_ok=True)
            fb_file_handler = logging.FileHandler(os.path.join(log_dir, f"{log_name}_fallback.log"))
            fb_file_handler.setFormatter(formatter)
            fb_logger.addHandler(fb_file_handler)
        except Exception as e_fb_log: print(f"Error setting up fallback file logger: {e_fb_log}", file=sys.stderr)
    return fb_logger

try:
    from utils.helpers import (
        load_configuration, setup_environment, setup_logger, allowed_file,
        convert_webpage_to_txt, save_uploaded_file, process_syllabus_pipeline,
        read_json, write_json
    )
    from utils.parsers.openai_parser import OpenAIParser as OpenAIParserClass_ForFields # To get field list
    ALL_POSSIBLE_CLASS_DATA_FIELDS = OpenAIParserClass_ForFields.CLASS_DATA_FIELDS
    HELPERS_AVAILABLE = True
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import core modules (utils.helpers or utils.parsers.openai_parser): {e}. Application will use fallbacks.", file=sys.stderr)
    def load_configuration(config_path=None, default_config=None): return default_config or {}
    def setup_environment(config, pr): pass
    setup_logger = _fallback_setup_logger
    def allowed_file(filename, allowed_extensions): return False
    def convert_webpage_to_txt(url, output_path=None): return None
    def save_uploaded_file(file_obj, upload_folder, unique_id_prefix): return None, None
    def process_syllabus_pipeline(input_path_or_text, config, parsers, original_filename_for_display=None, transient_run_id=None):
        return {"metadata": {"error": "Pipeline unavailable (helpers missing)", "unique_id": transient_run_id or "error_id", "missing_fields": [], "fields_to_prompt_user": []}, "class_data": {}}
    def read_json(file_path, fallback_data=None): return fallback_data or {}
    def write_json(file_path, data, create_dirs=True, backup_existing=True, indent=4): return False
    ALL_POSSIBLE_CLASS_DATA_FIELDS = [
        "School Name", "Term", "Course Title", "Course Code", "Instructor Name",
        "Instructor Email", "Class Time", "Time Zone", "Days of Week",
        "Term Start Date", "Term End Date", "Class Start Date", "Class End Date",
        "Office Hours", "Telephone", "Class Location", "Additional"
    ]

# --------------------------------------------------
# 3. Load Configuration
# --------------------------------------------------
config_path = os.path.join(project_root, "config.yaml")
app_default_config = {
    "project_root": project_root,
    "directories": {"logs": "logs", "uploads": "uploads", "converted": "converted", "output": "output",
                    "original_syllabus_dir": "syllabus_data/originals",
                    "converted_syllabus_dir": "syllabus_data/converted_text",
                    "parsed_syllabus_dir": "syllabus_data/parsed_results",
                    "main_data_dir": "syllabus_data/main_data_store"},
    "logging": {"level": "INFO", "format": '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s', "datefmt": '%Y-%m-%d %H:%M:%S'},
    "extraction": {"openai_model": "gpt-4.1-nano"},
    "flask": {"secret_key": "default_secret_key_CHANGE_ME", "host": "0.0.0.0", "port": 5000, "debug": True},
    "file_settings": {"max_content_length": 16 * 1024 * 1024, "allowed_extensions": ["pdf", "txt", "docx", "html", "htm"]},
    "required_fields": ["Term", "Course Title", "Course Code", "Instructor Name", "Class Time", "Days of Week", "Class Start Date", "Class End Date", "Time Zone"]
}
config = load_configuration(config_path=config_path, default_config=app_default_config)

# --------------------------------------------------
# 4. Setup Environment (Directories)
# --------------------------------------------------
if HELPERS_AVAILABLE: setup_environment(config, project_root)
else:
    for dir_path_str in config.get("directories", {}).values():
        try: Path(project_root).joinpath(dir_path_str).mkdir(parents=True, exist_ok=True)
        except Exception: pass

# --------------------------------------------------
# 5. Initialize Logger (Global)
# --------------------------------------------------
log_level_str = config.get("logging", {}).get("level", "DEBUG")
numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
logger = setup_logger(
    log_dir=config.get("directories", {}).get("logs", "logs"),
    log_name="SyllabusWebApp",
    log_level=numeric_level,
    config_format=config.get("logging", {}).get("format"),
    config_datefmt=config.get("logging", {}).get("datefmt")
)
if not HELPERS_AVAILABLE: logger.warning("Application is running with FALLBACK helper functions due to import errors.")

# --------------------------------------------------
# 6. Initialize Parsers (Global)
# --------------------------------------------------
parsers_dict: Dict[str, Any] = {}
date_parser, openai_parser, syllabus_parser, extractor, schedule_parser, assignment_parser, lab_parser, recitation_parser = None, None, None, None, None, None, None, None
openai_model_from_config = config.get("extraction", {}).get("openai_model", "gpt-4o")

try: from utils.parsers.date_parser import DateParser as SyllabusDateParser; date_parser = SyllabusDateParser(logger_instance=logger); parsers_dict['date_parser'] = date_parser
except Exception as e: logger.error(f"Failed to init DateParser: {e}", exc_info=True)
try: from utils.parsers.openai_parser import OpenAIParser; openai_parser = OpenAIParser(model=openai_model_from_config, logger_instance=logger, config=config); parsers_dict['openai_parser'] = openai_parser
except Exception as e: logger.error(f"Failed to init OpenAIParser: {e}", exc_info=True)
try:
    from utils.parsers.syllabus_parser import SyllabusParser
    syllabus_parser_base_dir = Path(config.get("directories", {}).get("parsed_syllabus_dir", Path(project_root) / "output" / "syllabus_parser_debug"))
    syllabus_parser = SyllabusParser(config=config, base_dir=str(syllabus_parser_base_dir), model=openai_model_from_config, logger_instance=logger)
    parsers_dict['syllabus_parser'] = syllabus_parser
except Exception as e: logger.error(f"Failed to init SyllabusParser: {e}", exc_info=True)
try:
    from utils.syllabus_extractor import SyllabusExtractor
    extractor = SyllabusExtractor(config=config, openai_model=openai_model_from_config, logger_instance=logger, openai_parser_instance=openai_parser)
    parsers_dict['extractor'] = extractor
except Exception as e: logger.error(f"Failed to init SyllabusExtractor: {e}", exc_info=True)
try:
    from utils.parsers.assignment_parser import AssignmentParser
    if date_parser and openai_parser:
        assignment_parser = AssignmentParser(logger_instance=logger, date_parser_instance=date_parser, openai_parser_instance=openai_parser, config=config)
        parsers_dict['assignment_parser'] = assignment_parser
    else: logger.warning("AssignmentParser not initialized due to missing dependencies (DateParser or OpenAIParser).")
except Exception as e: logger.error(f"Failed to init AssignmentParser: {e}", exc_info=True)
try:
    from utils.parsers.schedule_parser import ScheduleParser
    if date_parser and openai_parser:
        schedule_parser = ScheduleParser(logger_instance=logger, date_parser_instance=date_parser, openai_parser_instance=openai_parser, config=config)
        parsers_dict['schedule_parser'] = schedule_parser
    else: logger.warning("ScheduleParser not initialized due to missing dependencies.")
except Exception as e: logger.error(f"Failed to init ScheduleParser: {e}", exc_info=True)
try:
    from utils.parsers.lab_parser import LabParser
    if date_parser and openai_parser:
        lab_parser = LabParser(logger_instance=logger, date_parser_instance=date_parser, openai_parser_instance=openai_parser, config=config)
        parsers_dict['lab_parser'] = lab_parser
    else: logger.warning("LabParser not initialized due to missing dependencies.")
except Exception as e: logger.error(f"Failed to init LabParser: {e}", exc_info=True)
try:
    from utils.parsers.recitation_parser import RecitationParser
    if date_parser and openai_parser:
        recitation_parser = RecitationParser(logger_instance=logger, date_parser_instance=date_parser, openai_parser_instance=openai_parser, config=config)
        parsers_dict['recitation_parser'] = recitation_parser
    else: logger.warning("RecitationParser not initialized due to missing dependencies.")
except Exception as e: logger.error(f"Failed to init RecitationParser: {e}", exc_info=True)

parsers_dict = {k: v for k, v in parsers_dict.items() if v is not None}
logger.info(f"Initialized and available parsers after cleanup: {list(parsers_dict.keys())}")


# --------------------------------------------------
# 7. Initialize Flask App (Global)
# --------------------------------------------------
app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': str(Path(project_root) / config.get("directories", {}).get('uploads', 'uploads')),
    'CONVERTED_FOLDER': str(Path(project_root) / config.get("directories", {}).get('converted', 'converted')),
    'OUTPUT_FOLDER': str(Path(project_root) / config.get("directories", {}).get('output', 'output')),
    'PARSED_SYLLABUS_DIR': str(Path(project_root) / config.get("directories", {}).get('parsed_syllabus_dir', 'syllabus_data/parsed_results')),
    'MAX_CONTENT_LENGTH': config.get('file_settings', {}).get('max_content_length', 16 * 1024 * 1024),
    'SECRET_KEY': config.get('flask', {}).get('secret_key', 'default_secret_key_app_py')
})
for folder_key in ['UPLOAD_FOLDER', 'CONVERTED_FOLDER', 'OUTPUT_FOLDER', 'PARSED_SYLLABUS_DIR']:
    try: Path(app.config[folder_key]).mkdir(parents=True, exist_ok=True)
    except Exception as e_dir_create: logger.error(f"Error creating directory for key {folder_key}: {app.config[folder_key]}, Error: {e_dir_create}")

# --------------------------------------------------
# 8. Global Constants for Templates (Global)
# --------------------------------------------------
COMMON_TIME_ZONES = [
    {"value": "America/New_York", "label": "Eastern Time (ET)"},
    {"value": "America/Chicago", "label": "Central Time (CT)"},
    {"value": "America/Denver", "label": "Mountain Time (MT)"},
    {"value": "America/Los_Angeles", "label": "Pacific Time (PT)"},
    {"value": "America/Phoenix", "label": "Arizona (no DST)"},
    {"value": "UTC", "label": "Coordinated Universal Time (UTC/GMT)"},
    {"value": "Europe/London", "label": "London (GMT/BST)"},
    {"value": "Europe/Berlin", "label": "Central European Time (CET/CEST)"},
]

# --------------------------------------------------
# Route Definitions
# --------------------------------------------------
@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year, 'app_config': config}

@app.route('/')
def index():
    logger.debug(f"Request to / (index) from {request.remote_addr}")
    return render_template('index.html')

def determine_fields_to_prompt(class_data: Dict[str, Any], system_required_fields: List[str]) -> List[str]:
    """Helper function to determine which fields to prompt the user for."""
    fields_to_prompt = []

    # Priority 1: Class Start Date
    if "Class Start Date" in system_required_fields and not class_data.get("Class Start Date", "").strip():
        fields_to_prompt.append("Class Start Date")
    
    # Priority 2: Class End Date
    if "Class End Date" in system_required_fields and not class_data.get("Class End Date", "").strip():
        if "Class End Date" not in fields_to_prompt: # Avoid duplicates if logic changes
            fields_to_prompt.append("Class End Date")

    # Priority 3: Time Zone
    if "Time Zone" in system_required_fields and not class_data.get("Time Zone", "").strip():
        if "Time Zone" not in fields_to_prompt:
            fields_to_prompt.append("Time Zone")

    # Other required fields (excluding Term Dates and already handled fields)
    # These fields are essential for basic operation if not for date calculations.
    fields_to_definitely_not_prompt = ["Term Start Date", "Term End Date"]
    # Fields already handled or to be excluded from this generic loop:
    handled_or_excluded_fields = set(fields_to_definitely_not_prompt + ["Class Start Date", "Class End Date", "Time Zone"])

    for field in system_required_fields:
        if field not in handled_or_excluded_fields:
            if not class_data.get(field, "").strip():
                if field not in fields_to_prompt: # Ensure no duplicates
                    fields_to_prompt.append(field)
    
    return list(set(fields_to_prompt)) # Return unique list


@app.route('/upload', methods=['POST'])
def upload_file_route():
    logger.info("Processing /upload request")
    if not HELPERS_AVAILABLE: flash('Core application components missing.', 'error'); return redirect(url_for('index'))
    if not parsers_dict: flash('Parsing engine not fully initialized.', 'error'); return redirect(url_for('index'))

    transient_run_id = str(uuid.uuid4())
    input_for_pipeline: Union[str, Path, None] = None
    original_filename_for_display: Optional[str] = None
    content_hash_id = transient_run_id 

    try:
        url_input = request.form.get('url', '').strip()
        if url_input:
            logger.info(f"Processing URL: {url_input} (transient_id: {transient_run_id})")
            raw_text_from_url = convert_webpage_to_txt(url_input)
            if raw_text_from_url is None:
                flash(f"Failed to extract text from URL: {url_input}.", "error"); return redirect(url_for('index'))
            input_for_pipeline = raw_text_from_url
            original_filename_for_display = f"web_content_{transient_run_id[:8]}.txt"
        else:
            if 'file' not in request.files or not request.files['file'].filename:
                flash('No file selected.', 'error'); return redirect(url_for('index'))
            file_obj = request.files['file']
            allowed_exts = config.get('file_settings',{}).get('allowed_extensions', [])
            if not allowed_file(file_obj.filename, allowed_exts):
                flash(f'File type not allowed. Allowed: {", ".join(allowed_exts)}.', 'error'); return redirect(url_for('index'))
            
            saved_path, secure_name = save_uploaded_file(file_obj, Path(app.config['UPLOAD_FOLDER']), transient_run_id)
            if not saved_path:
                flash('Error saving uploaded file.', 'error'); return redirect(url_for('index'))
            input_for_pipeline = saved_path
            original_filename_for_display = secure_name
            logger.info(f"File uploaded to {saved_path} (transient_id: {transient_run_id})")
        
        if input_for_pipeline:
            logger.info(f"Starting pipeline for transient_id: {transient_run_id}")
            results = process_syllabus_pipeline(
                input_path_or_text=input_for_pipeline,
                config=config, parsers=parsers_dict,
                original_filename_for_display=original_filename_for_display,
                transient_run_id=transient_run_id
            )
            content_hash_id = results.get("metadata", {}).get("unique_id", content_hash_id)
            results.setdefault("metadata", {})["unique_id"] = content_hash_id

            system_required_fields = config.get("required_fields", [])
            class_data = results.get("class_data", {})
            
            # REVISED PROMPTING LOGIC
            fields_to_prompt_user = determine_fields_to_prompt(class_data, system_required_fields)
            results["metadata"]["fields_to_prompt_user"] = fields_to_prompt_user
            
            output_file_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{content_hash_id}_results.json"
            if HELPERS_AVAILABLE and callable(write_json):
                if not write_json(output_file_path, results):
                    logger.error(f"Failed to save updated results to {output_file_path} before redirecting.")
                    flash('Error saving intermediate processing results. Please try again.', 'error')
                    return redirect(url_for('index'))
                logger.info(f"Successfully saved updated results to {output_file_path} with fields_to_prompt_user: {fields_to_prompt_user} before redirect.")
            else:
                logger.error(f"write_json helper not available. Cannot save updated results to {output_file_path}.")
                flash('Critical error: JSON writing utility missing. Cannot proceed.', 'error')
                return redirect(url_for('index'))

            if results["metadata"]["fields_to_prompt_user"]:
                logger.info(f"Redirecting to missing fields for ID {content_hash_id}. Fields: {results['metadata']['fields_to_prompt_user']}")
                return redirect(url_for('missing_fields', unique_id=content_hash_id))
            
            logger.info(f"Redirecting to results for ID {content_hash_id}.")
            return redirect(url_for('show_results', unique_id=content_hash_id))
        else:
            flash('Could not obtain input from file or URL.', 'error'); return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Unhandled exception in /upload (transient_run_id: {transient_run_id}, content_hash_id: {content_hash_id if 'content_hash_id' in locals() and content_hash_id != transient_run_id else 'same as transient_run_id'}): {e}", exc_info=True)
        flash('An unexpected error occurred during processing. Please check server logs.', 'error')
        return redirect(url_for('index'))


@app.route('/missing-fields/<unique_id>')
def missing_fields(unique_id):
    logger.info(f"Displaying missing_fields page for ID: {unique_id}")
    results_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
    if not results_path.exists():
        results_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json" 
        if not results_path.exists():
            flash('Results file not found. Please re-process syllabus.', 'error'); return redirect(url_for('index'))
    
    data = read_json(results_path)
    if not data: flash('Error reading results data.', 'error'); return redirect(url_for('index'))
    
    fields_to_prompt = data.get('metadata', {}).get('fields_to_prompt_user', [])
    if not fields_to_prompt and data.get('class_data'): 
        logger.info(f"No fields to prompt for {unique_id}, redirecting to results.")
        return redirect(url_for('show_results', unique_id=unique_id))
            
    return render_template('missing_fields.html', unique_id=unique_id,
                           missing_fields=fields_to_prompt, 
                           class_data=data.get('class_data', {}),
                           time_zones=COMMON_TIME_ZONES)

@app.route('/complete-data/<unique_id>', methods=['POST'])
def complete_data(unique_id):
    logger.info(f"Processing /complete-data for ID: {unique_id}")
    try:
        results_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
        if not results_path.exists():
            results_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json"
            if not results_path.exists():
                flash('Original results file not found for update.', 'error'); return redirect(url_for('index'))
        
        data = read_json(results_path)
        if not data: flash('Error reading existing results for update.', 'error'); return redirect(url_for('index'))

        data.setdefault("class_data", {})
        data.setdefault("metadata", {})
        
        prompted_fields_this_round = data.get("metadata", {}).get("fields_to_prompt_user", [])
        
        for key in request.form:
            if key in ALL_POSSIBLE_CLASS_DATA_FIELDS or key in prompted_fields_this_round:
                data["class_data"][key] = request.form[key]
        
        term_for_date_parsing = data.get("class_data", {}).get("Term")
        
        current_date_parser = parsers_dict.get('date_parser')
        current_schedule_parser = parsers_dict.get('schedule_parser')
        current_assignment_parser = parsers_dict.get('assignment_parser')
        current_lab_parser = parsers_dict.get('lab_parser')
        current_recitation_parser = parsers_dict.get('recitation_parser')

        # IMPORTANT: The AttributeError in ScheduleParser (and potentially others if they use DateParser similarly)
        # needs to be fixed in those respective parser files.
        # Example: In schedule_parser.py, change self.date_parser.date_parser_module.parse(...)
        # to self.date_parser.parse(...)
        if current_date_parser: data = current_date_parser.process_dates(data, term_year_str=term_for_date_parsing)
        if current_schedule_parser: data = current_schedule_parser.process_schedule(data, unique_id)
        if current_assignment_parser: data = current_assignment_parser.process_tasks_from_structured_data(data)
        if current_lab_parser: data = current_lab_parser.process_labs_from_structured_data(data)
        if current_recitation_parser: data = current_recitation_parser.process_recitations_from_structured_data(data)

        data["metadata"]["processing_stage"] = "user_data_completed"
        data["metadata"]["last_updated_by_user"] = datetime.now().isoformat()
        
        system_required_fields = config.get("required_fields", [])
        current_class_data = data.get("class_data", {})
        
        # REVISED PROMPTING LOGIC
        fields_to_prompt_next = determine_fields_to_prompt(current_class_data, system_required_fields)
        
        data["metadata"]["missing_fields"] = [f for f in system_required_fields if not current_class_data.get(f, "").strip()] # All actual missing
        data["metadata"]["fields_to_prompt_user"] = fields_to_prompt_next # Fields chosen for next prompt

        write_json(results_path, data)
        transient_output_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json"
        write_json(transient_output_path, data)

        if data["metadata"]["fields_to_prompt_user"]:
            flash(f"Some required fields still missing. Please provide: {', '.join(data['metadata']['fields_to_prompt_user'])}.", "warning")
            logger.info(f"Redirecting to missing_fields (after complete_data) for ID {unique_id}. Fields: {data['metadata']['fields_to_prompt_user']}")
            return redirect(url_for('missing_fields', unique_id=unique_id))
        
        flash("Information updated successfully!", "success")
        logger.info(f"Data completion successful for {unique_id}, redirecting to results.")
        return redirect(url_for('show_results', unique_id=unique_id))
    except KeyError as ke:
        logger.error(f"KeyError in /complete-data/{unique_id}: {ke}", exc_info=True)
        flash(f'A data processing error occurred (KeyError: {ke}). Please try again.', 'error')
        return redirect(url_for('missing_fields', unique_id=unique_id))
    except Exception as e:
        logger.error(f"Error in /complete-data/{unique_id}: {e}", exc_info=True)
        flash('Error processing completed data. Please check logs and try again.', 'error')
        # Potentially redirect to index or a more general error page if missing_fields isn't appropriate
        return redirect(url_for('missing_fields', unique_id=unique_id))


@app.route('/results/<unique_id>')
def show_results(unique_id):
    logger.info(f"Displaying results page for ID: {unique_id}")
    results_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
    if not results_path.exists():
        results_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json" 
        if not results_path.exists():
            flash('Results file not found.', 'error'); return redirect(url_for('index'))
    data = read_json(results_path)
    if not data: flash('Error reading results data.', 'error'); return redirect(url_for('index'))
    return render_template('results.html', class_data=data.get('class_data', {}),
                           event_data=data.get('event_data', []), task_data=data.get('task_data', []),
                           lab_data=data.get('lab_data', []), recitation_data=data.get('recitation_data', []),
                           metadata=data.get('metadata', {}), unique_id=unique_id,
                           json_data=json.dumps(data, indent=2, default=str))


# --- API Routes ---
@app.route('/api/save-results/<unique_id>', methods=['POST'])
def save_results_api(unique_id):
    logger.info(f"API: Saving results for ID: {unique_id}")
    try:
        payload = request.json
        if not payload: return jsonify({"error": "No data provided"}), 400
        
        output_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
        transient_output_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json"
        if HELPERS_AVAILABLE and callable(write_json):
            if write_json(output_path, payload):
                write_json(transient_output_path, payload) 
                return jsonify({"success": True, "message": "Results saved."})
            else: return jsonify({"error": "Failed to write results"}), 500
        else: return jsonify({"error": "JSON writing utility missing"}), 500
    except Exception as e:
        logger.error(f"API Error in /api/save-results/{unique_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/results/<unique_id>')
def get_results_api(unique_id):
    logger.info(f"API: Request for results ID: {unique_id}")
    results_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
    if not results_path.exists(): results_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json"
    if not results_path.exists(): return jsonify({"error": "Results not found"}), 404
    data = read_json(results_path)
    return jsonify(data) if data else (jsonify({"error": "Failed to read results"}), 500)

@app.route('/export/calendar/<unique_id>', methods=['POST'])
def export_to_calendar_route(unique_id):
    logger.info(f"Calendar export requested for ID: {unique_id}")
    export_choices = request.json.get('choices', {})
    syllabus_data_for_export = request.json.get('syllabus_data', {})
    if not syllabus_data_for_export:
        results_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
        if results_path.exists(): syllabus_data_for_export = read_json(results_path)
        else: return jsonify({"success": False, "message": "Syllabus data not found."}), 404
        if not syllabus_data_for_export: return jsonify({"success": False, "message": "Error reading data."}), 500
    
    transient_id_for_filename = syllabus_data_for_export.get("metadata",{}).get("transient_run_id", unique_id)
    calendar_file_name = f"{secure_filename(transient_id_for_filename)}_calendar.ics"
    calendar_file_path = Path(app.config['OUTPUT_FOLDER']) / calendar_file_name
    try:
        calendar_content = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//SyllabusProcessor//EN\nEND:VCALENDAR" 
        calendar_file_path.write_text(calendar_content, encoding='utf-8')
        return jsonify({"success": True, "message": "Calendar file generated (placeholder).",
                        "download_url": url_for('download_calendar_route', filename=calendar_file_name)})
    except Exception as e_cal: 
        logger.error(f"Error generating calendar for {unique_id}: {e_cal}", exc_info=True)
        return jsonify({"success": False, "message": f"Error generating calendar: {str(e_cal)}"}), 500

@app.route('/download/calendar/<filename>')
def download_calendar_route(filename):
    logger.info(f"Calendar download requested for file: {filename}")
    safe_filename = secure_filename(filename)
    if not safe_filename == filename:
        flash('Invalid calendar filename.', 'error'); return redirect(url_for('index'))
    calendar_path = Path(app.config['OUTPUT_FOLDER']) / safe_filename
    if not calendar_path.exists():
        flash('Calendar file not found.', 'error'); return redirect(url_for('index'))
    try:
        return send_file(calendar_path.resolve(), as_attachment=True, download_name=safe_filename, mimetype='text/calendar')
    except Exception as e:
        logger.error(f"Error sending calendar file {safe_filename}: {e}", exc_info=True)
        flash('Error downloading calendar.', 'error'); return redirect(url_for('index'))

@app.route('/api/test-connection')
def test_connection(): return jsonify({"status": "success", "message": "API operational"})

@app.route('/api/pipeline-status')
def pipeline_status(): 
    return jsonify({"status": "success", "message": "Pipeline status nominal", "available_parsers": list(parsers_dict.keys())})


# --- Error Handlers ---
@app.errorhandler(404)
def page_not_found_error_handler(e):
    logger.warning(f"404 Not Found: {request.url} (Error: {e})")
    return render_template('error.html', error_code=404, error_message="Page Not Found"), 404

@app.errorhandler(500)
def internal_server_error_handler(e):
    logger.error(f"500 Internal Server Error: {request.url} (Error: {e})", exc_info=True)
    return render_template('error.html', error_code=500, error_message="Internal Server Error"), 500

# --- Main Execution ---
if __name__ == '__main__':
    if not HELPERS_AVAILABLE:
        logger.critical("Core helper functions could not be loaded. Aborting Flask app start.")
        sys.exit("Exiting due to missing core helpers.")
    
    critical_parsers = ['date_parser', 'openai_parser', 'syllabus_parser', 'extractor', 'schedule_parser', 'assignment_parser', 'lab_parser', 'recitation_parser']
    missing_critical_parsers = [p for p in critical_parsers if p not in parsers_dict or parsers_dict[p] is None]
    if missing_critical_parsers:
        logger.critical(f"One or more critical parsers failed to initialize: {missing_critical_parsers}. Check logs. Aborting Flask app start.")
        sys.exit(f"Exiting due to missing critical parsers: {missing_critical_parsers}")

    flask_host = config.get('flask',{}).get('host','0.0.0.0')
    flask_port = int(config.get('flask',{}).get('port', 5000))
    flask_debug = config.get('flask',{}).get('debug', True)

    logger.info(f"Starting Flask application on {flask_host}:{flask_port} with debug={flask_debug}")
    app.run(debug=flask_debug, host=flask_host, port=flask_port)

