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
# 2. Initialize Logger (Global) - MOVED UP
# --------------------------------------------------
def _app_fallback_setup_logger(log_dir_str="logs", log_name_str="AppFallbackLogger", log_level_val=logging.INFO):
    app_fb_logger = logging.getLogger(log_name_str)
    if not app_fb_logger.handlers:
        logging.basicConfig(level=log_level_val, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return app_fb_logger

logger = _app_fallback_setup_logger(log_name_str="SyllabusWebAppEarly") 

# --------------------------------------------------
# 3. Load Configuration 
# --------------------------------------------------
config_path_env = os.path.join(project_root, "config.yaml")
app_default_config_for_logger = {
    "project_root": project_root,
    "directories": {"logs": "logs"},
    "logging": {"level": "DEBUG", "format": '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s', "datefmt": '%Y-%m-%d %H:%M:%S'}
}
config_for_logger = app_default_config_for_logger 
config = {} # Will be fully populated after helper imports
HELPERS_CONFIG_LOADED = False

try:
    from utils.helpers import load_configuration as helper_load_configuration
    app_full_default_config = {
        "project_root": project_root,
        "directories": {"logs": "logs", "uploads": "uploads", "converted": "converted", "output": "output",
                        "original_syllabus_dir": "utils/parsing/syllabus_repository/original_syllabus",
                        "converted_syllabus_dir": "utils/parsing/syllabus_repository/converted_syllabus",
                        "parsed_syllabus_dir": "utils/parsing/syllabus_repository/parsed_syllabus",
                        "main_data_dir": "utils/parsing/syllabus_repository/main_data"},
        "logging": {"level": "DEBUG", "format": '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s', "datefmt": '%Y-%m-%d %H:%M:%S'},
        "extraction": {"openai_model": "gpt-4.1-nano"},
        "flask": {"secret_key": "default_secret_key_CHANGE_ME_IN_PROD", "host": "0.0.0.0", "port": 5000, "debug": True},
        "file_settings": {"max_content_length": 16 * 1024 * 1024, "allowed_extensions": ["pdf", "txt", "docx", "html", "htm"]},
        "required_fields": ['Time Zone', 'Class Start Date', 'Class End Date', 'Days of Week', 'Class Time', 'Course Code', 'Course Title']
    }
    config = helper_load_configuration(config_path=config_path_env, default_config=app_full_default_config)
    config_for_logger = config 
    HELPERS_CONFIG_LOADED = True
except ImportError:
    logger.warning("utils.helpers.load_configuration not found. Using basic default config for logger.")
    config = app_full_default_config 


try:
    from utils.helpers import setup_logger as helper_setup_logger
    log_level_str_main = config_for_logger.get("logging", {}).get("level", "DEBUG")
    numeric_level_main = getattr(logging, log_level_str_main.upper(), logging.DEBUG)
    logger = helper_setup_logger( 
        log_dir=str(Path(project_root) / config_for_logger.get("directories", {}).get("logs", "logs")),
        log_name="SyllabusWebApp",
        log_level=numeric_level_main,
        config_format=config_for_logger.get("logging", {}).get("format"),
        config_datefmt=config_for_logger.get("logging", {}).get("datefmt")
    )
    logger.info("Main application logger configured successfully using helper_setup_logger.")
except ImportError:
    logger.error("utils.helpers.setup_logger not found. Logger remains in basic fallback state.")
    log_level_str_main = config_for_logger.get("logging", {}).get("level", "DEBUG")
    numeric_level_main = getattr(logging, log_level_str_main.upper(), logging.DEBUG)
    logger = _app_fallback_setup_logger(
        log_dir_str=str(Path(project_root) / config_for_logger.get("directories", {}).get("logs", "logs")),
        log_name_str="SyllabusWebApp", 
        log_level_val=numeric_level_main
    )
    logger.warning("Main application logger using simplified _app_fallback_setup_logger due to missing helper.")


# --------------------------------------------------
# 4. Import Custom Helper Functions & Key Classes (with Fallbacks)
# --------------------------------------------------
HELPERS_AVAILABLE = False 
ALL_POSSIBLE_CLASS_DATA_FIELDS = [] 

def _fallback_read_json(file_path, fallback_data=None, local_logger=logger): 
    local_logger.warning(f"Using fallback read_json for {file_path}")
    return fallback_data or {}
def _fallback_write_json(file_path, data, create_dirs=True, backup_existing=True, indent=4, local_logger=logger): 
    local_logger.warning(f"Using fallback write_json for {file_path}")
    return False

try:
    from utils.helpers import (
        setup_environment, 
        allowed_file,
        convert_webpage_to_txt, save_uploaded_file, process_syllabus_pipeline,
        read_json, write_json 
    )
    try:
        from utils.parsers.openai_parser import OpenAIParser as OpenAIParserClass_ForFields 
        ALL_POSSIBLE_CLASS_DATA_FIELDS = OpenAIParserClass_ForFields.CLASS_DATA_FIELDS
    except ImportError:
        logger.warning("OpenAIParser class not found for field list, using fallback for ALL_POSSIBLE_CLASS_DATA_FIELDS.") 
        ALL_POSSIBLE_CLASS_DATA_FIELDS = [ 
            "School Name", "Term", "Course Title", "Course Code", "Instructor Name",
            "Instructor Email", "Class Time", "Time Zone", "Days of Week",
            "Term Start Date", "Term End Date", "Class Start Date", "Class End Date",
            "Office Hours", "Telephone", "Class Location", "Additional"
        ]
    HELPERS_AVAILABLE = True 
    if not HELPERS_CONFIG_LOADED: 
        logger.warning("Initial config loading used basic defaults because utils.helpers.load_configuration was not available at that stage.")
    logger.info("Core helper functions from utils.helpers imported successfully.")

except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Could not import core modules from utils.helpers: {e}. Application will use fallbacks.", exc_info=True)
    if 'load_configuration' not in locals() or not callable(helper_load_configuration): 
        def load_configuration(config_path=None, default_config=None): return default_config or {} 
    
    def setup_environment(cfg, pr_root): logger.warning("Using fallback setup_environment."); pass 
    def allowed_file(filename, allowed_extensions): logger.warning(f"Using fallback allowed_file for {filename}."); return False
    def convert_webpage_to_txt(url, output_path=None): logger.warning(f"Using fallback convert_webpage_to_txt for {url}."); return None
    def save_uploaded_file(file_obj, upload_folder, unique_id_prefix): logger.warning("Using fallback save_uploaded_file."); return None, None
    def process_syllabus_pipeline(input_path_or_text, cfg, prsrs, original_filename_for_display=None, transient_run_id=None):
        logger.error("Using fallback process_syllabus_pipeline due to import error.")
        return {"metadata": {"error": "Pipeline unavailable (helpers missing)", "unique_id": transient_run_id or "error_id", "missing_fields": [], "fields_to_prompt_user": []}, "class_data": {}}
    read_json = _fallback_read_json 
    write_json = _fallback_write_json
    ALL_POSSIBLE_CLASS_DATA_FIELDS = [ 
        "School Name", "Term", "Course Title", "Course Code", "Instructor Name",
        "Instructor Email", "Class Time", "Time Zone", "Days of Week",
        "Term Start Date", "Term End Date", "Class Start Date", "Class End Date",
        "Office Hours", "Telephone", "Class Location", "Additional"
    ]
    HELPERS_AVAILABLE = False 

if not HELPERS_AVAILABLE: 
    logger.warning("Application is running with FALLBACK helper functions due to import errors after full attempt.")


# --------------------------------------------------
# 5. Setup Environment (Directories)
# --------------------------------------------------
if HELPERS_AVAILABLE and callable(setup_environment): 
    setup_environment(config, project_root)
else: 
    logger.warning("Using fallback directory creation as setup_environment helper is unavailable.")
    for dir_key, dir_path_str in config.get("directories", {}).items(): 
        try: 
            actual_path = Path(project_root) / dir_path_str
            actual_path.mkdir(parents=True, exist_ok=True)
        except Exception as e_dir_fb: logger.error(f"Fallback: Error creating dir {dir_path_str}: {e_dir_fb}", exc_info=True)


# --------------------------------------------------
# 6. Initialize Parsers (Global)
# --------------------------------------------------
parsers_dict: Dict[str, Any] = {}
date_parser, openai_parser_main_instance, syllabus_parser, extractor, schedule_parser, assignment_parser, lab_parser, recitation_parser = None, None, None, None, None, None, None, None
openai_model_from_config = config.get("extraction", {}).get("openai_model", "gpt-4.1-nano") 

try: from utils.parsers.date_parser import DateParser as SyllabusDateParser; date_parser = SyllabusDateParser(logger_instance=logger, config=config.get("date_parser")); parsers_dict['date_parser'] = date_parser
except Exception as e: logger.error(f"Failed to init DateParser: {e}", exc_info=True)

try: from utils.parsers.openai_parser import OpenAIParser; openai_parser_main_instance = OpenAIParser(model=openai_model_from_config, logger_instance=logger, config=config); parsers_dict['openai_parser'] = openai_parser_main_instance
except Exception as e: logger.error(f"Failed to init OpenAIParser (main instance): {e}", exc_info=True)

try:
    from utils.parsers.syllabus_parser import SyllabusParser
    syllabus_parser_base_dir_str = str(Path(project_root) / config.get("directories", {}).get("parsed_syllabus_dir", "utils/parsing/syllabus_repository/parsed_syllabus"))
    syllabus_parser = SyllabusParser(config=config, base_dir=syllabus_parser_base_dir_str, model=openai_model_from_config, logger_instance=logger)
    parsers_dict['syllabus_parser'] = syllabus_parser
except Exception as e: logger.error(f"Failed to init SyllabusParser: {e}", exc_info=True)

try:
    from utils.syllabus_extractor import SyllabusExtractor
    extractor = SyllabusExtractor(config=config, openai_model=openai_model_from_config, logger_instance=logger, openai_parser_instance=openai_parser_main_instance)
    parsers_dict['extractor'] = extractor
except Exception as e: logger.error(f"Failed to init SyllabusExtractor: {e}", exc_info=True)

try:
    from utils.parsers.assignment_parser import AssignmentParser
    if date_parser and openai_parser_main_instance:
        assignment_parser = AssignmentParser(logger_instance=logger, date_parser_instance=date_parser, openai_parser_instance=openai_parser_main_instance, config=config)
        parsers_dict['assignment_parser'] = assignment_parser
    else: logger.warning("AssignmentParser not initialized due to missing dependencies (DateParser or main OpenAIParser).")
except Exception as e: logger.error(f"Failed to init AssignmentParser: {e}", exc_info=True)

try:
    from utils.parsers.schedule_parser import ScheduleParser
    if date_parser and openai_parser_main_instance: 
        schedule_parser = ScheduleParser(logger_instance=logger, date_parser_instance=date_parser, openai_parser_instance=openai_parser_main_instance, config=config)
        parsers_dict['schedule_parser'] = schedule_parser
    else: logger.warning("ScheduleParser not initialized due to missing dependencies.")
except Exception as e: logger.error(f"Failed to init ScheduleParser: {e}", exc_info=True)

try:
    from utils.parsers.lab_parser import LabParser
    if date_parser and openai_parser_main_instance: 
        lab_parser = LabParser(logger_instance=logger, date_parser_instance=date_parser, openai_parser_instance=openai_parser_main_instance, config=config)
        parsers_dict['lab_parser'] = lab_parser
    else: logger.warning("LabParser not initialized due to missing dependencies.")
except Exception as e: logger.error(f"Failed to init LabParser: {e}", exc_info=True)

try:
    from utils.parsers.recitation_parser import RecitationParser
    if date_parser and openai_parser_main_instance: 
        recitation_parser = RecitationParser(logger_instance=logger, date_parser_instance=date_parser, openai_parser_instance=openai_parser_main_instance, config=config)
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
    'PARSED_SYLLABUS_DIR': str(Path(project_root) / config.get("directories", {}).get('parsed_syllabus_dir', 'utils/parsing/syllabus_repository/parsed_syllabus')),
    'MAX_CONTENT_LENGTH': config.get('file_settings', {}).get('max_content_length', 16 * 1024 * 1024),
    'SECRET_KEY': config.get('flask', {}).get('secret_key', 'default_secret_key_app_py_CHANGE_ME_TOO')
})
for folder_key in ['UPLOAD_FOLDER', 'CONVERTED_FOLDER', 'OUTPUT_FOLDER', 'PARSED_SYLLABUS_DIR']:
    try: Path(app.config[folder_key]).mkdir(parents=True, exist_ok=True)
    except Exception as e_dir_create_app_cfg: logger.error(f"Error creating directory for Flask app.config key {folder_key}: {app.config[folder_key]}, Error: {e_dir_create_app_cfg}")

# --------------------------------------------------
# 8. Global Constants for Templates (Global)
# --------------------------------------------------
COMMON_TIME_ZONES = [ 
    {"value": "America/New_York", "label": "Eastern Time (ET)"},
    {"value": "America/Chicago", "label": "Central Time (CT)"},
    {"value": "America/Denver", "label": "Mountain Time (MT)"},
    {"value": "America/Los_Angeles", "label": "Pacific Time (PT)"},
    {"value": "America/Phoenix", "label": "Arizona (no DST)"},
    {"value": "America/Anchorage", "label": "Alaska Time (AKT)"},
    {"value": "America/Honolulu", "label": "Hawaii Time (HT)"},
    {"value": "UTC", "label": "Coordinated Universal Time (UTC/GMT)"},
    {"value": "Europe/London", "label": "London (GMT/BST)"},
    {"value": "Europe/Berlin", "label": "Central European Time (CET/CEST)"},
    {"value": "Asia/Tokyo", "label": "Japan Standard Time (JST)"},
    {"value": "Australia/Sydney", "label": "Australian Eastern Time (AET)"},
]

# --------------------------------------------------
# Route Definitions
# --------------------------------------------------
@app.context_processor
def inject_global_vars():
    return {'current_year': datetime.now().year, 'app_config': config}

@app.route('/')
def index():
    logger.debug(f"Request to / (index) from {request.remote_addr}")
    return render_template('index.html')

def determine_fields_to_prompt(class_data: Dict[str, Any], system_required_fields: List[str]) -> List[str]:
    """
    Helper function to determine which fields to prompt the user for,
    based on the system_required_fields list from config.
    """
    fields_to_prompt = []
    logger.debug(f"Determining fields to prompt. System required: {system_required_fields}. Class data received: {list(class_data.keys())}")
    for field in system_required_fields:
        if not class_data.get(field, "").strip():
            if field not in fields_to_prompt: 
                fields_to_prompt.append(field)
                logger.debug(f"Field '{field}' is missing or empty, adding to prompt list.")
    unique_fields_to_prompt = list(set(fields_to_prompt)) 
    logger.info(f"Fields determined for prompting: {unique_fields_to_prompt}")
    return unique_fields_to_prompt


@app.route('/upload', methods=['POST'])
def upload_file_route():
    logger.info(f"Processing /upload request from {request.remote_addr}")
    if not HELPERS_AVAILABLE: 
        flash('Core application components (helpers) are missing. Cannot process upload.', 'error')
        logger.error("Upload attempt failed: HELPERS_AVAILABLE is False.")
        return redirect(url_for('index'))
    if not parsers_dict: 
        flash('Parsing engine not fully initialized. Some parsers might be missing.', 'error')
        logger.error("Upload attempt failed: parsers_dict is empty or not fully populated.")
        return redirect(url_for('index'))

    transient_run_id = str(uuid.uuid4()) 
    input_for_pipeline: Union[str, Path, None] = None
    original_filename_for_display: Optional[str] = None
    
    try:
        url_input = request.form.get('url', '').strip()
        if url_input:
            logger.info(f"Processing URL: {url_input} (transient_id: {transient_run_id})")
            raw_text_from_url = convert_webpage_to_txt(url_input) 
            if raw_text_from_url is None or not raw_text_from_url.strip(): 
                flash(f"Failed to extract text from URL: {url_input}. The URL might be inaccessible or contain no parsable content.", "error")
                logger.warning(f"Failed to extract text from URL {url_input} or text was empty for transient_id {transient_run_id}.")
                return redirect(url_for('index'))
            input_for_pipeline = raw_text_from_url
            original_filename_for_display = f"web_content_{transient_run_id[:8]}.txt" 
            logger.info(f"Successfully extracted text from URL for transient_id {transient_run_id}. Length: {len(raw_text_from_url)}")
        else:
            if 'file' not in request.files or not request.files['file'].filename:
                flash('No file selected for upload. Please choose a file.', 'error')
                logger.warning(f"No file selected in upload request for transient_id {transient_run_id}.")
                return redirect(url_for('index'))
            
            file_obj = request.files['file']
            allowed_exts = config.get('file_settings',{}).get('allowed_extensions', [])
            if not allowed_file(file_obj.filename, allowed_exts): 
                flash(f'File type not allowed. Allowed types are: {", ".join(allowed_exts)}.', 'error')
                logger.warning(f"File type not allowed for '{file_obj.filename}' (transient_id: {transient_run_id}).")
                return redirect(url_for('index'))
            
            saved_path, secure_name = save_uploaded_file(file_obj, Path(app.config['UPLOAD_FOLDER']), transient_run_id)
            if not saved_path:
                flash('Error saving the uploaded file. Please try again.', 'error')
                logger.error(f"Error saving uploaded file '{file_obj.filename}' for transient_id {transient_run_id}.")
                return redirect(url_for('index'))
            input_for_pipeline = saved_path 
            original_filename_for_display = secure_name
            logger.info(f"File '{secure_name}' uploaded to {saved_path} (transient_id: {transient_run_id})")
        
        if input_for_pipeline:
            logger.info(f"Starting pipeline for input (transient_id: {transient_run_id}). Original filename: {original_filename_for_display}")
            results = process_syllabus_pipeline(
                input_path_or_text=input_for_pipeline,
                config=config, 
                parsers=parsers_dict,
                original_filename_for_display=original_filename_for_display,
                transient_run_id=transient_run_id
            )
            
            content_hash_id = results.get("metadata", {}).get("unique_id")
            if not content_hash_id:
                logger.error(f"Pipeline did not return a unique_id (content_hash) for transient_id {transient_run_id}. Using transient_id as fallback.")
                content_hash_id = transient_run_id 
                results.setdefault("metadata", {})["unique_id"] = content_hash_id

            system_required_fields = config.get("required_fields", []) 
            class_data_from_pipeline = results.get("class_data", {})
            
            fields_to_prompt_user = determine_fields_to_prompt(class_data_from_pipeline, system_required_fields)
            results["metadata"]["fields_to_prompt_user"] = fields_to_prompt_user 
            
            output_file_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{content_hash_id}_results.json"
            if HELPERS_AVAILABLE and callable(write_json):
                if not write_json(output_file_path, results): 
                    logger.error(f"Failed to save updated results (with prompt fields) to {output_file_path} for ID {content_hash_id}.")
                    flash('Error saving intermediate processing results. Please try again.', 'error')
                    return redirect(url_for('index'))
                logger.info(f"Successfully saved/updated results to {output_file_path} with fields_to_prompt_user: {fields_to_prompt_user} for ID {content_hash_id}.")
            else:
                logger.error(f"write_json helper not available. Cannot save updated results to {output_file_path}.")
                flash('Critical error: JSON writing utility missing. Cannot proceed.', 'error')
                return redirect(url_for('index'))

            if results["metadata"]["fields_to_prompt_user"]:
                logger.info(f"Redirecting to missing_fields page for ID {content_hash_id}. Fields to prompt: {results['metadata']['fields_to_prompt_user']}")
                return redirect(url_for('missing_fields', unique_id=content_hash_id))
            
            logger.info(f"No fields to prompt. Redirecting directly to results page for ID {content_hash_id}.")
            return redirect(url_for('show_results', unique_id=content_hash_id))
        else:
            flash('Could not obtain valid input from the uploaded file or URL.', 'error')
            logger.warning(f"No valid input_for_pipeline for transient_id {transient_run_id}.")
            return redirect(url_for('index'))

    except Exception as e:
        logger.error(f"Unhandled exception in /upload (transient_run_id: {transient_run_id}): {e}", exc_info=True)
        flash('An unexpected error occurred during file processing. Please check server logs for details.', 'error')
        return redirect(url_for('index'))


@app.route('/missing-fields/<unique_id>')
def missing_fields(unique_id):
    logger.info(f"Displaying missing_fields page for ID: {unique_id}")
    if not HELPERS_AVAILABLE or not callable(read_json): 
        flash("Critical error: JSON reading utility missing.", "error")
        logger.error("JSON reading utility (read_json) from helpers not available for missing_fields.")
        return redirect(url_for('index'))
        
    results_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
    if not results_path.exists():
        results_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json" 
        if not results_path.exists():
            flash('Results file not found. Please re-process the syllabus.', 'error')
            logger.warning(f"Results file not found for missing_fields page. ID: {unique_id}")
            return redirect(url_for('index'))
    
    data = read_json(results_path) 
    if not data: 
        flash('Error reading results data. The file might be empty or corrupted.', 'error')
        logger.error(f"Error reading results data from {results_path} for missing_fields page. ID: {unique_id}")
        return redirect(url_for('index'))
    
    fields_to_prompt = data.get('metadata', {}).get('fields_to_prompt_user', [])
    if not fields_to_prompt and data.get('class_data'): 
        logger.info(f"No fields to prompt for {unique_id} on missing_fields page, redirecting to results.")
        return redirect(url_for('show_results', unique_id=unique_id))
    
    current_class_data = data.get('class_data', {})
            
    return render_template('missing_fields.html', 
                           unique_id=unique_id,
                           missing_fields=fields_to_prompt, 
                           class_data=current_class_data,
                           time_zones=COMMON_TIME_ZONES, 
                           all_possible_fields=ALL_POSSIBLE_CLASS_DATA_FIELDS)

@app.route('/complete-data/<unique_id>', methods=['POST'])
def complete_data(unique_id):
    logger.info(f"Processing /complete-data for ID: {unique_id}")
    if not HELPERS_AVAILABLE or not callable(read_json) or not callable(write_json):
        flash("Critical error: JSON utilities missing for data completion.", "error")
        logger.error("JSON utilities (read_json or write_json) from helpers not available for complete_data.")
        return redirect(url_for('index'))

    try:
        results_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
        if not results_path.exists():
            results_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json"
            if not results_path.exists():
                flash('Original results file not found for update. Please re-process.', 'error')
                logger.warning(f"Original results file not found for /complete-data. ID: {unique_id}")
                return redirect(url_for('index'))
        
        data = read_json(results_path) 
        if not data: 
            flash('Error reading existing results for update. File might be corrupted.', 'error')
            logger.error(f"Error reading existing results from {results_path} for /complete-data. ID: {unique_id}")
            return redirect(url_for('index'))

        # TASK_TRACE: Log initial task_data
        logger.debug(f"TASK_TRACE (complete_data): Initial task_data loaded from JSON for ID {unique_id}: {json.dumps(data.get('task_data', 'KEY_NOT_FOUND_OR_NONE'))}")

        data.setdefault("class_data", {})
        data.setdefault("metadata", {})
        
        prompted_fields_this_round = data.get("metadata", {}).get("fields_to_prompt_user", [])
        logger.debug(f"Fields prompted in this round for {unique_id}: {prompted_fields_this_round}")
        
        updated_any_field = False
        for key in request.form:
            if key in ALL_POSSIBLE_CLASS_DATA_FIELDS or key in prompted_fields_this_round:
                new_value = request.form[key].strip()
                if data["class_data"].get(key, "") != new_value: 
                    logger.info(f"Updating field '{key}' for {unique_id} from '{data['class_data'].get(key, '')}' to '{new_value}'.")
                    data["class_data"][key] = new_value
                    updated_any_field = True
        
        should_reprocess_parsers = updated_any_field or bool(prompted_fields_this_round)

        if should_reprocess_parsers:
            logger.info(f"User input processed for {unique_id}. Re-processing relevant parser stages. Updated any field: {updated_any_field}")
            term_for_date_parsing = data.get("class_data", {}).get("Term") 

            current_date_parser = parsers_dict.get('date_parser')
            current_schedule_parser = parsers_dict.get('schedule_parser')
            current_assignment_parser = parsers_dict.get('assignment_parser')
            current_lab_parser = parsers_dict.get('lab_parser')
            current_recitation_parser = parsers_dict.get('recitation_parser')

            if current_date_parser:
                try: 
                    data = current_date_parser.process_dates(data, term_year_str=term_for_date_parsing)
                    logger.debug(f"TASK_TRACE (complete_data): Task data AFTER DateParser for ID {unique_id}: {json.dumps(data.get('task_data', 'KEY_NOT_FOUND_OR_NONE'))}")
                except Exception as e_dp: 
                    logger.error(f"Error re-running DateParser for {unique_id}: {e_dp}", exc_info=True)
                    data.setdefault("metadata",{}).setdefault("error_reprocessing_dateparser", str(e_dp))
            
            if current_schedule_parser:
                try: 
                    data = current_schedule_parser.process_schedule(data, unique_id) 
                    logger.debug(f"TASK_TRACE (complete_data): Task data AFTER ScheduleParser for ID {unique_id}: {json.dumps(data.get('task_data', 'KEY_NOT_FOUND_OR_NONE'))}")
                except Exception as e_sp: 
                    logger.error(f"Error re-running ScheduleParser for {unique_id}: {e_sp}", exc_info=True)
                    data.setdefault("metadata",{}).setdefault("error_reprocessing_scheduleparser", str(e_sp))

            # Ensure task_data is preserved if it exists before calling AssignmentParser
            # This is a safeguard in case previous parsers unintentionally removed it.
            # However, the root cause of its removal should be fixed in those parsers.
            if 'task_data' not in data and results_path.exists(): # If missing, try to re-read just task_data from original file
                original_data_for_task_recovery = read_json(results_path)
                if original_data_for_task_recovery and 'task_data' in original_data_for_task_recovery:
                    data['task_data'] = original_data_for_task_recovery['task_data']
                    logger.warning(f"TASK_TRACE (complete_data): task_data was missing, recovered from original JSON for ID {unique_id}.")


            if current_assignment_parser:
                logger.debug(f"TASK_TRACE (complete_data): Task data BEFORE AssignmentParser.process_tasks_from_structured_data for ID {unique_id}: {json.dumps(data.get('task_data', 'KEY_NOT_FOUND_OR_NONE'))}")
                try: 
                    data = current_assignment_parser.process_tasks_from_structured_data(data)
                    logger.debug(f"TASK_TRACE (complete_data): Task data AFTER AssignmentParser.process_tasks_from_structured_data for ID {unique_id}: {json.dumps(data.get('task_data', 'KEY_NOT_FOUND_OR_NONE'))}")
                except Exception as e_ap: 
                    logger.error(f"Error re-running AssignmentParser for {unique_id}: {e_ap}", exc_info=True)
                    data.setdefault("metadata",{}).setdefault("error_reprocessing_assignmentparser", str(e_ap))
            
            if current_lab_parser:
                try: 
                    data = current_lab_parser.process_labs_from_structured_data(data)
                    logger.debug(f"TASK_TRACE (complete_data): Task data AFTER LabParser for ID {unique_id}: {json.dumps(data.get('task_data', 'KEY_NOT_FOUND_OR_NONE'))}") # Though LabParser shouldn't affect task_data
                except Exception as e_lp: 
                    logger.error(f"Error re-running LabParser for {unique_id}: {e_lp}", exc_info=True)
                    data.setdefault("metadata",{}).setdefault("error_reprocessing_labparser", str(e_lp))

            if current_recitation_parser:
                try: 
                    data = current_recitation_parser.process_recitations_from_structured_data(data)
                    logger.debug(f"TASK_TRACE (complete_data): Task data AFTER RecitationParser for ID {unique_id}: {json.dumps(data.get('task_data', 'KEY_NOT_FOUND_OR_NONE'))}") # RecitationParser shouldn't affect task_data
                except Exception as e_rp: 
                    logger.error(f"Error re-running RecitationParser for {unique_id}: {e_rp}", exc_info=True)
                    data.setdefault("metadata",{}).setdefault("error_reprocessing_recitationparser", str(e_rp))
        else:
            logger.info(f"No fields were updated by the user for {unique_id} and no prior prompted fields. Skipping re-processing of parser stages in /complete-data.")

        data["metadata"]["processing_stage"] = "user_data_completed"
        data["metadata"]["last_updated_by_user"] = datetime.now().isoformat()
        
        system_required_fields = config.get("required_fields", [])
        current_class_data_after_update = data.get("class_data", {})
        
        fields_to_prompt_next = determine_fields_to_prompt(current_class_data_after_update, system_required_fields)
        
        data["metadata"]["missing_fields_after_user_input"] = [f for f in system_required_fields if not current_class_data_after_update.get(f, "").strip()]
        data["metadata"]["fields_to_prompt_user"] = fields_to_prompt_next 

        if callable(write_json): 
            if not write_json(results_path, data):
                 logger.error(f"Failed to save updated data to {results_path} after /complete-data for {unique_id}.")
            transient_output_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json"
            if not write_json(transient_output_path, data): 
                logger.warning(f"Failed to save updated data to transient output {transient_output_path} for {unique_id}.")
        else:
            logger.error("write_json function is not available. Cannot save data in /complete-data.")


        if data["metadata"].get("fields_to_prompt_user"): 
            reprocessing_errors = [k for k in data.get("metadata", {}) if k.startswith("error_reprocessing_")]
            if reprocessing_errors:
                flash(f"Errors occurred during data re-processing: {', '.join(reprocessing_errors)}. Some information might be incomplete or incorrect.", "error")
                logger.error(f"Errors during /complete-data re-processing for {unique_id}: {reprocessing_errors}")
            else:
                flash(f"Some required information is still missing. Please provide: {', '.join(data['metadata']['fields_to_prompt_user'])}.", "warning")
            
            logger.info(f"Redirecting to missing_fields (after /complete-data) for ID {unique_id}. Fields: {data['metadata']['fields_to_prompt_user']}")
            return redirect(url_for('missing_fields', unique_id=unique_id))
        
        reprocessing_errors = [k for k in data.get("metadata", {}) if k.startswith("error_reprocessing_")]
        if reprocessing_errors:
            flash(f"Data updated, but errors occurred during data re-processing: {', '.join(reprocessing_errors)}. Some information might be incomplete or incorrect. Please review the results carefully.", "warning")
            logger.error(f"Errors during /complete-data re-processing for {unique_id} (no further prompts): {reprocessing_errors}")
        else:
            flash("Information updated successfully!", "success")
            
        logger.info(f"Data completion successful for {unique_id}, all required fields provided or no critical reprocessing errors. Redirecting to results.")
        return redirect(url_for('show_results', unique_id=unique_id))

    except KeyError as ke:
        logger.error(f"KeyError in /complete-data/{unique_id}: {ke}", exc_info=True)
        flash(f'A data processing error occurred (KeyError: {ke}). Please check the input and try again.', 'error')
        return redirect(url_for('missing_fields', unique_id=unique_id))
    except Exception as e:
        logger.error(f"Error in /complete-data/{unique_id}: {e}", exc_info=True)
        flash('An error occurred while processing your completed data. Please check logs and try again.', 'error')
        return redirect(url_for('missing_fields', unique_id=unique_id))

# ... (rest of app.py, including show_results and other routes, error handlers, and __main__ block) ...

@app.route('/results/<unique_id>')
def show_results(unique_id):
    logger.info(f"Displaying results page for ID: {unique_id}")
    if not HELPERS_AVAILABLE or not callable(read_json):
        flash("Critical error: JSON reading utility missing.", "error")
        logger.error("JSON reading utility (read_json) from helpers not available for results page.")
        return redirect(url_for('index'))

    results_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
    if not results_path.exists():
        results_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json" 
        if not results_path.exists():
            flash('Results file not found. It might have been cleared or an error occurred.', 'error')
            logger.warning(f"Results file not found for /results page. ID: {unique_id}")
            return redirect(url_for('index'))
            
    data = read_json(results_path) 
    if not data: 
        flash('Error reading results data. The file may be corrupted or empty.', 'error')
        logger.error(f"Error reading results data from {results_path} for /results page. ID: {unique_id}")
        return redirect(url_for('index'))
        
    return render_template('results.html', 
                           class_data=data.get('class_data', {}),
                           event_data=data.get('event_data', []), 
                           task_data=data.get('task_data', []),
                           lab_data=data.get('lab_data', []), 
                           recitation_data=data.get('recitation_data', []),
                           metadata=data.get('metadata', {}), 
                           unique_id=unique_id,
                           json_data=json.dumps(data, indent=2, default=str)) 

# --- API Routes ---
@app.route('/api/save-results/<unique_id>', methods=['POST'])
def save_results_api(unique_id):
    logger.info(f"API: Saving results for ID: {unique_id}")
    if not HELPERS_AVAILABLE or not callable(write_json):
        return jsonify({"error": "JSON writing utility missing"}), 500
    try:
        payload = request.json
        if not payload: return jsonify({"error": "No data provided"}), 400
        
        output_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
        transient_output_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json" 
        
        if callable(write_json):
            if write_json(output_path, payload) and write_json(transient_output_path, payload):
                return jsonify({"success": True, "message": "Results saved to primary and transient locations."})
            else: 
                logger.error(f"API Error: Failed to write results for {unique_id} to one or both locations.")
                return jsonify({"error": "Failed to write results"}), 500
        else:
            logger.error("write_json function not available for API save.")
            return jsonify({"error": "JSON writing utility missing on server"}), 500
    except Exception as e:
        logger.error(f"API Error in /api/save-results/{unique_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/results/<unique_id>')
def get_results_api(unique_id):
    logger.info(f"API: Request for results ID: {unique_id}")
    if not HELPERS_AVAILABLE or not callable(read_json):
        return jsonify({"error": "JSON reading utility missing"}), 500

    results_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
    if not results_path.exists(): 
        results_path = Path(app.config['OUTPUT_FOLDER']) / f"{unique_id}_results.json" 
    
    if not results_path.exists(): 
        return jsonify({"error": "Results not found"}), 404
        
    data = read_json(results_path) 
    return jsonify(data) if data else (jsonify({"error": "Failed to read results data"}), 500)

@app.route('/export/calendar/<unique_id>', methods=['POST'])
def export_to_calendar_route(unique_id):
    logger.info(f"Calendar export requested for ID: {unique_id}")
    if not HELPERS_AVAILABLE or not callable(read_json) or not callable(write_json):
        return jsonify({"success": False, "message": "Core utilities missing for calendar export."}), 500

    export_choices = request.json.get('choices', {})
    syllabus_data_for_export = request.json.get('syllabus_data', {})
    
    if not syllabus_data_for_export: 
        results_path = Path(app.config['PARSED_SYLLABUS_DIR']) / f"{unique_id}_results.json"
        if results_path.exists(): 
            syllabus_data_for_export = read_json(results_path) 
        else: 
            return jsonify({"success": False, "message": "Syllabus data not found for export."}), 404
        if not syllabus_data_for_export: 
            return jsonify({"success": False, "message": "Error reading syllabus data for export."}), 500
    
    transient_id_part = syllabus_data_for_export.get("metadata",{}).get("transient_run_id", unique_id)
    calendar_file_name = f"{secure_filename(transient_id_part)}_calendar.ics"
    calendar_file_path = Path(app.config['OUTPUT_FOLDER']) / calendar_file_name 
    
    try:
        calendar_content = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//SyllabusProcessor//EN\nEND:VCALENDAR" 
        if callable(write_json): # Check if write_json is available (it should be if HELPERS_AVAILABLE)
            calendar_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
            calendar_file_path.write_text(calendar_content, encoding='utf-8') # Simple text write for .ics
        else:
            raise IOError("write_json utility not available to save calendar file.")

        return jsonify({
            "success": True, 
            "message": "Calendar file generated (placeholder). Ready for download.",
            "download_url": url_for('download_calendar_route', filename=calendar_file_name)
        })
    except Exception as e_cal: 
        logger.error(f"Error generating placeholder calendar for {unique_id}: {e_cal}", exc_info=True)
        return jsonify({"success": False, "message": f"Error generating calendar file: {str(e_cal)}"}), 500

@app.route('/download/calendar/<filename>')
def download_calendar_route(filename):
    logger.info(f"Calendar download requested for file: {filename}")
    safe_filename = secure_filename(filename) 
    if not safe_filename == filename: 
        flash('Invalid calendar filename provided.', 'error')
        logger.warning(f"Invalid calendar filename detected for download: original='{filename}', safe='{safe_filename}'")
        return redirect(url_for('index'))
        
    calendar_path = Path(app.config['OUTPUT_FOLDER']) / safe_filename 
    if not calendar_path.exists() or not calendar_path.is_file():
        flash('Calendar file not found or is invalid. It might have expired or an error occurred.', 'error')
        logger.warning(f"Calendar file not found for download: {calendar_path}")
        return redirect(url_for('index'))
    try:
        return send_file(str(calendar_path.resolve()), as_attachment=True, download_name=safe_filename, mimetype='text/calendar')
    except Exception as e_send_file:
        logger.error(f"Error sending calendar file {safe_filename}: {e_send_file}", exc_info=True)
        flash('Error downloading the calendar file.', 'error')
        return redirect(url_for('index'))

@app.route('/api/test-connection')
def test_connection(): 
    logger.debug("API test connection endpoint hit.")
    return jsonify({"status": "success", "message": "API operational", "timestamp": datetime.now().isoformat()})

@app.route('/api/pipeline-status')
def pipeline_status(): 
    logger.debug("API pipeline status endpoint hit.")
    return jsonify({
        "status": "success", 
        "message": "Pipeline status nominal (simulated)", 
        "available_parsers": list(parsers_dict.keys()),
        "helpers_available": HELPERS_AVAILABLE
    })

# --- Error Handlers ---
@app.errorhandler(404)
def page_not_found_error_handler(e):
    logger.warning(f"404 Not Found: {request.url} (Error: {e}) from {request.remote_addr}")
    return render_template('error.html', error_code=404, error_message="Page Not Found. The requested URL was not found on this server."), 404

@app.errorhandler(500)
def internal_server_error_handler(e):
    logger.error(f"500 Internal Server Error: {request.url} (Error: {e}) from {request.remote_addr}", exc_info=True)
    return render_template('error.html', error_code=500, error_message="An internal server error occurred. Please try again later."), 500

@app.errorhandler(400) 
def bad_request_error_handler(e):
    logger.warning(f"400 Bad Request: {request.url} (Error: {e}) from {request.remote_addr}", exc_info=True)
    description = getattr(e, 'description', "The browser (or proxy) sent a request that this server could not understand.")
    return render_template('error.html', error_code=400, error_message=f"Bad Request. {description}"), 400

@app.errorhandler(405) 
def method_not_allowed_error_handler(e):
    logger.warning(f"405 Method Not Allowed: {request.method} for {request.url} (Error: {e}) from {request.remote_addr}")
    return render_template('error.html', error_code=405, error_message="Method Not Allowed for the requested URL."), 405

# --- Main Execution ---
if __name__ == '__main__':
    if 'logger' not in globals() or not isinstance(logger, logging.Logger) or not logger.handlers:
        print("Re-initializing a basic logger for __main__ block as global 'logger' is not properly set.", file=sys.stderr)
        logger = _app_fallback_setup_logger(log_name_str="SyllabusWebAppMain")
        logger.warning("Logger was re-initialized in __main__ block using basic fallback.")

    if not HELPERS_AVAILABLE: 
        logger.critical("Core helper functions (utils.helpers) could not be loaded. Aborting Flask app start.")
        sys.exit("Exiting due to missing core helpers. Check imports and __init__.py files.")
    
    flask_host = config.get('flask',{}).get('host','0.0.0.0')
    flask_port = int(config.get('flask',{}).get('port', 5000))
    flask_debug = config.get('flask',{}).get('debug', True)

    logger.info(f"Attempting to start Flask application on {flask_host}:{flask_port} with debug={flask_debug}")
    try:
        app.run(debug=flask_debug, host=flask_host, port=flask_port)
    except Exception as e_flask_run:
        logger.critical(f"Failed to run Flask application: {e_flask_run}", exc_info=True)
        sys.exit(f"Exiting due to Flask app run failure: {e_flask_run}")
