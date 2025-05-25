import os
import sys
import logging
import traceback
from logging.handlers import RotatingFileHandler
from datetime import datetime

class SyllabusLogger:
    """
    Centralized logging utility for the Syllabus Parser Pipeline.
    Provides file and console logging with different levels of detail.
    """
    
    def __init__(self, log_dir='logs', log_level=logging.DEBUG):
        """
        Initialize the logger with file and console handlers.
        
        Args:
            log_dir (str): Directory to store log files
            log_level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = log_dir
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'syllabus_parser_{timestamp}.log')
        
        # Configure logger
        self.logger = logging.getLogger('syllabus_parser')
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create file handler with rotation (10 MB max size, keep 10 backup files)
        file_handler = RotatingFileHandler(
            self.log_file, 
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=10
        )
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Less verbose for console
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        
        # Add formatters to handlers
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized to file: {self.log_file}")
    
    def log_dict(self, level, message, data_dict):
        """
        Log a message with a dictionary (pretty-printed)
        
        Args:
            level (str): Log level ('debug', 'info', 'warning', 'error', 'critical')
            message (str): Log message
            data_dict (dict): Dictionary to log
        """
        import json
        
        log_func = getattr(self.logger, level.lower())
        log_func(f"{message}: {json.dumps(data_dict, indent=2)}")
    
    def log_exception(self, message="An exception occurred"):
        """
        Log exception information with traceback
        
        Args:
            message (str): Additional context message
        """
        exc_info = sys.exc_info()
        if exc_info[0] is not None:  # If there's an actual exception
            exception_message = f"{message}: {exc_info[1]}"
            exception_traceback = ''.join(traceback.format_exception(*exc_info))
            self.logger.error(f"{exception_message}\n{exception_traceback}")
        else:
            self.logger.error(f"{message} (no exception info available)")
    
    def start_operation(self, operation_name, **kwargs):
        """
        Log the start of an operation with parameters
        
        Args:
            operation_name (str): Name of the operation starting
            **kwargs: Operation parameters to log
        """
        if kwargs:
            params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.info(f"STARTED: {operation_name} with params: {params_str}")
        else:
            self.logger.info(f"STARTED: {operation_name}")
    
    def end_operation(self, operation_name, status="completed", **kwargs):
        """
        Log the end of an operation with result information
        
        Args:
            operation_name (str): Name of the operation ending
            status (str): Completion status ('completed', 'failed', etc.)
            **kwargs: Operation result information
        """
        if kwargs:
            results_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.info(f"ENDED: {operation_name} {status} with results: {results_str}")
        else:
            self.logger.info(f"ENDED: {operation_name} {status}")
    
    def log_file_operation(self, operation, filepath, status="success", details=None):
        """
        Log file operations with standard format
        
        Args:
            operation (str): File operation ('read', 'write', 'delete', etc.)
            filepath (str): Path to the file
            status (str): Operation status
            details (str, optional): Additional details
        """
        message = f"FILE {operation.upper()}: {filepath} - {status}"
        if details:
            message += f" - {details}"
        self.logger.info(message)
    
    def log_api_call(self, api_name, endpoint, status_code=None, response_size=None):
        """
        Log API calls with standard format
        
        Args:
            api_name (str): Name of the API (OpenAI, Google, etc.)
            endpoint (str): API endpoint or method called
            status_code (int, optional): HTTP status code
            response_size (int, optional): Size of response in bytes
        """
        message = f"API CALL: {api_name} - {endpoint}"
        if status_code:
            message += f" - Status: {status_code}"
        if response_size:
            message += f" - Size: {response_size} bytes"
        self.logger.info(message)

# Create a singleton instance
_logger_instance = None

def get_logger(log_dir='logs', log_level=logging.DEBUG):
    """
    Get or create the logger instance
    
    Args:
        log_dir (str): Directory to store log files
        log_level (int): Logging level
        
    Returns:
        SyllabusLogger: Logger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SyllabusLogger(log_dir, log_level)
    return _logger_instance