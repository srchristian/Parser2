"""
date_parser.py

Handles date normalization, days of week standardization, and class schedule calculation
for syllabus extraction. It aims for robust parsing and consistent date formatting.
No date inference or guessing is performed beyond standard date parsing rules.
"""

import re
import logging
import sys
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple

try:
    from dateutil import parser as dateutil_parser
    from dateutil.parser import ParserError as DateutilParserError # Explicit import
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    # This print is okay for module-level feedback during development/setup
    print("Warning (date_parser.py): python-dateutil library not found. Using simplified internal date parsing.")

    class DateutilParserError(ValueError): # Fallback error
        """Custom error to mimic dateutil.parser.ParserError for fallback."""
        pass

    class SimpleDateParser: # Fallback parser
        @staticmethod
        def _preprocess_date_string(date_string: str) -> str:
            if not date_string: return ""
            # Remove ordinal suffixes (st, nd, rd, th)
            return re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", date_string, flags=re.IGNORECASE).strip()

        @staticmethod
        def parse(date_string: str, default: Optional[datetime] = None, ignoretz: bool = False, **kwargs) -> datetime:
            if not date_string or not isinstance(date_string, str):
                raise ValueError("A non-empty string must be provided to SimpleDateParser.parse")
            
            processed_date_string = SimpleDateParser._preprocess_date_string(date_string)
            if not processed_date_string:
                 raise ValueError(f"Date string '{date_string}' became empty after preprocessing.")

            year_to_use = default.year if default and isinstance(default, (datetime, date)) else None
            
            # Expanded formats, especially for month-day without year
            formats_to_try = [
                "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", 
                "%d-%B-%Y", "%d-%b-%Y", "%d %B %Y", "%d %b %Y",
                "%m/%d/%y", "%Y/%m/%d", "%d.%m.%Y", "%Y.%m.%d",
                # Formats that might lack a year explicitly (strptime defaults to 1900)
                "%B %d", "%b %d", "%m/%d", "%m-%d", 
            ]
            for fmt in formats_to_try:
                try:
                    dt_obj = datetime.strptime(processed_date_string, fmt)
                    if dt_obj.year == 1900 and year_to_use:
                        dt_obj = dt_obj.replace(year=year_to_use)
                    # If ignoretz is True, ensure timezone is naive (consistent with dateutil behavior)
                    # datetime.strptime usually creates naive datetimes unless %z is used.
                    return dt_obj
                except ValueError:
                    continue
            # If all formats fail, raise an error similar to DateutilParserError
            raise DateutilParserError(f"SimpleDateParser: Date string '{date_string}' could not be parsed with any known format.")
    
    class DateutilParserFallback: # Wrapper for SimpleDateParser to mimic dateutil_parser module structure
        def parse(self, date_string: str, default: Optional[datetime] = None, ignoretz: bool = False, **kwargs) -> datetime:
            return SimpleDateParser.parse(date_string, default=default, ignoretz=ignoretz, **kwargs)

    # Assign the fallback parser module if dateutil is not available
    date_parser_module = DateutilParserFallback()

if DATEUTIL_AVAILABLE: # If real dateutil is available, use it
    date_parser_module = dateutil_parser 


class DateParser:
    def __init__(self, logger_instance: logging.Logger, config: Optional[Dict[str, Any]] = None): 
        self.logger = logger_instance
        if not self.logger.hasHandlers(): 
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s')
            self.logger = logging.getLogger(__name__) 
            self.logger.warning("DateParser initialized with a basic configured logger as provided logger_instance had no handlers.")
        
        self.config = config or {}
        self.output_date_format = self.config.get("output_date_format", "%B %d, %Y")
        
        self.weekday_map = { 
            "monday": 0, "mon": 0, "mo": 0,
            "tuesday": 1, "tue": 1, "tu": 1, "tues": 1,
            "wednesday": 2, "wed": 2, "we": 2, "weds": 2,
            "thursday": 3, "thu": 3, "th": 3, "r": 3, "thur": 3, "thurs": 3,
            "friday": 4, "fri": 4, "fr": 4,
            "saturday": 5, "sat": 5, "sa": 5,
            "sunday": 6, "sun": 6, "su": 6, "u": 6 
        }
        self.day_regex_map_ordered = [
            (r"monday", 0), (r"tuesday", 1), (r"wednesday", 2), (r"thursday", 3), (r"friday", 4), (r"saturday", 5), (r"sunday", 6),
            (r"mon", 0), (r"tues", 1), (r"tue", 1), (r"weds", 2), (r"wed", 2), (r"thurs", 3), (r"thur", 3), (r"thu", 3), (r"fri", 4), (r"sat", 5), (r"sun", 6),
            # Ensure single characters are distinct enough or use word boundaries if necessary in complex strings
            (r"mo", 0), (r"tu", 1), (r"we", 2), (r"th", 3), (r"fr", 4), (r"sa", 5), (r"su", 6), 
            (r"u", 6), # 'u' for Sunday
            (r"r", 3)  # 'r' for Thursday (common in academic schedules)
        ]
        self.day_combinations = { 
            "mwf": [0, 2, 4], "tr": [1, 3], "tth": [1, 3], "tuth": [1, 3], "thf": [3,4], 
            "mw": [0, 2], "mf": [0, 4], "wf": [2, 4],
            "mtwtf": [0, 1, 2, 3, 4], "mtwrf": [0, 1, 2, 3, 4], 
        }
        if DATEUTIL_AVAILABLE: self.logger.info("DateParser initialized using 'python-dateutil'.")
        else: self.logger.warning("DateParser initialized using simplified internal date parser (python-dateutil not found).")
    
    def parse(self, date_string: str, default: Optional[datetime] = None, ignoretz: bool = False, **kwargs) -> datetime:
        """
        Parses a date string into a datetime object using the available parser module (dateutil or fallback).
        This method is intended to be called by other parsers like ScheduleParser.
        """
        if not isinstance(date_string, str):
            self.logger.error(f"Invalid date_string type for parse: {type(date_string)}. Value: {date_string}")
            raise ValueError("Date string must be a string.")
        if not date_string.strip(): # Check for empty string after stripping
            self.logger.warning(f"Empty date_string provided to parse method (original: '{date_string}').")
            raise ValueError("Date string cannot be empty or just whitespace.")
            
        try:
            # Use the module-level date_parser_module (which is either dateutil.parser or the fallback)
            return date_parser_module.parse(date_string, default=default, ignoretz=ignoretz, **kwargs)
        except (DateutilParserError, ValueError, TypeError) as e: # Catch errors from both dateutil and fallback
            self.logger.warning(f"DateParser instance method: Could not parse date string '{date_string}' with default context '{default}'. Error: {e}")
            raise DateutilParserError(f"Failed to parse date: {date_string} - {e}") from e 
        except Exception as e_unexp:
            self.logger.error(f"DateParser instance method: Unexpected error parsing '{date_string}': {e_unexp}", exc_info=True)
            # Re-raise as a common error type if possible, or a generic one if not.
            raise DateutilParserError(f"Unexpected error parsing date: {date_string} - {e_unexp}") from e_unexp

    def process_dates(self, data: Dict[str, Any], term_year_str: Optional[str] = None) -> Dict[str, Any]:
        self.logger.debug(f"Processing dates. Context term_year_str: {term_year_str}")
        default_year_dt = None
        if term_year_str: 
            year_match = re.search(r'\b(20\d{2})\b', term_year_str) 
            if year_match:
                try: default_year_dt = datetime(int(year_match.group(1)), 1, 1)
                except ValueError: self.logger.warning(f"Could not parse year from term: {term_year_str}")
        
        class_data = data.get("class_data")
        if isinstance(class_data, dict):
            if "Days of Week" in class_data and isinstance(class_data["Days of Week"], str):
                class_data["Days of Week"] = self._standardize_days_of_week(class_data["Days of Week"])
            date_fields = ["Term Start Date", "Term End Date", "Class Start Date", "Class End Date"]
            for field in date_fields:
                if field in class_data and isinstance(class_data[field], str) and class_data[field].strip():
                    class_data[field] = self.normalize_date(class_data[field], default_datetime=default_year_dt)
        
        for data_key in ["event_data", "task_data", "lab_data", "recitation_data"]:
            item_list = data.get(data_key)
            if isinstance(item_list, list):
                date_key_map = {"event_data": "Event Date", "task_data": "Due Date", 
                                "lab_data": "Lab Date", "recitation_data": "Recitation Date"}
                raw_date_key_map = {"lab_data": "Lab Date String", "recitation_data": "Recitation Date String"}

                actual_date_key = date_key_map.get(data_key)
                raw_date_key_to_check = raw_date_key_map.get(data_key)

                for item_idx, item in enumerate(item_list):
                    if isinstance(item, dict):
                        date_val_str = None
                        key_used_for_update = None

                        if actual_date_key and actual_date_key in item and isinstance(item[actual_date_key], str) and item[actual_date_key].strip():
                            date_val_str = item[actual_date_key]
                            key_used_for_update = actual_date_key
                        elif raw_date_key_to_check and raw_date_key_to_check in item and isinstance(item[raw_date_key_to_check], str) and item[raw_date_key_to_check].strip():
                            date_val_str = item[raw_date_key_to_check]
                            key_used_for_update = raw_date_key_to_check 
                        
                        if date_val_str and key_used_for_update:
                            normalized_date = self.normalize_date(date_val_str, default_datetime=default_year_dt)
                            item[key_used_for_update] = normalized_date
                            if key_used_for_update != actual_date_key and actual_date_key:
                                item[actual_date_key] = normalized_date 
                        elif not date_val_str and actual_date_key in item and not item.get(actual_date_key):
                            item[actual_date_key] = "TBD" 
        self.logger.debug("Date processing and standardization completed.")
        return data
    
    def normalize_date(self, date_str: str, default_datetime: Optional[datetime] = None) -> str:
        if not isinstance(date_str, str): 
            self.logger.warning(f"normalize_date input not a string: {type(date_str)} '{date_str}'. Returning as is.")
            return str(date_str)
        stripped = date_str.strip()
        if not stripped or stripped.upper() in ["TBD", "TBA", "N/A", "UNKNOWN", "NOT SPECIFIED", "PENDING", "NONE"]:
            return "TBD" 
        try:
            parsed_dt = self.parse(stripped, default=default_datetime, ignoretz=True) # Uses instance method self.parse
            return parsed_dt.strftime(self.output_date_format)
        except (DateutilParserError, ValueError) as e: 
            self.logger.warning(f"DateParser: normalize_date could not parse '{stripped}' (default_dt: {default_datetime}). Error: {e}. Returning original.")
            return stripped 
        except Exception as e_unexp:
            self.logger.error(f"DateParser: Unexpected error in normalize_date for '{stripped}': {e_unexp}", exc_info=True)
            return stripped

    def _standardize_days_of_week(self, days_str: str) -> str:
        if not days_str or not isinstance(days_str, str): return ""
        original_input = days_str
        
        processed_str = days_str.lower()
        processed_str = re.sub(r'\s*&\s*|\s+and\s+|\s*to\s*', ' ', processed_str) 
        processed_str = re.sub(r'[./,-]+', ' ', processed_str) 
        processed_str = re.sub(r'\s+', ' ', processed_str).strip() 

        day_indices: Set[int] = set()
        
        compact_str = "".join(processed_str.split()) 
        if compact_str in self.day_combinations:
            day_indices.update(self.day_combinations[compact_str])
            self.logger.debug(f"Standardized '{original_input}' using combination: '{compact_str}' -> Indices: {sorted(list(day_indices))}")
        else:
            found_any_by_regex = False
            tokens = processed_str.split(' ')
            for token in tokens:
                if not token: continue
                # matched_token = False # Not strictly needed here due to break
                for day_pattern_str, day_idx_val in self.day_regex_map_ordered:
                    # For simple string patterns in day_regex_map_ordered (like "mon", "monday", "r", "u")
                    # a direct comparison or simpler regex might be more efficient than fullmatch with \b if tokens are clean.
                    # However, re.fullmatch(day_pattern_str, token) is fine if patterns are exact strings.
                    # If day_pattern_str is already a regex, re.fullmatch is appropriate.
                    # Example: if day_pattern_str is "mon", re.fullmatch("mon", "mon") is true.
                    if re.fullmatch(day_pattern_str, token): # Assuming patterns in map are exact or simple regex for full match
                        day_indices.add(day_idx_val)
                        found_any_by_regex = True
                        # matched_token = True
                        break 
            if found_any_by_regex:
                 self.logger.debug(f"Standardized '{original_input}' via token/regex matching -> Indices: {sorted(list(day_indices))}")

        if day_indices:
            full_day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            return ", ".join([full_day_names[idx] for idx in sorted(list(day_indices))])
        
        self.logger.warning(f"Could not reliably standardize days of week for: '{original_input}'. Returning original stripped string.")
        return original_input.strip()

    def parse_weekdays_to_indices(self, days_str: str) -> List[int]: 
        if not days_str or not isinstance(days_str, str): return []
        standardized_full_names_str = self._standardize_days_of_week(days_str)
        if not standardized_full_names_str: return []
        
        day_names_list = [name.strip().lower() for name in standardized_full_names_str.split(',') if name.strip()]
        indices: Set[int] = set()
        for name_lower in day_names_list:
            if name_lower in self.weekday_map: 
                indices.add(self.weekday_map[name_lower])
        return sorted(list(indices))

    def get_date_range(self, start_date_str: Optional[str], end_date_str: Optional[str], default_datetime: Optional[datetime] = None) -> List[date]:
        if not start_date_str or not end_date_str: 
            raise ValueError("Start and end date strings are required for get_date_range.")
        try:
            norm_start_str = self.normalize_date(start_date_str, default_datetime=default_datetime)
            norm_end_str = self.normalize_date(end_date_str, default_datetime=default_datetime)
            
            if norm_start_str == "TBD" or norm_end_str == "TBD":
                raise ValueError(f"Cannot create date range with TBD dates: Start='{norm_start_str}', End='{norm_end_str}'")

            start_date_obj = self.parse(norm_start_str, default=default_datetime, ignoretz=True).date() # Uses self.parse
            end_date_obj = self.parse(norm_end_str, default=default_datetime, ignoretz=True).date()   # Uses self.parse
            
            if start_date_obj > end_date_obj:
                self.logger.debug(f"Start date {start_date_obj} is after end date {end_date_obj}. Swapping them for range generation.")
                start_date_obj, end_date_obj = end_date_obj, start_date_obj
                
            date_range_list: List[date] = []
            curr = start_date_obj
            day_count = 0
            max_days_in_range = (365 * 5) 

            while curr <= end_date_obj:
                date_range_list.append(curr)
                curr += timedelta(days=1)
                day_count += 1
                if day_count > max_days_in_range:
                    self.logger.warning(f"Date range generation exceeded max_days_in_range ({max_days_in_range}). Breaking early.")
                    break
            return date_range_list
        except (DateutilParserError, ValueError) as e: 
            self.logger.error(f"Could not parse date range from '{start_date_str}' to '{end_date_str}'. Err: {e}")
            raise ValueError(f"Could not parse date range. Original strings: Start='{start_date_str}', End='{end_date_str}'. Error: {e}") from e
        except Exception as e_unexp:
            self.logger.error(f"Unexpected error in get_date_range for '{start_date_str}' to '{end_date_str}': {e_unexp}", exc_info=True)
            raise ValueError(f"Unexpected error creating date range. Original strings: Start='{start_date_str}', End='{end_date_str}'. Error: {e_unexp}") from e_unexp

    def get_class_dates(self, start_date_str: Optional[str], end_date_str: Optional[str], days_of_week_str: Optional[str], default_datetime: Optional[datetime] = None) -> List[date]:
        if not start_date_str or not end_date_str: 
            self.logger.warning("Start and/or End date strings are missing for get_class_dates.")
            return []
        
        norm_start_date = self.normalize_date(start_date_str, default_datetime=default_datetime)
        norm_end_date = self.normalize_date(end_date_str, default_datetime=default_datetime)

        if norm_start_date == "TBD" or norm_end_date == "TBD":
            self.logger.warning(f"Cannot get class dates with TBD start/end: Start='{norm_start_date}', End='{norm_end_date}'")
            return []

        try:
            weekday_indices = self.parse_weekdays_to_indices(days_of_week_str or "")
            if not weekday_indices and days_of_week_str and days_of_week_str.strip():
                self.logger.warning(f"Could not parse weekdays from '{days_of_week_str}'. Cannot determine specific class dates.")
                return [] 
            if not weekday_indices: 
                self.logger.info("No specific weekdays provided to filter by for get_class_dates.")
                return []

            full_term_range = self.get_date_range(norm_start_date, norm_end_date, default_datetime=default_datetime)
            return [d for d in full_term_range if d.weekday() in weekday_indices]
        except ValueError as ve: 
            self.logger.error(f"Failed to calculate class dates from '{norm_start_date}' to '{norm_end_date}' for days '{days_of_week_str}'. Err: {ve}")
            return [] 
        except Exception as e_unexp:
            self.logger.error(f"Unexpected error in get_class_dates: {e_unexp}", exc_info=True)
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)

    dp_config = {"output_date_format": "%Y-%m-%d"} 
    dp = DateParser(logger_instance=logger, config=dp_config)

    logger.info("--- Testing Date Normalization ---")
    test_dates = [
        ("Sept 7, 2024", datetime(2024,1,1), "2024-09-07"),
        ("Sep 7 2024", None, "2024-09-07"),
        ("09/07/2024", None, "2024-09-07"),
        ("TBD", None, "TBD"),
        ("7 September 2024", None, "2024-09-07"),
        ("July 4th", datetime(2023,1,1), "2023-07-04"), 
        ("Invalid Date", None, "Invalid Date"), 
        ("10/10", datetime(2025,1,1), "2025-10-10"), 
        ("", None, "TBD"),
        (None, None, "None"), 
    ]
    for dt_str, default_dt, expected in test_dates:
        result = dp.normalize_date(dt_str, default_datetime=default_dt) # type: ignore
        logger.info(f"Input: '{dt_str}', Default: {default_dt.year if default_dt else 'N/A'} -> Normalized: '{result}' (Expected: '{expected}') {'PASS' if result == expected else 'FAIL'}")

    logger.info("\n--- Testing Days of Week Standardization ---")
    test_days = [
        ("MWF", "Monday, Wednesday, Friday"), ("M W F", "Monday, Wednesday, Friday"),
        ("TR", "Tuesday, Thursday"), ("TTH", "Tuesday, Thursday"), ("TuTh", "Tuesday, Thursday"),
        ("Mon Wed Fri", "Monday, Wednesday, Friday"),
        ("Monday Wednesday", "Monday, Wednesday"),
        ("Tues", "Tuesday"), ("Th", "Thursday"), ("R", "Thursday"), ("U", "Sunday"),
        ("MTWRF", "Monday, Tuesday, Wednesday, Thursday, Friday"),
        ("Nonsense", "Nonsense"), 
        ("Mon & Wed & Fri", "Monday, Wednesday, Friday"),
        ("Mon/Wed/Fri", "Monday, Wednesday, Friday"),
        ("Mon, Weds, F", "Monday, Wednesday, Friday"),
        ("R F", "Thursday, Friday"),
        ("THF", "Thursday, Friday") 
    ]
    for days_input, expected_days in test_days:
        result_days = dp._standardize_days_of_week(days_input)
        logger.info(f"Days Input: '{days_input}' -> Standardized: '{result_days}' (Expected: '{expected_days}') {'PASS' if result_days == expected_days else 'FAIL'}")
        indices = dp.parse_weekdays_to_indices(days_input)
        logger.info(f"    -> Indices: {indices}")


    logger.info("\n--- Testing get_class_dates ---")
    try:
        start_norm = dp.normalize_date("Sept 1, 2024")
        end_norm = dp.normalize_date("Sept 15, 2024")
        class_dates_mwf = dp.get_class_dates(start_norm, end_norm, "MWF")
        logger.info(f"Class dates for Sept 1-15 2024 (MWF) with format {dp.output_date_format}:")
        for d in class_dates_mwf: logger.info(f"  {d.strftime(dp.output_date_format)}") 
                                                                                      
        class_dates_tr = dp.get_class_dates(start_norm, end_norm, "TR")
        logger.info(f"Class dates for Sept 1-15 2024 (TR) with format {dp.output_date_format}:")
        for d in class_dates_tr: logger.info(f"  {d.strftime(dp.output_date_format)}")
                                                                                     
        start_norm_dec = dp.normalize_date("December 20, 2024")
        end_norm_jan = dp.normalize_date("January 5, 2025")
        class_dates_cross_year = dp.get_class_dates(start_norm_dec, end_norm_jan, "MWF")
        logger.info(f"Class dates for Dec 20, 2024 - Jan 5, 2025 (MWF) with format {dp.output_date_format}:")
        for d in class_dates_cross_year: logger.info(f"  {d.strftime(dp.output_date_format)}")
                                                                                        
        logger.info("Testing get_class_dates with TBD start:")
        tbd_dates1 = dp.get_class_dates("TBD", "September 15, 2024", "MWF")
        logger.info(f"  Result (TBD start): {tbd_dates1} (Expected: []) {'PASS' if not tbd_dates1 else 'FAIL'}")
        
        logger.info("Testing get_class_dates with invalid day string:")
        tbd_dates2 = dp.get_class_dates(start_norm, end_norm, "InvalidDays")
        logger.info(f"  Result (InvalidDays): {tbd_dates2} (Expected: []) {'PASS' if not tbd_dates2 else 'FAIL'}")

    except Exception as e_test:
        logger.error(f"Error during get_class_dates testing: {e_test}", exc_info=True)

    logger.info("DateParser tests finished.")
