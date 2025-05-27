"""
date_parser.py

Handles date normalization, days of week standardization, class schedule calculation,
and ISO 8601 datetime string generation for syllabus extraction.
"""

import re
import logging
import sys
from datetime import datetime, date, time, timedelta # Added time
from typing import Dict, List, Any, Optional, Set, Tuple, Union # Added Union

try:
    from dateutil import parser as dateutil_parser
    from dateutil.parser import ParserError as DateutilParserError
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    print("Warning (date_parser.py): python-dateutil library not found. Using simplified internal date parsing.")

    class DateutilParserError(ValueError):
        """Custom error to mimic dateutil.parser.ParserError for fallback."""
        pass

    class SimpleDateParser:
        @staticmethod
        def _preprocess_date_string(date_string: str) -> str:
            if not date_string: return ""
            return re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", date_string, flags=re.IGNORECASE).strip()

        @staticmethod
        def parse(date_string: str, default: Optional[datetime] = None, ignoretz: bool = False, **kwargs) -> datetime:
            if not date_string or not isinstance(date_string, str):
                raise ValueError("A non-empty string must be provided to SimpleDateParser.parse")
            processed_date_string = SimpleDateParser._preprocess_date_string(date_string)
            if not processed_date_string:
                 raise ValueError(f"Date string '{date_string}' became empty after preprocessing.")
            year_to_use = default.year if default and isinstance(default, (datetime, date)) else None
            formats_to_try = [
                "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", 
                "%d-%B-%Y", "%d-%b-%Y", "%d %B %Y", "%d %b %Y",
                "%m/%d/%y", "%Y/%m/%d", "%d.%m.%Y", "%Y.%m.%d",
                "%B %d", "%b %d", "%m/%d", "%m-%d", 
            ]
            for fmt in formats_to_try:
                try:
                    dt_obj = datetime.strptime(processed_date_string, fmt)
                    if dt_obj.year == 1900 and year_to_use:
                        dt_obj = dt_obj.replace(year=year_to_use)
                    return dt_obj
                except ValueError:
                    continue
            raise DateutilParserError(f"SimpleDateParser: Date string '{date_string}' could not be parsed with any known format.")
    
    class DateutilParserFallback:
        def parse(self, date_string: str, default: Optional[datetime] = None, ignoretz: bool = False, **kwargs) -> datetime:
            return SimpleDateParser.parse(date_string, default=default, ignoretz=ignoretz, **kwargs)

    date_parser_module = DateutilParserFallback()

if DATEUTIL_AVAILABLE:
    date_parser_module = dateutil_parser 

# Attempt to import pytz for timezone handling
try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    print("Warning (date_parser.py): pytz library not found. Timezone conversions for ISO 8601 will be limited or may fail.")


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
            (r"mo", 0), (r"tu", 1), (r"we", 2), (r"th", 3), (r"fr", 4), (r"sa", 5), (r"su", 6), 
            (r"u", 6), (r"r", 3) # 'r' for Thursday is common in some academic contexts
        ]
        self.day_combinations = { 
            "mwf": [0, 2, 4], "tr": [1, 3], "tth": [1, 3], "tuth": [1, 3], "thf": [3,4], 
            "mw": [0, 2], "mf": [0, 4], "wf": [2, 4],
            "mtwtf": [0, 1, 2, 3, 4], "mtwrf": [0, 1, 2, 3, 4], 
        }
        if DATEUTIL_AVAILABLE: self.logger.info("DateParser initialized using 'python-dateutil'.")
        else: self.logger.warning("DateParser initialized using simplified internal date parser (python-dateutil not found).")
        if not PYTZ_AVAILABLE:
            self.logger.warning("pytz library not found. ISO 8601 timezone conversions will be unavailable.")

    def parse(self, date_string: str, default: Optional[datetime] = None, ignoretz: bool = False, **kwargs) -> datetime:
        if not isinstance(date_string, str):
            self.logger.error(f"Invalid date_string type for parse: {type(date_string)}. Value: {date_string}")
            raise ValueError("Date string must be a string.")
        if not date_string.strip():
            self.logger.warning(f"Empty date_string provided to parse method (original: '{date_string}').")
            raise ValueError("Date string cannot be empty or just whitespace.")
        try:
            return date_parser_module.parse(date_string, default=default, ignoretz=ignoretz, **kwargs)
        except (DateutilParserError, ValueError, TypeError) as e:
            self.logger.warning(f"DateParser: Could not parse date string '{date_string}' with default '{default}'. Error: {e}")
            raise DateutilParserError(f"Failed to parse date: {date_string} - {e}") from e
        except Exception as e_unexp:
            self.logger.error(f"DateParser: Unexpected error parsing '{date_string}': {e_unexp}", exc_info=True)
            raise DateutilParserError(f"Unexpected error parsing date: {date_string} - {e_unexp}") from e_unexp

    def parse_time_string_to_objects(self, time_str: str) -> Tuple[Optional[time], Optional[time]]:
        """
        Parses a time string (e.g., "10:00 AM - 11:15 AM", "5:00PM") into start and end time objects.
        Returns (start_time, end_time_or_none).
        """
        if not time_str or not isinstance(time_str, str):
            return None, None
        
        time_str_lower = time_str.lower().strip()
        
        time_pattern = r"(?P<start_H>\d{1,2})(?::(?P<start_M>\d{2}))?\s*(?P<start_AMPM>am|pm)?(?:[^\S\r\n]*-[^\S\r\n]*(?P<end_H>\d{1,2})(?::(?P<end_M>\d{2}))?\s*(?P<end_AMPM>am|pm)?)?"
        match = re.search(time_pattern, time_str_lower, re.IGNORECASE)

        if not match:
            self.logger.warning(f"Could not parse time string: '{time_str}'")
            return None, None

        g = match.groupdict()
        start_h, start_m_str, start_ampm = g.get('start_H'), g.get('start_M'), g.get('start_AMPM')
        end_h, end_m_str, end_ampm = g.get('end_H'), g.get('end_M'), g.get('end_AMPM')

        parsed_start_time = None
        parsed_end_time = None

        try:
            if start_h:
                s_hour = int(start_h)
                s_minute = int(start_m_str) if start_m_str else 0
                if start_ampm == 'pm' and s_hour != 12: s_hour += 12
                if start_ampm == 'am' and s_hour == 12: s_hour = 0 
                parsed_start_time = time(s_hour % 24, s_minute)
            
            if end_h: 
                e_hour = int(end_h)
                e_minute = int(end_m_str) if end_m_str else 0
                effective_end_ampm = end_ampm or start_ampm 
                if effective_end_ampm == 'pm' and e_hour != 12: e_hour += 12
                if effective_end_ampm == 'am' and e_hour == 12: e_hour = 0
                parsed_end_time = time(e_hour % 24, e_minute)
                if parsed_start_time and parsed_end_time < parsed_start_time and not end_ampm and start_ampm == 'am':
                     if (e_hour + 12) < 24 : 
                        parsed_end_time = time(e_hour + 12, e_minute)
        except ValueError as ve:
            self.logger.error(f"Error parsing time components from '{time_str}': {ve}")
            return None, None
            
        return parsed_start_time, parsed_end_time

    def get_iso_datetime_str(self, 
                             date_input: Union[str, date, datetime], 
                             time_input: Union[str, time], 
                             timezone_str: str, 
                             is_end_time_for_due_date: bool = False) -> Optional[str]:
        """
        Combines a date and time, localizes to the given timezone, and returns an ISO 8601 string.
        """
        if not PYTZ_AVAILABLE:
            self.logger.error("pytz library is not available. Cannot generate timezone-aware ISO 8601 strings.")
            return None
        if not timezone_str:
            self.logger.warning("No timezone_str provided for ISO datetime conversion. Cannot proceed.")
            return None

        try:
            target_timezone = pytz.timezone(timezone_str)
        except pytz.exceptions.UnknownTimeZoneError:
            self.logger.error(f"Unknown timezone string: '{timezone_str}'. Cannot generate ISO string.")
            return None

        try:
            naive_date_part: date
            if isinstance(date_input, datetime):
                naive_date_part = date_input.date()
            elif isinstance(date_input, date):
                naive_date_part = date_input
            else: 
                normalized_date_str = self.normalize_date(str(date_input)) # Uses updated normalize_date
                if normalized_date_str == "TBD": return None # Or handle as appropriate
                naive_date_part = self.parse(normalized_date_str, ignoretz=True).date()

            target_time_obj: Optional[time] = None
            if isinstance(time_input, time):
                target_time_obj = time_input
            elif isinstance(time_input, str) and time_input.strip():
                start_t, end_t = self.parse_time_string_to_objects(time_input)
                if is_end_time_for_due_date and end_t: 
                    target_time_obj = end_t
                else: 
                    target_time_obj = start_t
            
            if not target_time_obj: 
                if is_end_time_for_due_date: 
                    target_time_obj = time(23, 59, 59)
                else: 
                    self.logger.warning(f"No valid time_input resolved for date '{naive_date_part}' and timezone '{timezone_str}'.")
                    return naive_date_part.isoformat() 

            naive_dt = datetime.combine(naive_date_part, target_time_obj)
            localized_dt = target_timezone.localize(naive_dt)
            return localized_dt.isoformat()

        except (DateutilParserError, ValueError, TypeError) as e:
            self.logger.warning(f"Could not generate ISO datetime for date='{date_input}', time='{time_input}', tz='{timezone_str}'. Error: {e}")
            return None
        except Exception as e_unexp:
            self.logger.error(f"Unexpected error in get_iso_datetime_str: {e_unexp}", exc_info=True)
            return None

    def process_dates(self, data: Dict[str, Any], term_year_str: Optional[str] = None) -> Dict[str, Any]:
        self.logger.debug(f"Processing dates. Context term_year_str from process_dates: {term_year_str}") # Log received term_year_str
        
        # Determine default_year_dt based on term_year_str passed to this method first
        default_year_dt_for_normalize = None
        if term_year_str: 
            year_match = re.search(r'\b(20\d{2})\b', term_year_str) 
            if year_match:
                try: 
                    default_year_dt_for_normalize = datetime(int(year_match.group(1)), 1, 1)
                    self.logger.debug(f"process_dates: default_year_dt_for_normalize set to {default_year_dt_for_normalize} from term_year_str '{term_year_str}'.")
                except ValueError: 
                    self.logger.warning(f"Could not parse year from term_year_str: {term_year_str} in process_dates.")
        
        class_data = data.get("class_data")
        if isinstance(class_data, dict):
            if "Days of Week" in class_data and isinstance(class_data["Days of Week"], str):
                class_data["Days of Week"] = self._standardize_days_of_week(class_data["Days of Week"])
            
            date_fields = ["Term Start Date", "Term End Date", "Class Start Date", "Class End Date"]
            # Use term_year_str from class_data.get("Term") as context for individual field normalization
            # This allows normalize_date to use the most specific context if available
            contextual_term_for_fields = class_data.get("Term", term_year_str) # Prefer "Term" if present, else the overall one

            for field in date_fields:
                if field in class_data and isinstance(class_data[field], str) and class_data[field].strip():
                    class_data[field] = self.normalize_date(
                        class_data[field], 
                        default_datetime=default_year_dt_for_normalize, # Pass the default_year determined at method start
                        term_year_str=contextual_term_for_fields # Pass specific term context
                    )
        
        for data_key in ["event_data", "task_data", "lab_data", "recitation_data"]:
            item_list = data.get(data_key)
            if isinstance(item_list, list):
                date_key_map = {"event_data": "Event Date", "task_data": "Due Date", 
                                "lab_data": "Lab Date", "recitation_data": "Recitation Date"}
                raw_date_key_map = {"lab_data": "Lab Date String", "recitation_data": "Recitation Date String"}
                actual_date_key = date_key_map.get(data_key)
                raw_date_key_to_check = raw_date_key_map.get(data_key)

                for item in item_list:
                    if isinstance(item, dict):
                        date_val_str = None; key_used_for_update = None
                        if actual_date_key and item.get(actual_date_key) and isinstance(item[actual_date_key], str) and item[actual_date_key].strip():
                            date_val_str = item[actual_date_key]; key_used_for_update = actual_date_key
                        elif raw_date_key_to_check and item.get(raw_date_key_to_check) and isinstance(item[raw_date_key_to_check], str) and item[raw_date_key_to_check].strip():
                            date_val_str = item[raw_date_key_to_check]; key_used_for_update = raw_date_key_to_check
                        
                        if date_val_str and key_used_for_update:
                            # Use default_year_dt_for_normalize for list items too
                            # And term_year_str from class_data.get("Term") if available for specific item context
                            contextual_term_for_item = class_data.get("Term", term_year_str) if isinstance(class_data, dict) else term_year_str
                            normalized_date = self.normalize_date(date_val_str, default_datetime=default_year_dt_for_normalize, term_year_str=contextual_term_for_item)
                            item[key_used_for_update] = normalized_date
                            if key_used_for_update != actual_date_key and actual_date_key: item[actual_date_key] = normalized_date
                        elif not date_val_str and actual_date_key and not item.get(actual_date_key):
                            item[actual_date_key] = "TBD"
        self.logger.debug("Date processing and standardization completed.")
        return data
    
    def normalize_date(self, date_str: str, default_datetime: Optional[datetime] = None, term_year_str: Optional[str] = None) -> str: # ADDED term_year_str
        if not isinstance(date_str, str): 
            self.logger.warning(f"normalize_date input not a string: {type(date_str)} '{date_str}'. Returning as is.")
            return str(date_str)
        stripped = date_str.strip()
        if not stripped or stripped.upper() in ["TBD", "TBA", "N/A", "UNKNOWN", "NOT SPECIFIED", "PENDING", "NONE"]:
            return "TBD" 
        
        effective_default_dt = default_datetime
        if not effective_default_dt and term_year_str: # If no explicit default_datetime, try to derive from term_year_str
            year_match = re.search(r'\b(20\d{2})\b', term_year_str)
            if year_match:
                try:
                    effective_default_dt = datetime(int(year_match.group(1)), 1, 1) # Use Jan 1 of that year as default
                    self.logger.debug(f"normalize_date: Using year {effective_default_dt.year} from term_year_str ('{term_year_str}') as default for parsing '{stripped}'.")
                except ValueError:
                    self.logger.warning(f"normalize_date: Could not parse year from term_year_str: {term_year_str} for date '{stripped}'.")
        elif effective_default_dt:
             self.logger.debug(f"normalize_date: Using provided default_datetime: {effective_default_dt} for date '{stripped}'.")
        else:
             self.logger.debug(f"normalize_date: No default_datetime or usable term_year_str for date '{stripped}'.")


        try:
            parsed_dt = self.parse(stripped, default=effective_default_dt, ignoretz=True) # Pass the determined default
            return parsed_dt.strftime(self.output_date_format)
        except (DateutilParserError, ValueError) as e: 
            self.logger.warning(f"DateParser: normalize_date could not parse '{stripped}' (default_dt: {effective_default_dt}, term_year_str: {term_year_str}). Error: {e}. Returning original.")
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
            # Split by common separators that might still exist if not just spaces
            tokens = re.split(r'[\s,;/]+', processed_str)
            remaining_text_for_single_char_match = processed_str
            
            # First pass for multi-character abbreviations and full names
            for token in tokens:
                if not token: continue
                matched_in_token = False
                for day_pattern_str, day_idx_val in self.day_regex_map_ordered:
                    # Ensure we only match full tokens unless it's a single char pattern like 'r' or 'u'
                    if len(day_pattern_str) > 1 or day_pattern_str in ['r', 'u']: # Check if pattern is multi-char or specific single char
                        if re.fullmatch(day_pattern_str, token):
                            day_indices.add(day_idx_val)
                            found_any_by_regex = True
                            matched_in_token = True
                            # Remove matched part from remaining_text_for_single_char_match
                            # This needs careful regex to avoid issues if token is part of another word.
                            # Simpler: if a multi-char token matches, we assume it's distinct.
                            break 
                if matched_in_token:
                    # Try to remove the successfully matched token from the string for subsequent single-char check
                    # This helps avoid 'th' in 'thursday' matching 'h' again if 'h' was a pattern.
                    # This is tricky; for now, rely on ordered day_regex_map and fullmatch.
                    pass

            # Second pass for single characters if compact string wasn't a known combo
            # and if multi-char tokens didn't fully explain the string.
            # This is to catch "TR" or "MWF" if they were not space-separated.
            if not day_indices and not (compact_str in self.day_combinations):
                temp_compact_str = compact_str
                for char_idx in range(len(temp_compact_str)):
                    char = temp_compact_str[char_idx]
                    # Check if 'th' or 'tu' or 'su' (two-char abbrevs within compact string)
                    if char_idx + 1 < len(temp_compact_str):
                        two_char_token = temp_compact_str[char_idx:char_idx+2]
                        matched_two_char = False
                        for day_pattern_str, day_idx_val in self.day_regex_map_ordered:
                            if day_pattern_str == two_char_token: # like "th", "tu", "su"
                                day_indices.add(day_idx_val)
                                found_any_by_regex = True
                                matched_two_char = True
                                # Conceptually skip the next char, but loop handles index
                                break 
                        if matched_two_char:
                            # Advance index conceptually, or mark as processed
                            continue # Skips single char check for current char if two_char matched

                    # Check single character
                    for day_pattern_str, day_idx_val in self.day_regex_map_ordered:
                        if day_pattern_str == char:
                            day_indices.add(day_idx_val)
                            found_any_by_regex = True
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

    def get_date_range(self, start_date_str: Optional[str], end_date_str: Optional[str], 
                       default_datetime: Optional[datetime] = None, term_year_str: Optional[str] = None) -> List[date]: # Added term_year_str
        if not start_date_str or not end_date_str: 
            raise ValueError("Start and end date strings are required for get_date_range.")
        
        # Determine effective default for parsing
        effective_default_dt = default_datetime
        if not effective_default_dt and term_year_str:
            year_match = re.search(r'\b(20\d{2})\b', term_year_str)
            if year_match:
                try: effective_default_dt = datetime(int(year_match.group(1)), 1, 1)
                except ValueError: pass

        try:
            # Pass term_year_str to normalize_date for context if needed
            norm_start_str = self.normalize_date(start_date_str, default_datetime=default_datetime, term_year_str=term_year_str)
            norm_end_str = self.normalize_date(end_date_str, default_datetime=default_datetime, term_year_str=term_year_str)
            
            if norm_start_str == "TBD" or norm_end_str == "TBD":
                raise ValueError(f"Cannot create date range with TBD dates: Start='{norm_start_str}', End='{norm_end_str}'")

            start_date_obj = self.parse(norm_start_str, default=effective_default_dt, ignoretz=True).date()
            end_date_obj = self.parse(norm_end_str, default=effective_default_dt, ignoretz=True).date()   
            
            if start_date_obj > end_date_obj:
                self.logger.debug(f"Start date {start_date_obj} is after end date {end_date_obj}. Swapping them.")
                start_date_obj, end_date_obj = end_date_obj, start_date_obj
            
            date_range_list: List[date] = []
            curr = start_date_obj; day_count = 0; max_days_in_range = (365 * 5) 
            while curr <= end_date_obj:
                date_range_list.append(curr)
                curr += timedelta(days=1)
                day_count += 1
                if day_count > max_days_in_range: self.logger.warning(f"Date range exceeded {max_days_in_range} days."); break
            return date_range_list
        except (DateutilParserError, ValueError) as e: 
            self.logger.error(f"Could not parse date range from '{start_date_str}' to '{end_date_str}'. Default DT: {effective_default_dt}, Term: {term_year_str}. Err: {e}")
            raise ValueError(f"Could not parse date range. Original: Start='{start_date_str}', End='{end_date_str}'. Error: {e}") from e
        except Exception as e_unexp:
            self.logger.error(f"Unexpected error in get_date_range: {e_unexp}", exc_info=True)
            raise ValueError(f"Unexpected error in get_date_range. Original: Start='{start_date_str}', End='{end_date_str}'. Error: {e_unexp}") from e_unexp


    def get_class_dates(self, start_date_str: Optional[str], end_date_str: Optional[str], 
                        days_of_week_str: Optional[str], 
                        default_datetime: Optional[datetime] = None, term_year_str: Optional[str] = None) -> List[date]: # Added term_year_str
        if not start_date_str or not end_date_str: 
            self.logger.warning("Start and/or End date strings are missing for get_class_dates.")
            return []
        
        # Pass term_year_str for normalization context
        norm_start_date = self.normalize_date(start_date_str, default_datetime=default_datetime, term_year_str=term_year_str)
        norm_end_date = self.normalize_date(end_date_str, default_datetime=default_datetime, term_year_str=term_year_str)
        
        if norm_start_date == "TBD" or norm_end_date == "TBD":
            self.logger.warning(f"Cannot get class dates with TBD start/end: Start='{norm_start_date}', End='{norm_end_date}'")
            return []
        try:
            weekday_indices = self.parse_weekdays_to_indices(days_of_week_str or "")
            if not weekday_indices and days_of_week_str and days_of_week_str.strip():
                self.logger.warning(f"Could not parse weekdays from '{days_of_week_str}'. Cannot determine specific class dates.")
                return [] 
            if not weekday_indices: 
                self.logger.info("No specific weekdays provided for get_class_dates.")
                return [] 
            
            full_term_range = self.get_date_range(norm_start_date, norm_end_date, default_datetime=default_datetime, term_year_str=term_year_str) # Pass term_year_str
            return [d for d in full_term_range if d.weekday() in weekday_indices]
        except ValueError as ve: 
            self.logger.error(f"Failed to calculate class dates from '{norm_start_date}' to '{norm_end_date}' for days '{days_of_week_str}'. Err: {ve}")
            return [] 
        except Exception as e_unexp:
            self.logger.error(f"Unexpected error in get_class_dates: {e_unexp}", exc_info=True)
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)

    dp_config = {"output_date_format": "%Y-%m-%d"} 
    dp = DateParser(logger_instance=logger, config=dp_config)
    if not PYTZ_AVAILABLE:
        logger.error("PYTZ IS NOT AVAILABLE. Some tests for get_iso_datetime_str will be skipped or may not reflect full functionality.")

    logger.info("\n--- Testing parse_time_string_to_objects ---")
    time_tests = [
        ("10:00 AM - 11:15 AM", (time(10,0), time(11,15))),
        ("5:00PM", (time(17,0), None)),
        ("1:00 pm - 2pm", (time(13,0), time(14,0))),
        ("12:00 AM", (time(0,0), None)), 
        ("12:00 PM", (time(12,0), None)), 
        ("9AM", (time(9,0), None)),
        ("10:30 - 11:30 am", (time(10,30), time(11,30))), 
        ("11:00 am - 1:00 pm", (time(11,0), time(13,0))), 
        ("Invalid time string", (None, None)),
        ("", (None, None))
    ]
    for time_input_str, expected_times in time_tests:
        s_time, e_time = dp.parse_time_string_to_objects(time_input_str)
        logger.info(f"Input: '{time_input_str}' -> Start: {s_time}, End: {e_time} (Expected: Start={expected_times[0]}, End={expected_times[1]}) {'PASS' if (s_time, e_time) == expected_times else 'FAIL'}")

    logger.info("\n--- Testing get_iso_datetime_str ---")
    test_iso_date = "2024-09-05" 
    test_iso_date_winter = "2024-11-05" 

    iso_tests = []
    if PYTZ_AVAILABLE: 
        iso_tests.extend([
            (test_iso_date, "10:00 AM", "America/New_York", False, "2024-09-05T10:00:00-04:00"),
            (test_iso_date, time(14,30), "America/New_York", False, "2024-09-05T14:30:00-04:00"),
            (test_iso_date_winter, "10:00 AM", "America/New_York", False, "2024-11-05T10:00:00-05:00"), 
            (test_iso_date, "11:59 PM", "America/Los_Angeles", True, "2024-09-05T23:59:00-07:00"), 
            (date(2024, 7, 4), "8:00PM", "Europe/London", False, "2024-07-04T20:00:00+01:00"), 
            (test_iso_date, "Invalid Time", "America/New_York", False, test_iso_date), 
            (test_iso_date, "10:00 AM", "Invalid/Timezone", False, None),
            ("TBD", "10:00 AM", "America/New_York", False, None)
        ])
    else:
        logger.warning("Skipping some ISO datetime tests as pytz is not available.")
        iso_tests.extend([
             (test_iso_date, "10:00 AM", "Invalid/Timezone", False, None) 
        ])

    for dt_input, tm_input, tz_str, is_due_end, expected_iso in iso_tests:
        result_iso = dp.get_iso_datetime_str(dt_input, tm_input, tz_str, is_end_time_for_due_date=is_due_end)
        logger.info(f"Input: D='{dt_input}', T='{tm_input}', TZ='{tz_str}' -> ISO: '{result_iso}' (Expected: '{expected_iso}') {'PASS' if result_iso == expected_iso else 'FAIL'}")

    logger.info("\n--- Testing Date Normalization (with term_year_str) ---")
    test_dates_normalization = [
        ("Sept 7", datetime(2024,1,1), None, "2024-09-07"), # Default datetime provided
        ("Sep 7", None, "Fall 2024", "2024-09-07"),        # term_year_str provided
        ("Oct 10", None, "Spring 2025", "2025-10-10"),     # term_year_str provided
        ("11/15", None, "2023 Academic Year", "2023-11-15"),# term_year_str provided
        ("Jan 5th", None, None, "2025-01-05"),              # No year context (uses current year from dateutil default or 1900) -> this will depend on test run date, adjusted to current year for pass
        ("TBD", None, "Fall 2024", "TBD"),
        ("August 1st, 2022", None, "Fall 2024", "2022-08-01"), # Explicit year overrides term_year_str
        ("Dec 25", None, "NoYearHere", datetime.now().strftime("%Y") + "-12-25") # No year in term_year_str (uses current year)
    ]

    current_year = datetime.now().year
    for i, (dt_str, default_dt, term_str, expected_str) in enumerate(test_dates_normalization):
        if "Jan 5th" in dt_str: # Adjust expected for Jan 5th test
            expected_str = f"{current_year}-01-05"
        if "Dec 25" in dt_str and "NoYearHere" in term_str:
             expected_str = f"{current_year}-12-25"

        result = dp.normalize_date(dt_str, default_datetime=default_dt, term_year_str=term_str)
        is_pass = result == expected_str
        logger.info(f"Input: '{dt_str}', DefaultDT: {default_dt.year if default_dt else 'N/A'}, Term: '{term_str}' -> Normalized: '{result}' (Expected: '{expected_str}') {'PASS' if is_pass else 'FAIL'}")


    logger.info("\n--- Testing Days of Week Standardization ---")
    test_days = [
        ("MWF", "Monday, Wednesday, Friday"), 
        ("TR", "Tuesday, Thursday"),
        ("M W F", "Monday, Wednesday, Friday"),
        ("Mon,Wed,Fri", "Monday, Wednesday, Friday"),
        ("Tues/Thurs", "Tuesday, Thursday"),
        ("R", "Thursday"), 
        ("MTWRF", "Monday, Tuesday, Wednesday, Thursday, Friday"),
        ("Th", "Thursday"),
        ("F", "Friday"),
        ("SaSu", "Saturday, Sunday"),
        ("Mondays and Wednesdays", "Monday, Wednesday")
    ]
    for days_input, expected_days in test_days:
        result_days = dp._standardize_days_of_week(days_input)
        logger.info(f"Days Input: '{days_input}' -> Standardized: '{result_days}' (Expected: '{expected_days}') {'PASS' if result_days == expected_days else 'FAIL'}")

    logger.info("DateParser tests finished.")
