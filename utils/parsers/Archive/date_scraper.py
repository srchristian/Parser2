"""
date_scraper.py

Handles the scraping of academic term dates exclusively from official school websites.
Uses CSV lookup for school URLs and returns null when dates cannot be found with certainty.
Enhanced with regex matching for school names.
"""

import os
import json
import re
import csv
import requests
from datetime import datetime
from dateutil import parser as date_parser
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
import logging

class TermDateScraper:
    """
    Academic term date scraper that:
    1. Uses only official school websites from CSV lookup
    2. Never guesses dates - returns null if uncertain
    3. Uses explicit contextual clues for start/end dates
    4. Focuses on common academic calendar URL patterns
    5. Uses regex for flexible school name matching
    """
    
    def __init__(self, school_csv_path=None, logger=None):
        """
        Initialize the term date scraper.
        
        Args:
            school_csv_path: Path to CSV file with school URLs
            logger: Optional logger instance
        """
        # Set up logging - handle both standard logger and SyllabusLogger
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup cache for efficiency
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, 'term_dates_cache.json')
        self.date_cache = self._load_cache()
        
        # Load school URLs from CSV - use default path if none provided
        self.school_urls = {}
        
        # If no path provided, use the default path
        if school_csv_path is None:
            # Get the project root directory (2 levels up from this file)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            school_csv_path = os.path.join(project_root, 'utils', 'School List', 'schools_with_urls.csv')
            self._log('info', f"Using default school CSV path: {school_csv_path}")
        
        # Load the school URLs
        if os.path.exists(school_csv_path):
            self._load_school_urls(school_csv_path)
        else:
            self._log('error', f"School CSV file not found: {school_csv_path}")
        
        # Initialize the regex patterns cache for schools
        self.school_regex_patterns = {}
        self._compile_school_regex_patterns()
        
        # Key phrases that indicate term start/end dates
        self.start_indicators = [
            "classes begin", "first day of classes", "instruction begins", 
            "semester begins", "term begins", "classes start", 
            "beginning of term", "first class day", "start of instruction"
        ]
        
        self.end_indicators = [
            "classes end", "last day of classes", "instruction ends", 
            "semester ends", "term ends", "final day of classes",
            "end of term", "last class day", "end of instruction"
        ]
        
        # Common academic calendar URL patterns
        self.calendar_paths = [
            '/academic-calendar',
            '/calendar',
            '/academics/calendar',
            '/registrar/calendar',
            '/registrar/academic-calendar',
            '/academics/academic-calendar',
            '/important-dates',
            '/dates-and-deadlines',
            '/academic-dates'
        ]
        
        # Log initialization using logger.logger to handle SyllabusLogger
        if hasattr(self.logger, 'logger'):
            self.logger.logger.info("TermDateScraper initialized")
        else:
            self.logger.info("TermDateScraper initialized")

    def _log(self, level, message):
        """
        Log a message with appropriate logger method.
        
        Args:
            level: Log level ('info', 'error', 'warning', 'debug')
            message: Message to log
        """
        if hasattr(self.logger, 'logger'):
            # Using SyllabusLogger
            log_method = getattr(self.logger.logger, level, None)
            if log_method:
                log_method(message)
        else:
            # Using standard logger
            log_method = getattr(self.logger, level, None)
            if log_method:
                log_method(message)

    def _load_cache(self) -> Dict:
        """Load cached term dates from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self._log('error', f"Error loading term dates cache: {e}")
            return {}
    
    def _save_cache(self) -> None:
        """Save cached term dates to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.date_cache, f, indent=2)
        except Exception as e:
            self._log('error', f"Error saving term dates cache: {e}")
    
    def _load_school_urls(self, csv_path):
        """
        Load school URLs from CSV file.
        
        Args:
            csv_path: Path to CSV file with school URLs
        """
        try:
            self._log('info', f"Loading school URLs from {csv_path}")
            
            self.school_urls = {}  # Reset the dictionary
            
            # Try to read the file directly first
            with open(csv_path, 'r', encoding='utf-8') as file:
                # Read all content for debugging
                content = file.read()
                self._log('debug', f"CSV content length: {len(content)} characters")
                
            # Process as CSV
            with open(csv_path, 'r', encoding='utf-8') as file:
                # Use csv module to properly handle quoted fields
                reader = csv.reader(file)
                
                row_count = 0
                for row in reader:
                    row_count += 1
                    # Make sure we have at least 2 columns
                    if len(row) >= 2:
                        school_name = row[0].strip()
                        url = row[1].strip()
                        
                        if school_name and url:
                            # Store the school name in lowercase for case-insensitive lookups
                            self.school_urls[school_name.lower()] = self._normalize_url(url)
                            self._log('debug', f"Loaded school: {school_name} -> {url}")
            
            # Debug output to verify schools were loaded
            school_names = list(self.school_urls.keys())
            self._log('debug', f"Loaded {len(school_names)} school URLs")
            
            # Log first few school names for verification
            if school_names:
                sample = school_names[:5] if len(school_names) > 5 else school_names
                self._log('debug', f"Sample schools: {sample}")
                
                # Specifically check if URI is loaded
                if 'university of rhode island' in self.school_urls:
                    self._log('info', f"URI URL found: {self.school_urls['university of rhode island']}")
                else:
                    self._log('warning', "University of Rhode Island not found in loaded schools")
                    
            self._log('info', f"Processed {row_count} rows from CSV")
            
        except Exception as e:
            self._log('error', f"Error loading school URLs: {e}")
            import traceback
            self._log('error', traceback.format_exc())
    
    def _compile_school_regex_patterns(self):
        """
        Compile regex patterns for each school name for faster matching.
        This creates multiple regex patterns for each school to match common variations.
        """
        self._log('debug', "Compiling regex patterns for school names")
        
        for school_name, url in self.school_urls.items():
            patterns = []
            
            # Pattern 1: Exact match (case insensitive)
            patterns.append(re.compile(r'(?i)^' + re.escape(school_name) + r'$'))
            
            # Pattern 2: Remove common words like "University", "College", etc.
            simplified = re.sub(r'(?i)\b(university|college|institute|school|of|the)\b', '', school_name)
            simplified = re.sub(r'\s+', ' ', simplified).strip()
            if simplified and simplified != school_name:
                patterns.append(re.compile(r'(?i)' + re.escape(simplified)))
            
            # Pattern 3: Match initials (e.g. "URI" for "University of Rhode Island")
            parts = school_name.split()
            if len(parts) > 1:
                initials = ''.join(p[0] for p in parts if p.lower() not in ['of', 'the', 'and'])
                if len(initials) >= 2:
                    patterns.append(re.compile(r'(?i)\b' + re.escape(initials) + r'\b'))
            
            # Store the patterns
            self.school_regex_patterns[school_name] = patterns
            
        self._log('debug', f"Compiled regex patterns for {len(self.school_regex_patterns)} schools")
    
    def _normalize_url(self, url):
        """
        Ensure URL has proper formatting.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        if not url:
            return ""
            
        # Add https:// if no protocol is specified
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Remove trailing slash
        url = url.rstrip('/')
        
        return url
    
    def get_term_dates(self, school_name: str, term: str) -> Dict[str, str]:
        """
        Get the start and end dates for a specific school and term.
        
        Args:
            school_name: Name of the school
            term: Term name (e.g., "Fall 2024")
            
        Returns:
            Dictionary with school, term, year, start_date, end_date.
            Dates will be null if they couldn't be found with certainty.
        """
        self._log('info', f"Looking up term dates for {school_name}, {term}")
        
        try:
            # Normalize school name and term for caching
            school_key = school_name.lower().strip()
            term_key = term.lower().strip()
            cache_key = f"{school_key}_{term_key}"
            
            # Check if we have cached data for this school and term
            if cache_key in self.date_cache:
                self._log('info', f"Using cached dates for {school_name}, {term}")
                return self.date_cache[cache_key]
            
            # Parse term to extract season and year
            term_season, term_year = self._parse_term(term)
            
            # Prepare result structure with school and term info
            result = {
                "school": school_name,
                "term": term,
                "year": term_year,
                "start_date": None,
                "end_date": None
            }
            
            # Find school URL from our database using regex matching
            school_url = self._get_school_url_regex(school_name)
            
            if school_url:
                self._log('info', f"Found school URL: {school_url}")
                
                # Search school website for term dates
                dates = self._search_school_website(school_url, term_season, term_year)
                
                if dates and dates.get('start_date') and dates.get('end_date'):
                    result.update(dates)
            else:
                self._log('warning', f"No URL found for {school_name}, returning null dates")
            
            # Cache the result regardless of whether dates were found
            self.date_cache[cache_key] = result
            self._save_cache()
            
            return result
            
        except Exception as e:
            self._log('error', f"Error getting term dates for {school_name}, {term}: {e}")
            import traceback
            self._log('error', traceback.format_exc())
            
            # Return basic structure with null dates on error
            return {
                "school": school_name,
                "term": term,
                "year": self._parse_term(term)[1],
                "start_date": None,
                "end_date": None
            }
    
    def _parse_term(self, term: str) -> Tuple[str, int]:
        """
        Parse term string to extract season and year.
        
        Args:
            term: Term string (e.g., "Fall 2024")
            
        Returns:
            Tuple of (season, year)
        """
        # Default values
        season = "fall"
        year = datetime.now().year
        
        # Extract year (look for 4-digit number)
        year_match = re.search(r'20\d{2}', term)
        if year_match:
            year = int(year_match.group(0))
        
        # Extract season
        term_lower = term.lower()
        seasons = ["fall", "spring", "summer", "winter"]
        
        for s in seasons:
            if s in term_lower:
                season = s
                break
        
        return season, year
    
    def _get_school_url_regex(self, school_name):
        """
        Get school URL using regex pattern matching for flexible name comparison.
        
        Args:
            school_name: School name to find URL for
            
        Returns:
            School URL or None if not found
        """
        input_name = school_name.lower().strip()
        self._log('debug', f"Looking for school with regex: '{input_name}'")
        
        # Try direct match first (most efficient)
        if input_name in self.school_urls:
            self._log('info', f"Found direct URL match for {school_name}")
            return self.school_urls[input_name]
        
        # Now try regex matching
        match_scores = []
        
        # Common abbreviations dictionary - add any that might be in your data
        abbreviations = {
            'university': ['univ', 'u'],
            'college': ['coll', 'c'],
            'institute': ['inst', 'i'],
            'technology': ['tech', 't'],
            'rhode island': ['ri']
        }
        
        # Create regex patterns for the input name
        input_simplified = re.sub(r'(?i)\b(university|college|institute|school|of|the)\b', '', input_name)
        input_simplified = re.sub(r'\s+', ' ', input_simplified).strip()
        
        # Get possible initials
        input_parts = input_name.split()
        input_initials = ''.join(p[0] for p in input_parts if p.lower() not in ['of', 'the', 'and'])
        
        # Check each school in our database
        for db_name, url in self.school_urls.items():
            # Score starts at 0, higher is better
            score = 0
            
            # Score 1: Check for exact match
            if db_name == input_name:
                self._log('debug', f"Exact match found for {input_name}")
                return url  # Direct hit, return immediately
            
            # Score 2: Check if one contains the other
            if input_name in db_name:
                score += 10
            elif db_name in input_name:
                score += 8
                
            # Score 3: Check simplified versions (without common words)
            db_simplified = re.sub(r'(?i)\b(university|college|institute|school|of|the)\b', '', db_name)
            db_simplified = re.sub(r'\s+', ' ', db_simplified).strip()
            
            if input_simplified and db_simplified and (input_simplified in db_simplified or db_simplified in input_simplified):
                score += 7
                
            # Score 4: Check for abbreviations
            db_parts = db_name.split()
            db_initials = ''.join(p[0] for p in db_parts if p.lower() not in ['of', 'the', 'and'])
            
            # Check initials
            if input_initials and db_initials and (input_initials == db_initials):
                score += 15
            
            # Check common abbreviations
            for full, abbrs in abbreviations.items():
                if full in input_name and any(abbr in db_name for abbr in abbrs):
                    score += 3
                if full in db_name and any(abbr in input_name for abbr in abbrs):
                    score += 3
            
            # Add special check for URI
            if "rhode island" in input_name.lower() and "uri" in db_name.lower():
                score += 10
            if "rhode island" in db_name.lower() and "uri" in input_name.lower():
                score += 10
            
            # Only consider reasonable matches
            if score > 5:
                match_scores.append((db_name, url, score))
        
        # Sort by score (highest first)
        match_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Print top matches for debugging
        if match_scores:
            self._log('debug', f"Top matches for '{input_name}':")
            for db_name, url, score in match_scores[:3]:
                self._log('debug', f"  - {db_name}: score {score}")
            
            # Return the URL with highest score
            best_match = match_scores[0]
            if best_match[2] > 5:  # Minimum threshold
                self._log('info', f"Found regex match for {school_name}: {best_match[0]} (score: {best_match[2]})")
                return best_match[1]
        
        # Handle special case for URI
        if "rhode island" in input_name.lower() or "uri" in input_name.lower():
            for db_name, url in self.school_urls.items():
                if "rhode island" in db_name.lower():
                    self._log('info', f"Special case match for URI: {db_name}")
                    return url
        
        # No good match found
        self._log('warning', f"No regex match found for {school_name}")
        return None
    
    def _search_school_website(self, school_url, term_season, term_year):
        """
        Search only the official school website for term dates.
        
        Args:
            school_url: School website URL
            term_season: Term season (fall, spring, etc.)
            term_year: Term year
            
        Returns:
            Dictionary with dates or None if not found
        """
        self._log('info', f"Searching school website: {school_url}")
        
        try:
            # Strategy 1: Check common calendar URL patterns
            for path in self.calendar_paths:
                # Try direct path
                calendar_url = urljoin(school_url, path)
                self._log('debug', f"Checking URL: {calendar_url}")
                
                content = self._fetch_page_content(calendar_url)
                if content:
                    dates = self._extract_dates_from_text(content, term_season, term_year)
                    if dates and dates.get('start_date') and dates.get('end_date'):
                        self._log('info', f"Found dates on {calendar_url}: {dates}")
                        return dates
                
                # Try path with year
                calendar_url_with_year = urljoin(school_url, f"{path}/{term_year}")
                self._log('debug', f"Checking URL: {calendar_url_with_year}")
                
                content = self._fetch_page_content(calendar_url_with_year)
                if content:
                    dates = self._extract_dates_from_text(content, term_season, term_year)
                    if dates and dates.get('start_date') and dates.get('end_date'):
                        self._log('info', f"Found dates on {calendar_url_with_year}: {dates}")
                        return dates
            
            # Strategy 2: Try to find calendar links on main page
            main_page_content = self._fetch_page_content(school_url)
            if main_page_content:
                calendar_links = self._extract_calendar_links(school_url, main_page_content)
                
                for link in calendar_links:
                    self._log('debug', f"Checking calendar link: {link}")
                    
                    content = self._fetch_page_content(link)
                    if content:
                        dates = self._extract_dates_from_text(content, term_season, term_year)
                        if dates and dates.get('start_date') and dates.get('end_date'):
                            self._log('info', f"Found dates on {link}: {dates}")
                            return dates
            
            # Check registrar page as a last resort
            registrar_url = urljoin(school_url, '/registrar')
            registrar_content = self._fetch_page_content(registrar_url)
            
            if registrar_content:
                # Find calendar links on registrar page
                registrar_links = self._extract_calendar_links(registrar_url, registrar_content)
                
                for link in registrar_links:
                    self._log('debug', f"Checking registrar link: {link}")
                    
                    content = self._fetch_page_content(link)
                    if content:
                        dates = self._extract_dates_from_text(content, term_season, term_year)
                        if dates and dates.get('start_date') and dates.get('end_date'):
                            self._log('info', f"Found dates on {link}: {dates}")
                            return dates
            
            self._log('info', f"No dates found on school website")
            return None
            
        except Exception as e:
            self._log('error', f"Error searching school website: {e}")
            return None
    
    def _extract_calendar_links(self, base_url, content):
        """
        Extract calendar-related links from page content.
        
        Args:
            base_url: Base URL for resolving relative links
            content: Page content
            
        Returns:
            List of calendar-related URLs
        """
        links = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Keywords that indicate academic calendar links
            keywords = ['academic calendar', 'calendar', 'important dates', 
                       'deadlines', 'academic dates', 'term dates']
            
            for a in soup.find_all('a', href=True):
                href = a.get('href', '')
                text = a.get_text().lower()
                
                if any(keyword in text for keyword in keywords):
                    # Resolve relative URLs
                    if href.startswith('/'):
                        full_url = urljoin(base_url, href)
                    elif href.startswith(('http://', 'https://')):
                        full_url = href
                    else:
                        full_url = urljoin(base_url, href)
                    
                    links.append(full_url)
        except Exception as e:
            self._log('error', f"Error extracting calendar links: {e}")
            
        return links
    
    def _fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract text content from a webpage.
        
        Args:
            url: URL to fetch
            
        Returns:
            Extracted text content, or None on error
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            # For PDF files, return empty (we don't want to guess from complex PDFs)
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                self._log('debug', f"Skipping PDF file: {url}")
                return None
            
            # Process HTML content
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Extract text
                text = soup.get_text(" ", strip=True)
                text = re.sub(r'\s+', ' ', text)
                
                return text
            except ImportError:
                # Fallback if BeautifulSoup is not available
                text = re.sub(r'<[^>]+>', ' ', response.text)
                text = re.sub(r'\s+', ' ', text)
                return text
                
        except Exception as e:
            self._log('error', f"Error fetching page content from {url}: {e}")
            return None
    
    def _extract_dates_from_text(self, text: str, term_season: str, term_year: int) -> Optional[Dict[str, str]]:
        """
        Extract term start and end dates from text with high confidence.
        
        Args:
            text: Text to analyze
            term_season: Season of the term (fall, spring, etc.)
            term_year: Year of the term
            
        Returns:
            Dictionary with start_date and end_date, or None if not certain
        """
        text = text.lower()
        
        # Look for references to our specific term
        term_indicators = [
            f"{term_season} {term_year}",
            f"{term_season} semester {term_year}",
            f"{term_season} term {term_year}"
        ]
        
        # Check if text contains references to our term
        if not any(indicator in text for indicator in term_indicators):
            # If the term isn't mentioned, we can't be certain
            return None
        
        # Extract date contexts (date with surrounding text)
        date_contexts = self._find_dates_with_context(text)
        
        # Use explicit start/end indicators to find dates
        start_date = self._find_date_by_indicators(date_contexts, self.start_indicators, term_season, term_year)
        end_date = self._find_date_by_indicators(date_contexts, self.end_indicators, term_season, term_year)
        
        # Return both dates only if we found both with high confidence
        if start_date and end_date:
            try:
                # Validate dates make sense (start before end)
                start_dt = date_parser.parse(start_date)
                end_dt = date_parser.parse(end_date)
                
                if start_dt >= end_dt:
                    self._log('warning', f"Invalid date range: {start_date} to {end_date}")
                    return None
                
                # Format dates consistently
                start_formatted = start_dt.strftime("%B %d, %Y")
                end_formatted = end_dt.strftime("%B %d, %Y")
                
                return {
                    "start_date": start_formatted,
                    "end_date": end_formatted
                }
                
            except Exception as e:
                self._log('error', f"Error parsing dates: {e}")
                return None
        
        # If we don't have both dates, return None
        return None
    
    def _find_dates_with_context(self, text: str) -> List[Dict[str, str]]:
        """
        Find dates in text and extract surrounding context.
        
        Args:
            text: Text to search for dates
            
        Returns:
            List of dictionaries with date and context
        """
        date_contexts = []
        
        # Common date patterns
        patterns = [
            # MM/DD/YYYY
            (r'\b(\d{1,2}/\d{1,2}/\d{4})\b', 50),
            # Month DD, YYYY
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', 50),
            # Abbreviated month DD, YYYY
            (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)[\.]{0,1}\s+\d{1,2},\s+\d{4}\b', 50),
            # YYYY-MM-DD
            (r'\b(\d{4}-\d{1,2}-\d{1,2})\b', 50)
        ]
        
        for pattern, context_size in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date_str = match.group(0)
                
                try:
                    # Make sure this parses as a valid date
                    parsed_date = date_parser.parse(date_str)
                    
                    # Extract context around date
                    start_idx = max(0, match.start() - context_size)
                    end_idx = min(len(text), match.end() + context_size)
                    context = text[start_idx:end_idx]
                    
                    date_contexts.append({
                        "date": date_str,
                        "context": context.lower()
                    })
                except Exception:
                    # Skip invalid dates
                    continue
        
        return date_contexts
    
    def _find_date_by_indicators(self, date_contexts: List[Dict[str, str]], 
                               indicators: List[str],
                               term_season: str,
                               term_year: int) -> Optional[str]:
        """
        Find a date by checking its context for indicator phrases.
        
        Args:
            date_contexts: List of dates with contexts
            indicators: Phrases that indicate a date type (e.g., "classes begin")
            term_season: Season of the term
            term_year: Year of the term
            
        Returns:
            Date string if found with high confidence, None otherwise
        """
        candidates = []
        
        for date_context in date_contexts:
            date_str = date_context["date"]
            context = date_context["context"]
            
            # Check if context mentions both our term and an indicator
            term_mentioned = f"{term_season}" in context and str(term_year) in context
            
            # If the term season and year are mentioned near the date
            if term_mentioned:
                # Check for indicator phrases
                for indicator in indicators:
                    if indicator in context:
                        try:
                            # Parse and validate date
                            parsed_date = date_parser.parse(date_str)
                            
                            # Check if year matches term year (strict validation)
                            if parsed_date.year == term_year:
                                candidates.append({
                                    "date": date_str,
                                    "parsed": parsed_date,
                                    "indicator": indicator
                                })
                        except Exception:
                            # Skip invalid dates
                            continue
        
        # If we have candidates, return the one with highest confidence
        if candidates:
            # Sort by how specific the indicator is (longer phrases have more context)
            candidates.sort(key=lambda x: len(x["indicator"]), reverse=True)
            return candidates[0]["date"]
        
        return None