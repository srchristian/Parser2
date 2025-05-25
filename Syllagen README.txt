Syllabus Parser Project
Overview
The Syllabus Parser is a Flask-based web application that transforms unstructured academic syllabi into structured JSON data. The extraction pipeline takes input as a file (PDF, DOCX, TXT, HTML) or a URL, converts it to normalized text, segments the content, and then extracts structured information such as course metadata, class schedules, assignments, and events. The final output can be used for calendar integration, learning management systems, or administrative reporting.

Core Architecture & Pipeline
The system follows a modular, pipeline-based architecture with a clear separation of concerns. Configuration and logging are centralized (via config.yaml and a single logger created in app.py), and the same logger instance is passed to all specialized parser modules.

1. app.py – Application Orchestrator
Purpose:

Manages the web interface and file operations.

Loads configuration from config.yaml and sets up the environment.

Creates a single logger instance (using settings from config.yaml).

Coordinates the extraction pipeline by calling helper functions (from helpers.py) and instantiating specialized parsers.

Presents results to the user and handles missing-field corrections.

Input:

File uploads (PDF, DOCX, TXT, HTML) or URL submissions.

Output:

Rendered HTML templates and JSON files containing structured data.

2. helpers.py – Utility Functions & Pipeline Processing
Purpose:

Centralizes configuration loading and environment setup.

Provides utility functions for file conversion (PDF, DOCX, TXT, HTML), JSON read/write, and text normalization.

Encapsulates the overall extraction pipeline in process_syllabus_pipeline(), which routes text to the appropriate specialized parsers.

Integration:

Called directly by app.py to reduce redundancy in file operations and pipeline orchestration.

3. syllabus_parser.py – Text Segmenter
Purpose:

Converts raw syllabus documents into clean text.

Segments the text into logical sections (e.g., course info, schedule, assignments).

Produces a parsed JSON file that organizes the syllabus for subsequent extraction.

Output:

A JSON object with keys such as course_info, schedule, assignments, exams, readings, and special_sessions.

4. syllabus_extractor.py – Data Extraction Orchestrator
Purpose:

Coordinates the specialized parsers to extract structured data from the segmented syllabus text.

Merges and validates results from individual parsers.

Provides fallback methods if a specialized parser (e.g., OpenAIParser) fails.

Output:

A JSON object containing:

class_data: Core course metadata.

event_data: A list of class session events.

task_data: A list of assignments, exams, and tasks.

metadata: Extraction process details and confidence scores.

5. Specialized Parsers (located in utils/parsers/)
Each parser handles a specific portion of the extraction:

a. openai_parser.py – AI-Powered Metadata Extractor
Purpose:

Uses OpenAI’s API to extract course metadata.

Generates targeted prompts and handles API calls (with retries and error management).

Output:

A class_data dictionary with fields such as “School Name,” “Term,” “Course Title,” etc.

b. date_parser.py – Date Normalization and Day Standardization
Purpose:

Normalizes date strings into a standard format (e.g., "January 01, 2024").

Standardizes various representations of class days (e.g., “MWF” becomes “Monday, Wednesday, Friday”).

Provides utilities for generating date ranges and determining class dates.

Note:

The advanced date inference logic has been removed to simplify the pipeline. This parser now focuses solely on normalization.

c. schedule_parser.py – Class Schedule Generator
Purpose:

Generates a complete schedule of class session events based on term dates and days of the week.

Merges generated events with any pre-existing events.

Output:

A list of event objects (each with “Event Date,” “Event Title,” “Class Time,” etc.).

d. assignment_parser.py – Assignment and Task Extractor
Purpose:

Extracts and validates assignment, exam, and task information.

Normalizes task titles and orders assignments.

Links assignments with the corresponding class session events.

Output:

A list of task objects (each with “Task Title,” “Due Date,” “Task Description,” etc.).

Data Flow Diagram
java
Copy
User Upload
   │
   ▼
app.py → (helpers.py: File Conversion, Environment Setup, etc.)
   │
   ▼
syllabus_parser.py → Text Segmentation (Parsed JSON Output)
   │
   ▼
syllabus_extractor.py → Orchestration & Data Integration
   │
   ▼
Specialized Parsers:
   ├─ openai_parser.py → Extracts course metadata (class_data)
   ├─ date_parser.py   → Normalizes dates and days of week
   ├─ schedule_parser.py → Generates class session events (event_data)
   └─ assignment_parser.py → Extracts assignments and tasks (task_data)
   │
   ▼
Final Structured Data (JSON Output)
   │
   ▼
app.py → Display & Export (e.g., Calendar .ics)
Data Structure & Sample Outputs
The final JSON output is structured with four main components:

1. class_data
Contains core course metadata.

Sample Output:

json
Copy
{
  "School Name": "University of Example",
  "Term": "Fall 2024",
  "Course Title": "CSC101: Introduction to Computer Science",
  "Instructor Name": "Dr. Jane Doe",
  "Instructor Email": "jane.doe@example.edu",
  "Class Time": "10:00 AM - 11:15 AM",
  "Time Zone": "Eastern Time",
  "Days of Week": "Monday, Wednesday, Friday",
  "Term Start Date": "August 28, 2024",
  "Term End Date": "December 15, 2024",
  "Class Start Date": "August 30, 2024",
  "Class End Date": "December 12, 2024",
  "Office Hours": "Tuesdays 2:00 PM - 4:00 PM",
  "Telephone": "555-123-4567",
  "Class Location": "Science Building, Room 101",
  "Additional": "Additional course details",
  "Date Verification": "ai_generated"
}
2. event_data
An array of class session events.

Sample Output:

json
Copy
[
  {
    "Event Date": "August 30, 2024",
    "Event Title": "CSC101: Class",
    "Class Time": "10:00 AM - 11:15 AM",
    "Class Location": "Science Building, Room 101",
    "reading": ["Chapter 1: Introduction"],
    "assignment": ["Homework #1"],
    "assignment_description": "Complete problems 1-10 from Chapter 1",
    "test": null,
    "special": null
  },
  {
    "Event Date": "September 2, 2024",
    "Event Title": "CSC101: Class",
    "Class Time": "10:00 AM - 11:15 AM",
    "Class Location": "Science Building, Room 101",
    "reading": ["Chapter 1: Introduction"],
    "assignment": [],
    "assignment_description": "",
    "test": null,
    "special": null
  }
]
3. task_data
An array of assignments, exams, or other tasks.

Sample Output:

json
Copy
[
  {
    "Task Title": "CSC101: Homework #1",
    "Due Date": "September 6, 2024",
    "Task Description": "Complete exercises 1-10 from Chapter 1"
  },
  {
    "Task Title": "CSC101: Midterm Exam",
    "Due Date": "October 18, 2024",
    "Task Description": "Exam covering Chapters 1-4"
  }
]
4. metadata
Extraction process details and confidence scores.

Sample Output:

json
Copy
{
  "extraction_time": "2024-03-15T14:32:45.123456",
  "unique_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "text_length": 15243,
  "confidence_scores": {
    "class_data": 0.85,
    "event_data": 0.75,
    "task_data": 0.70,
    "overall": 0.77
  },
  "missing_fields": [],
  "parsed_syllabus_path": "utils/parsing/syllabus_repository/parsed_syllabus/a1b2c3d4.json",
  "original_file": "original_syllabus.pdf",
  "process_time": 3.45
}
The final JSON output is assembled as:

json
Copy
{
  "class_data": { ... },
  "event_data": [ ... ],
  "task_data": [ ... ],
  "metadata": { ... }
}
Configuration & Logging
Configuration:
All configuration settings—including directory paths, file settings, Flask parameters, logging level, and OpenAI settings—are managed via config.yaml.

Centralized Logging:
A single logger is instantiated in app.py using the setup_logger() function from helpers.py. This shared logger is then passed to every parser (e.g., OpenAIParser, DateParser, ScheduleParser, AssignmentParser, and SyllabusExtractor) so that all logging messages are consolidated and consistently formatted.

Pipeline Workflow
File Upload & Conversion:

app.py accepts file uploads or URL submissions.

Helper functions in helpers.py convert documents to text and save them using directory paths from config.yaml.

Text Segmentation:

syllabus_parser.py segments the normalized text into logical sections.

Data Extraction:

syllabus_extractor.py orchestrates the extraction by calling specialized parsers:

OpenAIParser: Extracts core course metadata.

DateParser: Normalizes dates and standardizes day formats.

ScheduleParser: Generates class session events.

AssignmentParser: Extracts assignments and exam details.

Fallback methods are used when a specialized parser fails to extract a required field (e.g., if "Days of Week" is missing, the user is prompted to provide it).

Data Integration & User Feedback:

The extracted data is merged into a final JSON structure and saved.

If critical fields are missing, users are prompted to supply corrections.

The final structured data is displayed via HTML templates and can be exported (e.g., to an iCalendar file).

Troubleshooting & Contributions
Troubleshooting:

Review logs in the logs directory for detailed error messages.

Verify that the directories specified in config.yaml exist and are writable.

Ensure the OpenAI API key is correctly set (via an environment variable or in config.yaml).

Contributing:
Contributions are welcome. Please adhere to the architectural principles and coding standards outlined in this document. When revising specific pipeline components, ensure that configuration and logging are consistently integrated.