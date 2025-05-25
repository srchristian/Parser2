# test_import.py
import sys
import os

# Print Python path for debugging
print("Python path:", sys.path)
print("Current directory:", os.getcwd())

try:
    # Try to import the module directly
    from utils.syllabus_extractor import SyllabusExtractor
    print("Import successful!")
    print("SyllabusExtractor found at:", SyllabusExtractor.__module__)
except ImportError as e:
    print(f"Import failed: {e}")
    # Show what's in the utils directory
    if os.path.exists("utils"):
        print("Contents of utils directory:", os.listdir("utils"))
        if os.path.exists("utils/syllabus_extractor.py"):
            print("syllabus_extractor.py exists in utils directory!")
        else:
            print("syllabus_extractor.py does NOT exist in utils directory!")