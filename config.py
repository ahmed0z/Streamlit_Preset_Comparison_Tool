"""
Configuration settings for the Preset Comparison Tool
"""
import os
from pathlib import Path

# Application Metadata
APP_TITLE = "🔍 Preset Comparison Tool"
APP_ICON = "🔍"
VERSION = "1.0.0"
DESCRIPTION = "Intelligent comparison tool for preset values using advanced NLP and fuzzy matching"

# File Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PRESET_DB_PATH = DATA_DIR / "preset_database.xlsx"
TEMPLATE_PATH = DATA_DIR / "templates" / "input_template.xlsx"

# Expected Column Names
PRESET_COLUMNS = {
    'category': 'Category',
    'sub_category': 'Sub-Category', 
    'attribute_name': 'Attribute Name',
    'preset_values': 'Preset values'  # Note: lowercase 'v' to match actual data
}

INPUT_COLUMNS = {
    'category': 'Category',
    'sub_category': 'Sub-Category',
    'attribute_name': 'Attribute Name',
    'input_values': 'Input values'  # User's values to compare against preset values
}

# Matching Configuration
MATCHING_THRESHOLD = 0.75  # Minimum similarity score (75%)
MAX_RESULTS_PER_INPUT = 5  # Maximum matching results to return per input
EXACT_MATCH_THRESHOLD = 0.99  # Threshold for considering a match "exact"

# Fuzzy Matching Weights
FUZZY_WEIGHTS = {
    'levenshtein': 0.3,
    'jaro_winkler': 0.25,
    'token_sort_ratio': 0.25,
    'token_set_ratio': 0.2
}

# Semantic Matching Configuration
SEMANTIC_MODEL = "all-MiniLM-L6-v2"  # Lightweight model for Streamlit Cloud
SEMANTIC_WEIGHT = 0.4  # Weight for semantic similarity in final score

# File Upload Limits
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = ['xlsx', 'xls']

# UI Configuration
ITEMS_PER_PAGE = 100
SIDEBAR_WIDTH = 300

# Performance Settings
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)
CHUNK_SIZE = 1000  # Processing chunk size for large datasets
MAX_MEMORY_USAGE_MB = 500  # Memory limit for Streamlit Cloud

# Status Types
class MatchStatus:
    EXACT_MATCH = "Exact Match"
    PARTIAL_MATCH = "Partial Match"  # Changed from SIMILAR_MATCH
    NOT_FOUND = "Not Found"

# Result Columns
RESULT_COLUMNS = [
    'Category',
    'Sub-Category', 
    'Attribute Name',
    'Original Input',
    'Matched Preset Value',
    'Similarity %',
    'Comment',
    'Suggested Value',
    'Status',
    'Composite Key'
]

# Colors for UI
COLORS = {
    'exact_match': '#4CAF50',     # Green
    'partial_match': '#FF9800',   # Orange  
    'not_found': '#F44336',       # Red
    'background': '#FAFAFA',      # Light Gray
    'primary': '#1976D2'          # Blue
}

# Environment Variables
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Sample Data for Testing
SAMPLE_INPUTS = [
    "20 kg",
    "0.181 inches",
    "Protective Cap", 
    "20 kg @ 30°C",
    "0.618 L x 78.740 W x 3.307 H (15.70mm x 200.00mm x 84.00mm)"
]