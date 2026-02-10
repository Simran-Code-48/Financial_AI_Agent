import os
from pathlib import Path
import streamlit as st

# Try to get API keys from Streamlit secrets first, then from environment
try:
    # For Streamlit Cloud
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", "")
except:
    # For local development
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# File paths
RAW_DATA_PATH = "data/raw/statement.csv"
PREPROCESSED_PATH = "data/processed/preprocessed.csv"
RULE_CATEGORIZED_PATH = "data/processed/categorized_rule_based.csv"
FINAL_CATEGORIZED_PATH = "data/processed/categorized_final.csv"
LLM_CATEGORIZED_PATH = "data/processed/categorized_llm_enriched.csv"
REPORTS_PATH = "data/reports/"
CATEGORIES_JSON_PATH = "categories.json"

# LLM settings
LLM_THRESHOLD = 0.7

# Categories
CATEGORY_OTHER = "OTHER"
SUBCATEGORY_UNCATEGORIZED = "UNCATEGORIZED"


# Add date-based naming for versioning
from datetime import datetime
def get_timestamped_filename(base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.csv"

# Processing settings
BATCH_SIZE = 30  # For LLM batch processing
MAX_RETRIES = 3   # For API call retries
API_DELAY = 1     # Delay between API calls in seconds