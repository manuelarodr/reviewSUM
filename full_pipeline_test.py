import json
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.summarizer import create_summarizer
from src.utils.data_loader import reviews_to_dataframe
from src.filtering import AdvancedReviewFilter, FilteringCriteria
from src.themes import create_theme_extractor

# Load sample data
with open('amasum-5productsample/0062820680.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== LOADING DATA ===")
reviews_df = reviews_to_dataframe(data['customer_reviews'])
print(f"Loaded {len(reviews_df)} reviews")

print("\n=== APPLYING FILTERING ===")
# Apply TF-IDF filtering
criteria = FilteringCriteria(
    token_limit=4000,
    min_reviews_per_rating=2,
    max_reviews_per_rating=50,
    tfidf_weight=0.7
)

filterer = AdvancedReviewFilter(criteria)
filtered_df, filter_stats = filterer.filter_reviews(reviews_df)
print(f"Filtered to {len(filtered_df)} reviews")

print("\n=== EXTRACTING THEMES ===")
# Extract themes
theme_extractor = create_theme_extractor(language="english")
themes_data = theme_extractor.extract_themes(filtered_df)
print(f"Extracted themes: {themes_data}")

print("\n=== RUNNING SUMMARIZER ===")
# Create summarizer and run it
product_name = data.get('product_meta', {}).get('title', 'product')
summarizer = create_summarizer()
result = summarizer.summarize_with_raw_output(filtered_df, themes_data, product_name)

print("\n=== RAW LLM OUTPUT ===")
print("=" * 80)
print(result['raw_output'])
print("=" * 80)

print("\n=== PARSED FINAL OUTPUT ===")
print(json.dumps(result, indent=2))
