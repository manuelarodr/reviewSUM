import json
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.filtering import AdvancedReviewFilter, FilteringCriteria
from src.sentence_index import build_or_load_sentence_index
from src.summarizer import create_summarizer
from src.utils.data_loader import reviews_to_dataframe

# Load sample data
with open('amasum-5productsample/0062820680.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== LOADING DATA ===")
reviews_df = reviews_to_dataframe(data['customer_reviews'])
print(f"Loaded {len(reviews_df)} reviews")

print("\n=== APPLYING FILTERING ===")
criteria = FilteringCriteria(token_limit=4000)

product_meta = data.get("product_meta", {}) or {}
product_id = product_meta.get("id") or Path("amasum-5productsample/0062820680.json").stem
sentences_df, sentence_embeddings = build_or_load_sentence_index(product_id, reviews_df)
review_id_to_idx = {rid: idx for idx, rid in enumerate(reviews_df["review_id"])}
sentence_to_review_mapping = [
    review_id_to_idx.get(rid, -1) for rid in sentences_df["review_id"].tolist()
]

filterer = AdvancedReviewFilter(criteria)
filtered_df, filter_stats = filterer.filter_reviews(
    reviews_df,
    sentence_embeddings=sentence_embeddings,
    sentence_to_review_mapping=sentence_to_review_mapping,
)
print(f"Filtered to {len(filtered_df)} reviews")

# Summarization without theme extraction
print("\n=== RUNNING SUMMARIZER ===")
# Create summarizer and run it
product_name = data.get('product_meta', {}).get('title', 'product')
summarizer = create_summarizer()
result = summarizer.summarize(filtered_df, product_name=product_name)

print("\n=== RAW LLM OUTPUT ===")
print("=" * 80)
print(result['raw_output'])
print("=" * 80)

print("\n=== PARSED FINAL OUTPUT ===")
print(json.dumps(result, indent=2))
