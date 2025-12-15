#!/usr/bin/env python3
"""
Build and cache ProductFacts for the AMASUM sample products.

For each product JSON in amasum-5productsample:
  - If a cached ProductFacts file exists, reuse it (unless --force is set).
  - Otherwise:
      * load reviews
      * build sentence index + embeddings
      * filter reviews to a token budget
      * run Chain-of-Density summarization (Groq)
      * ground entities to features with embeddings
      * save ProductFacts to data/cache/product_facts/<product_id>.json

"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Ensure environment variables (e.g., GROQ_API_KEY) are loaded
load_dotenv()

from src.utils.data_loader import load_product_data, reviews_to_dataframe
from src.product_facts_io import load_product_facts, save_product_facts
from src.sentence_index import build_or_load_sentence_index
from src.filtering import AdvancedReviewFilter, FilteringCriteria
from src.summarizer import create_summarizer
from src.cod_product_facts import build_product_facts_from_cod


def infer_product_id(data_path: Path, data: dict) -> str:
    """
    Derive a stable product_id from product_meta or file name.
    """
    meta = data.get("product_meta", {}) or {}
    for key in ("asin", "isbn", "id", "sku"):
        if key in meta and meta[key]:
            return str(meta[key])
    # Fallback: file stem
    return data_path.stem


def process_product_json(
    path: Path,
    *,
    force: bool = False,
) -> None:
    """
    Build or reuse ProductFacts for a single product JSON file.
    """
    print(f"\n=== Processing {path.name} ===")
    data = load_product_data(str(path))
    product_id = infer_product_id(path, data)

    # Try cache unless forcing recompute
    if not force:
        cached = load_product_facts(product_id)
        if cached is not None:
            print(f"Using cached ProductFacts for product_id={product_id}")
            return

    # Build reviews DataFrame
    reviews_df = reviews_to_dataframe(data["customer_reviews"])
    print(f"Reviews: {len(reviews_df)}")

    # Build or load sentence-level embeddings
    print("Building sentence index and embeddings...")
    sentences_df, sentence_embeddings = build_or_load_sentence_index(
        str(product_id), reviews_df
    )

    review_id_to_idx = {
        rid: idx for idx, rid in enumerate(reviews_df["review_id"])
    }
    sentence_to_review_mapping = [
        review_id_to_idx.get(rid, -1) for rid in sentences_df["review_id"].tolist()
    ]

    # Filter reviews to stay within token budget
    filterer = AdvancedReviewFilter(FilteringCriteria())
    filtered_df, filter_stats = filterer.filter_reviews(
        reviews_df,
        sentence_embeddings=sentence_embeddings,
        sentence_to_review_mapping=sentence_to_review_mapping,
    )
    print(
        f"Selected {len(filtered_df)} of {len(reviews_df)} reviews "
        f"({filter_stats.get('retention_rate', 0):.1%} retention)"
    )

    # Chain-of-Density summarization
    summarizer = create_summarizer()
    product_name = data.get("product_meta", {}).get("title", "product")
    summary_result = summarizer.summarize(
        filtered_df, product_name=product_name
    )

    # Ground entities to ProductFacts
    product_facts = build_product_facts_from_cod(
        product_id=product_id,
        reviews_df=filtered_df,
        summary_result=summary_result,
        similarity_threshold=0.3,
        max_hits=50,
        max_evidence=5,
        sentences_df=sentences_df,
        sentence_embeddings=sentence_embeddings,
    )

    # Cache result
    path_out = save_product_facts(product_facts)
    print(f"Saved ProductFacts for product_id={product_id} to {path_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and cache ProductFacts for AMASUM sample products."
    )
    parser.add_argument(
        "--data-dir",
        default="amasum-5productsample",
        help="Directory containing AMASUM product JSON files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute ProductFacts even if a cached file exists.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    print(f"Found {len(json_files)} product files in {data_dir}")

    for path in json_files:
        process_product_json(
            path,
            force=args.force,
        )


if __name__ == "__main__":
    main()
