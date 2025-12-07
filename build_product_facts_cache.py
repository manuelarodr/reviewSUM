#!/usr/bin/env python3
"""
Build and cache ProductFacts for the AMASUM sample products (Step 7).

For each product JSON in amasum-5productsample:
  - If a cached ProductFacts file exists, reuse it (unless --force is set).
  - Otherwise:
      * load reviews
      * compute feature-level facts using CoD entities + embeddings
      * save ProductFacts to data/cache/product_facts/<product_id>.json

This script is intended to be run occasionally (e.g., during experiments),
not on every request.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Ensure environment variables (e.g., GROQ_API_KEY) are loaded
load_dotenv()

from src.utils.data_loader import load_product_data, reviews_to_dataframe
from src.filtering import AdvancedReviewFilter, FilteringCriteria
from src.sentence_index import build_or_load_sentence_index
from src.summarizer import create_summarizer
from src.cod_product_facts import build_product_facts_from_cod
from src.product_facts_io import load_product_facts, save_product_facts


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
    token_limit: int = 4000,
    similarity_threshold: float = 0.3,
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

    # Sentence index (cached) for embeddings/sentiment
    sentences_df, sentence_embeddings = build_or_load_sentence_index(
        str(product_id), reviews_df
    )

    # Embedding-based filtering to respect token budget
    review_id_to_idx = {rid: idx for idx, rid in enumerate(reviews_df["review_id"])}
    sentence_to_review_mapping = [
        review_id_to_idx.get(rid, -1) for rid in sentences_df["review_id"].tolist()
    ]
    filterer = AdvancedReviewFilter(FilteringCriteria(token_limit=token_limit))
    filtered_df, filter_stats = filterer.filter_reviews(
        reviews_df,
        sentence_embeddings=sentence_embeddings,
        sentence_to_review_mapping=sentence_to_review_mapping,
    )

    if filtered_df.empty:
        print("Warning: No reviews remain after filtering; skipping.")
        return

    # CoD summarization over filtered reviews
    product_name = data.get("product_meta", {}).get("title", "product")
    summarizer = create_summarizer()
    summary_result = summarizer.summarize(
        filtered_df, themes_data={}, product_name=product_name
    )

    # Ground entities to build ProductFacts
    product_facts = build_product_facts_from_cod(
        product_id=product_id,
        reviews_df=filtered_df,
        summary_result=summary_result,
        similarity_threshold=similarity_threshold,
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
    parser.add_argument(
        "--token-limit",
        type=int,
        default=4000,
        help="Token budget for filtering before CoD.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.3,
        help="Cosine similarity threshold for grounding entities to sentences.",
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
            token_limit=args.token_limit,
            similarity_threshold=args.similarity_threshold,
        )


if __name__ == "__main__":
    main()
