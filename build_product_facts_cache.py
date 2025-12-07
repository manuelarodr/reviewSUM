#!/usr/bin/env python3
"""
Build and cache ProductFacts for the AMASUM sample products.

For each product JSON in amasum-5productsample:
  - If a cached ProductFacts file exists, reuse it (unless --force is set).
  - Otherwise:
      * load reviews
      * compute feature-level facts (local models only)
      * optionally call Groq once to generate a feature-grounded summary
      * save ProductFacts to data/cache/product_facts/<product_id>.json

"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Ensure environment variables (e.g., GROQ_API_KEY) are loaded
load_dotenv()

from src.utils.data_loader import load_product_data, reviews_to_dataframe
from src.feature_sentiment import build_product_facts_from_features
from src.feature_summary_groq import summarize_product_facts_with_groq
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
    summarize: bool = True,
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

    # Local feature detection + sentiment
    print("Computing feature-level facts locally...")
    product_facts = build_product_facts_from_features(
        product_id=product_id,
        reviews_df=reviews_df,
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
        "--no-summary",
        action="store_true",
        help="Skip Groq summarization; only compute feature facts.",
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
            summarize=not args.no_summary,
        )


if __name__ == "__main__":
    main()

