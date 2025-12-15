#!/usr/bin/env python3
"""
CLI entry point for the embedding-based Chain-of-Density review summarizer.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from dataclasses import asdict
from dotenv import load_dotenv

# Add src directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Load environment variables from .env if present
load_dotenv()

from filtering import AdvancedReviewFilter, FilteringCriteria
from sentence_index import build_or_load_sentence_index
from summarizer import create_summarizer
from utils.data_loader import load_product_data, reviews_to_dataframe
from cod_product_facts import build_product_facts_from_cod


def _coerce_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return value.lower() == "true"


def main() -> None:
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="AI-Powered Review Summarizer")

    # Input/Output
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Filtering options
    parser.add_argument("--token-limit", type=int, default=4000, help="Token budget for review selection")
    parser.add_argument("--tokens-per-char", type=float, default=0.25, help="Approximate tokens per character")
    parser.add_argument("--helpful-boost", type=float, default=0.5, help="Maximum helpful vote boost added to scores")
    parser.add_argument(
        "--use-verified",
        choices=["true", "false"],
        help="If set, include only verified (true) or only unverified (false) reviews",
    )
    parser.add_argument(
        "--use-useful",
        choices=["true", "false"],
        help="If set, include only reviews with helpful votes (true) or exclude them (false)",
    )

    # Summarization options
    parser.add_argument("--model", dest="model_name", default="llama-3.1-8b-instant", help="Groq model name")
    parser.add_argument("--max-length", type=int, default=120, help="Maximum summary length")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")

    args = parser.parse_args()

    try:
        # Load input data
        if args.verbose:
            print(f"Loading data from {args.input}")

        data = load_product_data(args.input)

        # Convert reviews to DataFrame
        reviews_df = reviews_to_dataframe(data.get("customer_reviews", []))

        if args.verbose:
            print(f"Loaded {len(reviews_df)} reviews")

        if reviews_df.empty:
            print("Error: No reviews found in the input file.")
            sys.exit(1)

        # Filtering Step
        if args.verbose:
            print("Applying embedding-based filtering...")

        product_meta = data.get("product_meta", {}) or {}
        product_id = product_meta.get("id") or Path(args.input).stem
        sentences_df, sentence_embeddings = build_or_load_sentence_index(product_id, reviews_df)
        review_id_to_idx = {rid: idx for idx, rid in enumerate(reviews_df["review_id"])}
        sentence_to_review_mapping = [
            review_id_to_idx.get(rid, -1) for rid in sentences_df["review_id"].tolist()
        ]

        criteria = FilteringCriteria(
            token_limit=args.token_limit,
            tokens_per_char=args.tokens_per_char,
            helpful_boost=args.helpful_boost,
            use_verified=_coerce_optional_bool(args.use_verified),
            use_useful=_coerce_optional_bool(args.use_useful),
        )

        filterer = AdvancedReviewFilter(criteria)
        filtered_df, filter_stats = filterer.filter_reviews(
            reviews_df,
            sentence_embeddings=sentence_embeddings,
            sentence_to_review_mapping=sentence_to_review_mapping,
        )

        if args.verbose:
            print(
                f"Filtered to {len(filtered_df)} reviews "
                f"({(len(filtered_df)/len(reviews_df)*100):.1f}% retention)"
            )
            if "rating" in filtered_df.columns and not filtered_df.empty:
                rating_dist = filtered_df["rating"].value_counts().sort_index()
                print("Rating distribution after filtering:")
                for rating, count in rating_dist.items():
                    print(f"  {int(rating)}: {count} reviews")

        if len(filtered_df) == 0:
            print("Error: No reviews remain after filtering")
            sys.exit(1)

        # Summarization step
        if args.verbose:
            print("Generating summary with Chain-of-Density prompting...")

        product_name = product_meta.get("title", "product")

        summarizer = create_summarizer(
            model_name=args.model_name,
            max_length=args.max_length,
        )

        summary_result = summarizer.summarize(filtered_df, product_name=product_name)

        if args.verbose:
            print("Summary generated successfully")
            print(f"Identified {len(summary_result.get('entities', []))} entities")

        # Ground entities to reviews to build feature list with evidence/sentiment
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

        # Prepare output
        output_data = {
            "summary_result": summary_result,
            "filter_stats": filter_stats,
            "product_facts": asdict(product_facts),
            "config": {
                "token_limit": args.token_limit,
                "tokens_per_char": args.tokens_per_char,
                "helpful_boost": args.helpful_boost,
                "use_verified": _coerce_optional_bool(args.use_verified),
                "use_useful": _coerce_optional_bool(args.use_useful),
                "model": args.model_name,
                "max_length": args.max_length,
                "temperature": args.temperature,
            },
        }

        # Output results
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            if args.verbose:
                print(f"Results saved to {args.output}")
        else:
            # Print summary to stdout
            print("\n" + "=" * 50)
            print("SUMMARY")
            print("=" * 50)
            print(summary_result["summary"])

            if summary_result.get("entities"):
                print("\n" + "=" * 50)
                print("ENTITIES & SUPPORTING REVIEWS")
                print("=" * 50)
                for i, entity in enumerate(summary_result["entities"], 1):
                    print(f"{i}. {entity['entity']}")
                    print(f"   Iteration: {entity['iteration']}")
                    print(f"   Supporting reviews: {', '.join(entity['supporting_reviews'])}")
                    print(f"   Review count: {entity['review_count']}")
                    print(f"   Percentage: {entity['percentage']:.1%}")
                    print()

            # Show filtering stats
            print("\n" + "=" * 50)
            print("FILTERING STATISTICS")
            print("=" * 50)
            print(f"Original reviews: {filter_stats.get('original_count', 0)}")
            print(f"Selected reviews: {filter_stats.get('after_ranking_count', 0)}")
            print(f"Retention rate: {filter_stats.get('retention_rate', 0.0):.1%}")
            print(f"Token usage: {filter_stats.get('token_usage', 0.0):.0f}")

            # Show grounded features/evidence
            if product_facts.features:
                print("\n" + "=" * 50)
                print("FEATURES FROM CoD ENTITIES")
                print("=" * 50)
                for feat in product_facts.features:
                    print(f"- {feat.name}: {feat.review_count} reviews")
                    print(
                        f"  Sentiment: +{feat.positive_count} / -{feat.negative_count} / °{feat.neutral_count}"
                    )
                    if feat.evidence_sentences:
                        print("  Evidence:")
                        for sent in feat.evidence_sentences[:3]:
                            print(f"    • {sent}")

    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"Error: {exc}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
