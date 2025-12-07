"""
Streamlit dashboard for review summarization and analysis (CoD + embeddings).

This UI lets you:
- Upload a product JSON (AMASUM-format)
- See basic product stats
- Generate:
  * Chain-of-Density summary from filtered reviews
  * Entity-grounded feature statistics using sentence embeddings
- Ask natural-language questions answered from review sentences
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Add project root to path for imports when running as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Load environment variables from .env at project root (e.g., GROQ_API_KEY)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.utils.data_loader import reviews_to_dataframe
from src.qa_engine import answer_question_with_embeddings
from src.filtering import AdvancedReviewFilter, FilteringCriteria
from src.sentence_index import build_or_load_sentence_index
from src.summarizer import create_summarizer
from src.cod_product_facts import build_product_facts_from_cod


st.set_page_config(
    page_title="Review Summarizer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("Review Summarizer")
    st.markdown(
        "AI-powered product review summarization using Chain-of-Density and "
        "embedding-grounded feature evidence."
    )

    with st.sidebar:
        st.header("File Upload")
        uploaded_file = st.file_uploader(
            "Upload Product Data (JSON)",
            type=["json"],
            help="Upload a JSON file containing product reviews and metadata.",
        )
        st.markdown("---")
        st.header("Filtering")
        verified_choice = st.selectbox(
            "Verified purchase filter",
            options=["All reviews", "Verified only", "Unverified only"],
            index=0,
            help="Choose whether to restrict to verified or unverified reviews.",
        )
        useful_choice = st.selectbox(
            "Helpful votes filter",
            options=["All reviews", "Helpful only", "No helpful votes"],
            index=0,
            help="Choose whether to restrict to reviews with helpful votes.",
        )

    use_verified = None
    if verified_choice == "Verified only":
        use_verified = True
    elif verified_choice == "Unverified only":
        use_verified = False

    use_useful = None
    if useful_choice == "Helpful only":
        use_useful = True
    elif useful_choice == "No helpful votes":
        use_useful = False

    if not uploaded_file:
        st.info("Please upload a product JSON file to get started.")
        return

    try:
        data = json.load(uploaded_file)
    except Exception as exc:
        st.error(f"Error reading JSON: {exc}")
        return

    reviews_df = reviews_to_dataframe(data.get("customer_reviews", []))
    if reviews_df.empty:
        st.warning("No customer reviews found in this file.")
        return

    display_product_info(data)

    # Human (website) summary, if available
    display_human_summary(data)

    # Reset cached ProductFacts when a new file is uploaded
    current_upload_name = getattr(uploaded_file, "name", None)
    last_upload_name = st.session_state.get("last_uploaded_name")
    if current_upload_name != last_upload_name:
        st.session_state["last_uploaded_name"] = current_upload_name
        st.session_state.pop("product_facts", None)
        st.session_state.pop("filter_stats", None)

    # CoD + embeddings pipeline: summary + feature stats + Q&A
    st.header("AI Summary (CoD + Embeddings)")
    if st.button("Generate Summary", type="primary"):
        product_id = infer_product_id(data, uploaded_file.name)
        try:
            with st.spinner(
                "Filtering reviews, running Chain-of-Density, and grounding entities..."
            ):
                # Build sentence index and embeddings once
                sentences_df, sentence_embeddings = build_or_load_sentence_index(
                    str(product_id), reviews_df
                )

                review_id_to_idx = {
                    rid: idx for idx, rid in enumerate(reviews_df["review_id"])
                }
                sentence_to_review_mapping = [
                    review_id_to_idx.get(rid, -1)
                    for rid in sentences_df["review_id"].tolist()
                ]

                # Filter reviews within token budget
                filterer = AdvancedReviewFilter(FilteringCriteria())
                filtered_df, filter_stats = filterer.filter_reviews(
                    reviews_df,
                    sentence_embeddings=sentence_embeddings,
                    sentence_to_review_mapping=sentence_to_review_mapping,
                    use_verified=use_verified,
                    use_useful=use_useful,
                )

                if filtered_df.empty:
                    st.warning("No reviews remain after filtering.")
                    st.session_state["product_facts"] = None
                    st.session_state["filter_stats"] = filter_stats
                else:
                    # Chain-of-Density summarization
                    summarizer = create_summarizer()
                    product_name = data.get("product_meta", {}).get("title", "product")
                    summary_result = summarizer.summarize(
                        filtered_df, themes_data={}, product_name=product_name
                    )

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
                    st.session_state["product_facts"] = product_facts
                    st.session_state["filter_stats"] = filter_stats

        except Exception as exc:
            st.warning(f"CoD pipeline failed; no AI summary is available. Details: {exc}")
            st.session_state["product_facts"] = None
            st.session_state["filter_stats"] = None

    # If we have cached ProductFacts, always display summary and Q&A
    product_facts_cached = st.session_state.get("product_facts")
    filter_stats_cached = st.session_state.get("filter_stats")
    if product_facts_cached is not None:
        display_product_facts_summary(product_facts_cached, reviews_df, filter_stats_cached)
        display_product_facts_qa(product_facts_cached, reviews_df)


def infer_product_id(data: Dict[str, Any], filename: str) -> str:
    """Infer a stable product_id for caching from metadata or filename."""
    meta = data.get("product_meta", {}) or {}
    for key in ("asin", "isbn", "id", "sku"):
        val = meta.get(key)
        if val:
            return str(val)

    specs = meta.get("specifications", {}) or {}
    for key in ("ASIN", "asin", "ISBN-10", "ISBN-13"):
        val = specs.get(key)
        if val:
            return str(val)

    stem, _ = os.path.splitext(filename or "uploaded")
    return stem


def display_product_info(data: Dict[str, Any]) -> None:
    """Display high-level product information and review stats."""
    st.header("Product Information")
    product_meta = data.get("product_meta", {}) or {}

    title = product_meta.get("title")
    if title:
        st.write(f"**Product:** {title}")

    reviews = data.get("customer_reviews", []) or []
    if not reviews:
        return

    total_reviews = len(reviews)
    avg_rating = sum(r.get("rating", 0.0) for r in reviews) / total_reviews
    helpful_count = sum(1 for r in reviews if r.get("helpful_votes", 0) > 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", total_reviews)
    with col2:
        st.metric("Average Rating", f"{avg_rating:.1f}/5")
    with col3:
        st.metric("Reviews with Helpful Votes", f"{helpful_count}/{total_reviews}")


def display_human_summary(data: Dict[str, Any]) -> None:
    """Display human-generated (website) summary, if present."""
    st.subheader("Human-Generated Summary (Website)")
    website_summaries = data.get("website_summaries", []) or []

    if not website_summaries:
        st.info("No human-generated summary available for this product.")
        return

    summary = website_summaries[0]
    verdict = summary.get("verdict")
    pros = summary.get("pros")
    cons = summary.get("cons")

    if verdict:
        st.write("**Verdict:**")
        st.write(verdict)

    if pros:
        st.write("**Pros:**")
        if isinstance(pros, list):
            for p in pros:
                st.write(f"- {p}")
        else:
            st.write(pros)

    if cons:
        st.write("**Cons:**")
        if isinstance(cons, list):
            for c in cons:
                st.write(f"- {c}")
        else:
            st.write(cons)


def display_product_facts_summary(
    product_facts, reviews_df: pd.DataFrame, filter_stats: Dict[str, Any] | None
) -> None:
    """Display CoD summary text and per-feature statistics."""
    summary = getattr(product_facts, "summary", None)
    if summary is None or not getattr(summary, "text", ""):
        st.info("No AI summary available for this product.")
        return

    st.subheader("Chain-of-Density Summary (Overview)")
    st.write(summary.text)

    if filter_stats:
        with st.expander("Filtering summary"):
            st.write(
                f"Selected {filter_stats.get('after_ranking_count', 0)} "
                f"of {filter_stats.get('original_count', 0)} reviews "
                f"({filter_stats.get('retention_rate', 0):.1%} retention)"
            )
            st.write(
                f"Token usage: {filter_stats.get('token_usage', 0.0):.0f} / "
                f"{filter_stats.get('token_limit', 0)}"
            )
            applied = []
            if filter_stats.get("selection_applied"):
                sel = filter_stats["selection_applied"]
                if sel.get("verified_only"):
                    applied.append("Verified only")
                if sel.get("exclude_verified"):
                    applied.append("Unverified only")
                if sel.get("useful_only"):
                    applied.append("Helpful only")
                if sel.get("exclude_useful"):
                    applied.append("No helpful votes")
            if applied:
                st.write("Filters: " + ", ".join(applied))

    features = getattr(product_facts, "features", []) or []
    if not features:
        st.info(
            "No feature-level statistics available "
            "(feature extraction may have failed or been skipped)."
        )
        return

    st.subheader("Feature Statistics from Reviews")
    total_reviews = getattr(product_facts, "total_reviews", 0) or 0
    review_lookup = (
        reviews_df.set_index("review_id")["text"].to_dict()
        if not reviews_df.empty and "review_id" in reviews_df
        else {}
    )

    for feat in features:
        if total_reviews:
            count_label = f"{feat.review_count} / {total_reviews} reviews"
        else:
            count_label = f"{feat.review_count} reviews"

        header = f"{feat.name} — {count_label}"
        with st.expander(header):
            st.write(
                f"Sentiment breakdown: {feat.positive_count} positive, "
                f"{feat.negative_count} negative, {feat.neutral_count} neutral."
            )

            full_reviews_shown = 0
            if getattr(feat, "supporting_review_ids", None):
                st.write("Supporting reviews (full text):")
                for rid in feat.supporting_review_ids:
                    full_text = review_lookup.get(rid)
                    if not full_text:
                        continue
                    st.markdown(f"- **{rid}**: {full_text}")
                    full_reviews_shown += 1
                    if full_reviews_shown >= 5:
                        break
            if full_reviews_shown == 0:
                st.write("No supporting reviews available.")


def display_product_facts_qa(product_facts, reviews_df: pd.DataFrame) -> None:
    """Q&A panel powered by sentence embeddings over reviews."""
    st.subheader("Ask Questions About This Product")
    question = st.text_input(
        "Ask what reviewers say about this product "
        "(e.g., 'What do people say about the images?', 'Is it good for beginners?')",
        key="qa_product_facts_question",
    )

    if not question:
        return

    # Answer immediately when a question is present (no extra button)
    with st.spinner("Searching in review sentences..."):
        qa_result = answer_question_with_embeddings(
            question=question,
            product_facts=product_facts,
            reviews_df=reviews_df,
        )

    st.markdown("**Answer:**")
    st.write(qa_result.get("answer", ""))

    evidence = qa_result.get("evidence", []) or []
    if evidence:
        st.write("Example review sentences used:")
        for i, sent in enumerate(evidence, 1):
            st.write(f"{i}. {sent}")


if __name__ == "__main__":
    main()
