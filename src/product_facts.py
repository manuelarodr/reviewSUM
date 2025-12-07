"""
Data structures and helpers for feature-grounded review summaries.

This module defines the core types used to represent:
- Feature-level aggregate facts derived from customer reviews
- Grounded summary text and per-claim metadata
- A simple sentence index over reviews for downstream feature detection

Steps 1–2 of the implementation plan focus on:
- Defining these data models (ProductFacts, FeatureFact, Claim)
- Providing a sentence index builder from the existing reviews DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import re


@dataclass
class Claim:
    """
    A single claim made in a summary about a product feature.

    The counts and supporting_review_ids fields are intended to be
    derived from the underlying feature facts / sentence-level analysis.
    """

    text: str
    feature_name: Optional[str] = None
    review_count: int = 0
    total_reviews: int = 0
    supporting_review_ids: List[str] = field(default_factory=list)


@dataclass
class FeatureFact:
    """
    Aggregated information about how reviews discuss a specific feature.
    """

    name: str
    description: str = ""

    # How many reviews mention this feature (at least once)
    review_count: int = 0

    # Basic sentiment breakdown at the feature level
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0

    # A subset of review IDs and example sentences used as evidence
    supporting_review_ids: List[str] = field(default_factory=list)
    evidence_sentences: List[str] = field(default_factory=list)


@dataclass
class SummaryObject:
    """
    Human-readable summary text plus structured claims metadata.
    """

    text: str = ""
    claims: List[Claim] = field(default_factory=list)


@dataclass
class ProductFacts:
    """
    Top-level container for all feature-grounded information about a product.

    This object is designed to be serializable (e.g., to JSON) so it can be
    cached and reused by both the summarization UI and Q&A components.
    """

    product_id: str
    total_reviews: int
    features: List[FeatureFact] = field(default_factory=list)
    summary: SummaryObject = field(default_factory=SummaryObject)


@dataclass
class SentenceRecord:
    """
    A single sentence extracted from a review.

    This is the basic unit used for feature detection and feature-level
    sentiment analysis in later steps of the implementation plan.
    """

    review_id: str
    sentence_id: int
    text: str


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def build_sentence_index(reviews_df: pd.DataFrame) -> List[SentenceRecord]:
    """
    Build a sentence-level index from a reviews DataFrame.

    Args:
        reviews_df: DataFrame with at least 'review_id' and 'text' columns.

    Returns:
        List of SentenceRecord objects, one per non-empty sentence.
    """
    records: List[SentenceRecord] = []

    if reviews_df is None or reviews_df.empty:
        return records

    for _, row in reviews_df.iterrows():
        review_id = str(row.get("review_id", "") or "")
        text = row.get("text", "") or ""

        if not review_id or not text:
            continue

        # Basic sentence splitting; this can be upgraded later if needed.
        raw_sentences = _SENTENCE_SPLIT_RE.split(text.strip())
        sentence_index = 0

        for sentence in raw_sentences:
            sentence = (sentence or "").strip()
            if not sentence:
                continue

            records.append(
                SentenceRecord(
                    review_id=review_id,
                    sentence_id=sentence_index,
                    text=sentence,
                )
            )
            sentence_index += 1

    return records


def build_sentence_dataframe(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper around build_sentence_index that returns a DataFrame.

    Columns:
        - review_id: ID of the source review
        - sentence_id: index of the sentence within the review (0-based)
        - text: sentence text
    """
    records = build_sentence_index(reviews_df)
    if not records:
        return pd.DataFrame(columns=["review_id", "sentence_id", "text"])

    data = [
        {"review_id": r.review_id, "sentence_id": r.sentence_id, "text": r.text}
        for r in records
    ]
    return pd.DataFrame(data)

