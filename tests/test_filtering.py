"""
Test suite for the embedding-based filtering module.
"""

import numpy as np
import pandas as pd
import pytest

from src.filtering import AdvancedReviewFilter, FilteringCriteria, create_filtering_criteria


def test_filtering_criteria_defaults():
    criteria = FilteringCriteria()
    assert criteria.token_limit == 4000
    assert criteria.tokens_per_char == 0.25
    assert criteria.helpful_boost == 0.5
    assert criteria.use_verified is None
    assert criteria.use_useful is None


def test_user_selection_filters():
    reviews_df = pd.DataFrame(
        [
            {"review_id": "r1", "text": "verified helpful", "verified": True, "helpful_votes": 2, "rating": 5},
            {"review_id": "r2", "text": "unverified helpful", "verified": False, "helpful_votes": 3, "rating": 4},
            {"review_id": "r3", "text": "verified not useful", "verified": True, "helpful_votes": 0, "rating": 3},
        ]
    )
    embeddings = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype="float32")
    mapping = [0, 1, 2]

    filterer = AdvancedReviewFilter(create_filtering_criteria(use_verified=True, use_useful=True))
    filtered_df, stats = filterer.filter_reviews_with_selection(
        reviews_df,
        sentence_embeddings=embeddings,
        sentence_to_review_mapping=mapping,
    )

    assert len(filtered_df) == 1
    assert filtered_df.loc[0, "review_id"] == "r1"
    assert stats["after_selection_count"] == 1
    assert stats["selection_applied"]["verified_only"] == 1
    assert stats["selection_applied"]["useful_only"] == 1


def test_embedding_ranking_respects_token_limit():
    reviews_df = pd.DataFrame(
        [
            {"review_id": "r1", "text": "abcdefghijkl", "verified": True, "helpful_votes": 1, "rating": 5},
            {"review_id": "r2", "text": "mnopqrstuvwx", "verified": True, "helpful_votes": 0, "rating": 4},
            {"review_id": "r3", "text": "shorttxt", "verified": True, "helpful_votes": 0, "rating": 3},
        ]
    )
    # Two reviews close to centroid, one further; r1 and r3 should fit token budget (3 + 2 tokens).
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype="float32",
    )
    mapping = [0, 1, 2]

    filterer = AdvancedReviewFilter(create_filtering_criteria(token_limit=5))
    filtered_df, stats = filterer.filter_reviews_with_selection(
        reviews_df,
        sentence_embeddings=embeddings,
        sentence_to_review_mapping=mapping,
    )

    selected_ids = filtered_df["review_id"].tolist()
    assert selected_ids == ["r1", "r3"]
    assert stats["after_ranking_count"] == 2
    assert stats["token_usage"] == 5.0
    assert "rating_5" in stats["rating_distribution"]
    assert stats["retention_rate"] == pytest.approx(2 / 3)
