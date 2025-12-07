"""
Embedding-based review filtering for Chain-of-Density summarization.

This module filters reviews in four stages:
- Optional user selection (verified/useful flags)
- Token budget check (early return if under budget)
- Semantic scoring via sentence embeddings aggregated to review level
- Selection of top-scoring reviews until the token budget is met
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FilteringCriteria:
    """Configuration for embedding-based filtering."""

    # User selection
    use_verified: Optional[bool] = None
    use_useful: Optional[bool] = None

    # Token management
    token_limit: int = 4000
    tokens_per_char: float = 0.25

    # Ranking
    helpful_boost: float = 0.5


class AdvancedReviewFilter:
    """Embedding-first review filter for the Chain-of-Density pipeline."""

    def __init__(self, criteria: Optional[FilteringCriteria] = None):
        self.criteria = criteria or FilteringCriteria()

    def normalize_column_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Map canonical column names to actual DataFrame columns.

        Returns:
            Dict[str, str]: Mapping of canonical names to DataFrame column names.
        """
        candidates = {
            "review_id": ["review_id", "id"],
            "text": ["text", "review_text", "content"],
            "rating": ["rating", "stars"],
            "verified": ["verified", "verified_purchase"],
            "helpful_votes": ["helpful_votes", "helpful", "useful_votes"],
        }

        lower_map = {col.lower(): col for col in df.columns}
        resolved: Dict[str, str] = {}

        for canon, options in candidates.items():
            for opt in options:
                match = lower_map.get(opt.lower())
                if match:
                    resolved[canon] = match
                    break

        return resolved

    def estimate_tokens(self, text: str) -> float:
        """Rough token estimate based on character count."""
        if isinstance(text, str):
            length = len(text)
        elif pd.isna(text):
            length = 0
        else:
            length = len(str(text))
        return float(length) * float(self.criteria.tokens_per_char)

    def aggregate_sentence_embeddings(
        self,
        reviews_df: pd.DataFrame,
        sentence_embeddings: np.ndarray,
        sentence_to_review_mapping: List[int],
    ) -> np.ndarray:
        """
        Aggregate sentence embeddings to review-level embeddings using mean pooling.

        Args:
            reviews_df: DataFrame of reviews (index expected to align with mapping)
            sentence_embeddings: (n_sentences, embedding_dim)
            sentence_to_review_mapping: list mapping sentence idx -> review idx

        Returns:
            np.ndarray of shape (n_reviews, embedding_dim)
        """
        if len(sentence_embeddings) != len(sentence_to_review_mapping):
            raise ValueError("Sentence embeddings and mapping length must match.")

        n_reviews = len(reviews_df)
        if n_reviews == 0:
            return np.zeros((0, 0), dtype="float32")

        if sentence_embeddings.size == 0:
            # No sentences to aggregate; return zeros
            return np.zeros((n_reviews, 0), dtype="float32")

        embedding_dim = sentence_embeddings.shape[1]
        review_embeddings = np.zeros((n_reviews, embedding_dim), dtype="float32")
        counts = np.zeros(n_reviews, dtype=np.int32)

        index_to_pos = {idx: pos for pos, idx in enumerate(reviews_df.index)}

        for sent_idx, review_idx in enumerate(sentence_to_review_mapping):
            pos = index_to_pos.get(review_idx)
            if pos is None or pos >= n_reviews or pos < 0:
                continue
            review_embeddings[pos] += sentence_embeddings[sent_idx]
            counts[pos] += 1

        for i in range(n_reviews):
            if counts[i] > 0:
                review_embeddings[i] /= float(counts[i])

        return review_embeddings

    def calculate_semantic_importance(
        self,
        review_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate semantic importance as closeness to the centroid (normalized to 0-1).
        """
        n_reviews = review_embeddings.shape[0]
        if n_reviews == 0 or review_embeddings.size == 0:
            return np.zeros(n_reviews, dtype="float32")

        # Normalize embeddings to unit length to use cosine similarity.
        norms = np.linalg.norm(review_embeddings, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1.0, norms)
        normalized = review_embeddings / safe_norms

        centroid = normalized.mean(axis=0, keepdims=True)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        similarities = np.dot(normalized, centroid.T).reshape(-1)
        similarities = np.clip(similarities, -1.0, 1.0)

        # Convert to a 0-1 score where 1 means most similar to centroid.
        min_sim, max_sim = float(similarities.min()), float(similarities.max())
        if max_sim > min_sim:
            scores = (similarities - min_sim) / (max_sim - min_sim)
        else:
            scores = np.zeros_like(similarities, dtype="float32")

        return scores.astype("float32")

    def add_helpful_boost(
        self,
        reviews_df: pd.DataFrame,
        semantic_scores: pd.Series,
        helpful_col: Optional[str],
    ) -> pd.Series:
        """
        Add a helpful-vote-based boost to semantic scores.
        """
        if helpful_col and helpful_col in reviews_df.columns:
            votes = reviews_df[helpful_col].fillna(0)
        else:
            votes = pd.Series([0] * len(reviews_df), index=reviews_df.index)

        if votes.max() > 0:
            normalized_votes = (votes / votes.max()).clip(0, 1)
        else:
            normalized_votes = pd.Series([0] * len(reviews_df), index=reviews_df.index)

        boost = normalized_votes * float(self.criteria.helpful_boost)
        return semantic_scores + boost

    def select_reviews_by_score(
        self,
        reviews_df: pd.DataFrame,
        scores: pd.Series,
        token_limit: int,
        text_col: str,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Select top-scoring reviews without exceeding the token limit.
        """
        sorted_idx = scores.sort_values(ascending=False).index
        selected_indices: List[int] = []
        tokens_used = 0.0

        for idx in sorted_idx:
            est_tokens = self.estimate_tokens(reviews_df.at[idx, text_col])
            if est_tokens <= 0:
                continue
            if tokens_used + est_tokens > token_limit:
                continue
            selected_indices.append(idx)
            tokens_used += est_tokens

        if not selected_indices and len(reviews_df) > 0:
            # Fallback: pick the shortest review if nothing fits.
            lengths = reviews_df[text_col].fillna("").apply(len)
            shortest_idx = lengths.idxmin()
            est_tokens = self.estimate_tokens(reviews_df.at[shortest_idx, text_col])
            if est_tokens <= token_limit:
                selected_indices.append(shortest_idx)
                tokens_used = est_tokens

        final_df = reviews_df.loc[selected_indices].reset_index(drop=True)
        return final_df, tokens_used

    def filter_reviews_with_selection(
        self,
        reviews_df: pd.DataFrame,
        sentence_embeddings: np.ndarray,
        sentence_to_review_mapping: List[int],
        use_verified: Optional[bool] = None,
        use_useful: Optional[bool] = None,
        token_limit: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Filter reviews using user selection + embedding ranking.
        """
        working_df = reviews_df.copy().reset_index(drop=True)
        original_count = len(working_df)
        limit = int(token_limit or self.criteria.token_limit)

        col_map = self.normalize_column_names(working_df)
        text_col = col_map.get("text", "text")
        verified_col = col_map.get("verified")
        helpful_col = col_map.get("helpful_votes")
        rating_col = col_map.get("rating")

        if text_col not in working_df.columns:
            raise ValueError("Input DataFrame must include a text column for token estimation.")

        stats: Dict[str, Any] = {
            "original_count": original_count,
            "after_selection_count": 0,
            "after_ranking_count": 0,
            "token_usage": 0.0,
            "token_limit": limit,
            "selection_applied": {
                "verified_only": 0,
                "useful_only": 0,
                "exclude_verified": 0,
                "exclude_useful": 0,
            },
            "rating_distribution": {},
            "retention_rate": 0.0,
        }

        # Stage 1: User selection filters.
        selection_verified = (
            self.criteria.use_verified if use_verified is None else use_verified
        )
        selection_useful = (
            self.criteria.use_useful if use_useful is None else use_useful
        )

        if selection_verified is not None and verified_col:
            before = len(working_df)
            if selection_verified:
                working_df = working_df[working_df[verified_col] == True]
                stats["selection_applied"]["verified_only"] = before - len(working_df)
            else:
                working_df = working_df[working_df[verified_col] != True]
                stats["selection_applied"]["exclude_verified"] = before - len(working_df)

        if selection_useful is not None:
            before = len(working_df)
            helpful_series = (
                working_df[helpful_col].fillna(0)
                if helpful_col
                else pd.Series([0] * len(working_df), index=working_df.index)
            )
            if selection_useful:
                working_df = working_df[helpful_series > 0]
                stats["selection_applied"]["useful_only"] = before - len(working_df)
            else:
                working_df = working_df[helpful_series <= 0]
                stats["selection_applied"]["exclude_useful"] = before - len(working_df)

        stats["after_selection_count"] = len(working_df)

        if len(working_df) == 0:
            return working_df, stats

        # Stage 2: Token budget check (early return if within budget).
        total_tokens = float(working_df[text_col].fillna("").apply(self.estimate_tokens).sum())
        if total_tokens <= limit:
            stats["after_ranking_count"] = len(working_df)
            stats["token_usage"] = total_tokens
            stats["retention_rate"] = (
                len(working_df) / original_count if original_count else 0.0
            )
            if rating_col and rating_col in working_df.columns:
                dist = working_df[rating_col].value_counts().sort_index()
                stats["rating_distribution"] = {
                    f"rating_{int(rating)}": {
                        "count": int(count),
                        "proportion": float(count / len(working_df)),
                    }
                    for rating, count in dist.items()
                }
            return working_df.reset_index(drop=True), stats

        # Stage 3: Embedding-based ranking.
        review_embeddings = self.aggregate_sentence_embeddings(
            working_df, sentence_embeddings, sentence_to_review_mapping
        )
        semantic_scores = self.calculate_semantic_importance(review_embeddings)
        semantic_series = pd.Series(semantic_scores, index=working_df.index)
        final_scores = self.add_helpful_boost(working_df, semantic_series, helpful_col)

        # Stage 4: Selection until token limit.
        selected_df, tokens_used = self.select_reviews_by_score(
            working_df, final_scores, limit, text_col
        )

        stats["after_ranking_count"] = len(selected_df)
        stats["token_usage"] = tokens_used
        stats["retention_rate"] = (
            len(selected_df) / original_count if original_count else 0.0
        )

        if rating_col and rating_col in selected_df.columns and len(selected_df) > 0:
            dist = selected_df[rating_col].value_counts().sort_index()
            stats["rating_distribution"] = {
                f"rating_{int(rating)}": {
                    "count": int(count),
                    "proportion": float(count / len(selected_df)),
                }
                for rating, count in dist.items()
            }

        return selected_df, stats

    def filter_reviews(
        self,
        reviews_df: pd.DataFrame,
        sentence_embeddings: Optional[np.ndarray] = None,
        sentence_to_review_mapping: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Backward-compatible wrapper that requires embeddings.
        """
        if sentence_embeddings is None or sentence_to_review_mapping is None:
            raise ValueError("sentence_embeddings and sentence_to_review_mapping are required.")
        return self.filter_reviews_with_selection(
            reviews_df,
            sentence_embeddings=sentence_embeddings,
            sentence_to_review_mapping=sentence_to_review_mapping,
            **kwargs,
        )


def create_filtering_criteria(
    token_limit: int = 4000,
    helpful_boost: float = 0.5,
    tokens_per_char: float = 0.25,
    use_verified: Optional[bool] = None,
    use_useful: Optional[bool] = None,
) -> FilteringCriteria:
    """
    Helper to build FilteringCriteria with common defaults.
    """
    return FilteringCriteria(
        token_limit=token_limit,
        helpful_boost=helpful_boost,
        tokens_per_char=tokens_per_char,
        use_verified=use_verified,
        use_useful=use_useful,
    )


# Backward compatibility alias
CredibilityFilter = AdvancedReviewFilter
