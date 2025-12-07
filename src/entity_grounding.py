"""
Ground entities from a summary to reviews using embeddings.

Given a list of entity strings and a sentence-level index with embeddings,
this module finds supporting sentences via cosine similarity and aggregates
per-entity stats (review counts, sentiment breakdown, evidence sentences).
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .product_facts import FeatureFact
from .sentence_index import embed_texts


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """Row-normalize embeddings to unit length."""
    if vecs.size == 0:
        return vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def ground_entities_with_embeddings(
    entities: Iterable[str],
    sentences_df: pd.DataFrame,
    sentence_embeddings: np.ndarray,
    similarity_threshold: float = 0.3,
    max_hits: int = 50,
    max_evidence: int = 5,
    entity_embeddings: Optional[np.ndarray] = None,
) -> List[FeatureFact]:
    """
    Map entities to supporting sentences using cosine similarity over embeddings.

    Args:
        entities: Iterable of entity strings from the final CoD summary.
        sentences_df: DataFrame with at least 'review_id' and 'text' columns.
                      Optional 'sentiment' column for pos/neg/neutral labels.
        sentence_embeddings: (n_sentences, dim) embeddings aligned to sentences_df rows.
        similarity_threshold: Minimum cosine similarity to count a sentence as evidence.
        max_hits: Maximum sentences to keep per entity for stats.
        max_evidence: Maximum sentences to return as evidence examples.
        entity_embeddings: Optional precomputed entity embeddings (len(entities), dim);
                           if None, entities are embedded with the same model.

    Returns:
        List[FeatureFact] where each feature corresponds to an entity.
    """
    entity_list = [e for e in entities if isinstance(e, str) and e.strip()]
    if not entity_list or sentence_embeddings.size == 0 or sentences_df.empty:
        return []

    # Embed entities if not provided
    if entity_embeddings is None:
        entity_embeddings = embed_texts(entity_list)

    sent_vecs = _normalize(sentence_embeddings)
    ent_vecs = _normalize(entity_embeddings)

    # Cosine similarities: [n_sentences, n_entities]
    sims = np.dot(sent_vecs, ent_vecs.T)

    feature_facts: List[FeatureFact] = []
    has_sentiment = "sentiment" in sentences_df.columns

    for ent_idx, entity in enumerate(entity_list):
        entity_sims = sims[:, ent_idx]

        # Candidates above threshold
        candidate_idxs = np.where(entity_sims >= similarity_threshold)[0]
        if candidate_idxs.size == 0:
            # Keep the single best sentence if nothing crosses threshold
            best_idx = int(entity_sims.argmax())
            if entity_sims[best_idx] > 0:
                candidate_idxs = np.array([best_idx], dtype=int)
            else:
                # No signal at all
                feature_facts.append(
                    FeatureFact(
                        name=entity,
                        description="",
                        review_count=0,
                        positive_count=0,
                        negative_count=0,
                        neutral_count=0,
                        supporting_review_ids=[],
                        evidence_sentences=[],
                    )
                )
                continue

        # Sort by similarity descending and cap
        sorted_idxs = candidate_idxs[np.argsort(entity_sims[candidate_idxs])[::-1]]
        top_idxs = sorted_idxs[:max_hits]

        top_rows = sentences_df.iloc[top_idxs]
        review_ids = top_rows["review_id"].astype(str).tolist()
        unique_review_ids = list(dict.fromkeys(review_ids))  # preserve order, remove dups

        if has_sentiment:
            sentiments = top_rows["sentiment"].fillna("neutral").str.lower()
        else:
            sentiments = pd.Series(["neutral"] * len(top_rows))

        pos = int((sentiments == "positive").sum())
        neg = int((sentiments == "negative").sum())
        neu = int((sentiments == "neutral").sum())

        # Evidence sentences: top few by similarity
        evidence_sentences: List[str] = []
        for idx in top_idxs[:max_evidence]:
            text = str(sentences_df.iloc[idx].get("text", "") or "")
            if text and text not in evidence_sentences:
                evidence_sentences.append(text)

        feature_facts.append(
            FeatureFact(
                name=entity,
                description="",
                review_count=len(unique_review_ids),
                positive_count=pos,
                negative_count=neg,
                neutral_count=neu,
                supporting_review_ids=unique_review_ids,
                evidence_sentences=evidence_sentences,
            )
        )

    return feature_facts
