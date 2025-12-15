"""
Build ProductFacts using Chain-of-Density entities grounded via embeddings.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .entity_grounding import ground_entities_with_embeddings
from .product_facts import ProductFacts, SummaryObject
from .sentence_index import build_or_load_sentence_index


def _extract_entity_names(summary_result: Dict[str, Any]) -> List[str]:
    """
    Extract entity names from a summarization result.
    Uses the `entities` list when present.
    """
    names: List[str] = []
    entities = summary_result.get("entities") or []
    for ent in entities:
        name = ent.get("entity") if isinstance(ent, dict) else ent
        if name:
            names.append(str(name))

    return names


def build_product_facts_from_cod(
    product_id: str,
    reviews_df: pd.DataFrame,
    summary_result: Dict[str, Any],
    similarity_threshold: float = 0.3,
    max_hits: int = 50,
    max_evidence: int = 5,
    entity_embeddings=None,
    sentences_df: Optional[pd.DataFrame] = None,
    sentence_embeddings=None,
) -> ProductFacts:
    """
    Build ProductFacts using CoD summary + entity grounding over embeddings.
    """
    total_reviews = int(reviews_df["review_id"].nunique()) if not reviews_df.empty else 0
    summary_text = str(summary_result.get("summary", "") or "").strip()
    entity_names = _extract_entity_names(summary_result)

    if total_reviews == 0 or not entity_names:
        return ProductFacts(
            product_id=str(product_id),
            total_reviews=total_reviews,
            features=[],
            summary=SummaryObject(text=summary_text),
        )

    if sentences_df is None or sentence_embeddings is None:
        sentences_df, sentence_embeddings = build_or_load_sentence_index(
            str(product_id), reviews_df
        )

    if (
        sentences_df is None
        or sentences_df.empty
        or sentence_embeddings is None
        or getattr(sentence_embeddings, "size", 0) == 0
    ):
        return ProductFacts(
            product_id=str(product_id),
            total_reviews=total_reviews,
            features=[],
            summary=SummaryObject(text=summary_text),
        )

    # Restrict to the selected reviews to keep grounding aligned with CoD input
    allowed_ids = set(reviews_df["review_id"].astype(str))
    mask = sentences_df["review_id"].astype(str).isin(allowed_ids)
    if mask.sum() == 0:
        return ProductFacts(
            product_id=str(product_id),
            total_reviews=total_reviews,
            features=[],
            summary=SummaryObject(text=summary_text),
        )
    if mask.sum() != len(sentences_df):
        sentences_df = sentences_df[mask].reset_index(drop=True)
        if (
            sentence_embeddings is not None
            and getattr(sentence_embeddings, "shape", (0,))[0] == len(mask)
        ):
            sentence_embeddings = sentence_embeddings[mask.values]

    feature_facts = ground_entities_with_embeddings(
        entity_names,
        sentences_df,
        sentence_embeddings=sentence_embeddings,
        similarity_threshold=similarity_threshold,
        max_hits=max_hits,
        max_evidence=max_evidence,
        entity_embeddings=entity_embeddings,
    )

    return ProductFacts(
        product_id=str(product_id),
        total_reviews=total_reviews,
        features=feature_facts,
        summary=SummaryObject(text=summary_text),
    )
