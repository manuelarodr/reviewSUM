"""
Embedding-based Q&A over review sentences for a single product.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .product_facts import ProductFacts
from .sentence_index import build_or_load_sentence_index, embed_texts


def answer_question_with_embeddings(
    question: str,
    product_facts: ProductFacts,
    reviews_df: pd.DataFrame,
    top_k: int = 4,
) -> Dict[str, any]:
    """
    Answer a question by retrieving semantically similar review snippets.

    Uses sentence embeddings to find the most relevant snippets and returns
    a short narrative answer plus full review texts as evidence (deduped).
    """
    if not question or not question.strip():
        return {"answer": "Please enter a question about this product.", "evidence": []}

    review_lookup: Dict[str, str] = {}
    if (
        isinstance(reviews_df, pd.DataFrame)
        and not reviews_df.empty
        and "review_id" in reviews_df
        and "text" in reviews_df
    ):
        review_lookup = (
            reviews_df.assign(review_id=reviews_df["review_id"].astype(str))
            .set_index("review_id")["text"]
            .apply(lambda t: t if isinstance(t, str) else str(t or ""))
            .to_dict()
        )

    product_id = getattr(product_facts, "product_id", "") or ""
    sentences_df, embeddings = build_or_load_sentence_index(product_id, reviews_df)

    if sentences_df.empty or embeddings.size == 0:
        return {
            "answer": "I do not have sentence-level information available for this product yet.",
            "evidence": [],
        }

    q_vec = embed_texts([question])
    if q_vec.size == 0:
        return {"answer": "I could not embed this question to search the reviews.", "evidence": []}

    scores = cosine_similarity(embeddings, q_vec).reshape(-1)

    if top_k <= 0:
        top_k = 5

    min_sim_threshold = 0.2
    candidate_indices = scores.argsort()[::-1]
    selected_indices = [idx for idx in candidate_indices if scores[idx] >= min_sim_threshold][:top_k]

    if not selected_indices:
        return {
            "answer": "I could not find much in the reviews that directly answers that question.",
            "evidence": [],
        }

    selected = sentences_df.iloc[selected_indices].copy()
    selected["score"] = scores[selected_indices]

    # Sentiment summary
    sentiments = selected.get("sentiment", pd.Series([], dtype=str)).astype(str)
    pos = int((sentiments == "positive").sum())
    neg = int((sentiments == "negative").sum())
    neu = int((sentiments == "neutral").sum())

    if neg > 0 and pos > 0:
        overall_sentiment = "mixed (some positive and some negative)"
    elif neg > 0 and pos == 0:
        overall_sentiment = "mostly negative"
    elif pos > 0 and neg == 0:
        overall_sentiment = "mostly positive"
    else:
        overall_sentiment = "fairly neutral"

    answer_lines: List[str] = [
        f"Looking at the most relevant review snippets, the overall tone is {overall_sentiment}."
    ]

    evidence_sentences: List[str] = []
    sorted_selected = selected.sort_values("score", ascending=False)
    for _, row in sorted_selected.iterrows():
        text = str(row.get("text", "") or "")
        if text and text not in evidence_sentences:
            evidence_sentences.append(text)
        if len(evidence_sentences) >= top_k:
            break

    evidence_reviews: List[str] = []
    seen_review_ids: set[str] = set()
    max_review_evidence = min(top_k, 3)
    for _, row in sorted_selected.iterrows():
        review_id = str(row.get("review_id", "") or "")
        if not review_id or review_id in seen_review_ids:
            continue
        seen_review_ids.add(review_id)
        full_review = review_lookup.get(review_id, "").strip()
        if full_review:
            evidence_reviews.append(full_review)
        else:
            fallback_sentence = str(row.get("text", "") or "")
            if fallback_sentence:
                evidence_reviews.append(fallback_sentence)
        if len(evidence_reviews) >= max_review_evidence:
            break

    if evidence_sentences:
        answer_lines.append(
            "For example, reviewers say things like "
            + " ".join(f'"{s}"' for s in evidence_sentences[:3])
        )

    return {"answer": " ".join(answer_lines), "evidence": evidence_reviews}
