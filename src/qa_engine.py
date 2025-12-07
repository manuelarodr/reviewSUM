"""
Local Q&A helpers for answering questions about a product.

Two modes are supported:
- Entity-based Q&A using Chain-of-Density entities (legacy)
- Feature-based Q&A using ProductFacts (preferred)

Both modes avoid additional LLM calls and rely on precomputed
statistics plus example review sentences.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .product_facts import ProductFacts, FeatureFact
from .sentence_index import build_or_load_sentence_index, embed_texts


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for simple matching."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _token_set(text: str) -> set:
    """Convert text to a set of simple alphanumeric tokens."""
    text = _normalize(text)
    if not text:
        return set()
    tokens = re.findall(r"[a-z0-9]+", text)
    return set(tokens)


def _score_entity_relevance(question: str, entity_name: str) -> int:
    """
    Simple relevance score between the user question and an entity name.

    Heuristics:
    - Overlap in word tokens
    - Substring matches
    """
    q_tokens = _token_set(question)
    e_tokens = _token_set(entity_name)
    if not e_tokens:
        return 0

    overlap = len(q_tokens & e_tokens)

    q_norm = _normalize(question)
    e_norm = _normalize(entity_name)

    # Bonus if entity name appears as a substring in the question
    if e_norm and e_norm in q_norm:
        overlap += 1

    return overlap


def _find_relevant_entities(
    question: str, entities: List[Dict[str, Any]], max_entities: int = 3
) -> List[Dict[str, Any]]:
    """Rank entities by relevance to the question using simple lexical heuristics."""
    scored: List[Tuple[int, Dict[str, Any]]] = []

    for entity in entities:
        name = entity.get("entity", "")
        score = _score_entity_relevance(question, name)
        if score <= 0:
            continue

        # Prefer entities supported by more reviews when scores tie
        review_count = int(entity.get("review_count", 0) or 0)
        scored.append((score * 1000 + review_count, entity))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:max_entities]]


def _extract_evidence_sentences(
    entity_name: str,
    supporting_review_ids: List[str],
    reviews_df: pd.DataFrame,
    max_sentences: int = 3,
) -> List[str]:
    """
    Extract representative sentences from supporting reviews that mention the entity.
    """
    if reviews_df.empty or not supporting_review_ids:
        return []

    review_map = {row["review_id"]: row for _, row in reviews_df.iterrows()}
    evidence: List[str] = []
    entity_norm = _normalize(entity_name)

    for review_id in supporting_review_ids:
        if len(evidence) >= max_sentences:
            break

        review = review_map.get(review_id)
        if not review:
            continue

        text = review.get("text", "") or ""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        # Prefer sentences that explicitly mention the entity text
        chosen_sentence = None
        for sentence in sentences:
            sent_norm = _normalize(sentence)
            if not sent_norm:
                continue
            if entity_norm and entity_norm in sent_norm:
                chosen_sentence = sentence.strip()
                break

        # Fallback: first non-empty sentence
        if not chosen_sentence:
            for sentence in sentences:
                sent_norm = _normalize(sentence)
                if sent_norm:
                    chosen_sentence = sentence.strip()
                    break

        if chosen_sentence:
            evidence.append(chosen_sentence)

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for s in evidence:
        if s not in seen:
            seen.add(s)
            unique.append(s)
        if len(unique) >= max_sentences:
            break

    return unique


def answer_question_from_entities(
    question: str,
    summary_result: Dict[str, Any],
    reviews_df: pd.DataFrame,
    max_entities: int = 2,
) -> Dict[str, Any]:
    """
    Answer a natural language question using precomputed entities and review evidence.

    This Q&A is fully local: it does not call any LLM and instead uses:
    - lexical matching between the question and entity names
    - supporting reviews for evidence sentences

    Returns a dictionary with:
    - 'answer': str         Human-readable answer text
    - 'entities_used': list Entities that contributed to the answer
    - 'evidence': list      Example sentences from reviews
    """
    entities: List[Dict[str, Any]] = summary_result.get("entities", []) or []
    if not entities or reviews_df.empty:
        return {
            "answer": "I could not find any entity-level information for this product's reviews.",
            "entities_used": [],
            "evidence": [],
        }

    relevant_entities = _find_relevant_entities(question, entities, max_entities=max_entities)
    if not relevant_entities:
        return {
            "answer": "I could not find much in the reviews that directly relates to that question.",
            "entities_used": [],
            "evidence": [],
        }

    total_reviews = len(reviews_df)
    parts: List[str] = []
    all_evidence: List[str] = []

    for entity in relevant_entities:
        name = entity.get("entity", "")
        review_count = int(entity.get("review_count", 0) or 0)
        percentage = float(entity.get("percentage", 0.0) or 0.0)
        supporting_review_ids = list(entity.get("supporting_reviews", []))

        if total_reviews > 0:
            parts.append(
                f"{review_count} of {total_reviews} reviews "
                f"(about {percentage:.0%}) mention {name}."
            )
        else:
            parts.append(f"{review_count} reviews mention {name}.")

        evidence = _extract_evidence_sentences(
            name, supporting_review_ids, reviews_df, max_sentences=2
        )
        all_evidence.extend(evidence)

    answer_text = " ".join(parts)

    if all_evidence:
        quoted = " ".join(f"“{s}”" for s in all_evidence[:3])
        answer_text = f"{answer_text} For example, reviewers say things like {quoted}"

    return {
        "answer": answer_text,
        "entities_used": relevant_entities,
        "evidence": all_evidence[:3],
    }


def _score_feature_relevance(question: str, feature_name: str, description: str = "") -> int:
    """
    Relevance score between a user question and a feature name/description.

    Heuristics:
    - Overlap in word tokens for feature name
    - Substring match for feature name
    - Small bonus for overlaps with description
    """
    q_tokens = _token_set(question)
    name_tokens = _token_set(feature_name)
    desc_tokens = _token_set(description)

    if not name_tokens:
        return 0

    overlap_name = len(q_tokens & name_tokens)
    overlap_desc = len(q_tokens & desc_tokens)

    score = overlap_name * 3 + overlap_desc  # favour direct name matches

    q_norm = _normalize(question)
    name_norm = _normalize(feature_name)
    if name_norm and name_norm in q_norm:
        score += 2

    return score


def _is_count_question(question: str) -> bool:
    """
    Heuristic check whether the user is explicitly asking for
    counts, proportions, or how common something is.
    """
    q = _normalize(question)
    count_keywords = [
        "how many",
        "what percentage",
        "what percent",
        "what fraction",
        "what proportion",
        "what share",
        "number of reviews",
        "how often",
        "how frequently",
        "how common",
        "most common",
        "most talked about",
        "a lot of",
        "do many",
        "do a lot of",
        "%",
    ]
    return any(kw in q for kw in count_keywords)


def answer_question_from_product_facts(
    question: str,
    product_facts: ProductFacts,
    max_features: int = 2,
) -> Dict[str, Any]:
    """
    Answer a natural language question using ProductFacts.

    Uses:
    - lexical matching between the question and feature names/descriptions
    - feature-level counts (positive/negative/neutral)
    - evidence_sentences from each FeatureFact

    Returns:
        {
          "answer": str,
          "features_used": [FeatureFact, ...],
          "evidence": [str, ...],
        }
    """
    features: List[FeatureFact] = getattr(product_facts, "features", []) or []
    total_reviews = getattr(product_facts, "total_reviews", 0) or 0

    if not features or total_reviews == 0:
        return {
            "answer": "I do not have feature-level information for this product's reviews yet.",
            "features_used": [],
            "evidence": [],
        }

    # Rank features by relevance to the question
    scored: List[Tuple[int, FeatureFact]] = []
    for feat in features:
        base_score = _score_feature_relevance(question, feat.name, feat.description)
        if base_score <= 0:
            continue
        # Prefer features that more reviews talk about
        importance = int(feat.review_count or 0)
        scored.append((base_score * 1000 + importance, feat))

    if not scored:
        return {
            "answer": "I could not find much in the reviews that directly relates to that question.",
            "features_used": [],
            "evidence": [],
        }

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen_features = [f for _, f in scored[: max_features or 2]]

    answer_parts: List[str] = []
    all_evidence: List[str] = []

    for feat in chosen_features:
        rc = int(feat.review_count or 0)
        pos = int(feat.positive_count or 0)
        neg = int(feat.negative_count or 0)
        neu = int(feat.neutral_count or 0)

        # Decide sentiment description
        if rc == 0:
            sentiment_text = "is not really discussed in the reviews."
        else:
            if neg > 0 and pos > 0:
                sentiment_label = "mixed"
            elif neg > 0 and pos == 0:
                sentiment_label = "mostly negative"
            elif pos > 0 and neg == 0:
                sentiment_label = "mostly positive"
            else:
                sentiment_label = "neutral"

            sentiment_text = (
                f"is {sentiment_label}: {pos} positive, {neg} negative, {neu} neutral"
            )

        if total_reviews > 0 and rc > 0:
            answer_parts.append(
                f"{rc} of {total_reviews} reviews mention {feat.name}, and the sentiment {sentiment_text}."
            )
        else:
            answer_parts.append(
                f"For {feat.name}, the sentiment {sentiment_text}"
            )

        # Collect evidence sentences (already curated in FeatureFact)
        for s in feat.evidence_sentences:
            if s and s not in all_evidence:
                all_evidence.append(s)
            if len(all_evidence) >= 5:
                break

    answer_text = " ".join(answer_parts)

    if all_evidence:
        quoted = " ".join(f"“{s}”" for s in all_evidence[:3])
        answer_text = (
            f"{answer_text} For example, reviewers say things like {quoted}"
        )

    return {
        "answer": answer_text,
        "features_used": chosen_features,
        "evidence": all_evidence[:3],
    }


# ---------------------------------------------------------------------------
# Embedding-based Q&A over raw sentences
# ---------------------------------------------------------------------------

def answer_question_with_embeddings(
    question: str,
    product_facts: ProductFacts,
    reviews_df: pd.DataFrame,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Answer a question by retrieving semantically similar sentences.

    Behaviour:
    - If the question is explicitly about counts/proportions, defer to
      feature-based counts via answer_question_from_product_facts.
    - Otherwise, use sentence embeddings to find the most relevant
      review snippets and compose a narrative answer grounded in them.

    Returns:
        {
          "answer": str,
          "evidence": [str, ...],
        }
    """
    if not question or not question.strip():
        return {
            "answer": "Please enter a question about this product.",
            "evidence": [],
        }

    # For explicitly quantitative questions, reuse feature-level counts.
    if _is_count_question(question):
        result = answer_question_from_product_facts(
            question=question,
            product_facts=product_facts,
        )
        return {
            "answer": result.get("answer", ""),
            "evidence": result.get("evidence", []) or [],
        }

    # Build or load sentence-level index and embeddings for this product
    product_id = getattr(product_facts, "product_id", "") or ""
    sentences_df, embeddings = build_or_load_sentence_index(product_id, reviews_df)

    if sentences_df.empty or embeddings.size == 0:
        return {
            "answer": "I do not have sentence-level information available for this product yet.",
            "evidence": [],
        }

    # Embed the question
    q_vec = embed_texts([question])
    if q_vec.size == 0:
        return {
            "answer": "I could not embed this question to search the reviews.",
            "evidence": [],
        }

    # Cosine similarity between all sentence embeddings and the question
    scores = cosine_similarity(embeddings, q_vec).reshape(-1)

    # Select top-k sentences with highest similarity, ignoring extremely low scores
    if top_k <= 0:
        top_k = 5

    # Filter out very low similarity to avoid random matches
    min_sim_threshold = 0.15
    candidate_indices = scores.argsort()[::-1]  # descending
    selected_indices = [
        idx
        for idx in candidate_indices
        if scores[idx] >= min_sim_threshold
    ][:top_k]

    if not selected_indices:
        return {
            "answer": "I could not find much in the reviews that directly answers that question.",
            "evidence": [],
        }

    selected = sentences_df.iloc[selected_indices].copy()
    selected["score"] = scores[selected_indices]

    # Basic sentiment tendency among the selected snippets
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

    # Compose a short narrative answer
    answer_lines: List[str] = []
    answer_lines.append(
        f"Looking at the most relevant review snippets, the overall tone is {overall_sentiment}."
    )

    evidence_sentences: List[str] = []
    for _, row in selected.sort_values("score", ascending=False).iterrows():
        text = str(row.get("text", "") or "")
        if text and text not in evidence_sentences:
            evidence_sentences.append(text)
        if len(evidence_sentences) >= top_k:
            break

    if evidence_sentences:
        answer_lines.append(
            "For example, reviewers say things like "
            + " ".join(f'"{s}"' for s in evidence_sentences[:3])
        )

    answer_text = " ".join(answer_lines)

    return {
        "answer": answer_text,
        "evidence": evidence_sentences[:3],
    }


# ---------------------------------------------------------------------------
# Updated feature-based Q&A behaviour
# ---------------------------------------------------------------------------

def answer_question_from_product_facts(  # type: ignore[no-redef]
    question: str,
    product_facts: ProductFacts,
    max_features: int = 2,
) -> Dict[str, Any]:
    """
    Answer a natural language question using ProductFacts.

    Behaviour:
    - By default, produce a narrative answer grounded in example review sentences,
      without surfacing explicit counts.
    - If the question is explicitly about quantities (e.g., \"how many\", \"what percentage\"),
      include feature-level counts and sentiment breakdown.

    Returns:
        {
          "answer": str,
          "features_used": [FeatureFact, ...],
          "evidence": [str, ...],
        }
    """
    wants_counts = _is_count_question(question)

    features: List[FeatureFact] = getattr(product_facts, "features", []) or []
    total_reviews = getattr(product_facts, "total_reviews", 0) or 0

    if not features or total_reviews == 0:
        return {
            "answer": "I do not have feature-level information for this product's reviews yet.",
            "features_used": [],
            "evidence": [],
        }

    # Rank features by relevance to the question
    scored: List[Tuple[int, FeatureFact]] = []
    for feat in features:
        base_score = _score_feature_relevance(question, feat.name, feat.description)
        if base_score <= 0:
            continue
        importance = int(feat.review_count or 0)
        scored.append((base_score * 1000 + importance, feat))

    if not scored:
        return {
            "answer": "I could not find much in the reviews that directly relates to that question.",
            "features_used": [],
            "evidence": [],
        }

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen_features = [f for _, f in scored[: max_features or 2]]

    all_evidence: List[str] = []

    if wants_counts:
        # Counts-focused answer: explicitly mention how many reviews talk about the feature.
        parts: List[str] = []

        for feat in chosen_features:
            rc = int(feat.review_count or 0)
            pos = int(feat.positive_count or 0)
            neg = int(feat.negative_count or 0)
            neu = int(feat.neutral_count or 0)

            if rc == 0:
                parts.append(
                    f"Very few reviews talk directly about {feat.name}."
                )
                continue

            if neg > 0 and pos > 0:
                sentiment_label = "mixed"
            elif neg > 0 and pos == 0:
                sentiment_label = "mostly negative"
            elif pos > 0 and neg == 0:
                sentiment_label = "mostly positive"
            else:
                sentiment_label = "neutral"

            if total_reviews > 0:
                parts.append(
                    f"About {rc} of {total_reviews} reviews mention {feat.name}, "
                    f"and the sentiment is {sentiment_label}: "
                    f"{pos} positive, {neg} negative, {neu} neutral."
                )
            else:
                parts.append(
                    f"{rc} reviews mention {feat.name}, with {sentiment_label} sentiment "
                    f"({pos} positive, {neg} negative, {neu} neutral)."
                )

            for s in feat.evidence_sentences:
                if s and s not in all_evidence:
                    all_evidence.append(s)
                if len(all_evidence) >= 5:
                    break

        answer_text = " ".join(parts)

    else:
        # Narrative-first answer: describe how people feel and ground it in example sentences.
        sections: List[str] = []

        for feat in chosen_features:
            rc = int(feat.review_count or 0)
            pos = int(feat.positive_count or 0)
            neg = int(feat.negative_count or 0)
            neu = int(feat.neutral_count or 0)

            if rc == 0:
                # Skip features that effectively have no support.
                continue

            if neg > 0 and pos > 0:
                sentiment_phrase = "mixed (some positive and some negative)"
            elif neg > 0 and pos == 0:
                sentiment_phrase = "mostly negative"
            elif pos > 0 and neg == 0:
                sentiment_phrase = "mostly positive"
            else:
                sentiment_phrase = "fairly neutral"

            lines: List[str] = []
            lines.append(
                f"For {feat.name}, reviewers generally have {sentiment_phrase} impressions."
            )

            feature_examples: List[str] = []
            for s in feat.evidence_sentences:
                if s and s not in all_evidence:
                    all_evidence.append(s)
                    feature_examples.append(s)
                if len(feature_examples) >= 2 or len(all_evidence) >= 5:
                    break

            if feature_examples:
                quoted = " ".join(f'"{s}"' for s in feature_examples)
                lines.append(f"For example: {quoted}")

            sections.append(" ".join(lines))

        if not sections:
            answer_text = (
                "I found some features related to your question, "
                "but they are not strongly discussed in the reviews."
            )
        else:
            answer_text = "\n\n".join(sections)

    return {
        "answer": answer_text,
        "features_used": chosen_features,
        "evidence": all_evidence[:3],
    }
