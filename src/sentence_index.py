"""
Sentence-level index with embeddings and sentiment.

This module provides utilities to:
- Split reviews into sentences (reusing build_sentence_dataframe)
- Compute sentence embeddings using a local Transformer model
- Compute per-sentence sentiment labels using a local classifier
- Cache the resulting index and embeddings per product on disk
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from .product_facts import build_sentence_dataframe


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SENTENCE_EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

SENTENCE_CACHE_DIR = Path("data") / "cache" / "sentence_index"

_EMB_TOKENIZER = None
_EMB_MODEL = None
_SENTIMENT_PIPE = None


def _get_embedding_model():
    """
    Lazily load the sentence embedding model and tokenizer.
    """
    global _EMB_TOKENIZER, _EMB_MODEL
    if _EMB_TOKENIZER is not None and _EMB_MODEL is not None:
        return _EMB_TOKENIZER, _EMB_MODEL

    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(SENTENCE_EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(SENTENCE_EMBEDDING_MODEL_NAME)
    if device >= 0:
        model = model.to(device)
    model.eval()

    _EMB_TOKENIZER = tokenizer
    _EMB_MODEL = model
    return tokenizer, model


def _get_sentiment_pipeline():
    """
    Lazily create and cache a local sentiment-analysis pipeline.
    """
    global _SENTIMENT_PIPE
    if _SENTIMENT_PIPE is not None:
        return _SENTIMENT_PIPE

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    _SENTIMENT_PIPE = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU
    )
    return _SENTIMENT_PIPE


def _encode_sentences(texts, batch_size: int = 64) -> np.ndarray:
    """
    Compute sentence embeddings for a list/Series of texts.

    Uses mean pooling over token embeddings.
    """
    if len(texts) == 0:
        return np.zeros((0, 0), dtype="float32")

    tokenizer, model = _get_embedding_model()
    device = next(model.parameters()).device

    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = [str(t) if t is not None else "" for t in texts[start : start + batch_size]]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            last_hidden = outputs.last_hidden_state  # [batch, seq, hidden]

            # Mean pooling over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1)  # [batch, seq, 1]
            masked = last_hidden * mask
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = masked.sum(dim=1) / lengths

        all_embeddings.append(pooled.cpu().numpy().astype("float32"))

    return np.vstack(all_embeddings)


def embed_texts(texts) -> np.ndarray:
    """
    Public helper to embed an iterable of texts using the same model
    as the sentence index. Returns a NumPy array of shape [n, dim].
    """
    return _encode_sentences(list(texts))


def _classify_sentence_sentiment(pipe, text: str) -> str:
    """Return 'positive', 'negative', or 'neutral' sentiment for a sentence."""
    if not text or not str(text).strip():
        return "neutral"

    result = pipe(str(text), truncation=True)[0]
    label = result["label"].lower()
    if label.startswith("neg"):
        return "negative"
    if label.startswith("pos"):
        return "positive"
    return "neutral"


def build_or_load_sentence_index(
    product_id: str,
    reviews_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build or load a sentence-level index for a product, including:
    - sentence text with review_id and sentence_id
    - per-sentence sentiment label
    - sentence embeddings matrix aligned with the DataFrame rows

    Results are cached under:
        data/cache/sentence_index/<product_id>_sentences.parquet
        data/cache/sentence_index/<product_id>_embeddings.npy
    """
    SENTENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sentences_path = SENTENCE_CACHE_DIR / f"{product_id}_sentences.parquet"
    embeddings_path = SENTENCE_CACHE_DIR / f"{product_id}_embeddings.npy"

    if sentences_path.exists() and embeddings_path.exists():
        sentences_df = pd.read_parquet(sentences_path)
        embeddings = np.load(embeddings_path)

        # Basic cache validation: if the number of reviews has changed for this
        # product_id, treat the cache as stale and rebuild the index so that
        # Q&A always reflects the currently loaded dataset.
        try:
            cached_reviews = int(sentences_df["review_id"].nunique())
            current_reviews = (
                int(reviews_df["review_id"].nunique()) if not reviews_df.empty else 0
            )
        except Exception:
            cached_reviews = -1
            current_reviews = -2

        if cached_reviews == current_reviews:
            return sentences_df, embeddings
        # else fall through to rebuild index and overwrite cache

    # Build sentence-level index from reviews
    sentences_df = build_sentence_dataframe(reviews_df)
    if sentences_df.empty:
        # Store empty artifacts to avoid recomputation
        sentences_df.to_parquet(sentences_path, index=False)
        np.save(embeddings_path, np.zeros((0, 0), dtype="float32"))
        return sentences_df, np.zeros((0, 0), dtype="float32")

    # Compute sentence sentiment
    sent_pipe = _get_sentiment_pipeline()
    sentences_df = sentences_df.copy()
    sentences_df["sentiment"] = sentences_df["text"].apply(
        lambda t: _classify_sentence_sentiment(sent_pipe, t)
    )

    # Compute embeddings
    embeddings = _encode_sentences(sentences_df["text"].tolist())

    # Persist
    sentences_df.to_parquet(sentences_path, index=False)
    np.save(embeddings_path, embeddings)

    return sentences_df, embeddings
