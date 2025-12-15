# -*- coding: utf-8 -*-
"""
Core summarization logic extracted from bart.py
Can be imported and used directly without saving to CSV
"""

import os
import re
import json
import warnings
from collections import Counter
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from transformers.utils import logging

# Suppress warnings
warnings.filterwarnings("ignore", message="Both `max_new_tokens`")
warnings.filterwarnings("ignore", message="Your max_length is set to")
logging.set_verbosity_error()

# =======================
# Config
# =======================
MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_CHARS = 4000
MIN_REVIEW_LEN = 30

THEMES = {
    "quality": ["quality","excellent","poor","great","good","bad","durable","fragile","strong","weak"],
    "value":   ["price","value","worth","cheap","expensive","deal","money","cost","refund"],
    "delivery": ["shipped","delivery","shipping","package","arrived","late","fast","slow","damage"],
    "content": ["article","topic","section","issue","story","coverage","writing"],
    "usability": ["easy","hard","use","setup","install","intuitive","comfortable"],
    "battery": ["battery","charge","charging","life","power"],
    "noise": ["loud","quiet","noise","noisy","silent"],
    "scent": ["smell","scent","fragrance","odor"],
    "fit": ["size","fit","fitting","tight","loose"],
    "safety": ["safety","hazard","danger","overheat","burn"]
}

POS_WORDS = set(["good","great","excellent","amazing","love","works","easy","smooth","well","best","perfect","reliable","recommend","worth"])
NEG_WORDS = set(["bad","poor","terrible","awful","hate","broken","hard","difficult","waste","disappointing","noisy","expensive","slow","late","smell"])

# =======================
# Helper Functions
# =======================
def count_word_boundary(text: str, word: str) -> int:
    return len(re.findall(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))

def theme_counts(text: str) -> dict:
    low = text.lower()
    return {k: sum(count_word_boundary(low, w) for w in words) for k, words in THEMES.items()}

def theme_strings(counts: dict) -> tuple:
    total = sum(counts.values()) or 1
    items = [(k, v, int(round(100.0 * v / total))) for k, v in counts.items() if v > 0]
    items.sort(key=lambda x: x[1], reverse=True)
    counts_str = ", ".join([f"{k.capitalize()} ({v})" for k, v, _ in items]) or "N/A"
    pct_str = ", ".join([f"{k.capitalize()} {p}%" for k, _, p in items]) or "N/A"
    return counts_str, pct_str

def split_sentences(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 10][:800]

def score_sentence_keywords(sent: str) -> int:
    lw = sent.lower()
    theme_score = sum(count_word_boundary(lw, w) for words in THEMES.values() for w in words)
    pos_score = sum(count_word_boundary(lw, w) for w in POS_WORDS)
    neg_score = sum(count_word_boundary(lw, w) for w in NEG_WORDS)
    return theme_score + pos_score + neg_score

def get_representative_sentences(text: str, top_k: int = 3):
    sents = split_sentences(text)
    if not sents:
        return []
    scored = [(score_sentence_keywords(s), s) for s in sents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_k]]

def get_sentiment_snippets(text: str, top_k: int = 2):
    sents = split_sentences(text)
    if not sents:
        return [], []
    def pos_score(s): return sum(count_word_boundary(s.lower(), w) for w in POS_WORDS)
    def neg_score(s): return sum(count_word_boundary(s.lower(), w) for w in NEG_WORDS)
    pos_sorted = sorted(sents, key=pos_score, reverse=True)
    neg_sorted = sorted(sents, key=neg_score, reverse=True)
    return pos_sorted[:top_k], neg_sorted[:top_k]

def credibility_score(row: dict) -> float:
    score = 1.0
    if row.get('verified_purchase') or row.get('verified'):
        score += 1.0
    hv = float(row.get('helpful_vote', row.get('helpful_votes', 0)) or 0)
    score += min(0.35, hv / 50.0)
    text_len = len(str(row.get('text', '')).split())
    if text_len > 20:
        score += 0.05
    text_lower = str(row.get('text', '')).lower()
    spam_words = ['viagra', 'casino', 'lottery', 'xxx']
    if any(w in text_lower for w in spam_words):
        score -= 0.4
    return score

# =======================
# Initialize Model (lazy loading)
# =======================
_summarizer = None
_tokenizer = None

def get_summarizer():
    global _summarizer, _tokenizer
    if _summarizer is None:
        device = 0 if torch.cuda.is_available() else -1
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, model_max_length=1024, truncation=True, use_fast=True
        )
        _summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            tokenizer=_tokenizer,
            device=device,
            dtype=torch.float32
        )
    return _summarizer, _tokenizer

def _tok_len(txt: str, tokenizer) -> int:
    return len(tokenizer.encode(txt, add_special_tokens=False))

def _chunk_by_tokens(text: str, tokenizer, max_tokens: int = 900) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if not s:
            continue
        candidate = (cur + " " + s).strip() if cur else s
        if _tok_len(candidate, tokenizer) <= max_tokens:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    safe = []
    for ch in chunks:
        if _tok_len(ch, tokenizer) <= max_tokens:
            safe.append(ch)
        else:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            for i in range(0, len(ids), max_tokens):
                safe.append(tokenizer.decode(ids[i:i+max_tokens]))
    return safe

def summarize_long_text(long_text: str, summarizer, tokenizer) -> str:
    text = long_text.strip()[:MAX_INPUT_CHARS]
    parts = _chunk_by_tokens(text, tokenizer, max_tokens=900)
    
    if not parts:
        return ""
    
    summaries = []
    for part in parts:
        if len(part.split()) < 10:
            continue
        try:
            result = summarizer(part, max_length=50, min_length=20, do_sample=False)
            if result and len(result) > 0:
                summaries.append(result[0]['summary_text'])
        except:
            summaries.append(part[:100])
    
    if not summaries:
        return ""
    
    combined = " ".join(summaries)
    if len(combined.split()) <= 50:
        return combined
    
    try:
        result = summarizer(combined, max_length=50, min_length=20, do_sample=False)
        return result[0]['summary_text'] if result else combined
    except:
        return combined

# =======================
# Main Summarization Function
# =======================
def summarize_reviews(
    df: pd.DataFrame,
    filter_mode: str = "strict",
    min_reviews: int = 50,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Main function to summarize reviews
    
    Args:
        df: DataFrame with review data (must have: text, rating, parent_asin)
        filter_mode: "strict" (verified only) or "inclusive" (all)
        min_reviews: minimum reviews per product
        show_progress: show progress bar
    
    Returns:
        DataFrame with summary results
    """
    
    # Get model
    summarizer, tokenizer = get_summarizer()
    
    # Add credibility score
    df = df.copy()
    df['credibility'] = df.apply(credibility_score, axis=1)
    
    # Filter
    if filter_mode == "strict":
        df = df[df['verified_purchase'] | df['verified']]
    
    df = df[df['text'].fillna('').str.len() >= MIN_REVIEW_LEN]
    
    print(f"Reviews after filtering: {len(df)}")
    
    # Group by product
    by_product = df.groupby('parent_asin')
    products = [p for p, g in by_product if len(g) >= min_reviews]
    print(f"🧾 Products to summarize (>= {min_reviews} reviews): {len(products)}")
    
    results = []
    
    iterator = tqdm(products) if show_progress else products
    for prod_id in iterator:
        group = by_product.get_group(prod_id)
        
        # Basic stats
        avg_rating = group['rating'].mean() if 'rating' in group.columns else np.nan
        review_count = len(group)
        avg_cred = group['credibility'].mean() if 'credibility' in group.columns else np.nan
        
        # Combine all reviews
        all_text = " ".join(group['text'].fillna('').astype(str))
        
        # Summarize
        summary = summarize_long_text(all_text, summarizer, tokenizer)
        
        # Extract themes
        counts = theme_counts(all_text)
        counts_str, pct_str = theme_strings(counts)
        
        # Get evidence sentences
        evidence_sents = get_representative_sentences(all_text, top_k=3)
        evidence = " | ".join(evidence_sents)
        
        # Get sentiment snippets
        pos_snippets, neg_snippets = get_sentiment_snippets(all_text, top_k=2)
        pos_str = " | ".join(pos_snippets)
        neg_str = " | ".join(neg_snippets)
        
        # Formatted summary
        formatted = f"Based on {review_count} reviews (avg rating {avg_rating:.1f}/5, credibility {avg_cred:.2f}). Top themes: {pct_str}. {summary}"
        
        results.append({
            'parent_asin': prod_id,
            'avg_rating': avg_rating,
            'review_count': review_count,
            'credibility_score': avg_cred,
            'themes_counts': counts_str,
            'themes_pct': pct_str,
            'summary': summary,
            'evidence': evidence,
            'pos_snippets': pos_str,
            'neg_snippets': neg_str,
            'formatted_summary': formatted
        })
    
    return pd.DataFrame(results)
