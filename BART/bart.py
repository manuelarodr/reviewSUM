# -*- coding: utf-8 -*-
"""
Credibility-Aware Review Summarizer (Backend-only, safe for long inputs)
"""

import os, re, json, time, warnings
from collections import Counter
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from transformers.utils import logging

# =======================
# Config
# =======================
INPUT_FILE = "All_Beauty.jsonl"
OUTPUT_FILE = "credibility_summaries_test_2.csv"

MODEL_NAME = "facebook/bart-large-cnn"
BATCH_SIZE = 32
SAVE_EVERY = 250
MIN_REVIEWS = 50
MAX_INPUT_CHARS = 4000
FILTER_MODE = "strict"       # "strict"=only verified；"inclusive"=all reviews
MIN_REVIEW_LEN = 30
MAX_CSV_ROWS = 5000

PROMPT = (
    "You are summarizing a batch of product reviews.\n"
    "Rules:\n"
    "- Summarize collectively (customers as a group), neutral and factual.\n"
    "- DO NOT use first-person words (I, my, me, we, our).\n"
    "- No external facts; only use information present in the input.\n"
    "- Prefer patterns/themes over one-off anecdotes.\n"
    "- Compress redundancy; keep 1–2 sentences.\n\n"
    "Input (reviews chunk):\n"
    )

# =======================
# Quiet logs
# =======================
warnings.filterwarnings("ignore", message="Both `max_new_tokens`")
warnings.filterwarnings("ignore", message="Your max_length is set to")
logging.set_verbosity_error()

# =======================
# Load
# =======================
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(rows)

df = load_jsonl(INPUT_FILE)
if "rating" not in df.columns:
    df["rating"] = np.nan
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

title = df["title"].fillna("")
text  = df["text"].fillna("")
df["review_text"] = (title + ". " + text).str.strip()
df = df[df["review_text"].str.len() > MIN_REVIEW_LEN].copy()

if "verified_purchase" in df.columns:
    df["verified"] = df["verified_purchase"].fillna(False).astype(bool)
elif "verified" in df.columns:
    df["verified"] = df["verified"].fillna(False).astype(bool)
else:
    df["verified"] = True

if "helpful_vote" not in df.columns:
    df["helpful_vote"] = 0
df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0)

# =======================
# Filtering
# =======================
SPAM_PHRASES = [
    "free sample","gifted","sponsored","promotion","in exchange for",
    "discounted for review","ad partnership","paid partnership","collaboration",
    "sponsored review"
]
def is_spam(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SPAM_PHRASES)

def normalize_for_dup(s: str) -> str:
    return re.sub(r"\W+", "", s.lower())

def remove_personal_sentences(text: str) -> str:
    return re.sub(r'\b(I|my|mine|me|we|our|us)\b[^.!?]*[.!?]', '', text, flags=re.IGNORECASE)

df = df[~df["review_text"].apply(is_spam)].copy()

seen = set()
mask = []
for s in df["review_text"].tolist():
    key = normalize_for_dup(s)
    if key in seen:
        mask.append(False)
    else:
        mask.append(True)
        seen.add(key)
df = df[mask].copy()

df["clean_text"] = df["review_text"].apply(remove_personal_sentences)

if FILTER_MODE == "strict":
    df = df[df["verified"] == True].copy()

print(f"Reviews after filtering: {len(df)}")

# =======================
# Credibility (0~1)
# =======================
def credibility_score(row) -> float:
    score = 1.0
    if not bool(row.get("verified", True)):
        score -= 0.5
    hv = float(row.get("helpful_vote", 0) or 0)
    score += min(hv / 50.0, 0.35)
    if is_spam(row.get("review_text", "")):
        score -= 0.4
    if len(row.get("review_text","")) > 300:
        score += 0.05
    return max(0.0, min(1.0, score))

df["credibility_score"] = df.apply(credibility_score, axis=1)

# =======================
# Aggregate by product
# =======================
stats = (
    df.groupby("parent_asin", dropna=False)
      .agg(
          avg_rating=("rating","mean"),
          review_count=("rating","count"),
          credibility_score=("credibility_score","mean")
      )
      .reset_index()
)
texts = (
    df.groupby("parent_asin", dropna=False)["clean_text"]
      .apply(lambda x: " ".join(x))
      .reset_index()
)
grouped = texts.merge(stats, on="parent_asin", how="left")
grouped = grouped[grouped["review_count"] >= MIN_REVIEWS].reset_index(drop=True)
print(f"🧾 Products to summarize (>= {MIN_REVIEWS} reviews): {len(grouped)}")

# =======================
# Themes & helpers
# =======================
THEMES = {
    "quality":   ["quality","durable","sturdy","material","build","craft","design"],
    "value":     ["price","worth","expensive","cheap","deal","overpriced","value"],
    "delivery":  ["shipping","delivery","arrived","delay","late","package"],
    "content":   ["article","topic","section","issue","story","coverage","writing"],
    "usability": ["easy","hard","use","setup","install","intuitive","comfortable"],
    "battery":   ["battery","charge","charging","life","power"],
    "noise":     ["loud","quiet","noise","noisy","silent"],
    "scent":     ["smell","scent","fragrance","odor"],
    "fit":       ["size","fit","fitting","tight","loose"],
    "safety":    ["safety","hazard","danger","overheat","burn"]
}
POS_WORDS = set(["good","great","excellent","amazing","love","works","easy","smooth","well","best","perfect","reliable","recommend","worth"])
NEG_WORDS = set(["bad","poor","terrible","awful","hate","broken","hard","difficult","waste","disappointing","noisy","expensive","slow","late","smell"])

def count_word_boundary(text: str, word: str) -> int:
    return len(re.findall(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))

def theme_counts(text: str) -> dict:
    low = text.lower()
    return {k: sum(count_word_boundary(low, w) for w in words) for k, words in THEMES.items()}

def theme_strings(counts: dict) -> tuple[str, str]:
    total = sum(counts.values()) or 1
    items = [(k, v, int(round(100.0 * v / total))) for k, v in counts.items() if v > 0]
    items.sort(key=lambda x: x[1], reverse=True)
    counts_str = ", ".join([f"{k.capitalize()} ({v})" for k, v, _ in items]) or "N/A"
    pct_str    = ", ".join([f"{k.capitalize()} {p}%" for k, _, p in items]) or "N/A"
    return counts_str, pct_str

def split_sentences(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 10][:800]

def score_sentence_keywords(sent: str) -> int:
    lw = sent.lower()
    theme_score = sum(count_word_boundary(lw, w) for words in THEMES.values() for w in words)
    pos_score   = sum(count_word_boundary(lw, w) for w in POS_WORDS)
    neg_score   = sum(count_word_boundary(lw, w) for w in NEG_WORDS)
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

# =======================
# Summarizer + tokenizer (safe)
# =======================
device = 0 if torch.cuda.is_available() else -1
print(f"Using {'GPU' if device == 0 else 'CPU'} for summarization")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, model_max_length=1024, truncation=True, use_fast=True
)
summarizer = pipeline(
    "summarization",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device=device,
    dtype=torch.float32
)

def _tok_len(txt: str) -> int:
    return len(tokenizer.encode(txt, add_special_tokens=False))

def _chunk_by_tokens(text: str, max_tokens: int = 900) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if not s: 
            continue
        candidate = (cur + " " + s).strip() if cur else s
        if _tok_len(candidate) <= max_tokens:
            cur = candidate
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)
    safe = []
    for ch in chunks:
        if _tok_len(ch) <= max_tokens:
            safe.append(ch)
        else:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            for i in range(0, len(ids), max_tokens):
                safe.append(tokenizer.decode(ids[i:i+max_tokens]))
    return safe

def summarize_long_text(long_text: str) -> str:
    text = long_text.strip()[:MAX_INPUT_CHARS]
    parts = _chunk_by_tokens(text, max_tokens=900)

    partial = []
    for p in parts:
        t = (PROMPT + p)
        w = len(t.split())
        max_len = min(140, max(48, int(w * 0.6)))
        min_len = max(32, int(max_len * 0.5))
        try:
            s = summarizer(t, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)[0]["summary_text"]
        except Exception:
            torch.cuda.empty_cache()
            cpu_sum = pipeline("summarization", model=MODEL_NAME, tokenizer=tokenizer, device=-1, dtype=torch.float32)
            s = cpu_sum(t, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)[0]["summary_text"]
        partial.append(s.strip())

    combined = " ".join(partial)
    final_input = (PROMPT + combined)[:3000]
    w2 = len(final_input.split())
    max_len2 = min(160, max(60, int(w2 * 0.6)))
    min_len2 = max(40, int(max_len2 * 0.5))
    final = summarizer(final_input, max_length=max_len2, min_length=min_len2, do_sample=False, truncation=True)[0]["summary_text"].strip()
    final = re.sub(r'\b(\w+\s+\w+)\s+\1\b', r'\1', final)
    if not final or "You are summarizing" in final or len(final) < 40:
        final = summarizer(final_input, max_length=120, min_length=48, do_sample=False, truncation=True)[0]["summary_text"].strip()
    return final

# =======================
# Resume checkpoint
# =======================
done = set()
if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE)
    if "parent_asin" in existing.columns:
        done = set(existing["parent_asin"].astype(str))
    print(f"Resuming from checkpoint: {len(done)} already summarized")
else:
    existing = pd.DataFrame(columns=[
        "parent_asin","avg_rating","review_count","credibility_score",
        "themes_counts","themes_pct","summary","evidence","pos_snippets","neg_snippets",
        "formatted_summary"
    ])

grouped = grouped[~grouped["parent_asin"].astype(str).isin(done)].reset_index(drop=True)
print(f"Remaining products: {len(grouped)}")

# =======================
# Main loop
# =======================
t0 = time.time()
buffer = []

for start in tqdm(range(0, len(grouped), BATCH_SIZE)):
    batch = grouped.iloc[start:start+BATCH_SIZE]
    for _, row in batch.iterrows():
        asin = row["parent_asin"]
        agg_text = str(row["clean_text"])[:MAX_INPUT_CHARS]

        counts = theme_counts(agg_text)
        counts_str, pct_str = theme_strings(counts)

        reps = get_representative_sentences(agg_text, top_k=3)
        pos_s, neg_s = get_sentiment_snippets(agg_text, top_k=2)

        summary = summarize_long_text(agg_text)

        formatted = (
            f"Based on {int(row['review_count'])} "
            f"{'verified ' if FILTER_MODE=='strict' else ''}reviews "
            f"(avg rating {row['avg_rating']:.1f}/5, credibility {row['credibility_score']:.2f}). "
            f"Top themes: {pct_str}. "
            f"{summary}"
        )

        buffer.append({
            "parent_asin": asin,
            "avg_rating": round(float(row["avg_rating"]), 3) if pd.notna(row["avg_rating"]) else None,
            "review_count": int(row["review_count"]),
            "credibility_score": round(float(row["credibility_score"]), 3) if pd.notna(row["credibility_score"]) else None,
            "themes_counts": counts_str,
            "themes_pct": pct_str,
            "summary": summary,
            "evidence": " | ".join(reps),
            "pos_snippets": " | ".join(pos_s),
            "neg_snippets": " | ".join(neg_s),
            "formatted_summary": formatted
        })

    if ((start + BATCH_SIZE) % SAVE_EVERY == 0) or (start + BATCH_SIZE >= len(grouped)):
        part = pd.DataFrame(buffer)
        updated = pd.concat([existing, part], ignore_index=True)
        updated.drop_duplicates(subset="parent_asin", keep="last", inplace=True)
        
        if len(updated) >= MAX_CSV_ROWS:
            updated = updated.iloc[:MAX_CSV_ROWS].copy()
        
        updated.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        existing = updated
        buffer = []
        torch.cuda.empty_cache()
        print(f"Progress saved: {len(existing)} rows")

        if len(existing) >= MAX_CSV_ROWS:
            print(f"Reached {MAX_CSV_ROWS} rows. Stopping early.")
            break

t1 = time.time()
print(f"Done in {(t1 - t0)/60:.1f} minutes. Output -> {OUTPUT_FILE}")
