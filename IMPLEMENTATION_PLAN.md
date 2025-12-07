# Implementation Plan: Review Summaries + Grounded Q&A

## 1. Goals

- Generate **credible, grounded summaries** of product reviews.
  - Focus on how customers feel about the **most important features**.
  - Each claim is backed by **counts** (e.g., “30 of 120 reviews”) and **evidence sentences**.
- Enable **natural language Q&A** about a product using its reviews.
  - Answers are grounded in the same evidence and counts.
- Respect **API limits** by doing heavy work locally and running Groq calls **infrequently** (batch / offline).
- Start with the **AMASUM static dataset** and a **Streamlit prototype**, but design so it can be moved to a web service later.

---

## 2. High-Level Architecture

### 2.1 Core Idea: Shared “Feature Facts” Layer

All functionality is built around a shared per-product facts object:

```json
{
  "product_id": "...",
  "total_reviews": 1000,
  "features": [
    {
      "name": "battery life",
      "description": "How long the device runs between charges, how often you must recharge it.",
      "review_count": 40,
      "positive_count": 10,
      "negative_count": 30,
      "neutral_count": 0,
      "supporting_review_ids": ["r23", "r157", "..."],
      "evidence_sentences": [
        "The battery always dies by afternoon.",
        "Have to charge it all the time."
      ]
    }
  ],
  "summary": {
    "text": "...final human-readable summary...",
    "claims": [
      {
        "text": "Many users complain that the battery runs out quickly.",
        "feature_name": "battery life",
        "review_count": 30,
        "total_reviews": 1000,
        "supporting_review_ids": ["r23", "r157", "..."]
      }
    ]
  }
}
```

- **Summaries** render `summary.text` plus `summary.claims` (for hover counts).
- **Q&A** uses `features` + `evidence_sentences` + raw reviews as context.

### 2.2 Components

- **Data ingestion layer**
  - Loads AMASUM JSON and (later) other sources into a unified internal format.
- **Feature & sentiment extraction (local)**
  - Splits reviews into sentences.
  - Detects which feature(s) each sentence is about.
  - Runs **local** sentiment per (sentence, feature).
  - Aggregates counts per feature.
- **LLM summarization (Groq CoD)**
  - Takes feature stats + evidence and produces a readable summary text and claims mapping.
  - Runs **offline / batch**, not per user request.
- **Q&A engine**
  - Uses cached feature facts + review snippets.
  - Optionally uses a small LLM call; can also answer with templates when budget is tight.
- **UI layer (Streamlit prototype)**
  - File upload → shows product dashboard, summary with hover counts, and Q&A panel.

---

## 3. Data & Storage Design

### 3.1 Internal Data Structures (initially in memory / JSON file)

For the AMASUM prototype, store per-product data in JSON or pickled objects on disk:

- `product_reviews.json` (existing AMASUM structure).
- `product_facts.json` (computed facts + summary):

```python
ProductFacts = TypedDict("ProductFacts", {
    "product_id": str,
    "total_reviews": int,
    "features": List[FeatureFact],
    "summary": SummaryObject,
})

FeatureFact = {
    "name": str,
    "description": str,
    "review_count": int,
    "positive_count": int,
    "negative_count": int,
    "neutral_count": int,
    "supporting_review_ids": List[str],
    "evidence_sentences": List[str],
}

SummaryObject = {
    "text": str,
    "claims": List[Claim],
}

Claim = {
    "text": str,
    "feature_name": str,
    "review_count": int,
    "total_reviews": int,
    "supporting_review_ids": List[str],
}
```

Later, this can map to DB tables (e.g., `products`, `reviews`, `features`, `summary_claims`).

### 3.2 Review Index (for Q&A)

- **Local embeddings** (if added later):
  - Sentence-level embeddings: per sentence, store vector + `review_id`.
- For the first iteration (no embeddings):
  - Use simple keyword + feature matching and a TF-IDF or BM25 index over sentences.

---

## 4. Pipelines

### 4.1 Offline / Batch: Product Facts + Summary

**Goal:** For each product, build `ProductFacts` and a grounded summary, using Groq only once per product (or per refresh).

Steps:

1. **Load product data**
   - Use existing `load_product_data` and `reviews_to_dataframe`.
   - Ensure each review has `review_id`, `rating`, `text`, `timestamp`.

2. **Sentence splitting**
   - For each review, split `text` into sentences.
   - Keep `(review_id, sentence_id, sentence_text)`.

3. **Feature “schema” discovery**
   - For now, define a **small, manual feature set per domain** (e.g., electronics):
     - battery life, build quality, ease of use, sound quality, comfort, price/value, etc.
     - Each with a short **description** and some **keywords**.
   - Later, optionally:
     - Use the existing CoD summarizer to propose candidate features per product.
     - Cluster them and map them into a canonical feature list.

4. **Feature detection per sentence (local)**
   - For each sentence:
     - Use:
       - Keyword match against feature keywords.
       - (Optional) local embedding similarity between sentence and feature description.
     - Assign one or more `feature_name` tags if similarity exceeds a threshold.
   - Example: “have to charge it all the time” → `battery life`.

5. **Feature-level sentiment (local)**
   - For each `(sentence, feature_name)`:
     - Run a local sentiment classifier:
       - Simple option: rule-based using sentiment lexicons.
       - Better: `transformers.pipeline("sentiment-analysis")` model (CPU is fine).
     - Get labels: `positive`, `negative`, `neutral`, (optionally `mixed`).
   - Aggregate per **review + feature**:
     - If any sentence is strongly negative → review’s stance on that feature = negative.
     - Similarly for positive/neutral.

6. **Aggregate per product feature**
   - For each `feature_name`:
     - Count:
       - `review_count`
       - `positive_count`
       - `negative_count`
       - `neutral_count`
     - Pick a small set of `supporting_review_ids` and `evidence_sentences`:
       - Top N representative sentences (balanced across positive/negative).
   - Construct the `features` list.

7. **Generate summary with Groq CoD (offline)**
   - Use `ChainOfDensitySummarizer` but change the prompt/usage:
     - Instead of sending all reviews, send:
       - High-level feature stats (`review_count`, sentiment counts).
       - A few evidence sentences per feature.
   - Prompt the model to:
     - Focus on **most important features** (by review_count).
     - Include both pros and cons.
     - Do **not invent numbers**; trust the supplied counts.
   - Parse model output into:
     - `summary.text` (plain summary).
     - `summary.claims` (per-claim linkage to a `feature_name` and counts).
       - If needed, have the model output JSON with a list of claims.

8. **Persist ProductFacts**
   - Save `ProductFacts` to disk:
     - e.g., `data/cache/{product_id}_facts.json`.
   - This is the main artifact consumed by UI and Q&A.

9. **Refresh strategy**
   - For static AMASUM: recompute only when code changes.
   - Later:
     - Mark a product as “stale” when enough new reviews arrive.
     - Re-run steps 4–8 in a batch job (nightly or on-demand).

---

### 4.2 Online: Summary UI

**Goal:** Fast retrieval and display of a precomputed summary with hoverable evidence.

1. **Load ProductFacts**
   - When a product JSON is uploaded in Streamlit:
     - Compute or load `ProductFacts` from cache.
   - In a future web app:
     - Fetch from DB/API by `product_id`.

2. **Render summary text**
   - Show `summary.text` as paragraphs or bullet points.

3. **Render claims with hover info**
   - For each claim in `summary.claims`:
     - Display the claim text.
     - On hover or click:
       - Show:
         - `review_count` / `total_reviews` (e.g., “Backed by 30 of 120 reviews”).
         - Optionally: 1–3 `evidence_sentences`.
   - In Streamlit:
     - Use `st.expander` or a small clickable icon next to each claim to reveal the details.

---

### 4.3 Online: Q&A Engine

**Goal:** Answer natural language questions about a product using existing `ProductFacts` and raw reviews, with ~1–2s latency.

1. **UI**
   - On the Streamlit dashboard:
     - Add a Q&A panel:
       - A text input: “Ask about this product’s reviews…”
       - A “Ask” button.

2. **Intent & feature detection**
   - On question submit:
     - Map the question to one or more `feature_name`s.
       - First iteration: simple keyword mapping (“battery”, “charge” → “battery life”).
       - Later: local embeddings for better mapping.

3. **Gather evidence**
   - Based on detected features:
     - From `ProductFacts.features`, pull:
       - Counts (`review_count`, `positive_count`, `negative_count`).
       - `evidence_sentences`.
       - Additional raw review snippets (if needed) from the original dataset.

4. **Answer generation**
   - Cheapest mode (no LLM call):
     - Use templated answers:
       - “About battery life, 30 of 120 reviews mention it; most are negative, saying things like ‘the battery always dies by afternoon’.”
       - Encode answer purely from counts + evidence.
   - Enhanced mode (small LLM call):
     - If Groq budget allows:
       - Send a compact prompt with feature stats + evidence sentences for relevant features.
       - Ask for a short, neutral answer using only this information.
   - Always keep the counts as authoritative and do not let the LLM invent numbers.

5. **Display answer with grounding**
   - Show the answer text.
   - Underneath or on hover:
     - Show which features and how many reviews were used.
     - Optionally list a few evidence sentences.

---

## 5. Model & Resource Strategy

- **Groq (llama-3.1-8b-instant)**
  - Used only in batch/slow paths:
    - Initial feature discovery (optional).
    - Generating the final summary text and structured claims.
  - Not used per Q&A request in the minimal implementation.

- **Local models (preferred)**
  - Sentence splitting: Python regex / spaCy.
  - Feature detection:
    - Rules + keywords in the first iteration.
    - Later, a local sentence-embedding model (e.g., sentence-transformers) for semantic matching.
  - Sentiment analysis:
    - A small `pipeline("sentiment-analysis")` model, or a compact classifier.
  - BART:
    - Optional baseline; can be used for alternative summaries or internal comparison, but not required for final UX.

---

## 6. Evaluation & Iteration

- **Automatic metrics**
  - Continue using `structure_aware_evaluation.py` for ROUGE/BERTScore/coverage on AMASUM.
  - Compare:
    - Vanilla vs CoD vs new “feature-grounded” summaries.

- **Human evaluation**
  - For a small set of products:
    - Manually rate:
      - Factual correctness.
      - Per-feature coverage.
      - Clarity and perceived trustworthiness.

- **Q&A evaluation**
  - Create a small benchmark of questions per product with expected answer behaviors.
  - Check:
    - Whether answers are grounded (can you trace claims to evidence?).
    - Whether counts correspond to actual review distributions.

---

## 7. Future Extensions

- Move from Streamlit prototype to a **web service** (FastAPI + React or similar).
- Add **multilingual support** (feature detection + sentiment per language).
- Improve feature discovery via clustering of review sentences and LLM labeling.
- Add **user feedback** (“Was this summary helpful?”) and use it to refine prompts and thresholds.

---