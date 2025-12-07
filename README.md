Input & preprocessing

Load AMASUM-format JSON; convert reviews to reviews_df with review_ids added if missing.
Build/load sentence index (sentence_index.py): sentence split reviews, classify per-sentence sentiment (CardiffNLP Twitter-RoBERTa sentiment), and compute sentence embeddings (sentence-transformers/all-MiniLM-L6-v2). Cache to data/cache/sentence_index/ for reuse.
Filtering (token budget control)

Module: filtering.AdvancedReviewFilter
Inputs: reviews_df, sentence_embeddings, sentence→review mapping; optional sidebar flags (verified/useful).
Steps: apply verified/useful filters; estimate tokens via chars×0.25; if under limit, return; else aggregate sentence embeddings to review embeddings (mean); score reviews by centroid cosine similarity + helpful-vote boost; select top reviews until token_limit. Output: filtered_df, filter_stats.
Chain-of-Density summarization

Module: summarizer.ChainOfDensitySummarizer (Groq API, model llama-3.1-8b-instant, temp 0.1, max_tokens 4000).
Prompt: 5 iterations of “add 1–3 missing entities” at fixed length (~120 words start), increasing density; asks for final JSON {summary, entity_log} (legacy prompt).
Output used: summary text and entity_log/entities (entity names only; review_ids ignored downstream).
Entity grounding to features

Modules: cod_product_facts.py + entity_grounding.py.
Steps: embed entity strings (same MiniLM model), cosine-match to sentence embeddings (threshold 0.3, top-K capped), aggregate per entity:
unique supporting review_ids
sentiment counts from sentence sentiments (pos/neg/neutral)
evidence sentences (top scored)
counts stored in FeatureFacts.
Result packaged as ProductFacts with summary text + grounded features.
QA over embeddings

Module: qa_engine.answer_question_with_embeddings.
Uses the same sentence embeddings/index to embed the user question, cosine search sentences, and synthesize an answer with evidence sentences; prefers ProductFacts features when available.
Models and settings (key choices):

Sentence embeddings: sentence-transformers/all-MiniLM-L6-v2 (fast, light).
Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest (CPU).
CoD LLM: Groq llama-3.1-8b-instant, temp 0.1, max_tokens 4000; prompt targets ~80-word fixed-length summaries over 5 iterations.
Similarity thresholds: 0.3 for entity→sentence grounding; helpful_boost 0.5; tokens_per_char 0.25; default token_limit 4000.