"""
Theme extraction module for review summarization.

This module extracts product themes using BERTopic to identify key aspects mentioned in customer reviews.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

try:
    from bertopic import BERTopic  # type: ignore
    HAS_BERTOPIC = True
except Exception:
    HAS_BERTOPIC = False


class ThemeExtractor:
    """
    Extracts themes from customer reviews using BERTopic.
    """
    def __init__(self, language: str = "english"):
        self.num_themes = 5
        self.language = language
        self.model = BERTopic(nr_topics=5, language=language) if HAS_BERTOPIC else None

    def extract_themes(self, reviews_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract themes from reviews using BERTopic if available; otherwise, use a simple TF-IDF fallback.

        Args:
            reviews_df: DataFrame with a 'text' column containing review texts.

        Returns:
            Dict with 'theme_stats' mapping theme name to prevalence, plus optional 'raw' details.
        """
        texts = reviews_df['text'].dropna().tolist()
        if not texts:
            return {"theme_stats": {}}

        if self.model is not None:
            topics, probs = self.model.fit_transform(texts)
            topic_info = self.model.get_topic_info()
            # Build theme_stats from top topics excluding outlier -1
            topic_info = topic_info[topic_info['Topic'] != -1]
            top_n = topic_info.head(self.num_themes)
            total = topic_info['Count'].sum() or 1
            theme_stats = {row['Name']: float(row['Count']) / float(total) for _, row in top_n.iterrows()}
            return {"theme_stats": theme_stats, "raw": topic_info.to_dict(orient='records')}

        # Fallback: TF-IDF top keywords as pseudo-themes
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())
        # Compute mean tf-idf score per term
        means = np.asarray(X.mean(axis=0)).ravel()
        top_idx = np.argsort(means)[::-1][: self.num_themes]
        top_terms = feature_names[top_idx]
        # Prevalence: fraction of docs containing the term
        term_counts = (X[:, top_idx] > 0).sum(axis=0).A1
        total_docs = max(1, X.shape[0])
        theme_stats = {term: float(count) / float(total_docs) for term, count in zip(top_terms, term_counts)}
        return {"theme_stats": theme_stats, "raw": {"top_terms": top_terms.tolist()}}

def create_theme_extractor(
    language: str = "english"
) -> ThemeExtractor:
    """
    Factory for ThemeExtractor with BERTopic. Always extracts 5 themes.

    Args:
        language: Language for BERTopic.

    Returns:
        ThemeExtractor instance.
    """
    return ThemeExtractor(language=language)
