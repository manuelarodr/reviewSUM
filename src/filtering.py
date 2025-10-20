"""
Advanced filtering module for review summarization.

This module implements text preprocessing, TF-IDF based importance scoring,
and stratified sampling for optimal review selection.
"""

import pandas as pd
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

@dataclass
class FilteringCriteria:
    """Configuration for review filtering criteria."""
    min_review_length: int = 20
    max_review_length: int = 2000
    token_limit: int = 50000  # Removed token limit for testing
    min_reviews_per_rating: int = 2  # Minimum reviews to keep per rating
    max_reviews_per_rating: int = 200  # Increased max reviews per rating
    tfidf_weight: float = 0.7  # Weight for TF-IDF in hybrid scoring
    length_weight: float = 0.3  # Weight for length normalization

class AdvancedReviewFilter:
    """
    Advanced review filtering using text preprocessing, TF-IDF scoring, and stratified sampling.
    """
    
    def __init__(self, criteria: Optional[FilteringCriteria] = None):
        """
        Initialize the advanced review filter.
        
        Args:
            criteria: Filtering criteria configuration
        """
        self.criteria = criteria or FilteringCriteria()
        
        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Compile regex patterns for text cleaning
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.emoji_pattern = re.compile(r'[^\w\s]', re.UNICODE)  # Removes emojis and punctuation
        self.digit_pattern = re.compile(r'\d+')
        
        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing HTML tags, emojis, punctuation, digits, URLs,
        converting to lowercase, removing stopwords, and lemmatizing.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text as bag of words
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # 1. Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        # 2. Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # 3. Remove emojis and punctuation
        text = self.emoji_pattern.sub(' ', text)
        
        # 4. Remove digits
        text = self.digit_pattern.sub(' ', text)
        
        # 5. Convert to lowercase
        text = text.lower()
        
        # 6. Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # 7. Lemmatize to root form
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def calculate_hybrid_tfidf_score(self, reviews_df: pd.DataFrame) -> pd.Series:
        """
        Calculate hybrid TF-IDF scores for reviews.
        
        Args:
            reviews_df: DataFrame containing reviews
            
        Returns:
            Series of hybrid TF-IDF scores
        """
        # Preprocess all review texts
        processed_texts = reviews_df['text'].apply(self.preprocess_text)
        
        # Fit TF-IDF vectorizer
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # Calculate average TF-IDF score for each review
        avg_tfidf_scores = np.array(tfidf_matrix.mean(axis=1)).flatten()
        
        # Normalize scores to 0-1 range
        if avg_tfidf_scores.max() > 0:
            avg_tfidf_scores = avg_tfidf_scores / avg_tfidf_scores.max()
        
        return pd.Series(avg_tfidf_scores, index=reviews_df.index)
    
    def calculate_review_importance(self, reviews_df: pd.DataFrame) -> pd.Series:
        """
        Calculate hybrid importance scores combining TF-IDF and length normalization.
        
        Args:
            reviews_df: DataFrame containing reviews
            
        Returns:
            Series of importance scores
        """
        # Calculate TF-IDF scores
        tfidf_scores = self.calculate_hybrid_tfidf_score(reviews_df)
        
        # Calculate length normalization scores
        text_lengths = reviews_df['text'].str.len()
        max_length = text_lengths.max()
        min_length = text_lengths.min()
        
        if max_length > min_length:
            length_scores = (text_lengths - min_length) / (max_length - min_length)
        else:
            length_scores = pd.Series([0.5] * len(reviews_df), index=reviews_df.index)
        
        # Combine scores with weights
        importance_scores = (
            self.criteria.tfidf_weight * tfidf_scores + 
            self.criteria.length_weight * length_scores
        )
        
        return importance_scores
    
    def filter_reviews(self, reviews_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Filter and rank reviews using advanced preprocessing and TF-IDF scoring.
        
        Args:
            reviews_df: DataFrame containing reviews with columns:
                       review_id, text, rating, etc.
        
        Returns:
            Tuple of (filtered_dataframe, filtering_stats)
        """
        original_count = len(reviews_df)
        filtered_df = reviews_df.copy()
        
        # Track filtering statistics
        stats = {
            'original_count': original_count,
            'filtered_count': 0,
            'filter_reasons': {},
            'rating_distribution': {},
            'token_usage': 0
        }
        
        # 1. Basic length filtering
        if self.criteria.min_review_length > 0:
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['text'].str.len() >= self.criteria.min_review_length]
            removed_count = before_count - len(filtered_df)
            stats['filter_reasons']['too_short'] = removed_count
        
        if self.criteria.max_review_length > 0:
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['text'].str.len() <= self.criteria.max_review_length]
            removed_count = before_count - len(filtered_df)
            stats['filter_reasons']['too_long'] = removed_count
        
        # 2. Calculate importance scores
        importance_scores = self.calculate_review_importance(filtered_df)
        filtered_df['importance_score'] = importance_scores
        
        # 3. Rank within each rating (stratified ranking)
        filtered_df['rank_within_rating'] = filtered_df.groupby('rating')['importance_score'].rank(
            method='dense', ascending=False
        )
        
        # 4. Calculate proportional selection based on original distribution
        # First, calculate the original distribution proportions
        original_distribution = reviews_df['rating'].value_counts().sort_index()
        total_original = len(reviews_df)
        original_proportions = original_distribution / total_original
        
        # Calculate target number of reviews to select (based on token limit)
        # Estimate average tokens per review
        avg_tokens_per_review = filtered_df['text'].str.len().mean() / 4
        target_total_reviews = min(
            int(self.criteria.token_limit / avg_tokens_per_review),
            len(filtered_df)
        )
        
        # Calculate target reviews per rating based on original proportions
        target_reviews_per_rating = {}
        for rating in sorted(filtered_df['rating'].unique()):
            proportion = original_proportions.get(rating, 0)
            target_count = max(
                self.criteria.min_reviews_per_rating,
                min(
                    int(target_total_reviews * proportion),
                    self.criteria.max_reviews_per_rating
                )
            )
            target_reviews_per_rating[rating] = target_count
        
        # Round-robin selection to ensure fair representation across all ratings
        final_reviews = []
        total_tokens = 0
        
        # Prepare rating groups with sorted reviews
        rating_groups = {}
        for rating in sorted(filtered_df['rating'].unique()):
            rating_reviews = filtered_df[filtered_df['rating'] == rating].copy()
            rating_reviews = rating_reviews.sort_values('rank_within_rating')
            rating_reviews['estimated_tokens'] = rating_reviews['text'].str.len() / 4
            rating_groups[rating] = {
                'reviews': rating_reviews,
                'target_count': target_reviews_per_rating[rating],
                'selected_count': 0,
                'tokens_used': 0,
                'current_index': 0
            }
        
        # Round-robin selection: alternate between rating groups
        ratings_list = sorted(filtered_df['rating'].unique())
        round_count = 0
        
        while True:
            round_count += 1
            added_any = False
            
            # Go through each rating group in this round
            for rating in ratings_list:
                group = rating_groups[rating]
                
                # Check if this group has reached its target or has no more reviews
                if (group['selected_count'] >= group['target_count'] or 
                    group['current_index'] >= len(group['reviews'])):
                    continue
                
                # Get the next review for this rating
                review = group['reviews'].iloc[group['current_index']]
                review_tokens = review['estimated_tokens']
                
                # Check if adding this review would exceed token limit
                if total_tokens + review_tokens > self.criteria.token_limit:
                    continue  # Skip this review, try next rating
                
                # Add the review
                final_reviews.append(review)
                group['selected_count'] += 1
                group['tokens_used'] += review_tokens
                group['current_index'] += 1
                total_tokens += review_tokens
                added_any = True
            
            # If no reviews were added in this round, we're done
            if not added_any:
                break
        
        # Update stats for each rating group
        for rating in ratings_list:
            group = rating_groups[rating]
            stats['rating_distribution'][f'rating_{int(rating)}'] = {
                'total_reviews': len(group['reviews']),
                'selected_reviews': group['selected_count'],
                'target_reviews': group['target_count'],
                'original_proportion': original_proportions.get(rating, 0),
                'tokens_used': group['tokens_used']
            }
        
        # Create final DataFrame
        if final_reviews:
            final_df = pd.DataFrame(final_reviews)
        else:
            final_df = pd.DataFrame(columns=filtered_df.columns)
        
        stats['filtered_count'] = len(final_df)
        stats['token_usage'] = total_tokens
        stats['retention_rate'] = len(final_df) / original_count if original_count > 0 else 0
        
        return final_df, stats
    
    def get_preprocessed_text(self, text: str) -> str:
        """
        Get preprocessed text for a single review.
        
        Args:
            text: Raw review text
            
        Returns:
            Preprocessed text
        """
        return self.preprocess_text(text)
    
    def get_review_importance_score(self, review_text: str, all_reviews: List[str]) -> float:
        """
        Calculate importance score for a single review given a corpus.
        
        Args:
            review_text: Text of the review to score
            all_reviews: List of all review texts in the corpus
            
        Returns:
            Importance score between 0 and 1
        """
        # Preprocess the review
        processed_text = self.preprocess_text(review_text)
        
        # Create a temporary DataFrame for TF-IDF calculation
        temp_df = pd.DataFrame({'text': all_reviews + [review_text]})
        temp_df['processed_text'] = temp_df['text'].apply(self.preprocess_text)
        
        # Calculate TF-IDF scores
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(temp_df['processed_text'])
        
        # Get the score for the last review (our target review)
        review_score = np.array(tfidf_matrix[-1].mean()).flatten()[0]
        
        # Normalize
        all_scores = np.array(tfidf_matrix.mean(axis=1)).flatten()
        if all_scores.max() > 0:
            review_score = review_score / all_scores.max()
        
        return float(review_score)

def create_filtering_criteria(
    token_limit: int = 4000,
    min_reviews_per_rating: int = 2,
    max_reviews_per_rating: int = 50,
    tfidf_weight: float = 0.7
) -> FilteringCriteria:
    """
    Create filtering criteria with common defaults for API token limits.
    
    Args:
        token_limit: Maximum tokens to use for API calls
        min_reviews_per_rating: Minimum reviews to keep per rating
        max_reviews_per_rating: Maximum reviews per rating
        tfidf_weight: Weight for TF-IDF in hybrid scoring
        
    Returns:
        FilteringCriteria object
    """
    return FilteringCriteria(
        token_limit=token_limit,
        min_reviews_per_rating=min_reviews_per_rating,
        max_reviews_per_rating=max_reviews_per_rating,
        tfidf_weight=tfidf_weight
    )

# Backward compatibility alias
CredibilityFilter = AdvancedReviewFilter
