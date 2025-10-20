"""
Evaluation module for review summarization.

This module computes ROUGE scores, factual consistency, and coverage metrics
to evaluate the quality of generated summaries against human-written gold standards.
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    rouge_types: List[str] = None
    compute_factual_consistency: bool = True
    compute_coverage: bool = True
    compute_diversity: bool = True
    min_claim_support: float = 0.1
    
    def __post_init__(self):
        if self.rouge_types is None:
            self.rouge_types = ["rouge1", "rouge2", "rougeL"]

class SummaryEvaluator:
    """
    Evaluates generated summaries using multiple metrics.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize the evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.rouge_scorer = rouge_scorer.RougeScorer(self.config.rouge_types, use_stemmer=True)
        self.stop_words = set(stopwords.words('english'))
    
    def evaluate_summary(
        self,
        generated_summary: str,
        gold_summary: str,
        claims: List[Dict[str, Any]],
        reviews_df: pd.DataFrame,
        themes_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a generated summary.
        
        Args:
            generated_summary: Generated summary text
            gold_summary: Human-written gold standard summary
            claims: List of extracted claims with supporting reviews
            reviews_df: DataFrame containing source reviews
            themes_data: Theme extraction results
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        evaluation_results = {}
        
        # 1. ROUGE scores
        rouge_scores = self._compute_rouge_scores(generated_summary, gold_summary)
        evaluation_results['rouge_scores'] = rouge_scores
        
        # 2. Factual consistency
        if self.config.compute_factual_consistency:
            factual_consistency = self._compute_factual_consistency(claims, reviews_df)
            evaluation_results['factual_consistency'] = factual_consistency
        
        # 3. Coverage metrics
        if self.config.compute_coverage:
            coverage_metrics = self._compute_coverage_metrics(claims, reviews_df, themes_data)
            evaluation_results['coverage'] = coverage_metrics
        
        # 4. Diversity metrics
        if self.config.compute_diversity:
            diversity_metrics = self._compute_diversity_metrics(generated_summary, gold_summary)
            evaluation_results['diversity'] = diversity_metrics
        
        # 5. Overall quality score
        evaluation_results['overall_score'] = self._compute_overall_score(evaluation_results)
        
        return evaluation_results
    
    def _compute_rouge_scores(self, generated: str, gold: str) -> Dict[str, float]:
        """
        Compute ROUGE scores between generated and gold summaries.
        
        Args:
            generated: Generated summary
            gold: Gold standard summary
            
        Returns:
            Dictionary of ROUGE scores
        """
        scores = self.rouge_scorer.score(gold, generated)
        
        rouge_results = {}
        for rouge_type in self.config.rouge_types:
            rouge_results[f"{rouge_type}_precision"] = scores[rouge_type].precision
            rouge_results[f"{rouge_type}_recall"] = scores[rouge_type].recall
            rouge_results[f"{rouge_type}_fmeasure"] = scores[rouge_type].fmeasure
        
        return rouge_results
    
    def _compute_factual_consistency(self, claims: List[Dict[str, Any]], reviews_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute factual consistency of claims against source reviews.
        
        Args:
            claims: List of claims with supporting reviews
            reviews_df: DataFrame containing source reviews
            
        Returns:
            Dictionary containing consistency metrics
        """
        if not claims:
            return {
                'consistency_score': 0.0,
                'supported_claims': 0,
                'total_claims': 0,
                'claim_details': []
            }
        
        supported_claims = 0
        claim_details = []
        
        for claim in claims:
            claim_text = claim.get('text', '')
            supporting_reviews = claim.get('supporting_reviews', [])
            
            # Check if claim is supported by its cited reviews
            is_supported = self._validate_claim_support(claim_text, supporting_reviews, reviews_df)
            
            if is_supported:
                supported_claims += 1
            
            claim_details.append({
                'claim': claim_text,
                'supporting_reviews': supporting_reviews,
                'is_supported': is_supported,
                'percentage': claim.get('percentage', 0.0)
            })
        
        consistency_score = supported_claims / len(claims) if claims else 0.0
        
        return {
            'consistency_score': consistency_score,
            'supported_claims': supported_claims,
            'total_claims': len(claims),
            'claim_details': claim_details
        }
    
    def _validate_claim_support(self, claim: str, supporting_reviews: List[str], reviews_df: pd.DataFrame) -> bool:
        """
        Validate if a claim is supported by its cited reviews.
        
        Args:
            claim: Claim text
            supporting_reviews: List of review IDs
            reviews_df: DataFrame containing reviews
            
        Returns:
            True if claim is supported, False otherwise
        """
        if not supporting_reviews:
            return False
        
        # Get review texts for supporting reviews
        supporting_texts = []
        for review_id in supporting_reviews:
            review_row = reviews_df[reviews_df['review_id'] == review_id]
            if not review_row.empty:
                supporting_texts.append(review_row.iloc[0]['text'])
        
        if not supporting_texts:
            return False
        
        # Check if claim keywords appear in supporting texts
        claim_keywords = self._extract_keywords(claim)
        
        for text in supporting_texts:
            text_keywords = self._extract_keywords(text)
            
            # Check for keyword overlap
            overlap = len(set(claim_keywords) & set(text_keywords))
            if overlap >= len(claim_keywords) * 0.3:  # At least 30% keyword overlap
                return True
        
        return False
    
    def _compute_coverage_metrics(self, claims: List[Dict[str, Any]], reviews_df: pd.DataFrame, themes_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute coverage metrics for themes and reviews.
        
        Args:
            claims: List of claims
            reviews_df: DataFrame containing reviews
            themes_data: Theme extraction results
            
        Returns:
            Dictionary containing coverage metrics
        """
        total_reviews = len(reviews_df)
        theme_stats = themes_data.get('theme_stats', {})
        
        # Coverage of themes
        covered_themes = 0
        theme_coverage_details = {}
        
        for theme, prevalence in theme_stats.items():
            # Check if theme is mentioned in claims
            theme_mentioned = any(theme.lower() in claim.get('text', '').lower() for claim in claims)
            
            if theme_mentioned:
                covered_themes += 1
            
            theme_coverage_details[theme] = {
                'prevalence': prevalence,
                'covered': theme_mentioned
            }
        
        theme_coverage = covered_themes / len(theme_stats) if theme_stats else 0.0
        
        # Coverage of reviews (how many reviews are cited)
        cited_reviews = set()
        for claim in claims:
            cited_reviews.update(claim.get('supporting_reviews', []))
        
        review_coverage = len(cited_reviews) / total_reviews if total_reviews > 0 else 0.0
        
        return {
            'theme_coverage': theme_coverage,
            'review_coverage': review_coverage,
            'covered_themes': covered_themes,
            'total_themes': len(theme_stats),
            'cited_reviews': len(cited_reviews),
            'total_reviews': total_reviews,
            'theme_coverage_details': theme_coverage_details
        }
    
    def _compute_diversity_metrics(self, generated: str, gold: str) -> Dict[str, float]:
        """
        Compute diversity metrics for generated summary.
        
        Args:
            generated: Generated summary
            gold: Gold standard summary
            
        Returns:
            Dictionary containing diversity metrics
        """
        # Tokenize summaries
        generated_tokens = word_tokenize(generated.lower())
        gold_tokens = word_tokenize(gold.lower())
        
        # Remove stopwords
        generated_tokens = [token for token in generated_tokens if token not in self.stop_words]
        gold_tokens = [token for token in gold_tokens if token not in self.stop_words]
        
        # Type-token ratio (lexical diversity)
        generated_ttr = len(set(generated_tokens)) / len(generated_tokens) if generated_tokens else 0
        gold_ttr = len(set(gold_tokens)) / len(gold_tokens) if gold_tokens else 0
        
        # Unique word overlap
        generated_unique = set(generated_tokens)
        gold_unique = set(gold_tokens)
        overlap = len(generated_unique & gold_unique)
        union = len(generated_unique | gold_unique)
        jaccard_similarity = overlap / union if union > 0 else 0
        
        return {
            'generated_ttr': generated_ttr,
            'gold_ttr': gold_ttr,
            'jaccard_similarity': jaccard_similarity,
            'unique_words_generated': len(generated_unique),
            'unique_words_gold': len(gold_unique)
        }
    
    def _compute_overall_score(self, evaluation_results: Dict[str, Any]) -> float:
        """
        Compute overall quality score from individual metrics.
        
        Args:
            evaluation_results: Dictionary containing all evaluation metrics
            
        Returns:
            Overall score between 0 and 1
        """
        scores = []
        
        # ROUGE F1 scores
        rouge_scores = evaluation_results.get('rouge_scores', {})
        rouge_f1_scores = [score for key, score in rouge_scores.items() if key.endswith('_fmeasure')]
        if rouge_f1_scores:
            scores.append(np.mean(rouge_f1_scores))
        
        # Factual consistency
        factual_consistency = evaluation_results.get('factual_consistency', {})
        if factual_consistency:
            scores.append(factual_consistency.get('consistency_score', 0.0))
        
        # Coverage
        coverage = evaluation_results.get('coverage', {})
        if coverage:
            coverage_score = (coverage.get('theme_coverage', 0.0) + coverage.get('review_coverage', 0.0)) / 2
            scores.append(coverage_score)
        
        # Diversity (use TTR as proxy)
        diversity = evaluation_results.get('diversity', {})
        if diversity:
            scores.append(diversity.get('generated_ttr', 0.0))
        
        return np.mean(scores) if scores else 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Tokenize and remove stopwords
        tokens = word_tokenize(text.lower())
        keywords = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return keywords

def create_evaluator(
    rouge_types: List[str] = None,
    compute_factual_consistency: bool = True
) -> SummaryEvaluator:
    """
    Create an evaluator with common defaults.
    
    Args:
        rouge_types: List of ROUGE types to compute
        compute_factual_consistency: Whether to compute factual consistency
        
    Returns:
        SummaryEvaluator instance
    """
    config = EvaluationConfig(
        rouge_types=rouge_types or ["rouge1", "rouge2", "rougeL"],
        compute_factual_consistency=compute_factual_consistency
    )
    
    return SummaryEvaluator(config)
