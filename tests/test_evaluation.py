"""
Test suite for the evaluation module.
"""

import pytest
import pandas as pd
from src.evaluation import SummaryEvaluator, EvaluationConfig, create_evaluator

def test_evaluation_config():
    """Test EvaluationConfig dataclass."""
    config = EvaluationConfig(
        rouge_types=["rouge1", "rouge2"],
        compute_factual_consistency=True
    )
    
    assert config.rouge_types == ["rouge1", "rouge2"]
    assert config.compute_factual_consistency == True

def test_evaluator_initialization():
    """Test SummaryEvaluator initialization."""
    config = EvaluationConfig()
    evaluator = SummaryEvaluator(config)
    
    assert evaluator.config is not None
    assert evaluator.rouge_scorer is not None
    assert len(evaluator.stop_words) > 0

def test_rouge_scores():
    """Test ROUGE score computation."""
    evaluator = create_evaluator()
    
    generated = "This is a great product with excellent quality."
    gold = "This product has great quality and is excellent."
    
    scores = evaluator._compute_rouge_scores(generated, gold)
    
    assert 'rouge1_precision' in scores
    assert 'rouge1_recall' in scores
    assert 'rouge1_fmeasure' in scores
    assert all(0 <= score <= 1 for score in scores.values())

def test_factual_consistency():
    """Test factual consistency computation."""
    reviews_data = [
        {
            'review_id': 'r001',
            'text': 'This product has excellent quality and durability.',
            'rating': 5.0,
            'verified': True,
            'helpful_votes': 10
        },
        {
            'review_id': 'r002',
            'text': 'Great quality product, very durable.',
            'rating': 4.0,
            'verified': True,
            'helpful_votes': 8
        }
    ]
    
    reviews_df = pd.DataFrame(reviews_data)
    
    claims = [
        {
            'text': 'Product has good quality',
            'supporting_reviews': ['r001', 'r002'],
            'percentage': 0.8
        },
        {
            'text': 'Product is expensive',
            'supporting_reviews': ['r001'],  # Not supported by review text
            'percentage': 0.3
        }
    ]
    
    evaluator = create_evaluator()
    consistency = evaluator._compute_factual_consistency(claims, reviews_df)
    
    assert consistency['total_claims'] == 2
    assert consistency['supported_claims'] >= 1  # At least one should be supported
    assert 0 <= consistency['consistency_score'] <= 1

def test_claim_validation():
    """Test claim validation against reviews."""
    reviews_data = [
        {
            'review_id': 'r001',
            'text': 'This product has excellent quality and durability.',
            'rating': 5.0,
            'verified': True,
            'helpful_votes': 10
        }
    ]
    
    reviews_df = pd.DataFrame(reviews_data)
    
    evaluator = create_evaluator()
    
    # Valid claim
    is_supported = evaluator._validate_claim_support(
        "Product has good quality",
        ['r001'],
        reviews_df
    )
    assert is_supported == True
    
    # Invalid claim
    is_supported = evaluator._validate_claim_support(
        "Product is very expensive",
        ['r001'],
        reviews_df
    )
    assert is_supported == False

def test_coverage_metrics():
    """Test coverage metrics computation."""
    reviews_data = [
        {
            'review_id': 'r001',
            'text': 'Great quality product',
            'rating': 5.0,
            'verified': True,
            'helpful_votes': 10
        },
        {
            'review_id': 'r002',
            'text': 'Good design',
            'rating': 4.0,
            'verified': True,
            'helpful_votes': 8
        }
    ]
    
    reviews_df = pd.DataFrame(reviews_data)
    
    claims = [
        {
            'text': 'Product has good quality',
            'supporting_reviews': ['r001'],
            'percentage': 0.5
        }
    ]
    
    themes_data = {
        'theme_stats': {
            'quality': 0.5,
            'design': 0.5
        }
    }
    
    evaluator = create_evaluator()
    coverage = evaluator._compute_coverage_metrics(claims, reviews_df, themes_data)
    
    assert 'theme_coverage' in coverage
    assert 'review_coverage' in coverage
    assert 0 <= coverage['theme_coverage'] <= 1
    assert 0 <= coverage['review_coverage'] <= 1

def test_diversity_metrics():
    """Test diversity metrics computation."""
    evaluator = create_evaluator()
    
    generated = "This is a great product with excellent quality and durability."
    gold = "This product has great quality and is excellent."
    
    diversity = evaluator._compute_diversity_metrics(generated, gold)
    
    assert 'generated_ttr' in diversity
    assert 'gold_ttr' in diversity
    assert 'jaccard_similarity' in diversity
    assert 0 <= diversity['generated_ttr'] <= 1
    assert 0 <= diversity['jaccard_similarity'] <= 1

def test_keyword_extraction():
    """Test keyword extraction functionality."""
    evaluator = create_evaluator()
    
    text = "This is a great product with excellent quality and durability."
    keywords = evaluator._extract_keywords(text)
    
    assert 'great' in keywords
    assert 'product' in keywords
    assert 'quality' in keywords
    assert 'durability' in keywords
    assert 'this' not in keywords  # Stop word should be removed
    assert 'is' not in keywords  # Stop word should be removed

def test_overall_score():
    """Test overall score computation."""
    evaluator = create_evaluator()
    
    evaluation_results = {
        'rouge_scores': {
            'rouge1_fmeasure': 0.8,
            'rouge2_fmeasure': 0.7
        },
        'factual_consistency': {
            'consistency_score': 0.9
        },
        'coverage': {
            'theme_coverage': 0.6,
            'review_coverage': 0.7
        },
        'diversity': {
            'generated_ttr': 0.5
        }
    }
    
    overall_score = evaluator._compute_overall_score(evaluation_results)
    
    assert 0 <= overall_score <= 1

def test_create_evaluator():
    """Test create_evaluator helper function."""
    evaluator = create_evaluator(
        rouge_types=["rouge1"],
        compute_factual_consistency=False
    )
    
    assert evaluator.config.rouge_types == ["rouge1"]
    assert evaluator.config.compute_factual_consistency == False
