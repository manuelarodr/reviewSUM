"""
Test suite for the filtering module.
"""

import pytest
import pandas as pd
from src.filtering import CredibilityFilter, FilteringCriteria, create_filtering_criteria

def test_filtering_criteria():
    """Test FilteringCriteria dataclass."""
    criteria = FilteringCriteria(
        min_helpful_votes=5,
        require_verified=True,
        min_review_length=50
    )
    
    assert criteria.min_helpful_votes == 5
    assert criteria.require_verified == True
    assert criteria.min_review_length == 50

def test_credibility_filter_initialization():
    """Test CredibilityFilter initialization."""
    criteria = create_filtering_criteria()
    filterer = CredibilityFilter(criteria)
    
    assert filterer.criteria is not None
    assert len(filterer.spam_patterns) > 0

def test_filter_reviews():
    """Test review filtering functionality."""
    # Create sample reviews DataFrame
    reviews_data = [
        {
            'review_id': 'r001',
            'text': 'This is a great product with excellent quality and durability.',
            'rating': 5.0,
            'verified': True,
            'helpful_votes': 10
        },
        {
            'review_id': 'r002',
            'text': 'Bad product, click here to buy better one!',
            'rating': 1.0,
            'verified': False,
            'helpful_votes': 1
        },
        {
            'review_id': 'r003',
            'text': 'Average product, nothing special.',
            'rating': 3.0,
            'verified': True,
            'helpful_votes': 5
        }
    ]
    
    reviews_df = pd.DataFrame(reviews_data)
    
    criteria = create_filtering_criteria(min_helpful_votes=5, require_verified=True)
    filterer = CredibilityFilter(criteria)
    
    filtered_df, stats = filterer.filter_reviews(reviews_df)
    
    # Should filter out r002 (low helpful votes, unverified, spam)
    assert len(filtered_df) == 2
    assert 'r001' in filtered_df['review_id'].values
    assert 'r003' in filtered_df['review_id'].values
    assert 'r002' not in filtered_df['review_id'].values
    
    # Check stats
    assert stats['original_count'] == 3
    assert stats['filtered_count'] == 2

def test_credibility_score():
    """Test credibility score calculation."""
    criteria = create_filtering_criteria()
    filterer = CredibilityFilter(criteria)
    
    # High credibility review
    high_cred_review = {
        'verified': True,
        'helpful_votes': 50,
        'text': 'This is a detailed review with lots of information about the product.',
        'rating': 4.0
    }
    
    score = filterer.get_credibility_score(high_cred_review)
    assert score > 0.5
    
    # Low credibility review
    low_cred_review = {
        'verified': False,
        'helpful_votes': 0,
        'text': 'Bad',
        'rating': 1.0
    }
    
    score = filterer.get_credibility_score(low_cred_review)
    assert score < 0.5

def test_spam_pattern_filtering():
    """Test spam pattern filtering."""
    criteria = create_filtering_criteria()
    filterer = CredibilityFilter(criteria)
    
    spam_texts = [
        'Click here to buy now!',
        'Visit our website http://example.com',
        'THIS IS SPAM!!!',
        'What??? Really???'
    ]
    
    clean_texts = [
        'This is a good product',
        'I recommend this item',
        'Quality is excellent'
    ]
    
    # Test spam detection
    for text in spam_texts:
        assert filterer._filter_spam_patterns(pd.DataFrame({'text': [text]})).empty
    
    # Test clean text passes
    for text in clean_texts:
        result = filterer._filter_spam_patterns(pd.DataFrame({'text': [text]}))
        assert not result.empty
