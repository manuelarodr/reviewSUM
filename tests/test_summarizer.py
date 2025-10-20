"""
Test suite for the summarizer module.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.summarizer import ChainOfDensitySummarizer, SummarizationConfig, create_summarizer

def test_summarization_config():
    """Test SummarizationConfig dataclass."""
    config = SummarizationConfig(
        backend="groq",
        model_name="llama3-8b-8192",
        max_summary_length=120
    )
    
    assert config.backend == "groq"
    assert config.model_name == "llama3-8b-8192"
    assert config.max_summary_length == 120

def test_summarizer_initialization():
    """Test ChainOfDensitySummarizer initialization."""
    config = SummarizationConfig(backend="local")
    
    with patch('src.summarizer.pipeline') as mock_pipeline:
        mock_pipeline.return_value = Mock()
        summarizer = ChainOfDensitySummarizer(config)
        
        assert summarizer.config.backend == "local"
        assert summarizer.local_pipeline is not None

def test_prepare_review_texts():
    """Test review text preparation."""
    reviews_data = [
        {
            'review_id': 'r001',
            'text': 'Great product!',
            'rating': 5.0,
            'verified': True,
            'helpful_votes': 10
        },
        {
            'review_id': 'r002',
            'text': 'Not bad.',
            'rating': 3.0,
            'verified': False,
            'helpful_votes': 5
        }
    ]
    
    reviews_df = pd.DataFrame(reviews_data)
    
    config = SummarizationConfig(backend="local")
    with patch('src.summarizer.pipeline') as mock_pipeline:
        mock_pipeline.return_value = Mock()
        summarizer = ChainOfDensitySummarizer(config)
        
        prepared_text = summarizer._prepare_review_texts(reviews_df)
        
        assert "Review r001 (Rating: 5.0/5): Great product!" in prepared_text
        assert "Review r002 (Rating: 3.0/5): Not bad." in prepared_text

def test_chain_of_density_prompt():
    """Test Chain-of-Density prompt creation."""
    config = SummarizationConfig()
    with patch('src.summarizer.pipeline') as mock_pipeline:
        mock_pipeline.return_value = Mock()
        summarizer = ChainOfDensitySummarizer(config)
    
    review_texts = "Review r001: Great product!"
    product_name = "test_product"
    
    prompt = summarizer._create_chain_of_density_prompt(review_texts, product_name)
    
    assert "test_product" in prompt
    assert "Chain-of-Density" in prompt or "entity-dense" in prompt
    assert "Review r001: Great product!" in prompt
    assert "JSON" in prompt

def test_extract_summary_from_text():
    """Test summary extraction from text response."""
    config = SummarizationConfig()
    with patch('src.summarizer.pipeline') as mock_pipeline:
        mock_pipeline.return_value = Mock()
        summarizer = ChainOfDensitySummarizer(config)
    
    # Test with JSON-like response
    json_response = '{"summary": "This is a great product with excellent quality."}'
    result = summarizer._extract_summary_from_text(json_response)
    
    assert result['summary'] == "This is a great product with excellent quality."
    
    # Test with plain text response
    plain_response = "This is a great product with excellent quality."
    result = summarizer._extract_summary_from_text(plain_response)
    
    assert result['summary'] == plain_response

def test_extract_claims():
    """Test claim extraction and validation."""
    reviews_data = [
        {
            'review_id': 'r001',
            'text': 'Great product!',
            'rating': 5.0,
            'verified': True,
            'helpful_votes': 10
        },
        {
            'review_id': 'r002',
            'text': 'Not bad.',
            'rating': 3.0,
            'verified': False,
            'helpful_votes': 5
        }
    ]
    
    reviews_df = pd.DataFrame(reviews_data)
    
    config = SummarizationConfig()
    with patch('src.summarizer.pipeline') as mock_pipeline:
        mock_pipeline.return_value = Mock()
        summarizer = ChainOfDensitySummarizer(config)
    
    summary_result = {
        'claims': [
            {
                'text': 'Product has good quality',
                'supporting_reviews': ['r001', 'r002'],
                'percentage': 0.8
            },
            {
                'text': 'Product is expensive',
                'supporting_reviews': ['r999'],  # Invalid review ID
                'percentage': 0.3
            }
        ]
    }
    
    claims = summarizer._extract_claims(summary_result, reviews_df)
    
    # Should only include claims with valid supporting reviews
    assert len(claims) == 1
    assert claims[0]['text'] == 'Product has good quality'
    assert claims[0]['supporting_reviews'] == ['r001', 'r002']

def test_empty_reviews_summarization():
    """Test summarization with empty reviews."""
    empty_df = pd.DataFrame(columns=['review_id', 'text', 'rating', 'verified', 'helpful_votes'])
    themes_data = {'theme_stats': {}}
    
    config = SummarizationConfig()
    with patch('src.summarizer.pipeline') as mock_pipeline:
        mock_pipeline.return_value = Mock()
        summarizer = ChainOfDensitySummarizer(config)
    
    result = summarizer.summarize(empty_df, themes_data, "test_product")
    
    assert result['summary'] == "No reviews available for summarization."
    assert result['claims'] == []
    assert result['theme_stats'] == {}

def test_create_summarizer():
    """Test create_summarizer helper function."""
    with patch('src.summarizer.pipeline') as mock_pipeline:
        mock_pipeline.return_value = Mock()
        summarizer = create_summarizer(
            backend="local",
            model_name="test-model",
            max_length=100
        )
        
        assert summarizer.config.backend == "local"
        assert summarizer.config.model_name == "test-model"
        assert summarizer.config.max_summary_length == 100
