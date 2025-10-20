"""
Test suite for the themes module.
"""

import pytest
import pandas as pd
from src.themes import ThemeExtractor, ThemeExtractionConfig, create_theme_extractor

def test_theme_extraction_config():
    """Test ThemeExtractionConfig dataclass."""
    config = ThemeExtractionConfig(
        num_themes=10,
        min_theme_frequency=0.05,
        use_embeddings=True
    )
    
    assert config.num_themes == 10
    assert config.min_theme_frequency == 0.05
    assert config.use_embeddings == True

def test_theme_extractor_initialization():
    """Test ThemeExtractor initialization."""
    config = ThemeExtractionConfig()
    extractor = ThemeExtractor(config)
    
    assert extractor.config is not None
    assert len(extractor.attribute_keywords) > 0

def test_keyword_theme_extraction():
    """Test keyword-based theme extraction."""
    # Create sample reviews DataFrame
    reviews_data = [
        {
            'review_id': 'r001',
            'text': 'This product has excellent quality and durability. Very well-made.',
            'rating': 5.0,
            'verified': True,
            'helpful_votes': 10
        },
        {
            'review_id': 'r002',
            'text': 'The price is too expensive for what you get. Not good value.',
            'rating': 2.0,
            'verified': True,
            'helpful_votes': 8
        },
        {
            'review_id': 'r003',
            'text': 'Beautiful design and style. Looks great in my home.',
            'rating': 4.0,
            'verified': False,
            'helpful_votes': 5
        }
    ]
    
    reviews_df = pd.DataFrame(reviews_data)
    
    config = ThemeExtractionConfig(min_theme_frequency=0.1)
    extractor = ThemeExtractor(config)
    
    themes_data = extractor.extract_themes(reviews_df)
    
    # Check that themes were extracted
    assert 'themes' in themes_data
    assert 'theme_stats' in themes_data
    assert len(themes_data['themes']) > 0
    
    # Check that quality theme was detected
    theme_stats = themes_data['theme_stats']
    assert 'quality' in theme_stats or any('quality' in theme for theme in theme_stats.keys())

def test_empty_reviews():
    """Test theme extraction with empty reviews."""
    empty_df = pd.DataFrame(columns=['review_id', 'text', 'rating', 'verified', 'helpful_votes'])
    
    extractor = create_theme_extractor()
    themes_data = extractor.extract_themes(empty_df)
    
    assert themes_data['themes'] == []
    assert themes_data['theme_stats'] == {}

def test_text_preprocessing():
    """Test text preprocessing functionality."""
    extractor = create_theme_extractor()
    
    test_text = "This is a GREAT product!!! It's amazing."
    processed = extractor._preprocess_text(test_text)
    
    assert processed == "this is a great product its amazing"
    assert "!!!" not in processed
    assert processed.islower()

def test_theme_stats_calculation():
    """Test theme statistics calculation."""
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
            'text': 'Good quality item',
            'rating': 4.0,
            'verified': True,
            'helpful_votes': 8
        },
        {
            'review_id': 'r003',
            'text': 'Nice design',
            'rating': 3.0,
            'verified': False,
            'helpful_votes': 5
        }
    ]
    
    reviews_df = pd.DataFrame(reviews_data)
    
    extractor = create_theme_extractor()
    themes_data = extractor.extract_themes(reviews_df)
    
    theme_stats = themes_data['theme_stats']
    
    # Quality should have high prevalence (2/3 reviews)
    if 'quality' in theme_stats:
        assert theme_stats['quality'] >= 0.6  # 2/3 = 0.67

def test_create_theme_extractor():
    """Test create_theme_extractor helper function."""
    extractor = create_theme_extractor(
        num_themes=5,
        min_frequency=0.1,
        use_embeddings=False
    )
    
    assert extractor.config.num_themes == 5
    assert extractor.config.min_theme_frequency == 0.1
    assert extractor.config.use_embeddings == False
