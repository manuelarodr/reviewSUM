"""
Utility functions for data loading and processing.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

def load_product_data(file_path: str) -> Dict[str, Any]:
    """
    Load product data from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing website_summaries, customer_reviews, and product_meta
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def reviews_to_dataframe(reviews: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert reviews list to pandas DataFrame.
    
    Args:
        reviews: List of review dictionaries
        
    Returns:
        DataFrame with review data
    """
    df = pd.DataFrame(reviews)
    
    # Add review IDs if not present
    if 'review_id' not in df.columns:
        df['review_id'] = [f"r{i:03d}" for i in range(len(df))]
    
    return df

def extract_review_texts(reviews_df: pd.DataFrame) -> List[str]:
    """
    Extract review texts from DataFrame.
    
    Args:
        reviews_df: DataFrame containing reviews
        
    Returns:
        List of review texts
    """
    return reviews_df['text'].tolist()

def get_website_summary_verdict(website_summaries: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract the verdict (gold summary) from website summaries.
    
    Args:
        website_summaries: List of website summary dictionaries
        
    Returns:
        The verdict text or None if not found
    """
    if not website_summaries:
        return None
    
    # Use the first summary's verdict as gold standard
    return website_summaries[0].get('verdict', None)

def validate_data_structure(data: Dict[str, Any]) -> bool:
    """
    Validate that the data has the expected structure.
    
    Args:
        data: Dictionary containing product data
        
    Returns:
        True if structure is valid, False otherwise
    """
    required_keys = ['website_summaries', 'customer_reviews', 'product_meta']
    
    for key in required_keys:
        if key not in data:
            print(f"Missing required key: {key}")
            return False
    
    # Check that customer_reviews is a list
    if not isinstance(data['customer_reviews'], list):
        print("customer_reviews must be a list")
        return False
    
    # Check that reviews have required fields
    if data['customer_reviews']:
        required_review_fields = ['title', 'text', 'rating', 'verified', 'helpful_votes']
        first_review = data['customer_reviews'][0]
        
        for field in required_review_fields:
            if field not in first_review:
                print(f"Missing required field in reviews: {field}")
                return False
    
    return True

def load_sample_dataset(data_dir: str = "data/amasum-5productsample") -> List[str]:
    """
    Load all JSON files from the sample dataset directory.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        List of file paths
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    
    return [str(f) for f in json_files]
