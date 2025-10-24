#!/usr/bin/env python3
"""
Test script to compare vanilla prompt vs chain-of-density summarization
using the first product from amasum-5productsample dataset.
"""

import json
import pandas as pd
import os
from dotenv import load_dotenv
from src.summarizer import ChainOfDensitySummarizer, SummarizationConfig
from src.utils.data_loader import load_product_data, reviews_to_dataframe

def load_first_product_data():
    """Load the first product from amasum-5productsample."""
    sample_dir = "amasum-5productsample"
    files = [f for f in os.listdir(sample_dir) if f.endswith('.json')]
    
    if not files:
        raise FileNotFoundError("No JSON files found in amasum-5productsample directory")
    
    first_file = sorted(files)[0]  # Get first file alphabetically
    file_path = os.path.join(sample_dir, first_file)
    
    print(f"Loading data from: {file_path}")
    
    # Use the data_loader function
    data = load_product_data(file_path)
    
    # Extract product name from the data
    product_name = "BOSH! cookbook"  # Based on the sample data we saw
    
    # Convert to DataFrame format expected by the summarizer
    reviews_df = reviews_to_dataframe(data['customer_reviews'])
    print(f"Loaded {len(reviews_df)} reviews for product: {product_name}")
    
    return reviews_df, product_name

def main():
    """Main function to run the comparison test."""
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        print("Please set your Groq API key in a .env file or environment variable")
        return
    
    try:
        # Load the first product data
        reviews_df, product_name = load_first_product_data()
        
        # Initialize components
        config = SummarizationConfig(
            model_name="llama-3.1-8b-instant",
            max_summary_length=120,
            temperature=0.1
        )
        summarizer = ChainOfDensitySummarizer(config)
        
        # Create dummy themes data (since we're focusing on summarization)
        themes_data = {"theme_stats": {}}
        
        # Prepare review texts to see what's being sent to LLM
        review_texts = summarizer._prepare_review_texts(reviews_df)
        review_text_length = len(review_texts)
        review_text_words = len(review_texts.split())
        
        print("\n" + "="*80)
        print("COMPARISON: VANILLA PROMPT vs CHAIN-OF-DENSITY SUMMARIZATION")
        print("="*80)
        print(f"Product: {product_name}")
        print(f"Number of reviews: {len(reviews_df)}")
        print(f"Total text length sent to LLM: {review_text_length:,} characters")
        print(f"Total word count sent to LLM: {review_text_words:,} words")
        print(f"Model: {config.model_name}")
        print("="*80)
        
        # Generate vanilla summary
        print("\n[1/2] Generating VANILLA PROMPT summary...")
        vanilla_result = summarizer.summarize_vanilla(reviews_df, themes_data, product_name)
        vanilla_summary = vanilla_result['summary']
        
        # Generate chain-of-density summary
        print("[2/2] Generating CHAIN-OF-DENSITY summary...")
        cod_result = summarizer.summarize(reviews_df, themes_data, product_name)
        cod_summary = cod_result['summary']
        
        # Display results
        print("\n" + "="*80)
        print("RESULTS COMPARISON")
        print("="*80)
        
        print("\n[VANILLA PROMPT SUMMARY]:")
        print("-" * 50)
        print(vanilla_summary)
        print(f"\nWord count: {len(vanilla_summary.split())}")
        
        print("\n[CHAIN-OF-DENSITY SUMMARY]:")
        print("-" * 50)
        print(cod_summary)
        print(f"\nWord count: {len(cod_summary.split())}")
        
        # Show entities extracted by CoD with supporting reviews
        if 'entities' in cod_result and cod_result['entities']:
            print("\n[ENTITIES EXTRACTED BY CHAIN-OF-DENSITY]:")
            print("-" * 50)
            for entity in cod_result['entities']:
                print(f"\n• {entity['entity']} (mentioned in {entity['review_count']} reviews)")
                print(f"  Supporting review IDs: {entity['supporting_reviews']}")
                
                # Show actual review texts for this entity
                print("  Supporting review texts:")
                for review_id in entity['supporting_reviews'][:3]:  # Show first 3 reviews
                    review_text = reviews_df[reviews_df['review_id'] == review_id]['text'].iloc[0]
                    print(f"    - {review_id}: {review_text}")
                
                if len(entity['supporting_reviews']) > 3:
                    print(f"    ... and {len(entity['supporting_reviews']) - 3} more reviews")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
