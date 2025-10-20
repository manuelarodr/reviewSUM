#!/usr/bin/env python3
"""
Main entry point for the AI-powered review summarizer.

This script provides a command-line interface for running the complete summarization pipeline
with Chain-of-Density prompting, round-robin filtering, and entity traceability.
"""

import argparse
from dotenv import load_dotenv
import json
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from .env if present
load_dotenv()

from filtering import AdvancedReviewFilter, FilteringCriteria
from themes import ThemeExtractor, create_theme_extractor
from summarizer import ChainOfDensitySummarizer, create_summarizer
from utils.data_loader import load_product_data, reviews_to_dataframe

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="AI-Powered Review Summarizer")
    
    # Input/Output
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Filtering options
    parser.add_argument("--token-limit", type=int, default=50000, help="Token limit for API calls")
    parser.add_argument("--min-reviews-per-rating", type=int, default=2, help="Minimum reviews per rating")
    parser.add_argument("--max-reviews-per-rating", type=int, default=200, help="Maximum reviews per rating")
    parser.add_argument("--tfidf-weight", type=float, default=0.7, help="TF-IDF weight for hybrid scoring")
    
    # Summarization options
    parser.add_argument("--model", dest="model_name", default="llama-3.1-8b-instant", help="Groq model name")
    parser.add_argument("--max-length", type=int, default=120, help="Maximum summary length")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    
    # Theme extraction options
    parser.add_argument("--language", default="english", help="Language for theme extraction")
    
    args = parser.parse_args()
    
    try:
        # Load input data
        if args.verbose:
            print(f"Loading data from {args.input}")
        
        data = load_product_data(args.input)
        
        # Convert reviews to DataFrame
        reviews_df = reviews_to_dataframe(data['customer_reviews'])
        
        if args.verbose:
            print(f"Loaded {len(reviews_df)} reviews")
        
        # Step 1: Advanced Filtering with Round-Robin
        if args.verbose:
            print("Applying advanced filtering with round-robin selection...")
        
        criteria = FilteringCriteria(
            token_limit=args.token_limit,
            min_reviews_per_rating=args.min_reviews_per_rating,
            max_reviews_per_rating=args.max_reviews_per_rating,
            tfidf_weight=args.tfidf_weight
        )
        
        filterer = AdvancedReviewFilter(criteria)
        filtered_df, filter_stats = filterer.filter_reviews(reviews_df)
        
        if args.verbose:
            print(f"Filtered to {len(filtered_df)} reviews ({len(filtered_df)/len(reviews_df)*100:.1f}% retention)")
            print("Rating distribution after filtering:")
            rating_dist = filtered_df['rating'].value_counts().sort_index()
            for rating, count in rating_dist.items():
                print(f"  {int(rating)}⭐: {count} reviews")
        
        if len(filtered_df) == 0:
            print("Error: No reviews remain after filtering")
            sys.exit(1)
        
        # Step 2: Theme extraction
        if args.verbose:
            print("Extracting themes...")
        
        theme_extractor = create_theme_extractor(language=args.language)
        themes_data = theme_extractor.extract_themes(filtered_df)
        
        if args.verbose:
            print(f"Extracted themes: {list(themes_data.get('theme_stats', {}).keys())}")
        
        # Step 3: Chain-of-Density Summarization
        if args.verbose:
            print("Generating summary with Chain-of-Density prompting...")
        
        product_name = data.get('product_meta', {}).get('title', 'product')
        
        summarizer = create_summarizer(
            model_name=args.model_name,
            max_length=args.max_length
        )
        
        summary_result = summarizer.summarize(filtered_df, themes_data, product_name)
        
        if args.verbose:
            print("Summary generated successfully")
            print(f"Identified {len(summary_result.get('entities', []))} entities")
        
        # Prepare output
        output_data = {
            'summary_result': summary_result,
            'filter_stats': filter_stats,
            'themes_data': themes_data,
            'config': {
                'token_limit': args.token_limit,
                'min_reviews_per_rating': args.min_reviews_per_rating,
                'max_reviews_per_rating': args.max_reviews_per_rating,
                'tfidf_weight': args.tfidf_weight,
                'model': args.model_name,
                'max_length': args.max_length,
                'temperature': args.temperature,
                'language': args.language
            }
        }
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            if args.verbose:
                print(f"Results saved to {args.output}")
        else:
            # Print summary to stdout
            print("\n" + "="*50)
            print("SUMMARY")
            print("="*50)
            print(summary_result['summary'])
            
            if summary_result.get('entities'):
                print("\n" + "="*50)
                print("ENTITIES & SUPPORTING REVIEWS")
                print("="*50)
                for i, entity in enumerate(summary_result['entities'], 1):
                    print(f"{i}. {entity['entity']}")
                    print(f"   Iteration: {entity['iteration']}")
                    print(f"   Supporting reviews: {', '.join(entity['supporting_reviews'])}")
                    print(f"   Review count: {entity['review_count']}")
                    print(f"   Percentage: {entity['percentage']:.1%}")
                    print()
            
            # Show filtering stats
            print("\n" + "="*50)
            print("FILTERING STATISTICS")
            print("="*50)
            print(f"Original reviews: {filter_stats['original_count']}")
            print(f"Filtered reviews: {filter_stats['filtered_count']}")
            print(f"Retention rate: {filter_stats['retention_rate']:.1%}")
            print(f"Token usage: {filter_stats['token_usage']:.0f}")
            
            # Show theme stats
            if themes_data.get('theme_stats'):
                print("\n" + "="*50)
                print("THEME STATISTICS")
                print("="*50)
                for theme, score in themes_data['theme_stats'].items():
                    print(f"{theme}: {score:.3f}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
