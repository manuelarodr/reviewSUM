"""
Interactive dashboard for the credibility-aware review summarizer.

This module provides a Streamlit-based interface for exploring summaries,
filtering by credibility, and analyzing results.
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from summarizer import ChainOfDensitySummarizer, create_summarizer
from utils.data_loader import load_product_data, reviews_to_dataframe, get_website_summary_verdict
from utils.config import config
from filtering import AdvancedReviewFilter, FilteringCriteria

# Page configuration
st.set_page_config(
    page_title="Review Summarizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main dashboard application."""
    st.title("Review Summarizer")
    st.markdown("AI-powered product review summarization using Groq's Llama-3.1-8b-instant")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("File Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Product Data (JSON)",
            type=['json'],
            help="Upload a JSON file containing product reviews and metadata"
        )
        
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load and process data
            data = json.load(uploaded_file)
            
            # Display product information
            display_product_info(data)
            
            # Process reviews
            reviews_df = reviews_to_dataframe(data['customer_reviews'])
            
            # Apply TF-IDF filtering
            st.header("🔍 Review Filtering & Selection")
            criteria = FilteringCriteria(
                token_limit=4000,
                min_reviews_per_rating=2,
                max_reviews_per_rating=50,
                tfidf_weight=0.7
            )
            
            filterer = AdvancedReviewFilter(criteria)
            filtered_df, filter_stats = filterer.filter_reviews(reviews_df)
            
            # Display filtering breakdown
            display_filtering_breakdown(reviews_df, filter_stats)
            
            # Display human-generated summary
            display_human_summary(data)
            
            # Summarization
            st.header("📝 AI Summary Generation")
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    # Create empty themes data since we're not using theme extraction
                    themes_data = {"theme_stats": {}}
                    
                    summary_result = generate_summary(
                        filtered_df, themes_data, data.get('product_meta', {}).get('title', 'product'),
                        120  # Default summary length
                    )
                    
                    display_summary_results(summary_result, filtered_df)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Display sample data information
        st.info("👆 Please upload a JSON file to get started")
        display_sample_data_info()

def display_product_info(data: Dict[str, Any]):
    """Display product information."""
    st.header("📦 Product Information")
    product_meta = data.get('product_meta', {})
        # Display product title
    if 'title' in product_meta:
        st.write(f"**Product:** {product_meta['title']}")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Reviews", len(data['customer_reviews']))
    
    with col2:
        avg_rating = sum(review['rating'] for review in data['customer_reviews']) / len(data['customer_reviews'])
        st.metric("Average Rating", f"{avg_rating:.1f}/5")
    
    with col3:
        helpful_count = sum(1 for review in data['customer_reviews'] if review.get('helpful_votes', 0) > 0)
        st.metric("Reviews with Helpful Votes", f"{helpful_count}/{len(data['customer_reviews'])}")
    

def display_filtering_breakdown(reviews_df: pd.DataFrame, filter_stats: Dict[str, Any]):
    """Display filtering breakdown by rating strata showing TF-IDF sampling results."""
    
    # Calculate original rating distribution
    original_counts = reviews_df['rating'].value_counts().sort_index()
    total_original = len(reviews_df)
    
    # Create breakdown data
    breakdown_data = []
    for rating in sorted(original_counts.index):
        original_count = original_counts[rating]
        original_percentage = (original_count / total_original) * 100
        
        # Get filtered stats for this rating
        rating_key = f'rating_{int(rating)}'
        if rating_key in filter_stats.get('rating_distribution', {}):
            rating_stats = filter_stats['rating_distribution'][rating_key]
            selected_count = rating_stats.get('selected_reviews', 0)
            target_count = rating_stats.get('target_reviews', 0)
            original_proportion = rating_stats.get('original_proportion', 0) * 100
        else:
            selected_count = 0
            target_count = 0
            original_proportion = original_percentage
        
        breakdown_data.append({
            'Rating': f"{int(rating)}⭐",
            'Original': original_count,
            'Original %': f"{original_percentage:.1f}%",
            'Target': target_count,
            'Selected': selected_count,
            'Selection Rate': f"{(selected_count/original_count*100):.1f}%" if original_count > 0 else "0%"
        })
    
    breakdown_df = pd.DataFrame(breakdown_data)
    st.dataframe(breakdown_df, width='stretch')
    
    # Show summary statistics
    total_selected = sum(row['Selected'] for _, row in breakdown_df.iterrows())
    st.info(f"**Total Reviews Used for Summary:** {total_selected} out of {total_original} original reviews")

def display_human_summary(data: Dict[str, Any]):
    """Display human-generated summary from website."""
    st.subheader("👤 Human-Generated Summary")
    
    website_summaries = data.get('website_summaries', [])
    
    if website_summaries and len(website_summaries) > 0:
        # Get the first summary from the list
        summary = website_summaries[0]
        
        if 'verdict' in summary and summary['verdict']:
            st.write("**Verdict:**")
            st.write(summary['verdict'])
        
        if 'pros' in summary and summary['pros']:
            st.write("**Pros:**")
            if isinstance(summary['pros'], list):
                for pro in summary['pros']:
                    st.write(f"• {pro}")
            else:
                st.write(summary['pros'])
        
        if 'cons' in summary and summary['cons']:
            st.write("**Cons:**")
            if isinstance(summary['cons'], list):
                for con in summary['cons']:
                    st.write(f"• {con}")
            else:
                st.write(summary['cons'])
    else:
        st.info("No human-generated summary available for this product.")

def generate_summary(reviews_df: pd.DataFrame, themes_data: Dict[str, Any], 
                    product_name: str, max_length: int) -> Dict[str, Any]:
    """Generate summary using Chain-of-Density with Groq."""
    summarizer = create_summarizer(
        model_name="llama-3.1-8b-instant",
        max_length=max_length
    )
    
    return summarizer.summarize(reviews_df, themes_data, product_name)

def display_summary_generation_stats(summary_result: Dict[str, Any], reviews_df: pd.DataFrame):
    """Display statistics about summary generation."""
    st.subheader("📈 Summary Generation Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Reviews Analyzed", len(reviews_df))
    
    with col2:
        entities_count = len(summary_result.get('entities', []))
        st.metric("Entities Identified", entities_count)
    
    with col3:
        st.metric("Summary Length", f"{len(summary_result.get('summary', ''))} chars")

def display_summary_results(summary_result: Dict[str, Any], reviews_df: pd.DataFrame):
    """Display summary results with generation statistics."""
    
    # Show generation statistics first
    display_summary_generation_stats(summary_result, reviews_df)
    
    # Display the generated summary
    st.subheader("📝 Generated Summary")
    st.write(summary_result['summary'])
    
    # Display entities with full review text
    entities = summary_result.get('entities', [])
    if entities:
        st.subheader("🔍 Identified Entities & Supporting Reviews")
        
        # Create a mapping of review_id to review data for quick lookup
        review_map = {row['review_id']: row for _, row in reviews_df.iterrows()}
        
        for i, entity in enumerate(entities, 1):
            st.markdown(f"### Entity {i}: {entity['entity']}")
            
            # Entity metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Iteration Added", entity['iteration'])
            with col2:
                st.metric("Review Count", entity['review_count'])
            with col3:
                st.metric("Percentage", f"{entity['percentage']:.1%}")
            
            # Show all supporting reviews with full text
            st.write("**Supporting Reviews:**")
            for j, review_id in enumerate(entity['supporting_reviews'], 1):
                if review_id in review_map:
                    review = review_map[review_id]
                    
                    # Create an expandable section for each review
                    with st.expander(f"Review {j}: {review_id} (⭐{review['rating']})"):
                        st.write(f"**Review ID:** {review_id}")
                        st.write(f"**Rating:** {review['rating']}/5")
                        st.write(f"**Author:** {review.get('author', 'Unknown')}")
                        st.write(f"**Date:** {review.get('publication_date', 'Unknown')}")
                        st.write(f"**Helpful Votes:** {review.get('helpful_votes', 0)}")
                        
                        st.write("**Full Review Text:**")
                        st.write(review['text'])
                        
                        # Show review title if available
                        if 'title' in review and review['title']:
                            st.write(f"**Title:** {review['title']}")
                else:
                    st.warning(f"Review {review_id} not found in dataset")
            
            st.divider()  # Add separator between entities
    
    # Display entity log for debugging
    entity_log = summary_result.get('entity_log', [])
    if entity_log:
        with st.expander("🔧 Entity Development Log (Debug)"):
            st.write("**Chain-of-Density Entity Development:**")
            for entry in entity_log:
                st.write(f"**Iteration {entry.get('iteration', '?')}:** {entry.get('entity', '')}")
                st.write(f"  Supporting Reviews: {', '.join(entry.get('review_ids', []))}")
                st.write("---")

def display_sample_data_info():
    """Display information about sample data format."""
    st.header("📋 Sample Data Format")
    
    st.markdown("""
    The system expects JSON files with the following structure:
    
    ```json
    {
      "website_summaries": [
        {
          "verdict": "Human-written summary",
          "pros": ["pro1", "pro2"],
          "cons": ["con1", "con2"],
          "source": "source_name"
        }
      ],
      "customer_reviews": [
        {
          "title": "Review title",
          "text": "Review content",
          "rating": 5.0,
          "verified": true,
          "author": "Author name",
          "helpful_votes": 79,
          "publication_date": 20180501
        }
      ],
      "product_meta": {
        "title": "Product title",
        "rating": 4.5,
        "categories": ["category1", "category2"]
      }
    }
    ```
    
    You can use the sample files in the `amasum-5productsample` directory for testing.
    """)

if __name__ == "__main__":
    main()
