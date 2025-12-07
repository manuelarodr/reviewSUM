# -*- coding: utf-8 -*-
"""
Streamlit UI for Credibility-Aware Review Summarizer
Direct in-memory processing without CSV files
"""

import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
from summarizer import summarize_reviews

st.set_page_config(page_title="Review Summarizer", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if "results_df" not in st.session_state:
    st.session_state.results_df = None

st.title("📊 Credibility-Aware Review Summarizer")
st.markdown("Upload a JSON/JSONL file containing product reviews to generate AI-powered summaries (no CSV files saved)")

# =====================
# Sidebar Configuration
# =====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    filter_mode = st.selectbox(
        "Filter Mode",
        ["strict", "inclusive"],
        help="strict: only verified purchases | inclusive: all reviews"
    )
    
    st.divider()
    st.subheader("📋 Data Format Requirements")
    st.markdown("""
    **Required fields:**
    - `title` - Review title
    - `text` - Review content
    - `rating` - Rating (1-5)
    - `parent_asin` - Product ID (or derived from filename/metadata)
    
    **Optional fields:**
    - `verified_purchase` or `verified` - Verified purchase flag
    - `helpful_vote` or `helpful_votes` - Helpful votes count
    
    **Single Product JSON:**
    Can upload a single product JSON with `customer_reviews` array
    """)

# =====================
# Main Content
# =====================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload & Process")
    
    uploaded_file = st.file_uploader(
        "Choose a JSON or JSONL file",
        type=["jsonl", "json"],
        help="JSON array, JSONL (one object per line), or single-product JSON with customer_reviews"
    )
    
    if uploaded_file:
        st.success(f"✅ File loaded: {uploaded_file.name}")
        
        # Preview data
        with st.expander("👁️ Preview Data"):
            try:
                file_content = uploaded_file.getvalue().decode('utf-8')
                
                # Try parsing as JSON first
                try:
                    obj = json.loads(file_content)
                    if isinstance(obj, dict) and 'customer_reviews' in obj:
                        st.write(f"Single product JSON with {len(obj.get('customer_reviews', []))} reviews")
                        st.json(obj.get('customer_reviews', [])[:1])
                    elif isinstance(obj, list):
                        st.write(f"JSON array with {len(obj)} items")
                        st.json(obj[:1])
                    else:
                        st.json(obj)
                except:
                    # Try JSONL
                    lines = [l for l in file_content.split('\n') if l.strip()]
                    st.write(f"JSONL file with {len(lines)} lines")
                    for i, line in enumerate(lines[:2]):
                        st.json(json.loads(line))
                        
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Process button
        if st.button("🚀 Run Summarization", key="process_btn", use_container_width=True):
            try:
                status_text = st.empty()
                
                status_text.text("⏳ Loading file...")
                
                # Parse input file
                file_content = uploaded_file.getvalue().decode('utf-8')
                
                # Try as JSON first
                try:
                    obj = json.loads(file_content)
                    
                    # Single product JSON with customer_reviews
                    if isinstance(obj, dict) and 'customer_reviews' in obj:
                        reviews = obj.get('customer_reviews', [])
                        pm = obj.get('product_meta', {}) or {}
                        specs = pm.get('specifications', {}) or {}
                        parent_asin = specs.get('ISBN-10') or specs.get('ISBN-13') or pm.get('title') or os.path.splitext(uploaded_file.name)[0]
                        
                        rows = []
                        for r in reviews:
                            rows.append({
                                'parent_asin': str(parent_asin),
                                'title': r.get('title', ''),
                                'text': r.get('text', ''),
                                'rating': r.get('rating') or r.get('rating_value'),
                                'verified_purchase': r.get('verified', r.get('verified_purchase', False)),
                                'verified': r.get('verified', r.get('verified_purchase', False)),
                                'helpful_vote': r.get('helpful_votes', r.get('helpful_vote', 0))
                            })
                        df = pd.DataFrame(rows)
                    
                    # JSON array
                    elif isinstance(obj, list):
                        df = pd.DataFrame(obj)
                    
                    else:
                        st.error("Unknown JSON structure")
                        st.stop()
                
                except json.JSONDecodeError:
                    # Try JSONL
                    rows = []
                    for line in file_content.split('\n'):
                        if line.strip():
                            rows.append(json.loads(line))
                    df = pd.DataFrame(rows)
                
                # Ensure required columns
                for col in ['text', 'rating', 'parent_asin']:
                    if col not in df.columns:
                        if col == 'parent_asin':
                            df['parent_asin'] = os.path.splitext(uploaded_file.name)[0]
                        else:
                            st.error(f"Missing required column: {col}")
                            st.stop()
                
                # Add missing columns with defaults
                if 'verified_purchase' not in df.columns:
                    df['verified_purchase'] = False
                if 'verified' not in df.columns:
                    df['verified'] = df.get('verified_purchase', False)
                
                status_text.text(f"⏳ Processing {len(df)} reviews...")
                
                # Run summarization
                results_df = summarize_reviews(
                    df,
                    filter_mode=filter_mode,
                    min_reviews=1,
                    show_progress=True
                )
                
                st.session_state.results_df = results_df
                
                status_text.text("✅ Processing completed!")
                st.success(f"✨ Generated summaries for {len(results_df)} product(s)")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

with col2:
    st.subheader("📊 Results Preview")
    
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        
        st.metric("Total Products", len(df))
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.metric("Avg Rating", f"{df['avg_rating'].mean():.2f}")
        with col_right:
            st.metric("Avg Credibility", f"{df['credibility_score'].mean():.2f}")
        
        # Display dataframe (compact view)
        display_cols = ['parent_asin', 'avg_rating', 'review_count', 'credibility_score', 'themes_pct']
        st.dataframe(df[display_cols], use_container_width=True, height=300)
        
        # Download button
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="review_summaries.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("⬅️ Upload and process a file to see results here")

# =====================
# Detailed View
# =====================
if st.session_state.results_df is not None:
    st.divider()
    st.subheader("🔍 Detailed Summary View")
    
    df = st.session_state.results_df
    
    # Select product
    product_id = st.selectbox(
        "Select a product to view details",
        df['parent_asin'].tolist(),
        key="product_select"
    )
    
    if product_id:
        product_row = df[df['parent_asin'] == product_id].iloc[0]
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Summary", "📊 Metrics", "💬 Evidence", "🎯 Themes"])
        
        with tab1:
            st.subheader("Formatted Summary")
            st.markdown(f"**{product_row['formatted_summary']}**")
            
            st.divider()
            
            st.markdown("<span style='color: #888; font-size: 0.85em;'><b>Our Summary</b></span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #666; font-size: 0.9em;'>{product_row['summary']}</span>", unsafe_allow_html=True)
        
        with tab2:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Average Rating", f"{product_row['avg_rating']:.1f}/5")
            with col_m2:
                st.metric("Review Count", int(product_row['review_count']))
            with col_m3:
                st.metric("Credibility Score", f"{product_row['credibility_score']:.2f}")
            with col_m4:
                st.metric("Product ID", product_row['parent_asin'])
        
        with tab3:
            col_ev1, col_ev2 = st.columns(2)
            
            with col_ev1:
                st.subheader("✅ Positive Snippets")
                pos_snippets = [s.strip() for s in str(product_row['pos_snippets']).split('|') if s.strip()]
                if pos_snippets:
                    for i, snippet in enumerate(pos_snippets, 1):
                        st.markdown(f"**{i}.** {snippet}")
                else:
                    st.write("No positive snippets found")
            
            with col_ev2:
                st.subheader("❌ Negative Snippets")
                neg_snippets = [s.strip() for s in str(product_row['neg_snippets']).split('|') if s.strip()]
                if neg_snippets:
                    for i, snippet in enumerate(neg_snippets, 1):
                        st.markdown(f"**{i}.** {snippet}")
                else:
                    st.write("No negative snippets found")
            
            st.subheader("🎯 Representative Sentences")
            evidence = [s.strip() for s in str(product_row['evidence']).split('|') if s.strip()]
            if evidence:
                for i, sent in enumerate(evidence, 1):
                    st.markdown(f"**{i}.** {sent}")
            else:
                st.write("No evidence sentences found")
        
        with tab4:
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.subheader("Count by Theme")
                st.write(product_row['themes_counts'])
            
            with col_t2:
                st.subheader("Percentage Distribution")
                st.write(product_row['themes_pct'])

# =====================
# Footer
# =====================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Made with Streamlit | BART-Large-CNN Summarization | In-Memory Processing (No CSV files)
</div>
""", unsafe_allow_html=True)
