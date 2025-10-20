"""
Credibility-aware summarization module using Chain-of-Density prompting.

This module implements transparent, factual summarization from customer reviews using
a Chain-of-Density (CoD) procedure that iteratively compresses content while preserving
all distinct factual entities. Each claim maps to supporting review IDs and includes
prevalence data.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from groq import Groq
import os

@dataclass
class SummarizationConfig:
    """Configuration for summarization."""
    model_name: str = "llama-3.1-8b-instant"  # Groq model
    max_summary_length: int = 120
    chain_of_density_iterations: int = 5
    temperature: float = 0.1
    max_tokens: int = 4000  # Increased token limit
    max_prompt_chars: int = 50000  # Removed prompt character limit

class ChainOfDensitySummarizer:
    """
    Implements Chain-of-Density summarization for customer reviews.
    """
    
    def __init__(self, config: Optional[SummarizationConfig] = None):
        """
        Initialize the summarizer.
        Args:
            config: Summarization configuration
        """
        self.config = config or SummarizationConfig()
        self.groq_client = None
        self._initialize_groq()
    
    def _initialize_groq(self):
        """Initialize Groq client."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.groq_client = Groq(api_key=api_key)
    
    
    def summarize(
        self,
        reviews_df: pd.DataFrame,
        themes_data: Dict[str, Any],
        product_name: str = "product"
    ) -> Dict[str, Any]:
        """
        Generate summary using Chain-of-Density prompting.
        Args:
            reviews_df: DataFrame containing filtered reviews
            themes_data: Theme extraction results
            product_name: Name of the product being summarized
        Returns:
            Dictionary containing summary, claims, and metadata
        """
        if len(reviews_df) == 0:
            return {
                "summary": "No reviews available for summarization.",
                "claims": [],
                "theme_stats": {},
                "model_used": self.config.model_name
            }
        
        # Prepare review texts for summarization
        review_texts = self._prepare_review_texts(reviews_df)
        
        # Generate summary using Groq
        summary_result = self._summarize_with_groq(review_texts, product_name)
        
        # Extract entities and map to supporting reviews
        entities = self._extract_entities(summary_result, reviews_df)
        
        # Extract the actual summary text and entity_log from the nested structure
        if isinstance(summary_result.get("summary"), dict):
            actual_summary = summary_result["summary"].get("summary", "")
            entity_log = summary_result["summary"].get("entity_log", [])
        else:
            actual_summary = summary_result.get("summary", "")
            entity_log = summary_result.get("entity_log", [])
        
        return {
            "summary": actual_summary,
            "entities": entities,
            "entity_log": entity_log,
            "theme_stats": themes_data.get("theme_stats", {}),
            "model_used": self.config.model_name
        }
    
    def summarize_with_raw_output(
        self,
        reviews_df: pd.DataFrame,
        themes_data: Dict[str, Any],
        product_name: str = "product"
    ) -> Dict[str, Any]:
        """
        Generate summary and return raw LLM output for testing purposes.
        Args:
            reviews_df: DataFrame containing filtered reviews
            themes_data: Theme extraction results
            product_name: Name of the product being summarized
        Returns:
            Dictionary containing summary, raw output, and metadata
        """
        if len(reviews_df) == 0:
            return {
                "summary": "No reviews available for summarization.",
                "raw_output": "",
                "theme_stats": {},
                "model_used": self.config.model_name
            }
        
        # Prepare review texts for summarization
        review_texts = self._prepare_review_texts(reviews_df)
        
        # Generate summary using Groq and get raw output
        summary_result, raw_output = self._summarize_with_groq_raw(review_texts, product_name)
        
        # Extract entities and map to supporting reviews
        entities = self._extract_entities(summary_result, reviews_df)
        
        # Extract the actual summary text and entity_log from the nested structure
        if isinstance(summary_result.get("summary"), dict):
            actual_summary = summary_result["summary"].get("summary", "")
            entity_log = summary_result["summary"].get("entity_log", [])
        else:
            actual_summary = summary_result.get("summary", "")
            entity_log = summary_result.get("entity_log", [])
        
        return {
            "summary": actual_summary,
            "raw_output": raw_output,
            "entities": entities,
            "entity_log": entity_log,
            "theme_stats": themes_data.get("theme_stats", {}),
            "model_used": self.config.model_name
        }

    def _prepare_review_texts(self, reviews_df: pd.DataFrame) -> str:
        """
        Prepare review texts for summarization.
        
        Args:
            reviews_df: DataFrame containing reviews
            
        Returns:
            Formatted string of review texts
        """
        review_texts = []
        
        for _, review in reviews_df.iterrows():
            # Format each review with metadata
            review_text = f"Review_id: {review['review_id']}: {review['text']}"
            review_texts.append(review_text)
        
        combined = "\n\n".join(review_texts)
        # No truncation - use all reviews
        return combined
    
    def _summarize_with_groq(self, review_texts: str, product_name: str) -> Dict[str, Any]:
        """
        Generate summary using Groq API with Chain-of-Density prompting.
        
        Args:
            review_texts: Formatted review texts
            product_name: Name of the product
            
        Returns:
            Dictionary containing summary and claims
        """
        prompt = self._create_chain_of_density_prompt(review_texts, product_name)
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, extract summary from text
                return self._extract_summary_from_text(content)
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error with Groq API: {e}\nDetails:\n{error_details}")
            return {"summary": f"Error generating summary: {e}", "claims": [], "error_details": error_details}
    
    def _summarize_with_groq_raw(self, review_texts: str, product_name: str) -> Tuple[Dict[str, Any], str]:
        """
        Generate summary using Groq API with Chain-of-Density prompting and return raw output.
        
        Args:
            review_texts: Formatted review texts
            product_name: Name of the product
            
        Returns:
            Tuple of (parsed_result, raw_output)
        """
        prompt = self._create_chain_of_density_prompt(review_texts, product_name)
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                result = json.loads(content)
                return result, content
            except json.JSONDecodeError:
                # If JSON parsing fails, extract summary from text
                parsed_result = self._extract_summary_from_text(content)
                return parsed_result, content
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error with Groq API: {e}\nDetails:\n{error_details}")
            error_result = {"summary": f"Error generating summary: {e}", "claims": [], "error_details": error_details}
            return error_result, f"Error: {e}\n{error_details}"
    
    
    def _create_chain_of_density_prompt(self, review_texts: str, product_name: str) -> str:
        """
        Create Chain-of-Density prompt for summarization.
        
        Args:
            review_texts: Formatted review texts
            product_name: Name of the product
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You will generate increasingly concise, entity-dense summaries of the customer reviews of the {product_name} product.

Repeat the following two steps 5 times:

1. Identify 1–3 informative entities from the reviews which are missing from the previously generated summary. 
    For each new entity:
        - Record the `review_id`s for all reviews that mention the entity.
        - Add a record to a running `entity_log` in the format:
  ```json
  {{"iteration": <int>, "entity": "<entity_name>", "review_ids": ["<id1>", "<id2>", ...]}}
2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

An entity is any functional or non-functional feature of the product that users mention in their reviews and perceive to either harm or enhance their overall experience.

A missing entity is:
• must berelevant to the product's functionality,
• specific yet concise (≤5 words),
• novel (not in the previous summary),
• faithful (present in the reviews),
• anywhere (can occur anywhere in the reviews).

Guidelines:
- The first summary should be long (≈120 words, 4–5 sentences) and intentionally vague, containing few specifics beyond the identified missing entities. Use filler phrases ("the reviews mention...") to reach 120 words.
- Each subsequent summary must maintain identical length while increasing information density through fusion, compression, and removal of redundant language.
- Never drop entities from previous summaries.
- When space is limited, add fewer new entities.
- The final summary must be highly dense yet self-contained and understandable without reading the reviews.
- Avoid product names other than {product_name}.
- Exclude personal names, locations, URLs, or emails.

Only output the final(5th) dense summary and the entity_log. Structured as JSON:

{{
    "summary": "<final dense summary text>",
    "entity_log": [
    {{"iteration": 1, "entity": "<string>", "review_ids": ["<id1>", "<id2>"]}},
    {{"iteration": 2, "entity": "<string>", "review_ids": ["<id3>"]}},
    ...
  ],
}}

Return only valid JSON.

Customer Reviews:
{review_texts}"""
        
        return prompt
    
    def _extract_summary_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract summary from non-JSON response text.
        
        Args:
            text: Response text from model
            
        Returns:
            Dictionary with extracted summary and entity_log
        """
        # First try to extract JSON from the text (in case it's wrapped in other text)
        json_match = re.search(r'\{.*"summary".*"entity_log".*\}', text, re.DOTALL)
        if json_match:
            try:
                json_text = json_match.group(0)
                result = json.loads(json_text)
                return result
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON structure with regex patterns
        summary_match = re.search(r'"summary":\s*"([^"]*)"', text)
        entity_log_match = re.search(r'"entity_log":\s*(\[.*?\])', text, re.DOTALL)
        
        if summary_match and entity_log_match:
            try:
                summary_text = summary_match.group(1)
                entity_log_text = entity_log_match.group(1)
                entity_log = json.loads(entity_log_text)
                
                return {
                    "summary": summary_text,
                    "entity_log": entity_log
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to find the final summary in Chain-of-Density format
        lines = text.split('\n')
        
        # Look for explicit final summary markers
        final_summary = None
        for i, line in enumerate(lines):
            if any(marker in line.lower() for marker in ['final summary', 'summary 5', 'dense summary']):
                # Take the next few lines as the summary
                summary_lines = []
                for j in range(i+1, min(i+6, len(lines))):
                    if lines[j].strip():
                        summary_lines.append(lines[j].strip())
                if summary_lines:
                    final_summary = ' '.join(summary_lines)
                    break
        
        # If no explicit marker found, try to extract the last substantial paragraph
        if not final_summary:
            # Look for the longest paragraph that seems like a summary
            paragraphs = text.split('\n\n')
            for paragraph in reversed(paragraphs):
                paragraph = paragraph.strip()
                if len(paragraph) > 50 and not paragraph.startswith('{') and not paragraph.startswith('['):
                    final_summary = paragraph
                    break
        
        # Final fallback: use entire text
        if not final_summary:
            final_summary = text
        
        return {
            "summary": final_summary,
            "entity_log": []
        }
    def _extract_entities(self, summary_result: Dict[str, Any], reviews_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract and validate entities from summary result.
        
        Args:
            summary_result: Result from summarization
            reviews_df: DataFrame containing reviews
            
        Returns:
            List of validated entities with supporting reviews
        """
        # Handle nested structure - the actual summary and entity_log might be nested
        if isinstance(summary_result.get("summary"), dict):
            # If summary is a dict, extract the actual content
            actual_summary = summary_result["summary"].get("summary", "")
            entity_log = summary_result["summary"].get("entity_log", [])
        else:
            # If summary is a string, use the top-level entity_log
            actual_summary = summary_result.get("summary", "")
            entity_log = summary_result.get("entity_log", [])
        
        validated_entities = []
        
        for entity_entry in entity_log:
            entity_name = entity_entry.get("entity", "")
            review_ids = entity_entry.get("review_ids", [])
            
            # Validate supporting reviews exist
            valid_reviews = [review_id for review_id in review_ids if review_id in reviews_df['review_id'].values]
            
            if valid_reviews and entity_name:  # Only include entities with valid supporting reviews
                validated_entity = {
                    "entity": entity_name,
                    "iteration": entity_entry.get("iteration", 0),
                    "supporting_reviews": valid_reviews,
                    "review_count": len(valid_reviews),
                    "percentage": len(valid_reviews) / len(reviews_df) if len(reviews_df) > 0 else 0.0
                }
                validated_entities.append(validated_entity)
        
        return validated_entities

def create_summarizer(
    model_name: str = "llama-3.1-8b-instant",
    max_length: int = 120
    ) -> ChainOfDensitySummarizer:
    """
    Create a summarizer with common defaults.
    Args:
        model_name: Model name for Groq API
        max_length: Maximum summary length
    Returns:
        ChainOfDensitySummarizer instance
    """
    config = SummarizationConfig(
        model_name=model_name,
        max_summary_length=max_length
    )
    return ChainOfDensitySummarizer(config)
                # Remove stray duplicate backend selection logic at end of file
