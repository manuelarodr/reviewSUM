"""
summarization module using Chain-of-Density prompting.

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
        product_name: str = "product"
    ) -> Dict[str, Any]:
        """
        Generate summary using Chain-of-Density prompting.
        Args:
            reviews_df: DataFrame containing filtered reviews
            product_name: Name of the product being summarized
        Returns:
            Dictionary containing summary, claims, and metadata
        """
        if len(reviews_df) == 0:
            return {
                "summary": "No reviews available for summarization.",
                "entities": [],
                "model_used": self.config.model_name
            }
        
        # Prepare review texts for summarization
        review_texts = self._prepare_review_texts(reviews_df)
        
        # Generate summary using Groq
        summary_result = self._summarize_with_groq(review_texts, product_name)
        
        # Extract entities and map to supporting reviews
        entities = self._extract_entities(summary_result, reviews_df)
        
        actual_summary = summary_result.get("summary", "")

        return {
            "summary": actual_summary,
            "entities": entities,
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
            return {"summary": f"Error generating summary: {e}", "entities": [], "error_details": error_details}
    
    def summarize_vanilla(
        self,
        reviews_df: pd.DataFrame,
        product_name: str = "product"
    ) -> Dict[str, Any]:
        """
        Generate summary using vanilla prompt (simple summarization).
        Args:
            reviews_df: DataFrame containing filtered reviews
            product_name: Name of the product being summarized
        Returns:
            Dictionary containing summary and metadata
        """
        if len(reviews_df) == 0:
            return {
                "summary": "No reviews available for summarization.",
                "model_used": self.config.model_name
            }
        
        # Prepare review texts for summarization
        review_texts = self._prepare_review_texts(reviews_df)
        
        # Generate summary using vanilla prompt
        summary_result = self._summarize_with_vanilla_prompt(review_texts, product_name)
        
        return {
            "summary": summary_result,
            "model_used": self.config.model_name
        }
    
    def _summarize_with_vanilla_prompt(self, review_texts: str, product_name: str) -> str:
        """
        Generate summary using vanilla prompt.
        
        Args:
            review_texts: Formatted review texts
            product_name: Name of the product
            
        Returns:
            Summary text
        """
        prompt = f"""Summarize the following product reviews of {product_name} into a single, coherent summary of under 120 words. Capture the main themes, common praises, and criticisms mentioned by users. Avoid repetition, personal opinions, or unrelated details. Maintain a neutral and informative tone.

Customer reviews:
{review_texts}"""
        
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
            return content.strip()
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error with Groq API: {e}\nDetails:\n{error_details}")
            return f"Error generating summary: {e}"

    def _create_chain_of_density_prompt(self, review_texts: str, product_name: str) -> str:
        """
        Create Chain-of-Density prompt for summarization.
        """
        prompt = f"""You will generate increasingly concise, entity-dense summaries of the customer reviews of the {product_name} product.

Repeat the following two steps 5 times:

1. Identify 1-3 informative entities from the reviews which are missing from the previously generated summary.
2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

An entity is any functional or non-functional feature of the product that customers mention in their reviews and perceive to either harm or enhance their overall experience with the product.

A missing entity is:
- relevant to the product functionality and useful dimension for a potential shopper,
- specific yet concise (<=5 words),
- novel (not in the previous summary),
- faithful (present in the reviews),
- anywhere (can occur anywhere in the reviews).

Guidelines:
- The first summary should be long (~80 words, 4-5 sentences) and intentionally vague, containing few specifics beyond the identified missing entities. Use filler phrases ("the reviews mention...") to reach 80 words.
- Each subsequent summary must maintain identical length while increasing information density through fusion, compression, and removal of redundant language.
- Never drop entities from previous summaries.
- When space is limited, add fewer new entities.
- The final summary must be highly dense yet self-contained and understandable without reading the reviews.
- Avoid product names.
- Exclude personal names, locations, URLs, or emails.

Only output the final (5th) dense summary and the unique list of entities you identified and are present in the summary. Structured as JSON:

{{
    "summary": "<final dense summary text>",
    "entities": ["<entity_1>", "<entity_2>", "..."]
}}

Return only valid JSON.

Customer Reviews:
{review_texts}"""

        return prompt

    def _extract_summary_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract summary and entities from non-JSON response text.
        """
        # Try to extract JSON containing summary/entities
        json_match = re.search(r'\{.*"summary".*"entities".*\}', text, re.DOTALL)
        if json_match:
            try:
                json_text = json_match.group(0)
                result = json.loads(json_text)
                return result
            except json.JSONDecodeError:
                pass

        summary_match = re.search(r'"summary":\\s*"([^"]*)"', text)
        entities_match = re.search(r'"entities":\\s*(\\[.*?\\])', text, re.DOTALL)

        if summary_match:
            summary_text = summary_match.group(1)
        else:
            summary_text = ""

        entities: List[str] = []
        if entities_match:
            try:
                entities = json.loads(entities_match.group(1))
            except json.JSONDecodeError:
                entities = []

        if not summary_text:
            # Fallback: take longest paragraph-like chunk
            paragraphs = text.split("\\n\\n")
            summary_text = max((p.strip() for p in paragraphs), key=len, default=text).strip()

        return {
            "summary": summary_text or text,
            "entities": entities,
        }
    def _extract_entities(self, summary_result: Dict[str, Any], reviews_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract entities from summary result (expects a flat list of strings).
        """
        entities_raw = summary_result.get("entities") or []
        entities: List[Dict[str, Any]] = []

        for ent in entities_raw:
            name = ent.get("entity") if isinstance(ent, dict) else ent
            if not name:
                continue
            entities.append(
                {
                    "entity": str(name),
                    "supporting_reviews": [],
                    "review_count": 0,
                    "percentage": 0.0,
                }
            )

        return entities

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
                
