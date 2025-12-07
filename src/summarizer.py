"""
Chain-of-Density summarization for customer reviews.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from groq import Groq


@dataclass
class SummarizationConfig:
    model_name: str = "llama-3.1-8b-instant"
    max_summary_length: int = 120
    chain_of_density_iterations: int = 5
    temperature: float = 0.1
    max_tokens: int = 4000
    max_prompt_chars: int = 50000


class ChainOfDensitySummarizer:
    """Implements Chain-of-Density summarization for customer reviews."""

    def __init__(self, config: Optional[SummarizationConfig] = None):
        self.config = config or SummarizationConfig()
        self.groq_client: Optional[Groq] = None
        self._initialize_groq()

    def _initialize_groq(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.groq_client = Groq(api_key=api_key)

    def summarize(
        self,
        reviews_df: pd.DataFrame,
        themes_data: Dict[str, Any],
        product_name: str = "product",
    ) -> Dict[str, Any]:
        if len(reviews_df) == 0:
            return {
                "summary": "No reviews available for summarization.",
                "entities": [],
                "entity_log": [],
                "theme_stats": {},
                "model_used": self.config.model_name,
            }

        review_texts = self._prepare_review_texts(reviews_df)
        summary_result = self._summarize_with_groq(review_texts, product_name)
        entities = self._extract_entities(summary_result)

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
            "model_used": self.config.model_name,
        }

    def summarize_with_raw_output(
        self,
        reviews_df: pd.DataFrame,
        themes_data: Dict[str, Any],
        product_name: str = "product",
    ) -> Dict[str, Any]:
        if len(reviews_df) == 0:
            return {
                "summary": "No reviews available for summarization.",
                "raw_output": "",
                "theme_stats": {},
                "model_used": self.config.model_name,
            }

        review_texts = self._prepare_review_texts(reviews_df)
        summary_result, raw_output = self._summarize_with_groq_raw(review_texts, product_name)
        entities = self._extract_entities(summary_result)

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
            "model_used": self.config.model_name,
        }

    def _prepare_review_texts(self, reviews_df: pd.DataFrame) -> str:
        review_texts = []
        for _, review in reviews_df.iterrows():
            review_text = f"Review_id: {review['review_id']}: {review['text']}"
            review_texts.append(review_text)
        return "\n\n".join(review_texts)

    def _summarize_with_groq(self, review_texts: str, product_name: str) -> Dict[str, Any]:
        prompt = self._create_chain_of_density_prompt(review_texts, product_name)
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = response.choices[0].message.content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return self._extract_summary_from_text(content)
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return {"summary": f"Error generating summary: {e}", "claims": [], "error_details": error_details}

    def _summarize_with_groq_raw(
        self, review_texts: str, product_name: str
    ) -> Tuple[Dict[str, Any], str]:
        prompt = self._create_chain_of_density_prompt(review_texts, product_name)
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = response.choices[0].message.content
            try:
                result = json.loads(content)
                return result, content
            except json.JSONDecodeError:
                parsed_result = self._extract_summary_from_text(content)
                return parsed_result, content
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            error_result = {"summary": f"Error generating summary: {e}", "claims": [], "error_details": error_details}
            return error_result, f"Error: {e}\n{error_details}"

    def summarize_vanilla(
        self,
        reviews_df: pd.DataFrame,
        themes_data: Dict[str, Any],
        product_name: str = "product",
    ) -> Dict[str, Any]:
        if len(reviews_df) == 0:
            return {
                "summary": "No reviews available for summarization.",
                "theme_stats": {},
                "model_used": self.config.model_name,
            }

        review_texts = self._prepare_review_texts(reviews_df)
        summary_result = self._summarize_with_vanilla_prompt(review_texts, product_name)

        return {
            "summary": summary_result,
            "theme_stats": themes_data.get("theme_stats", {}),
            "model_used": self.config.model_name,
        }

    def _summarize_with_vanilla_prompt(self, review_texts: str, product_name: str) -> str:
        prompt = f"""Summarize the following product reviews of {product_name} into a single, coherent summary of under 120 words. Capture the main themes, common praises, and criticisms mentioned by users. Avoid repetition, personal opinions, or unrelated details. Maintain a neutral and informative tone.

Customer reviews:
{review_texts}"""
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return f"Error generating summary: {e}"

    def _create_chain_of_density_prompt(self, review_texts: str, product_name: str) -> str:
        prompt = f"""You will generate increasingly concise, entity-dense summaries of the customer reviews of the {product_name} product.

Repeat the following two steps 5 times:
1) Identify 1–3 informative entities from the reviews which are missing from the previously generated summary.
2) Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

Entities: any functional or non-functional feature of the product that customers mention and that affects their experience. A missing entity is relevant to shoppers (helps a potential buyer understand customer opinion on that feature), specific (≤5 words), novel vs. the previous summary, faithful to the reviews, and can come from anywhere in the reviews.

Guidelines:
- The first summary should be long (~120 words, 4–5 sentences) and intentionally vague except for the identified entities; use filler (“Customers mention...”) to reach length.
- Subsequent summaries: identical length, increase density via fusion/compression, remove redundancy.
- Never drop entities from previous summaries; when space is limited, add fewer new entities.
- The final summary must be dense, self-contained, and understandable without reading the reviews.
- Avoid product names other than {product_name}. Exclude personal names, locations, URLs, or emails.

Only output the final (5th) dense summary and the entities mentioned in that final summary. Return valid JSON in this structure:
{{
  "summary": "<final dense summary text>",
  "entities": ["<entity_1>", "<entity_2>", ...]
}}

Customer Reviews:
{review_texts}"""
        return prompt

    def _extract_summary_from_text(self, text: str) -> Dict[str, Any]:
        # Try to extract JSON from the text
        json_match = re.search(r'\{.*"summary".*?\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try to find summary field explicitly
        summary_match = re.search(r'"summary":\s*"([^"]*)"', text)
        if summary_match:
            return {"summary": summary_match.group(1), "entity_log": []}

        # Fallback: longest paragraph
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        paragraphs.sort(key=len, reverse=True)
        fallback = paragraphs[0] if paragraphs else text
        return {"summary": fallback, "entity_log": []}

    def _extract_entities(self, summary_result: Dict[str, Any]) -> List[str]:
        entities_raw = summary_result.get("entities") or []
        names: List[str] = []

        for ent in entities_raw:
            if isinstance(ent, dict):
                name = ent.get("entity") or ent.get("name") or ent.get("text")
                if name:
                    names.append(str(name))
            elif isinstance(ent, str):
                ent = ent.strip()
                if ent:
                    names.append(ent)

        if names:
            return names

        # Fallback: derive from entity_log if present
        entity_log = summary_result.get("entity_log") or []
        for entry in entity_log:
            name = entry.get("entity")
            if name:
                names.append(str(name))

        return names


def create_summarizer(
    model_name: str = "llama-3.1-8b-instant", max_length: int = 120
) -> ChainOfDensitySummarizer:
    config = SummarizationConfig(model_name=model_name, max_summary_length=max_length)
    return ChainOfDensitySummarizer(config)
