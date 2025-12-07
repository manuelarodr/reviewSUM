"""
Persistence helpers for ProductFacts.

This module provides simple JSON-based caching so that expensive
feature detection, sentiment analysis, and Groq summarization
can be reused across runs.

Cache layout (relative to project root):
    data/cache/product_facts/<product_id>.json
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List

from .product_facts import ProductFacts, FeatureFact, SummaryObject, Claim


CACHE_DIR = Path("data") / "cache" / "product_facts"


def save_product_facts(
    product_facts: ProductFacts, cache_dir: Path = CACHE_DIR
) -> Path:
    """
    Serialize ProductFacts to JSON on disk.

    File path:
        <cache_dir>/<product_id>.json
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{product_facts.product_id}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(product_facts), f, ensure_ascii=False, indent=2)

    return path


def load_product_facts(
    product_id: str, cache_dir: Path = CACHE_DIR
) -> Optional[ProductFacts]:
    """
    Load ProductFacts from cache; return None if not present.
    """
    path = cache_dir / f"{product_id}.json"
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct nested dataclasses
    features: List[FeatureFact] = [
        FeatureFact(**feat) for feat in data.get("features", [])
    ]

    summary_data = data.get("summary", {}) or {}
    claims_data = summary_data.get("claims", []) or []
    claims: List[Claim] = [Claim(**c) for c in claims_data]

    summary = SummaryObject(
        text=summary_data.get("text", "") or "",
        claims=claims,
    )

    return ProductFacts(
        product_id=str(data.get("product_id", "")),
        total_reviews=int(data.get("total_reviews", 0)),
        features=features,
        summary=summary,
    )

