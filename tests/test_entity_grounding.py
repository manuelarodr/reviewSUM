import numpy as np
import pandas as pd

from src.entity_grounding import ground_entities_with_embeddings


def test_ground_entities_with_embeddings_threshold_and_counts():
    # Three sentences, simple 2D embeddings
    sentences_df = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "text": ["great battery life", "bad screen quality", "battery okay"],
            "sentiment": ["positive", "negative", "neutral"],
        }
    )
    sentence_embeddings = np.array(
        [
            [1.0, 0.0],  # battery direction
            [0.0, 1.0],  # screen direction
            [0.8, 0.2],  # mixed, closer to battery
        ],
        dtype="float32",
    )

    entities = ["battery life", "screen"]
    # Precomputed entity embeddings to avoid model dependency in tests
    entity_embeddings = np.array(
        [
            [1.0, 0.0],  # matches battery sentences
            [0.0, 1.0],  # matches screen sentence
        ],
        dtype="float32",
    )

    feature_facts = ground_entities_with_embeddings(
        entities,
        sentences_df,
        sentence_embeddings=sentence_embeddings,
        entity_embeddings=entity_embeddings,
        similarity_threshold=0.3,  # default
        max_hits=5,
        max_evidence=3,
    )

    # Battery entity should capture sentences 0 and 2 (both above threshold)
    battery_fact = next(f for f in feature_facts if f.name == "battery life")
    assert battery_fact.review_count == 2
    assert battery_fact.positive_count == 1
    assert battery_fact.neutral_count == 1
    assert battery_fact.negative_count == 0
    assert battery_fact.supporting_review_ids == ["r1", "r3"]
    assert len(battery_fact.evidence_sentences) >= 1

    # Screen entity should capture sentence 1 only
    screen_fact = next(f for f in feature_facts if f.name == "screen")
    assert screen_fact.review_count == 1
    assert screen_fact.negative_count == 1
    assert screen_fact.positive_count == 0
    assert screen_fact.neutral_count == 0
    assert screen_fact.supporting_review_ids == ["r2"]
