"""Unit tests for Human-vs-LLM Judge Agreement workflow and calculations."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.compare_human_llm_judge import (
    compute_dimension_metrics,
    compute_agreement,
)
from scripts.select_human_subset import generate_human_subset


@pytest.fixture
def golden_ids_set():
    """Load all 200 Golden IDs for verification."""
    golden_path = Path("data/golden/golden_evaluation_200.jsonl")
    ids = set()
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["golden_id"])
    return ids


def test_subset_contains_exactly_50_unique_ids(golden_ids_set):
    """Verify results/human_judge_subset.json has exactly 50 valid Golden IDs."""
    subset_path = Path("results/human_judge_subset.json")
    assert subset_path.exists(), "results/human_judge_subset.json does not exist"

    with open(subset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = data["selected_ids"]
    assert len(ids) == 50
    assert len(set(ids)) == 50
    # All IDs must exist in Golden Set
    for gid in ids:
        assert gid in golden_ids_set, f"{gid} not found in Golden Set"


def test_annotation_template_schema_and_valid_values(tmp_path: Path):
    """Verify data/golden/human_reply_annotations.jsonl has 50 records and valid human fields."""
    ann_path = Path("data/golden/human_reply_annotations.jsonl")
    assert ann_path.exists()

    records = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 50
    human_score_keys = [
        "human_correctness",
        "human_relevance",
        "human_historical_grounding",
        "human_helpfulness",
        "human_unsupported_claims",
        "human_escalation_appropriateness",
    ]

    for r in records:
        assert "golden_id" in r
        assert "customer_text" in r
        assert "pred_reply" in r
        for hk in human_score_keys:
            assert hk in r
            if r[hk] is not None:
                assert 1 <= r[hk] <= 5
        if r.get("human_overall_score") is not None:
            assert 1.0 <= r["human_overall_score"] <= 5.0
        if r.get("human_decision") is not None:
            assert r["human_decision"] in {"pass", "borderline", "fail"}


def test_weighted_kappa_and_spearman_calculation():
    """Test dimension agreement calculation on synthetic scores."""
    # Perfect agreement
    h_scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    l_scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    res_perfect = compute_dimension_metrics(h_scores, l_scores)
    assert res_perfect["exact_agreement_pct"] == 100.0
    assert res_perfect["weighted_cohen_kappa"] == 1.0
    assert res_perfect["spearman_correlation"] == 1.0
    assert res_perfect["mean_absolute_difference"] == 0.0

    # Slight disagreement
    l_scores_slight = [4.0, 4.0, 3.0, 2.0, 2.0]
    res_slight = compute_dimension_metrics(h_scores, l_scores_slight)
    assert res_slight["exact_agreement_pct"] == 60.0
    assert res_slight["weighted_cohen_kappa"] > 0.7
    assert res_slight["mean_absolute_difference"] == 0.4


def test_compute_agreement_on_synthetic_annotations():
    """Test compute_agreement with small synthetic completed annotations."""
    annotations = [
        {
            "golden_id": "gold_001",
            "gold_intent": "delivery_status_tracking",
            "gold_escalation": "escalate",
            "has_retrieved_evidence": True,
            "human_correctness": 5,
            "human_relevance": 5,
            "human_historical_grounding": 4,
            "human_helpfulness": 5,
            "human_unsupported_claims": 5,
            "human_escalation_appropriateness": 5,
            "human_overall_score": 4.83,
            "human_decision": "pass",
            "human_notes": "Great answer",
        },
        {
            "golden_id": "gold_002",
            "gold_intent": "payment_billing_promotions",
            "gold_escalation": "auto_handle",
            "has_retrieved_evidence": True,
            "human_correctness": 3,
            "human_relevance": 4,
            "human_historical_grounding": 4,
            "human_helpfulness": 3,
            "human_unsupported_claims": 4,
            "human_escalation_appropriateness": 3,
            "human_overall_score": 3.5,
            "human_decision": "borderline",
            "human_notes": "Missed link",
        },
    ]

    judgments = {
        "gold_001": {
            "golden_id": "gold_001",
            "overall_score": 5.0,
            "decision": "pass",
            "correctness": {"score": 5},
            "relevance": {"score": 5},
            "historical_grounding": {"score": 5},
            "helpfulness": {"score": 5},
            "unsupported_claims": {"score": 5},
            "escalation_appropriateness": {"score": 5},
        },
        "gold_002": {
            "golden_id": "gold_002",
            "overall_score": 3.83,
            "decision": "borderline",
            "correctness": {"score": 4},
            "relevance": {"score": 4},
            "historical_grounding": {"score": 4},
            "helpfulness": {"score": 4},
            "unsupported_claims": {"score": 4},
            "escalation_appropriateness": {"score": 3},
        },
    }

    preds = {"gold_001": {}, "gold_002": {}}

    res = compute_agreement(annotations, judgments, preds)
    assert res["completed_count"] == 2
    assert res["decision"]["exact_agreement_pct"] == 100.0
    assert "correctness" in res["dimensions"]
    assert res["overall_score"]["mean_absolute_difference"] > 0.0


def test_missing_human_annotations_detected_gracefully():
    """Verify that empty annotations return INCOMPLETE status."""
    empty_annotations = [
        {"golden_id": "gold_001", "human_overall_score": None},
        {"golden_id": "gold_002", "human_overall_score": None},
    ]
    judgments = {"gold_001": {}, "gold_002": {}}
    preds = {"gold_001": {}, "gold_002": {}}

    res = compute_agreement(empty_annotations, judgments, preds)
    assert res["status"] == "INCOMPLETE"
    assert res["completed_count"] == 0
