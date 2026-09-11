"""Unit tests for the LLM-as-a-judge evaluation module (offline / mocked)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.judge import (
    DIMENSIONS,
    calculate_decision,
    parse_judge_response,
    build_judge_prompt,
    ResponseJudge,
)
from scripts.judge_golden_replies import (
    load_existing_judgments,
    compute_aggregate_summary,
)


def test_parse_valid_judge_response():
    """Test parsing a well-formed JSON judge output."""
    raw = json.dumps({
        "correctness": {"score": 5, "reason": "Accurate guidance."},
        "relevance": {"score": 4, "reason": "Directly addresses delivery delay."},
        "historical_grounding": {"score": 4, "reason": "Matches typical help flow."},
        "helpfulness": {"score": 5, "reason": "Gives clear link and next step."},
        "unsupported_claims": {"score": 5, "reason": "No false promises."},
        "escalation_appropriateness": {"score": 4, "reason": "Appropriately directs to support."},
        "overall_score": 4.5,
        "decision": "pass",
    })

    res = parse_judge_response(raw)
    assert res["overall_score"] == 4.5
    assert res["decision"] == "pass"
    for dim in DIMENSIONS:
        assert dim in res
        assert "score" in res[dim]
        assert "reason" in res[dim]


def test_parse_markdown_wrapped_json():
    """Test parsing JSON wrapped in markdown code blocks."""
    raw = """```json
    {
      "correctness": {"score": 4, "reason": "Good"},
      "relevance": {"score": 4, "reason": "Good"},
      "historical_grounding": {"score": 4, "reason": "Good"},
      "helpfulness": {"score": 4, "reason": "Good"},
      "unsupported_claims": {"score": 4, "reason": "Good"},
      "escalation_appropriateness": {"score": 4, "reason": "Good"},
      "overall_score": 4.0,
      "decision": "pass"
    }
    ```"""
    res = parse_judge_response(raw)
    assert res["decision"] == "pass"
    assert res["overall_score"] == 4.0


def test_invalid_score_rejection():
    """Test rejection of non-numeric scores."""
    raw = json.dumps({
        "correctness": {"score": "not_a_number", "reason": "Bad"},
        "relevance": {"score": 4, "reason": "Good"},
        "historical_grounding": {"score": 4, "reason": "Good"},
        "helpfulness": {"score": 4, "reason": "Good"},
        "unsupported_claims": {"score": 4, "reason": "Good"},
        "escalation_appropriateness": {"score": 4, "reason": "Good"},
    })
    with pytest.raises(ValueError, match="is not numeric"):
        parse_judge_response(raw)


def test_missing_required_dimension():
    """Test rejection when a dimension is omitted."""
    raw = json.dumps({
        "correctness": {"score": 5, "reason": "Good"},
        # missing other dimensions
    })
    with pytest.raises(ValueError, match="Missing required dimension"):
        parse_judge_response(raw)


def test_decision_rules():
    """Test deterministic decision rule logic."""
    # Clean Pass
    scores_pass = {d: 4.5 for d in DIMENSIONS}
    ov, dec = calculate_decision(scores_pass)
    assert ov == 4.5
    assert dec == "pass"

    # Borderline (overall 3.5)
    scores_border = {d: 3.5 for d in DIMENSIONS}
    ov, dec = calculate_decision(scores_border)
    assert ov == 3.5
    assert dec == "borderline"

    # Borderline because one dimension < 3 even though overall >= 4.0
    scores_mixed = {d: 5.0 for d in DIMENSIONS}
    scores_mixed["historical_grounding"] = 2.0
    ov, dec = calculate_decision(scores_mixed)
    assert ov >= 4.0
    assert dec == "borderline"

    # Fail because overall < 3.0
    scores_fail = {d: 2.5 for d in DIMENSIONS}
    ov, dec = calculate_decision(scores_fail)
    assert dec == "fail"

    # Fail due to severe hallucination (unsupported_claims <= 1)
    scores_hallucinate = {d: 5.0 for d in DIMENSIONS}
    scores_hallucinate["unsupported_claims"] = 1.0
    ov, dec = calculate_decision(scores_hallucinate)
    assert dec == "fail"


def test_build_judge_prompt_structure():
    """Test that build_judge_prompt formats all evidence fields cleanly."""
    rec = {
        "golden_id": "gold_001",
        "customer_text": "I was charged twice for Prime.",
        "conversation_context": "None",
        "gold_intent": "payment_billing_promotions",
        "gold_escalation": "escalate",
        "pred_intent_tfidf_lr": "payment_billing_promotions",
        "pred_escalation": "escalate",
        "pred_reply": "Please DM us so we can review the duplicate charge.",
        "support_historical_response": "Send us a DM with your account email.",
        "retrieved_examples": [
            {
                "historical_customer_text": "Why double charge?",
                "historical_amazonhelp_response": "We'd like to check that. Please DM us.",
                "similarity_score": 0.42,
            }
        ],
    }

    prompt = build_judge_prompt(rec)
    assert "Customer Message: I was charged twice for Prime." in prompt
    assert "Human Gold Intent: payment_billing_promotions" in prompt
    assert "Human Gold Escalation: escalate" in prompt
    assert "AI GENERATED REPLY TO EVALUATE:" in prompt
    assert "Please DM us so we can review the duplicate charge." in prompt
    assert "Retrieved Historical AmazonHelp Evidence" in prompt
    assert "Similarity: 0.42" in prompt


def test_resume_behavior(tmp_path: Path):
    """Test that existing valid judgments are loaded and skipped."""
    out_file = tmp_path / "test_judgments.jsonl"
    existing_rec = {
        "golden_id": "gold_001",
        "overall_score": 4.5,
        "decision": "pass",
        "correctness": {"score": 5, "reason": "ok"},
    }
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(existing_rec) + "\n")

    loaded = load_existing_judgments(out_file)
    assert "gold_001" in loaded
    assert loaded["gold_001"]["decision"] == "pass"


def test_pred_reply_immutability(tmp_path: Path):
    """Ensure the judging process never alters pred_reply in the prediction file."""
    preds_file = tmp_path / "preds.jsonl"
    original_reply = "We apologize for the inconvenience. Please send us a DM. ^RC"
    with open(preds_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "golden_id": "gold_001",
            "customer_text": "Help with order",
            "pred_reply": original_reply,
            "gold_intent": "delivery_status_tracking",
            "gold_escalation": "escalate",
        }) + "\n")

    # Read back and assert equality
    with open(preds_file, "r", encoding="utf-8") as f:
        reloaded = json.loads(f.readline())
    assert reloaded["pred_reply"] == original_reply
