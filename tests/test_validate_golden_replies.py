"""Unit tests for Golden Set reply validation logic."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.validate_golden_replies import validate_golden_replies, load_jsonl


@pytest.fixture
def temp_workspace(tmp_path: Path):
    """Create a temporary mock workspace with valid dataset splits and predictions."""
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    golden_file = tmp_path / "golden.jsonl"
    preds_file = tmp_path / "preds.jsonl"

    # Train data
    train_records = [
        {
            "conversation_id": f"conv_tr_{i}",
            "support_tweet_id": f"supp_tr_{i}",
            "customer_tweet_id": f"cust_tr_{i}",
        }
        for i in range(1, 10)
    ]
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    # Val & Test data
    with open(val_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"conversation_id": "conv_val_1", "support_tweet_id": "supp_val_1"}) + "\n")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"conversation_id": "conv_test_1", "support_tweet_id": "supp_test_1"}) + "\n")

    # Golden data (200 records)
    golden_records = [
        {"golden_id": f"gold_{i:03d}", "conversation_id": f"gold_conv_{i}", "tweet_id": f"gold_tw_{i}"}
        for i in range(1, 201)
    ]
    with open(golden_file, "w", encoding="utf-8") as f:
        for r in golden_records:
            f.write(json.dumps(r) + "\n")

    # Predictions data (200 valid records)
    preds_records = [
        {
            "golden_id": f"gold_{i:03d}",
            "pred_reply": f"Hello, we would be happy to look into this order issue for you! ^AG {i}",
            "retrieved_examples": [
                {
                    "conversation_id": "conv_tr_1",
                    "tweet_id": "supp_tr_1",
                    "historical_customer_text": "Where is my item?",
                    "historical_amazonhelp_response": "We can help you track that. Please share order number.",
                    "similarity_score": 0.55 - (j * 0.05),
                }
                for j in range(3)
            ],
        }
        for i in range(1, 201)
    ]
    with open(preds_file, "w", encoding="utf-8") as f:
        for r in preds_records:
            f.write(json.dumps(r) + "\n")

    return {
        "train": train_file,
        "val": val_file,
        "test": test_file,
        "golden": golden_file,
        "preds": preds_file,
        "tmp_path": tmp_path,
    }


def test_validation_clean_pass(temp_workspace):
    """Test that a compliant dataset passes all critical checks."""
    res = validate_golden_replies(
        predictions_path=temp_workspace["preds"],
        golden_path=temp_workspace["golden"],
        train_path=temp_workspace["train"],
        val_path=temp_workspace["val"],
        test_path=temp_workspace["test"],
    )

    checks = res["checks"]
    assert checks["check_01_exact_200_records"]["status"] == "PASS"
    assert checks["check_02_all_golden_ids_present"]["status"] == "PASS"
    assert checks["check_03_no_duplicate_golden_ids"]["status"] == "PASS"
    assert checks["check_04_non_empty_reply_and_evidence"]["status"] == "PASS"
    assert checks["check_05_valid_retrieved_examples_structure"]["status"] == "PASS"
    assert checks["check_06_retrieval_evidence_train_only"]["status"] == "PASS"
    assert checks["check_07_zero_leakage"]["status"] == "PASS"
    assert checks["check_08_no_secret_leaks"]["status"] == "PASS"
    assert checks["check_09_valid_jsonl"]["status"] == "PASS"
    assert checks["check_12_no_suspicious_replies"]["status"] == "PASS"
    assert res["summary"]["ready_for_judge"] is True


def test_validation_catches_duplicate_golden_id(temp_workspace):
    """Test duplicate golden_id detection."""
    bad_preds = temp_workspace["tmp_path"] / "bad_dup.jsonl"
    records = load_jsonl(temp_workspace["preds"])
    records[1]["golden_id"] = "gold_001"  # duplicate gold_001
    with open(bad_preds, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    res = validate_golden_replies(
        predictions_path=bad_preds,
        golden_path=temp_workspace["golden"],
        train_path=temp_workspace["train"],
        val_path=temp_workspace["val"],
        test_path=temp_workspace["test"],
    )
    assert res["checks"]["check_03_no_duplicate_golden_ids"]["status"] == "FAIL"


def test_validation_catches_leakage(temp_workspace):
    """Test that evidence leaking from test partition is flagged immediately."""
    bad_preds = temp_workspace["tmp_path"] / "bad_leak.jsonl"
    records = load_jsonl(temp_workspace["preds"])
    records[0]["retrieved_examples"][0]["conversation_id"] = "conv_test_1"  # test leakage!
    with open(bad_preds, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    res = validate_golden_replies(
        predictions_path=bad_preds,
        golden_path=temp_workspace["golden"],
        train_path=temp_workspace["train"],
        val_path=temp_workspace["val"],
        test_path=temp_workspace["test"],
    )
    assert res["checks"]["check_07_zero_leakage"]["status"] == "FAIL"
    assert len(res["leakage_incidents"]) > 0


def test_validation_catches_suspicious_meta_language(temp_workspace):
    """Test detection of meta language ('As an AI')."""
    bad_preds = temp_workspace["tmp_path"] / "bad_meta.jsonl"
    records = load_jsonl(temp_workspace["preds"])
    records[5]["pred_reply"] = "As an AI language model, I cannot check your order status."
    with open(bad_preds, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    res = validate_golden_replies(
        predictions_path=bad_preds,
        golden_path=temp_workspace["golden"],
        train_path=temp_workspace["train"],
        val_path=temp_workspace["val"],
        test_path=temp_workspace["test"],
    )
    assert res["checks"]["check_12_no_suspicious_replies"]["status"] == "FAIL"
    assert len(res["suspicious_replies"]) == 1


def test_validation_catches_secret_leak(temp_workspace):
    """Test detection of API key leaks."""
    bad_preds = temp_workspace["tmp_path"] / "bad_secret.jsonl"
    records = load_jsonl(temp_workspace["preds"])
    records[2]["pred_reply"] = "Contact us with gsk_abcdef1234567890abcdef123456"
    with open(bad_preds, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    res = validate_golden_replies(
        predictions_path=bad_preds,
        golden_path=temp_workspace["golden"],
        train_path=temp_workspace["train"],
        val_path=temp_workspace["val"],
        test_path=temp_workspace["test"],
    )
    assert res["checks"]["check_08_no_secret_leaks"]["status"] == "FAIL"
