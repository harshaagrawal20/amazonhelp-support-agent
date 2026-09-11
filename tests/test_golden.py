"""
Tests for Golden Evaluation Set (Phase 4):
- Exact 200 examples requirement.
- ID uniqueness (golden_id, conversation_id, tweet_id).
- Test-split-only provenance.
- Zero contamination with train and val splits.
- Schema completeness and null human annotation fields.
- Deterministic sampling behavior.
- Annotation read/write and validation integrity.
"""

from collections import Counter
import json
from pathlib import Path
import pytest

from src.intents import INTENT_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FILE = PROJECT_ROOT / "data/golden/golden_evaluation_200.jsonl"
TEST_FILE = PROJECT_ROOT / "data/processed/amazonhelp_test.jsonl"
TRAIN_FILE = PROJECT_ROOT / "data/processed/amazonhelp_train.jsonl"
VAL_FILE = PROJECT_ROOT / "data/processed/amazonhelp_val.jsonl"


@pytest.fixture(scope="module")
def golden_records():
    """Load the golden records fixture."""
    if not GOLDEN_FILE.exists():
        pytest.skip(f"Golden dataset file not found at {GOLDEN_FILE}")
    records = []
    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_golden_set_exact_count(golden_records):
    """Verify that the Golden Set contains exactly 200 records."""
    assert len(golden_records) == 200, f"Expected 200 records, got {len(golden_records)}"


def test_golden_set_id_uniqueness(golden_records):
    """Verify that golden_id, conversation_id, and tweet_id are strictly unique."""
    golden_ids = [r["golden_id"] for r in golden_records]
    conv_ids = [r["conversation_id"] for r in golden_records]
    tweet_ids = [r["tweet_id"] for r in golden_records]

    assert len(set(golden_ids)) == 200, "Duplicate golden_id found!"
    assert len(set(conv_ids)) == 200, "Duplicate conversation_id found (must be strictly 1 example per conversation)!"
    assert len(set(tweet_ids)) == 200, "Duplicate tweet_id found!"


def test_golden_set_test_provenance_and_zero_leakage(golden_records):
    """Verify all 200 examples originate from test split and have zero overlap with train/val."""
    golden_cids = set(r["conversation_id"] for r in golden_records)

    # Check test provenance
    test_cids = set()
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            test_cids.add(json.loads(line)["conversation_id"])

    assert golden_cids.issubset(test_cids), "Found golden conversations not in test split!"

    # Check train leakage
    train_cids = set()
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            train_cids.add(json.loads(line)["conversation_id"])

    assert len(golden_cids.intersection(train_cids)) == 0, "Contamination: Golden Set contains training conversations!"

    # Check val leakage
    val_cids = set()
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            val_cids.add(json.loads(line)["conversation_id"])

    assert len(golden_cids.intersection(val_cids)) == 0, "Contamination: Golden Set contains validation conversations!"


def test_golden_set_schema_and_null_placeholders(golden_records):
    """Verify required schema fields and ensure human labels are null initially."""
    for r in golden_records:
        assert "golden_id" in r
        assert "conversation_id" in r
        assert "tweet_id" in r
        assert "customer_text" in r
        assert "conversation_context" in r
        assert "weak_label" in r
        assert "model_prediction" in r

        # Weak label and model prediction must be valid taxonomy intents
        assert r["weak_label"] in INTENT_NAMES
        assert r["model_prediction"] in INTENT_NAMES

        # Status must be consistent with gold_intent
        if r["gold_intent"] is None:
            assert r["annotation_status"] == "unlabeled"
        else:
            assert r["annotation_status"] == "labeled"
            assert r["gold_intent"] in INTENT_NAMES


def test_golden_set_strata_coverage(golden_records):
    """Verify that targeted strata (rare intents, multilingual, disagreements, follow-ups) are represented."""
    strata = Counter(r["sampling_stratum"] for r in golden_records)
    assert strata["model_rule_disagreement"] >= 30, "Insufficient model/rule disagreements"
    assert strata["multilingual_inquiry"] >= 20, "Insufficient multilingual examples"
    assert strata["context_dependent_followup"] >= 20, "Insufficient follow-up examples"

    # Verify multilingual representation
    langs = Counter(r["detected_language"] for r in golden_records)
    assert langs["ja"] >= 10, "Insufficient Japanese examples"
    assert langs["es"] >= 5, "Insufficient Spanish examples"
    assert langs["de"] >= 3, "Insufficient German examples"


def test_golden_validation_script():
    """Verify that validate_golden.py runs and reports success."""
    from scripts.validate_golden import validate_golden_dataset
    res = validate_golden_dataset(GOLDEN_FILE)
    assert res["total"] == 200
    assert res["unique_conversations"] == 200
    assert res["zero_contamination"] is True


# ==============================================================================
# Assisted Annotation Mode Tests
# ==============================================================================

from scripts.annotate_golden import (
    suggest_intent,
    suggest_escalation,
    is_already_annotated,
    run_annotator,
    load_golden_set,
    save_golden_set,
)


def test_suggest_intent_and_escalation_heuristics():
    """Verify that deterministic heuristics produce valid canonical intents and escalation choices."""
    cases = [
        ("where is my tracking number for package", "delivery_status_tracking", "auto_handle"),
        ("someone changed my email address and I cannot log in help!", "account_access_security", "escalate"),
        ("cancel my order immediately please", "order_cancellation_modification", "escalate"),
        ("the screen arrived completely broken and shattered", "damaged_defective_wrong_item", "escalate"),
        ("hi", "other_unclear", "unclear"),
        ("thank you so much for the quick help!", "feedback_complaint_chatter", "auto_handle"),
    ]

    for text, expected_intent, expected_esc in cases:
        record = {"customer_text": text, "cleaned_text": text}
        intent = suggest_intent(record)
        esc, rationale = suggest_escalation(record, intent)

        assert intent in INTENT_NAMES
        assert esc in ["auto_handle", "escalate", "unclear"]
        assert intent == expected_intent
        assert esc == expected_esc
        assert len(rationale) > 0


def test_is_already_annotated_protection():
    """Verify that is_already_annotated correctly identifies labeled records for both annotators."""
    labeled_rec = {
        "golden_id": "g1",
        "gold_intent": "delivery_status_tracking",
        "gold_intent_annotator_1": "delivery_status_tracking",
        "gold_intent_annotator_2": None,
    }
    unlabeled_rec = {
        "golden_id": "g2",
        "gold_intent": None,
        "gold_intent_annotator_1": None,
        "gold_intent_annotator_2": None,
    }
    ann2_rec = {
        "golden_id": "g3",
        "gold_intent": "delivery_status_tracking",
        "gold_intent_annotator_1": "delivery_status_tracking",
        "gold_intent_annotator_2": "return_refund_exchange",
    }

    # Annotator 1 checks
    assert is_already_annotated(labeled_rec, "annotator_1") is True
    assert is_already_annotated(unlabeled_rec, "annotator_1") is False

    # Annotator 2 checks
    assert is_already_annotated(labeled_rec, "annotator_2") is False
    assert is_already_annotated(ann2_rec, "annotator_2") is True


def _make_mock_records():
    return [
        {
            "golden_id": "gold_test_01",
            "conversation_id": 99901,
            "root_id": 99901,
            "tweet_id": 99901,
            "turn_index": 1,
            "is_followup": False,
            "customer_text": "Existing human label that must never be overwritten",
            "cleaned_text": "Existing human label that must never be overwritten",
            "conversation_context": [],
            "support_historical_response": "We are here to help",
            "sampling_stratum": "core_intent",
            "detected_language": "en",
            "weak_label": "other_unclear",
            "model_prediction": "other_unclear",
            "gold_intent": "account_access_security",
            "gold_intent_annotator_1": "account_access_security",
            "gold_intent_annotator_2": None,
            "gold_reply_quality": None,
            "gold_escalation": "escalate",
            "gold_notes": "Existing human notes",
            "annotator": "annotator_1",
            "annotation_status": "labeled",
            "annotation_method": "manual",
        },
        {
            "golden_id": "gold_test_02",
            "conversation_id": 99902,
            "root_id": 99902,
            "tweet_id": 99902,
            "turn_index": 1,
            "is_followup": False,
            "customer_text": "Where is my package? Need tracking status",
            "cleaned_text": "Where is my package? Need tracking status",
            "conversation_context": [],
            "support_historical_response": "Track here",
            "sampling_stratum": "core_intent",
            "detected_language": "en",
            "weak_label": "delivery_status_tracking",
            "model_prediction": "delivery_status_tracking",
            "gold_intent": None,
            "gold_intent_annotator_1": None,
            "gold_intent_annotator_2": None,
            "gold_reply_quality": None,
            "gold_escalation": None,
            "gold_notes": "",
            "annotator": None,
            "annotation_status": "unlabeled",
        },
        {
            "golden_id": "gold_test_03",
            "conversation_id": 99903,
            "root_id": 99903,
            "tweet_id": 99903,
            "turn_index": 1,
            "is_followup": False,
            "customer_text": "Cancel my order right now, I made a mistake",
            "cleaned_text": "Cancel my order right now, I made a mistake",
            "conversation_context": [],
            "support_historical_response": "We can help cancel",
            "sampling_stratum": "core_intent",
            "detected_language": "en",
            "weak_label": "order_cancellation_modification",
            "model_prediction": "order_cancellation_modification",
            "gold_intent": None,
            "gold_intent_annotator_1": None,
            "gold_intent_annotator_2": None,
            "gold_reply_quality": None,
            "gold_escalation": None,
            "gold_notes": "",
            "annotator": None,
            "annotation_status": "unlabeled",
        },
    ]


def test_assisted_mode_accept_saves_annotation_and_protects_existing(tmp_path, monkeypatch):
    """Verify assisted mode accepts suggestions with 'y', saves annotation, and never overwrites existing."""
    mock_file = tmp_path / "mock_golden.jsonl"
    records = _make_mock_records()
    save_golden_set(records, mock_file)

    monkeypatch.setattr("scripts.annotate_golden.GOLDEN_FILE", mock_file)

    # Simulated inputs: 'y' to accept gold_test_02, then 'q' to quit on gold_test_03
    input_values = iter(["y", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(input_values))

    run_annotator("annotator_1", assisted_mode=True)

    updated_records = load_golden_set(mock_file)

    # 1. Existing record 01 must be completely untouched
    rec1 = updated_records[0]
    assert rec1["golden_id"] == "gold_test_01"
    assert rec1["gold_intent"] == "account_access_security"
    assert rec1["gold_notes"] == "Existing human notes"
    assert rec1["annotation_status"] == "labeled"

    # 2. Record 02 must now be labeled via assisted_accept
    rec2 = updated_records[1]
    assert rec2["golden_id"] == "gold_test_02"
    assert rec2["gold_intent"] == "delivery_status_tracking"
    assert rec2["gold_escalation"] == "auto_handle"
    assert rec2["annotation_status"] == "labeled"
    assert rec2["annotation_method"] == "assisted_accept"
    assert rec2["gold_intent_annotator_1"] == "delivery_status_tracking"
    assert rec2["gold_reply_quality"] is None  # Must remain null

    # 3. Record 03 was quit before labeling, must remain unlabeled
    rec3 = updated_records[2]
    assert rec3["golden_id"] == "gold_test_03"
    assert rec3["gold_intent"] is None
    assert rec3["annotation_status"] == "unlabeled"


def test_assisted_mode_reject_enters_manual_mode(tmp_path, monkeypatch):
    """Verify that choosing 'n' in assisted mode allows custom manual selection."""
    mock_file = tmp_path / "mock_golden.jsonl"
    records = _make_mock_records()
    save_golden_set(records, mock_file)

    monkeypatch.setattr("scripts.annotate_golden.GOLDEN_FILE", mock_file)

    # Inputs:
    # On gold_test_02: 'n' (reject suggestion) -> '2' (return_refund_exchange) -> '2' (escalate) -> 'user manual note'
    # On gold_test_03: 'q' (quit)
    input_values = iter(["n", "2", "2", "user manual note", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(input_values))

    run_annotator("annotator_1", assisted_mode=True)

    updated = load_golden_set(mock_file)
    rec2 = updated[1]
    assert rec2["gold_intent"] == "return_refund_exchange"
    assert rec2["gold_escalation"] == "escalate"
    assert rec2["gold_notes"] == "user manual note"
    assert rec2["annotation_status"] == "labeled"
    assert rec2["annotation_method"] == "manual"


def test_assisted_mode_skip_leaves_unlabeled(tmp_path, monkeypatch):
    """Verify that 's' skips without modifying or labeling the record."""
    mock_file = tmp_path / "mock_golden.jsonl"
    records = _make_mock_records()
    save_golden_set(records, mock_file)

    monkeypatch.setattr("scripts.annotate_golden.GOLDEN_FILE", mock_file)

    # Inputs: 's' to skip gold_test_02, 'q' to quit on gold_test_03
    input_values = iter(["s", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(input_values))

    run_annotator("annotator_1", assisted_mode=True)

    updated = load_golden_set(mock_file)
    rec2 = updated[1]
    assert rec2["gold_intent"] is None
    assert rec2["annotation_status"] == "unlabeled"


def test_manual_mode_still_works(tmp_path, monkeypatch):
    """Verify that standard manual mode (assisted_mode=False) functions as before."""
    mock_file = tmp_path / "mock_golden.jsonl"
    records = _make_mock_records()
    save_golden_set(records, mock_file)

    monkeypatch.setattr("scripts.annotate_golden.GOLDEN_FILE", mock_file)

    # Inputs in standard manual mode:
    # gold_test_02: '1' (delivery_status_tracking), '1' (auto_handle), '' (no notes)
    # gold_test_03: 'q' (quit)
    input_values = iter(["1", "1", "", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(input_values))

    run_annotator("annotator_1", assisted_mode=False)

    updated = load_golden_set(mock_file)
    rec2 = updated[1]
    assert rec2["gold_intent"] == "delivery_status_tracking"
    assert rec2["gold_escalation"] == "auto_handle"
    assert rec2["annotation_status"] == "labeled"
    assert rec2["annotation_method"] == "manual"

