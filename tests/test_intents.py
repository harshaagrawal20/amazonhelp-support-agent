"""
Tests for Phase 3: Intent taxonomy, development labeling rules,
baseline models, evaluation metrics, and split contamination.
"""

from collections import Counter
import json
from pathlib import Path
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.intents import (
    INTENT_NAMES,
    INTENT_TAXONOMY,
    build_dev_record,
    clean_tweet_text,
    classify_message_intent,
    extract_dev_dataset_from_conversations,
    normalize_intent_label,
)


def test_intent_taxonomy_completeness():
    """Verify all 10 candidate intents exist and have full specifications."""
    assert len(INTENT_TAXONOMY) == 10
    assert len(INTENT_NAMES) == 10

    for name in INTENT_NAMES:
        assert name in INTENT_TAXONOMY, f"Missing {name} in taxonomy"
        defn = INTENT_TAXONOMY[name]
        assert defn.name == name
        assert len(defn.display_name.strip()) > 0
        assert len(defn.department.strip()) > 0
        assert len(defn.description.strip()) > 20
        assert len(defn.inclusion_criteria.strip()) > 10
        assert len(defn.exclusion_criteria.strip()) > 10
        assert len(defn.likely_confusions) > 0
        assert len(defn.representative_examples) >= 3


def test_normalize_intent_label():
    """Verify label normalization maps aliases and canonical names properly."""
    # Canonical names
    for name in INTENT_NAMES:
        assert normalize_intent_label(name) == name

    # Aliases
    assert normalize_intent_label("delivery") == "delivery_status_tracking"
    assert normalize_intent_label("shipping") == "delivery_status_tracking"
    assert normalize_intent_label("refund") == "return_refund_exchange"
    assert normalize_intent_label("cancellation") == "order_cancellation_modification"
    assert normalize_intent_label("billing") == "payment_billing_promotions"
    assert normalize_intent_label("prime") == "prime_digital_services"
    assert normalize_intent_label("security") == "account_access_security"
    assert normalize_intent_label("seller") == "product_seller_inquiry"
    assert normalize_intent_label("chatter") == "feedback_complaint_chatter"
    assert normalize_intent_label("other") == "other_unclear"

    # Invalid
    with pytest.raises(ValueError, match="Unknown intent label"):
        normalize_intent_label("flight_booking")


def test_clean_tweet_text():
    """Verify handle and URL stripping while retaining text semantics."""
    raw = "@AmazonHelp @115821 My package #123-456 is delayed! Please help https://t.co/abcXYZ"
    cleaned = clean_tweet_text(raw)
    assert "@AmazonHelp" not in cleaned
    assert "@115821" not in cleaned
    assert "https://" not in cleaned
    assert "package #123-456 is delayed!" in cleaned


def test_classify_message_intent_rules():
    """Verify representative customer messages are mapped to expected intents."""
    cases = [
        ("Where is my order? Tracking says out for delivery but hasn't arrived", "delivery_status_tracking"),
        ("I would like to return this shirt and get a full refund please", "return_refund_exchange"),
        ("The box was completely smashed and the laptop inside is shattered and broken", "damaged_defective_wrong_item"),
        ("Please cancel my order # 123-456 immediately, placed by mistake", "order_cancellation_modification"),
        ("My credit card was charged twice for the same purchase", "payment_billing_promotions"),
        ("Prime video app keeps crashing on my firetv stick", "prime_digital_services"),
        ("I forgot my password and cannot receive the OTP verification code to login", "account_access_security"),
        ("When will the Nintendo Switch console be back in stock with third party sellers?", "product_seller_inquiry"),
        ("Worst customer service experience ever! Your agent was extremely rude and unhelpful", "feedback_complaint_chatter"),
        ("hi I need Help", "other_unclear"),
        ("__email__", "other_unclear"),
    ]

    for text, expected_intent in cases:
        intent, conf, method = classify_message_intent(text)
        assert intent == expected_intent, f"Failed for '{text}': got '{intent}', expected '{expected_intent}'"
        if expected_intent == "other_unclear":
            assert conf < 0.50
            assert method == "fallback_unmatched"
        else:
            assert conf >= 0.80
            assert method.startswith("regex_rule_")


def test_priority_rule_disambiguation():
    """Verify that multi-keyword collisions are resolved according to priority order."""
    # Defect + Delivery: Damaged should win over tracking
    text = "The courier delivered a broken and smashed box with shattered screen"
    intent, _, _ = classify_message_intent(text)
    assert intent == "damaged_defective_wrong_item"

    # Cancellation + Delivery: Cancellation should win over tracking
    text = "Please cancel my order because shipping is delayed"
    intent, _, _ = classify_message_intent(text)
    assert intent == "order_cancellation_modification"


def test_build_dev_record_preserves_context():
    """Verify dev record schema contains customer message, context, and intent."""
    sample_conv = {
        "conversation_id": 1001,
        "root_id": 1001,
    }
    sample_pair = {
        "turn_index": 2,
        "customer_tweet_id": 2002,
        "customer_author_id": "cust_1",
        "customer_text": "@AmazonHelp here is my order #123",
        "support_tweet_id": 3003,
        "support_text": "@cust_1 thanks, let us check",
        "prior_context": [{"tweet_id": 1001, "author_id": "cust_1", "text": "where is my package?"}],
    }

    record = build_dev_record(sample_conv, sample_pair)
    assert record["conversation_id"] == 1001
    assert record["root_id"] == 1001
    assert record["turn_index"] == 2
    assert record["customer_tweet_id"] == 2002
    assert record["customer_message"] == "@AmazonHelp here is my order #123"
    assert "order #123" in record["cleaned_text"]
    assert len(record["prior_context"]) == 1
    assert record["prior_context"][0]["text"] == "where is my package?"
    assert record["intent"] in INTENT_NAMES


def test_baseline_majority_class_deterministic():
    """Verify majority baseline predicts training mode and behaves deterministically."""
    y_train = ["delivery_status_tracking"] * 10 + ["other_unclear"] * 20 + ["return_refund_exchange"] * 5
    majority_class = Counter(y_train).most_common(1)[0][0]
    assert majority_class == "other_unclear"

    y_test = ["delivery_status_tracking"] * 5 + ["other_unclear"] * 5
    preds = [majority_class] * len(y_test)
    assert all(p == "other_unclear" for p in preds)


def test_tfidf_pipeline_reproducibility():
    """Verify TF-IDF + Logistic Regression pipeline fits and predicts deterministically."""
    train_texts = [
        "Where is my package delivery status?",
        "I need a refund for my return",
        "The screen is broken and damaged",
        "Cancel my order please",
        "Charged twice on my credit card",
    ] * 5
    train_labels = [
        "delivery_status_tracking",
        "return_refund_exchange",
        "damaged_defective_wrong_item",
        "order_cancellation_modification",
        "payment_billing_promotions",
    ] * 5

    vec = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)

    clf1 = LogisticRegression(random_state=42)
    clf1.fit(X_train, train_labels)

    test_msg = ["package delivery delayed"]
    X_test = vec.transform(test_msg)
    pred1 = clf1.predict(X_test)[0]

    clf2 = LogisticRegression(random_state=42)
    clf2.fit(X_train, train_labels)
    pred2 = clf2.predict(X_test)[0]

    assert pred1 == pred2
    assert pred1 == "delivery_status_tracking"


def test_dev_datasets_no_contamination():
    """Verify zero conversation_id overlap between train, val, and test dev sets."""
    dev_train = Path("data/processed/amazonhelp_dev_train.jsonl")
    dev_val = Path("data/processed/amazonhelp_dev_val.jsonl")
    dev_test = Path("data/processed/amazonhelp_dev_test.jsonl")

    if not (dev_train.exists() and dev_val.exists() and dev_test.exists()):
        pytest.skip("Dev dataset files not yet generated.")

    train_cids = set()
    with open(dev_train, "r", encoding="utf-8") as f:
        for line in f:
            train_cids.add(json.loads(line)["conversation_id"])

    val_cids = set()
    with open(dev_val, "r", encoding="utf-8") as f:
        for line in f:
            val_cids.add(json.loads(line)["conversation_id"])

    test_cids = set()
    with open(dev_test, "r", encoding="utf-8") as f:
        for line in f:
            test_cids.add(json.loads(line)["conversation_id"])

    assert len(train_cids.intersection(val_cids)) == 0, "Train and Val share conversation IDs!"
    assert len(train_cids.intersection(test_cids)) == 0, "Train and Test share conversation IDs!"
    assert len(val_cids.intersection(test_cids)) == 0, "Val and Test share conversation IDs!"
