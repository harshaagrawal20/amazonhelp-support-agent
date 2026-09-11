"""
Unit tests for conversation graph reconstruction, chronological sorting,
interaction extraction, and leak-free dataset partitioning.
"""

from datetime import datetime
import pytest

from src.conversations import (
    build_conversation_roots,
    order_conversation_messages,
    extract_direct_interactions,
    build_conversation_record,
    split_conversations,
    verify_zero_leakage,
    parse_twitter_timestamp,
)
import pandas as pd
import numpy as np


def test_parse_twitter_timestamp():
    ts_str = "Tue Oct 31 22:10:47 +0000 2017"
    dt = parse_twitter_timestamp(ts_str)
    assert dt.year == 2017
    assert dt.month == 10
    assert dt.day == 31
    assert dt.hour == 22
    assert dt.minute == 10
    assert dt.second == 47


def test_build_conversation_roots_complex_tree():
    # Tree:
    # 100 (root)
    #   ├── 101
    #   │     └── 102
    #   └── 103
    # 200 (separate root)
    #   └── 201
    df = pd.DataFrame({
        "tweet_id": [100, 101, 102, 103, 200, 201],
        "in_response_to_tweet_id": [np.nan, 100.0, 101.0, 100.0, np.nan, 200.0],
    })
    roots = build_conversation_roots(df)
    assert roots[100] == 100
    assert roots[101] == 100
    assert roots[102] == 100
    assert roots[103] == 100
    assert roots[200] == 200
    assert roots[201] == 200


def test_order_conversation_messages():
    # Messages provided out of chronological order
    raw_msgs = [
        {"tweet_id": 3, "timestamp_epoch": 300.0, "text": "Third"},
        {"tweet_id": 1, "timestamp_epoch": 100.0, "text": "First"},
        {"tweet_id": 2, "timestamp_epoch": 200.0, "text": "Second"},
    ]
    ordered = order_conversation_messages(raw_msgs)
    assert [m["tweet_id"] for m in ordered] == [1, 2, 3]


def test_extract_direct_interactions_and_multi_turn():
    # Conversation:
    # Customer: "Where is my package?" (id=1)
    # AmazonHelp: "Please check your tracking link." (id=2, in_reply_to=1)
    # Customer: "Tracking says delivered but not here!" (id=3, in_reply_to=2)
    # AmazonHelp: "We understand. Please send us a DM with order ID." (id=4, in_reply_to=3)
    raw_msgs = [
        {
            "tweet_id": 1,
            "author_id": "cust1",
            "inbound": True,
            "created_at": "Tue Oct 31 10:00:00 +0000 2017",
            "text": "Where is my package?",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": 2,
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Tue Oct 31 10:05:00 +0000 2017",
            "text": "Please check your tracking link.",
            "in_response_to_tweet_id": 1,
        },
        {
            "tweet_id": 3,
            "author_id": "cust1",
            "inbound": True,
            "created_at": "Tue Oct 31 10:10:00 +0000 2017",
            "text": "Tracking says delivered but not here!",
            "in_response_to_tweet_id": 2,
        },
        {
            "tweet_id": 4,
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Tue Oct 31 10:15:00 +0000 2017",
            "text": "We understand. Please send us a DM with order ID.",
            "in_response_to_tweet_id": 3,
        },
    ]

    record = build_conversation_record(root_id=1, raw_messages=raw_msgs)
    assert record["conversation_id"] == 1
    assert record["total_messages"] == 4
    assert record["customer_messages_count"] == 2
    assert record["support_messages_count"] == 2
    assert record["is_multi_turn"] is True
    assert record["usable_for_training"] is True
    assert record["exclusion_reason"] is None

    # Verify extracted direct interaction pairs
    pairs = record["interaction_pairs"]
    assert len(pairs) == 2

    # Turn 1
    assert pairs[0]["turn_index"] == 1
    assert pairs[0]["customer_tweet_id"] == 1
    assert pairs[0]["support_tweet_id"] == 2
    assert len(pairs[0]["prior_context"]) == 0

    # Turn 2 has prior context
    assert pairs[1]["turn_index"] == 2
    assert pairs[1]["customer_tweet_id"] == 3
    assert pairs[1]["support_tweet_id"] == 4
    assert len(pairs[1]["prior_context"]) >= 2


def test_split_conversations_zero_leakage():
    # Create 20 mock conversations
    mock_convs = []
    for cid in range(1, 21):
        mock_convs.append({
            "conversation_id": cid,
            "root_id": cid,
            "messages": [
                {"tweet_id": cid * 10 + 1, "text": "Customer query"},
                {"tweet_id": cid * 10 + 2, "text": "Support reply"},
            ]
        })

    splits = split_conversations(mock_convs, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(splits["train"]) == 14
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 3

    # Leakage check must pass
    verify_zero_leakage(splits)

    # Intentionally inject leakage to ensure verifier catches it
    leaky_splits = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"] + [splits["train"][0]],  # overlap!
    }
    with pytest.raises(AssertionError, match="CRITICAL LEAKAGE DETECTED"):
        verify_zero_leakage(leaky_splits)


def test_split_determinism():
    mock_convs = [{"conversation_id": cid, "root_id": cid, "messages": []} for cid in range(50)]
    s1 = split_conversations(mock_convs, seed=123)
    s2 = split_conversations(mock_convs, seed=123)
    s3 = split_conversations(mock_convs, seed=999)

    s1_train_ids = [c["conversation_id"] for c in s1["train"]]
    s2_train_ids = [c["conversation_id"] for c in s2["train"]]
    s3_train_ids = [c["conversation_id"] for c in s3["train"]]

    assert s1_train_ids == s2_train_ids
    assert s1_train_ids != s3_train_ids
