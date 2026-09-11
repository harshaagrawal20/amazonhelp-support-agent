"""
Core module for conversation graph reconstruction, chronological ordering,
AmazonHelp filtering, interaction pair extraction, and leak-free dataset partitioning.
"""

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


TWITTER_DATETIME_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"


def parse_twitter_timestamp(ts_str: str) -> datetime:
    """Parse Twitter raw created_at string into timezone-aware datetime."""
    return datetime.strptime(ts_str, TWITTER_DATETIME_FORMAT)


def build_conversation_roots(df_links: pd.DataFrame) -> Dict[int, int]:
    """
    Build a mapping from tweet_id to its conversation root tweet_id.
    Each conversation is a directed tree where edges point from child -> parent.
    Uses path compression to guarantee O(1) amortized root lookups and cycle prevention.
    """
    valid = df_links.dropna(subset=["in_response_to_tweet_id"])
    parent_map = dict(zip(
        valid["tweet_id"].astype(np.int64),
        valid["in_response_to_tweet_id"].astype(np.int64)
    ))

    root_map: Dict[int, int] = {}

    def find_root(node: int) -> int:
        path = []
        curr = node
        visited = set()
        while curr in parent_map and curr not in visited:
            visited.add(curr)
            path.append(curr)
            curr = parent_map[curr]

        # Path compression
        for p in path:
            root_map[p] = curr
        root_map[curr] = curr
        return curr

    all_tweet_ids = df_links["tweet_id"].astype(np.int64).values
    for tid in all_tweet_ids:
        if tid not in root_map:
            find_root(tid)

    return root_map


def order_conversation_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort conversation messages chronologically by timestamp.
    Tie-break stably using tweet_id.
    """
    return sorted(messages, key=lambda m: (m["timestamp_epoch"], m["tweet_id"]))


def extract_direct_interactions(ordered_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract all direct customer -> AmazonHelp interaction pairs in the conversation.
    A direct pair occurs when an AmazonHelp reply directly responds to a customer message.
    Preserves prior conversation context (history) for each turn.
    """
    msg_by_id = {m["tweet_id"]: m for m in ordered_messages}
    pairs = []
    turn_counter = 1

    for idx, msg in enumerate(ordered_messages):
        # Look for AmazonHelp responses
        if msg["author_id"] == "AmazonHelp" and not msg["inbound"]:
            parent_id = msg.get("in_response_to_tweet_id")
            if parent_id and parent_id in msg_by_id:
                parent_msg = msg_by_id[parent_id]
                # Is the parent an inbound customer message?
                if parent_msg["inbound"]:
                    # Context is all messages preceding this AmazonHelp response
                    prior_context = [
                        {
                            "tweet_id": m["tweet_id"],
                            "author_id": m["author_id"],
                            "inbound": m["inbound"],
                            "text": m["text"],
                        }
                        for m in ordered_messages[:idx]
                        if m["tweet_id"] != parent_id
                    ]
                    pairs.append({
                        "turn_index": turn_counter,
                        "customer_tweet_id": parent_msg["tweet_id"],
                        "customer_author_id": parent_msg["author_id"],
                        "customer_text": parent_msg["text"],
                        "support_tweet_id": msg["tweet_id"],
                        "support_author_id": msg["author_id"],
                        "support_text": msg["text"],
                        "prior_context": prior_context,
                    })
                    turn_counter += 1

    return pairs


def build_conversation_record(root_id: int, raw_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a complete, structured conversation record from a group of raw message dicts.
    """
    # Parse timestamps for sorting
    processed_messages = []
    for m in raw_messages:
        try:
            dt = parse_twitter_timestamp(m["created_at"])
            epoch = dt.timestamp()
            iso = dt.isoformat()
        except Exception:
            epoch = 0.0
            iso = m["created_at"]

        parent_val = m.get("in_response_to_tweet_id")
        parent_id = int(parent_val) if pd.notnull(parent_val) and str(parent_val).strip() != "" else None

        processed_messages.append({
            "tweet_id": int(m["tweet_id"]),
            "author_id": str(m["author_id"]),
            "inbound": bool(m["inbound"]),
            "created_at": str(m["created_at"]),
            "timestamp_epoch": epoch,
            "timestamp_iso": iso,
            "text": str(m["text"]),
            "in_response_to_tweet_id": parent_id,
        })

    ordered_messages = order_conversation_messages(processed_messages)

    # Message counts
    cust_msgs = [m for m in ordered_messages if m["inbound"]]
    ah_msgs = [m for m in ordered_messages if m["author_id"] == "AmazonHelp" and not m["inbound"]]
    other_brand_msgs = [m for m in ordered_messages if not m["inbound"] and m["author_id"] != "AmazonHelp"]

    # Direct customer -> support pairs
    interaction_pairs = extract_direct_interactions(ordered_messages)

    # Determine usability & exclusion reasons
    exclusion_reason = None
    if len(cust_msgs) == 0:
        exclusion_reason = "no_customer_message"
    elif len(ah_msgs) == 0:
        exclusion_reason = "no_amazonhelp_response"
    elif len(interaction_pairs) == 0:
        exclusion_reason = "no_direct_customer_support_pair"
    elif len(ordered_messages) > 100:
        exclusion_reason = "anomalous_large_viral_thread"
    elif all(len(m["text"].strip()) < 5 for m in cust_msgs):
        exclusion_reason = "empty_or_trivial_noise"

    usable = (exclusion_reason is None)

    # Temporal bounds
    timestamps = [m["timestamp_epoch"] for m in ordered_messages if m["timestamp_epoch"] > 0]
    min_epoch = min(timestamps) if timestamps else 0.0
    max_epoch = max(timestamps) if timestamps else 0.0
    duration_sec = max_epoch - min_epoch

    min_iso = ordered_messages[0]["timestamp_iso"] if ordered_messages else ""
    max_iso = ordered_messages[-1]["timestamp_iso"] if ordered_messages else ""

    return {
        "conversation_id": int(root_id),
        "root_id": int(root_id),
        "total_messages": len(ordered_messages),
        "customer_messages_count": len(cust_msgs),
        "support_messages_count": len(ah_msgs),
        "other_brand_messages_count": len(other_brand_msgs),
        "created_at_min": min_iso,
        "created_at_max": max_iso,
        "duration_seconds": round(duration_sec, 2),
        "is_multi_turn": (len(cust_msgs) > 1 or len(ah_msgs) > 1),
        "has_direct_customer_support_pair": len(interaction_pairs) > 0,
        "usable_for_training": usable,
        "exclusion_reason": exclusion_reason,
        "messages": ordered_messages,
        "interaction_pairs": interaction_pairs,
    }


def split_conversations(
    conversations: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split conversations deterministically at the conversation (root_id) level.
    Guarantees that all tweets belonging to a conversation remain in exactly one split.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Sort conversations by conversation_id first for absolute determinism before shuffle
    sorted_convs = sorted(conversations, key=lambda c: c["conversation_id"])
    rng = random.Random(seed)
    shuffled_convs = sorted_convs.copy()
    rng.shuffle(shuffled_convs)

    n_total = len(shuffled_convs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_set = shuffled_convs[:n_train]
    val_set = shuffled_convs[n_train : n_train + n_val]
    test_set = shuffled_convs[n_train + n_val :]

    splits = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }

    # Verify zero leakage
    verify_zero_leakage(splits)
    return splits


def verify_zero_leakage(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Verify that root_ids and tweet_ids have strictly zero overlap across train, val, and test.
    Raises AssertionError if any data leakage is detected.
    """
    train_roots = set(c["root_id"] for c in splits["train"])
    val_roots = set(c["root_id"] for c in splits["val"])
    test_roots = set(c["root_id"] for c in splits["test"])

    train_val_root_overlap = train_roots.intersection(val_roots)
    train_test_root_overlap = train_roots.intersection(test_roots)
    val_test_root_overlap = val_roots.intersection(test_roots)

    if train_val_root_overlap or train_test_root_overlap or val_test_root_overlap:
        raise AssertionError(
            f"CRITICAL LEAKAGE DETECTED across conversation roots! "
            f"train-val: {len(train_val_root_overlap)}, "
            f"train-test: {len(train_test_root_overlap)}, "
            f"val-test: {len(val_test_root_overlap)}"
        )

    # Also verify individual tweet_id overlap
    train_tweets = set(m["tweet_id"] for c in splits["train"] for m in c["messages"])
    val_tweets = set(m["tweet_id"] for c in splits["val"] for m in c["messages"])
    test_tweets = set(m["tweet_id"] for c in splits["test"] for m in c["messages"])

    train_val_tweet_overlap = train_tweets.intersection(val_tweets)
    train_test_tweet_overlap = train_tweets.intersection(test_tweets)
    val_test_tweet_overlap = val_tweets.intersection(test_tweets)

    if train_val_tweet_overlap or train_test_tweet_overlap or val_test_tweet_overlap:
        raise AssertionError(
            f"CRITICAL LEAKAGE DETECTED across tweet IDs! "
            f"train-val: {len(train_val_tweet_overlap)}, "
            f"train-test: {len(train_test_tweet_overlap)}, "
            f"val-test: {len(val_test_tweet_overlap)}"
        )
