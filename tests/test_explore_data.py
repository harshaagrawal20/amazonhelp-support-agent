"""
Unit tests for data exploration and conversation graph functions.
"""

import pandas as pd
import numpy as np
import pytest

from scripts.explore_data import (
    build_conversation_roots,
    ESCALATION_REGEX,
)


def test_build_conversation_roots_linear_thread():
    # Tree: 1 <- 2 <- 3
    # Separate: 4
    df_mock = pd.DataFrame({
        "tweet_id": [1, 2, 3, 4],
        "in_response_to_tweet_id": [np.nan, 1.0, 2.0, np.nan],
    })
    root_map = build_conversation_roots(df_mock)

    assert root_map[1] == 1
    assert root_map[2] == 1
    assert root_map[3] == 1
    assert root_map[4] == 4


def test_build_conversation_roots_branching_thread():
    # Tree: 10 has two children: 20 and 30
    # 20 has child 25
    df_mock = pd.DataFrame({
        "tweet_id": [10, 20, 25, 30],
        "in_response_to_tweet_id": [np.nan, 10.0, 20.0, 10.0],
    })
    root_map = build_conversation_roots(df_mock)

    assert root_map[10] == 10
    assert root_map[20] == 10
    assert root_map[25] == 10
    assert root_map[30] == 10


def test_escalation_regex():
    # Matches
    assert ESCALATION_REGEX.search("Please send us a DM with your account email.")
    assert ESCALATION_REGEX.search("Kindly call our support helpline.")
    assert ESCALATION_REGEX.search("Please private message us your tracking ID.")
    assert ESCALATION_REGEX.search("Reach out to our team at http://help.example.com")

    # Non-matches (standard resolution guidance)
    assert not ESCALATION_REGEX.search("Your package has been dispatched and is arriving tomorrow.")
    assert not ESCALATION_REGEX.search("You can restart your device by holding the power button.")
