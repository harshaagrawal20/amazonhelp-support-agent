"""
Unit Tests for Historical Retriever and Support Agent.

Ensures:
1. Retriever loads training data and fits TF-IDF matrix.
2. Retriever returns exactly k results with valid descending similarity scores.
3. Retrieved examples come strictly from the training partition.
4. Empty or whitespace queries are handled safely without crashing.
5. Agent prompt includes customer message, context, and retrieved evidence.
6. Agent generate_reply outputs valid schema with pred_reply and retrieved_examples.
7. Unit tests run completely offline without making live Groq API calls.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.retrieval import HistoricalRetriever
from src.agent import SupportAgent, build_grounded_prompt


@pytest.fixture
def mock_training_records():
    """Sample of 4 distinct training interaction pairs."""
    return [
        {
            "conversation_id": 1001,
            "customer_tweet_id": 5001,
            "customer_message": "Where is my package? The tracking number says delayed in transit.",
            "cleaned_text": "Where is my package? The tracking number says delayed in transit.",
            "support_response": "Hi! You can track your package in real-time here: https://amazon.com/orders",
            "intent": "delivery_status_tracking",
        },
        {
            "conversation_id": 1002,
            "customer_tweet_id": 5002,
            "customer_message": "How do I return an unopened book for a refund?",
            "cleaned_text": "How do I return an unopened book for a refund?",
            "support_response": "You can start a return via our Online Returns Center here: https://amazon.com/returns",
            "intent": "return_refund_exchange",
        },
        {
            "conversation_id": 1003,
            "customer_tweet_id": 5003,
            "customer_message": "My Kindle screen is completely frozen and won't restart.",
            "cleaned_text": "My Kindle screen is completely frozen and won't restart.",
            "support_response": "Try holding down the power button for 40 full seconds to hard reset the device.",
            "intent": "prime_digital_services",
        },
        {
            "conversation_id": 1004,
            "customer_tweet_id": 5004,
            "customer_message": "Someone hacked my account and changed the login email!",
            "cleaned_text": "Someone hacked my account and changed the login email!",
            "support_response": "Please reach our Account Security specialists immediately via: https://amazon.com/help",
            "intent": "account_access_security",
        },
    ]


def test_retriever_fit_and_retrieve_k(mock_training_records):
    """Test that retriever fits and returns exactly k items when available."""
    retriever = HistoricalRetriever()
    retriever.fit(mock_training_records)

    assert retriever.is_fitted
    assert len(retriever.records) == 4

    results = retriever.retrieve("tracking package delivery status", k=2)
    assert len(results) == 2

    # Top result should be the delivery tracking record
    top = results[0]
    assert "package" in top["historical_customer_text"].lower()
    assert top["conversation_id"] == 1001
    assert "similarity_score" in top
    assert top["similarity_score"] > 0


def test_retriever_similarity_scores_descending(mock_training_records):
    """Test that retrieved items are ordered by similarity score descending."""
    retriever = HistoricalRetriever()
    retriever.fit(mock_training_records)

    results = retriever.retrieve("Kindle screen frozen restart", k=3)
    assert len(results) >= 2

    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_provenance_strictly_training(mock_training_records):
    """Test that retrieved examples originate strictly from the fitted training records."""
    retriever = HistoricalRetriever()
    retriever.fit(mock_training_records)

    results = retriever.retrieve("return refund book", k=3)
    valid_cids = {1001, 1002, 1003, 1004}
    for item in results:
        assert item["conversation_id"] in valid_cids


def test_retriever_empty_and_short_queries_safe(mock_training_records):
    """Test that empty, whitespace, and out-of-vocab queries return empty list safely."""
    retriever = HistoricalRetriever()
    retriever.fit(mock_training_records)

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("xyzqwertyuiop") == []


def test_agent_build_grounded_prompt_structure():
    """Test that build_grounded_prompt formats context, inquiry, and evidence correctly."""
    context = [{"author_id": "User", "inbound": True, "text": "Prior turn message"}]
    retrieved = [
        {
            "historical_customer_text": "Sample historical customer text",
            "historical_amazonhelp_response": "Sample historical resolution",
            "similarity_score": 0.85,
        }
    ]

    messages = build_grounded_prompt(
        customer_text="Where is my parcel?",
        conversation_context=context,
        pred_intent="delivery_status_tracking",
        pred_escalation="auto_handle",
        escalation_rationale="Tracking link FAQ",
        retrieved_examples=retrieved,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "AmazonHelp" in messages[0]["content"]

    user_prompt = messages[1]["content"]
    assert "Where is my parcel?" in user_prompt
    assert "Prior turn message" in user_prompt
    assert "delivery_status_tracking" in user_prompt
    assert "auto_handle" in user_prompt
    assert "Sample historical customer text" in user_prompt
    assert "Sample historical resolution" in user_prompt


@patch("groq.Groq")
def test_agent_generate_reply_mocked(mock_groq_class, mock_training_records):
    """Test agent generation pipeline offline using mocked Groq response."""
    # Mock Groq API response
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "We'd be glad to check on this! You can track your package at https://amazon.com/orders ^JD"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_groq_class.return_value = mock_client

    retriever = HistoricalRetriever()
    retriever.fit(mock_training_records)

    agent = SupportAgent(
        retriever=retriever,
        api_key="gsk_mock_api_key_for_unit_tests",
    )

    reply, retrieved = agent.generate_reply(
        customer_text="Where is my package? It is late",
        pred_intent="delivery_status_tracking",
        pred_escalation="auto_handle",
        k=2,
    )

    assert isinstance(reply, str)
    assert len(reply) > 10
    assert "track" in reply.lower()
    assert len(retrieved) == 2
    assert retrieved[0]["conversation_id"] == 1001
