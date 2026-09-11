"""
Customer Support Agent Module for AmazonHelp.

Constructs grounded prompts using retrieved historical resolutions and calls Groq
(llama-3.3-70b-versatile) to draft safe, brand-aligned, customer-facing replies.
"""

import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import groq

from src.intents import classify_message_intent, HybridIntentClassifier
from scripts.annotate_golden import suggest_escalation
from src.retrieval import HistoricalRetriever

load_dotenv()

# ---------------------------------------------------------------------------
# Production intent classifier (Candidate D: Regex ≥ 0.85 Cascade + LinearSVC)
# Trained lazily on first call from data/processed/amazonhelp_dev_train.jsonl
# ---------------------------------------------------------------------------
_production_intent_clf = HybridIntentClassifier()


def build_grounded_prompt(
    customer_text: str,
    conversation_context: Optional[List[Dict[str, Any]]] = None,
    pred_intent: Optional[str] = None,
    pred_escalation: Optional[str] = None,
    escalation_rationale: Optional[str] = None,
    retrieved_examples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """
    Construct a structured prompt with grounding evidence and anti-hallucination constraints.
    """
    system_content = (
        "You are an AmazonHelp customer-support drafting assistant for Twitter/X.\n"
        "Your role is to draft a helpful, concise, and professional customer support tweet responding to the customer's message.\n\n"
        "Core Guidelines:\n"
        "1. Ground your response in the historical AmazonHelp resolution examples provided below.\n"
        "2. Do NOT invent policies, refund amounts, delivery dates, order statuses, guarantees, or private account details.\n"
        "3. Do NOT claim an action has already been performed (e.g. 'I have refunded your order' or 'I cancelled your order') unless the customer or evidence confirms it.\n"
        "4. If the inquiry requires private account access, order database modification, or escalation (as indicated by the escalation assessment), direct the customer to contact Amazon via official secure support channels (e.g. Amazon website/chat/help link) without making false promises.\n"
        "5. Keep the reply concise, polite, empathetic, and natural for public Twitter support.\n"
        "6. Use the historical replies as guidance on resolution style, phrasing, and links, but adapt naturally to the specific inquiry.\n"
        "7. If the customer wrote in a foreign language (e.g. Spanish, German, French, Japanese, Italian, Portuguese), reply in that same language.\n"
        "8. Never mention that you are an AI, a language model, retrieval context, training data, or an evaluation set.\n"
        "9. Output ONLY the customer-facing reply message text. Do not include labels like 'Reply:' or surrounding quotes."
    )

    parts = []
    if conversation_context:
        parts.append("PRIOR CONVERSATION CONTEXT (Thread History):")
        for turn in conversation_context:
            author = turn.get("author_id", "User")
            prefix = "[Customer]" if turn.get("inbound", False) else f"[{author}]"
            text = turn.get("text", "").replace("\n", " ")
            parts.append(f"  {prefix}: {text}")
        parts.append("")

    parts.append(f"CURRENT CUSTOMER INQUIRY:\n\"{customer_text}\"\n")

    if pred_intent:
        parts.append(f"Classified Operational Intent: {pred_intent}")
    if pred_escalation:
        parts.append(f"Escalation Assessment: {pred_escalation}")
        if escalation_rationale:
            parts.append(f"Escalation Rationale: {escalation_rationale}")
    parts.append("")

    if retrieved_examples:
        parts.append("HISTORICAL AMAZONHELP RESOLUTIONS (Grounding Evidence):")
        for idx, ex in enumerate(retrieved_examples, 1):
            cust = ex.get("historical_customer_text", "").replace("\n", " ")
            resp = ex.get("historical_amazonhelp_response", "").replace("\n", " ")
            score = ex.get("similarity_score", 0.0)
            parts.append(f"Example {idx} (Similarity: {score:.2f}):")
            parts.append(f"  Customer Inquiry: {cust}")
            parts.append(f"  AmazonHelp Resolution: {resp}")
        parts.append("")
    else:
        parts.append("No historical examples retrieved. Provide a safe, standard guidance response.")

    parts.append("Draft the final AmazonHelp customer-facing reply following all guidelines above. Output ONLY the reply text:")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(parts)},
    ]


class SupportAgent:
    """
    AmazonHelp Support Agent integrating retrieval and Groq LLM reply drafting.
    """

    def __init__(
        self,
        retriever: Optional[HistoricalRetriever] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 250,
        max_retries: int = 3,
    ):
        self.retriever = retriever
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment or arguments.")

        self.client = groq.Groq(api_key=self.api_key)

    def generate_reply(
        self,
        customer_text: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        pred_intent: Optional[str] = None,
        pred_escalation: Optional[str] = None,
        escalation_rationale: Optional[str] = None,
        retrieved_examples: Optional[List[Dict[str, Any]]] = None,
        k: int = 3,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a customer-facing support reply grounded in retrieved historical evidence.
        Returns: (pred_reply, retrieved_examples)
        """
        # 1. Classify intent using production Candidate D hybrid classifier;
        #    fall back to pure-regex baseline if the ML model is unavailable.
        if pred_intent is None:
            try:
                pred_intent, _, _ = _production_intent_clf.classify(customer_text)
            except Exception:  # pragma: no cover
                pred_intent, _, _ = classify_message_intent(customer_text)
        if pred_escalation is None:
            record_stub = {"customer_text": customer_text, "cleaned_text": customer_text}
            pred_escalation, escalation_rationale = suggest_escalation(record_stub, pred_intent)

        # 2. Retrieve historical grounding if not provided
        if retrieved_examples is None:
            if self.retriever is not None:
                retrieved_examples = self.retriever.retrieve(customer_text, k=k)
            else:
                retrieved_examples = []

        # 3. Build prompt
        messages = build_grounded_prompt(
            customer_text=customer_text,
            conversation_context=conversation_context,
            pred_intent=pred_intent,
            pred_escalation=pred_escalation,
            escalation_rationale=escalation_rationale,
            retrieved_examples=retrieved_examples,
        )

        # 4. Call Groq with exponential backoff retry
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw_reply = response.choices[0].message.content.strip()

                # Clean stray labels
                reply = re.sub(r"^(Reply|AmazonHelp|Draft):\s*", "", raw_reply, flags=re.I).strip()
                if reply.startswith("\"") and reply.endswith("\"") and len(reply) > 2:
                    reply = reply[1:-1].strip()

                return reply, retrieved_examples
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                    fallback = "qwen/qwen3.8-27b"
                    if self.model_name != fallback:
                        self.model_name = fallback
                        continue
                wait_sec = 2 ** attempt
                if attempt < self.max_retries - 1:
                    time.sleep(wait_sec)

        raise RuntimeError(f"Failed to generate reply after {self.max_retries} attempts: {last_err}")

