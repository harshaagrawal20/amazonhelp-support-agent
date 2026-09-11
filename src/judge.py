"""LLM-as-a-judge evaluation module for customer support responses.

Evaluates generated customer support replies on 6 dimensions using Groq LLM:
1. Correctness
2. Relevance
3. Historical Grounding
4. Helpfulness
5. Unsupported Claims
6. Escalation Appropriateness
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DIMENSIONS = [
    "correctness",
    "relevance",
    "historical_grounding",
    "helpfulness",
    "unsupported_claims",
    "escalation_appropriateness",
]

RUBRIC_VERSION = "v1.0"

JUDGE_SYSTEM_PROMPT = """You are an expert, impartial evaluator judging AI-generated Twitter/X customer support replies for AmazonHelp.

EVALUATION PRINCIPLES:
1. Primary Truth: The customer's message and issue are the primary truth source.
2. Grounding vs Exact Matching: Retrieved historical AmazonHelp responses provide evidence of standard policies, tone, and resolution patterns. Do NOT expect or require the generated reply to match the historical response verbatim. Multiple valid responses can resolve the same issue differently. Different valid wording, tone, and actions should receive high scores.
3. Conciseness: Twitter support replies are naturally short (under 280 characters). Do NOT reward verbosity or penalize concise answers if they give correct, clear guidance.
4. No Hallucinations / Unsupported Claims: Agents cannot see private user accounts directly on Twitter. A reply must NOT claim that a specific action (refund, cancellation, shipment change, credit) has already been executed on the user's account. It must direct the customer to DM, order details, carrier, or customer service page.
5. Multilingual Handling: If the customer writes in a language other than English (e.g. Japanese, Spanish, Portuguese, French), evaluate semantic appropriateness in that language. Do not assume English is required.
6. Boilerplate: Standard AmazonHelp practices (asking customer to send a direct message / DM with details, providing official help links like https://t.co/... or amazon.com/help, or signing with an agent tag like ^XX) are authentic and should be scored favorably when appropriate.

SCORING RUBRIC (Each dimension scored 1 to 5):
1. correctness (1-5): Does the response correctly address the customer's actual problem without factual errors or misleading claims?
2. relevance (1-5): Is the response directly relevant to the customer's specific situation and conversation context?
3. historical_grounding (1-5): Is the response consistent with typical AmazonHelp resolution patterns and policies shown in historical evidence? (Do not penalize solely for differing wording). If no historical evidence was retrieved, judge whether it reflects standard, realistic Amazon customer service behavior.
4. helpfulness (1-5): Does the response provide clear next steps, actionable guidance, or appropriate resolution path for the customer?
5. unsupported_claims (1-5): IMPORTANT DIRECTION:
   - 5 = Fully safe: Zero invented claims, refunds, credits, guarantees, or false promises.
   - 4 = Minor unverified assumption that does not harm the customer.
   - 3 = Ambiguous promise or slightly overconfident claim.
   - 2 = Significant unsupported claim (e.g., claiming a package has definitely shipped or refund is guaranteed without checking).
   - 1 = Severe hallucination (e.g., claiming an account-specific action has already been performed or inventing non-existent policies/credits).
6. escalation_appropriateness (1-5): Alignment with the gold escalation expectation:
   - auto_handle: Response should reasonably answer or guide without unnecessarily escalating to human agents.
   - escalate: Response should recognize need for private account inspection/human intervention and direct to DM/secure channel without falsely claiming it was already resolved.
   - unclear: Response should handle ambiguity safely without overclaiming.

OUTPUT FORMAT:
You MUST respond with a single, strictly valid JSON object. No Markdown backticks, no text before or after.
Schema:
{
  "correctness": {"score": 1-5, "reason": "brief explanation (1-2 sentences)"},
  "relevance": {"score": 1-5, "reason": "brief explanation (1-2 sentences)"},
  "historical_grounding": {"score": 1-5, "reason": "brief explanation (1-2 sentences)"},
  "helpfulness": {"score": 1-5, "reason": "brief explanation (1-2 sentences)"},
  "unsupported_claims": {"score": 1-5, "reason": "brief explanation (1-2 sentences)"},
  "escalation_appropriateness": {"score": 1-5, "reason": "brief explanation (1-2 sentences)"},
  "overall_score": 1.0-5.0,
  "decision": "pass|borderline|fail"
}
"""


def calculate_decision(scores: Dict[str, float]) -> Tuple[float, str]:
    """Deterministically calculate overall score and decision rule.

    Decision Rules:
    - PASS: overall >= 4.0 and no dimension < 3
    - BORDERLINE: overall >= 3.0 and not PASS, and no critical hallucination (unsupported_claims >= 2)
    - FAIL: overall < 3.0 OR any critical dimension <= 1 (e.g., severe hallucination)
    """
    valid_scores = [scores[dim] for dim in DIMENSIONS if dim in scores]
    if not valid_scores:
        return 0.0, "fail"

    overall = round(sum(valid_scores) / len(valid_scores), 2)

    # Severe hallucination or failure
    if scores.get("unsupported_claims", 5) <= 1 or scores.get("correctness", 5) <= 1:
        return overall, "fail"

    if overall < 3.0:
        decision = "fail"
    elif overall >= 4.0 and all(scores.get(dim, 0) >= 3 for dim in DIMENSIONS):
        decision = "pass"
    else:
        decision = "borderline"

    return overall, decision


def parse_judge_response(raw_text: str) -> Dict[str, Any]:
    """Parse and strictly validate the JSON response from the LLM judge."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Judge response is empty")

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge output is not valid JSON: {exc} | Raw text: {raw_text[:200]}") from exc

    extracted_scores: Dict[str, float] = {}
    parsed_result: Dict[str, Any] = {}

    for dim in DIMENSIONS:
        if dim not in data or not isinstance(data[dim], dict):
            raise ValueError(f"Missing required dimension '{dim}' in judge output")
        dim_data = data[dim]
        if "score" not in dim_data or "reason" not in dim_data:
            raise ValueError(f"Dimension '{dim}' missing 'score' or 'reason'")

        try:
            score = float(dim_data["score"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Score for '{dim}' is not numeric: {dim_data['score']}") from exc

        # Clamp between 1.0 and 5.0
        score = max(1.0, min(5.0, score))
        extracted_scores[dim] = score
        parsed_result[dim] = {
            "score": score,
            "reason": str(dim_data["reason"]).strip(),
        }

    overall_score, decision = calculate_decision(extracted_scores)
    parsed_result["overall_score"] = overall_score
    parsed_result["decision"] = decision

    return parsed_result


def build_judge_prompt(record: Dict[str, Any]) -> str:
    """Build the user prompt presenting all case evidence to the judge."""
    customer_text = record.get("customer_text") or record.get("text") or ""
    context = record.get("conversation_context") or "None (First turn)"
    gold_intent = record.get("gold_intent") or "unknown"
    gold_escalation = record.get("gold_escalation") or "unknown"
    pred_intent = record.get("pred_intent_tfidf_lr") or record.get("pred_intent") or "unknown"
    pred_escalation = record.get("pred_escalation") or "unknown"
    pred_reply = record.get("pred_reply") or ""
    hist_ref = record.get("support_historical_response") or "None available"
    retrieved = record.get("retrieved_examples") or []

    prompt = (
        f"EVALUATION TASK:\n"
        f"Customer Message: {customer_text}\n"
        f"Conversation Context: {context}\n"
        f"Human Gold Intent: {gold_intent}\n"
        f"Human Gold Escalation: {gold_escalation}\n"
        f"Model Predicted Intent: {pred_intent}\n"
        f"Model Predicted Escalation: {pred_escalation}\n\n"
        f"Reference Historical Resolution (for context/comparison only, do NOT expect exact match):\n"
        f"\"{hist_ref}\"\n\n"
    )

    if retrieved:
        prompt += "Retrieved Historical AmazonHelp Evidence (top examples from training data):\n"
        for i, ev in enumerate(retrieved[:3], 1):
            c_text = ev.get("historical_customer_text", "").strip()
            a_text = ev.get("historical_amazonhelp_response", "").strip()
            sim = ev.get("similarity_score", 0.0)
            prompt += f"Example {i} (Similarity: {sim:.2f}):\n"
            prompt += f"  Past Customer: {c_text}\n"
            prompt += f"  Past Agent Reply: {a_text}\n"
        prompt += "\n"
    else:
        prompt += "Retrieved Historical AmazonHelp Evidence: None retrieved for this query (e.g., non-Latin script).\n\n"

    prompt += (
        f"AI GENERATED REPLY TO EVALUATE:\n"
        f"\"{pred_reply}\"\n\n"
        f"Please score this generated reply across all 6 dimensions according to the rubric and output strict JSON."
    )

    return prompt


class ResponseJudge:
    """Judge client using Groq API for response evaluation."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_JUDGE_MODEL") or "openai/gpt-oss-20b"
        self.temperature = temperature
        self.provider = "groq"
        self.rubric_version = RUBRIC_VERSION

        if not self.api_key:
            logger.warning("No GROQ_API_KEY set. Live calls will fail unless mocked.")
            self.client = None
        else:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)

    def judge_example(self, record: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Evaluate a single record using the LLM judge."""
        if not self.client:
            raise RuntimeError("Cannot execute judge_example: GROQ_API_KEY is not configured.")

        prompt = build_judge_prompt(record)
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                raw_content = response.choices[0].message.content
                parsed = parse_judge_response(raw_content)

                # Attach metadata
                parsed["golden_id"] = record.get("golden_id")
                parsed["judge_model"] = self.model
                parsed["judge_provider"] = self.provider
                parsed["judge_temperature"] = self.temperature
                parsed["rubric_version"] = self.rubric_version
                parsed["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                return parsed

            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                # If model not found, fallback to qwen/qwen3.8-27b
                if "model_not_found" in err_str or "not found" in err_str:
                    logger.warning("Model %s not found. Falling back to qwen/qwen3.8-27b", self.model)
                    self.model = "qwen/qwen3.8-27b"

                wait_time = (2 ** attempt) + 0.5
                time.sleep(wait_time)

        raise RuntimeError(f"Failed to judge record {record.get('golden_id')} after {max_retries} attempts: {last_error}")
