#!/usr/bin/env python3
"""Interactive CLI for human evaluation of generated customer support replies.

Blinded evaluation:
- Shows customer text, conversation context, gold intent/escalation, generated reply, and historical references.
- Does NOT show LLM judge scores or rationales.
- Scores 6 dimensions (1-5), calculates overall score and decision (pass/borderline/fail), and saves atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIMENSIONS = [
    ("human_correctness", "1. Correctness (1-5, addresses problem factually)"),
    ("human_relevance", "2. Relevance (1-5, directly pertinent to issue)"),
    ("human_historical_grounding", "3. Historical Grounding (1-5, AmazonHelp style & policy)"),
    ("human_helpfulness", "4. Helpfulness (1-5, actionable next steps)"),
    ("human_unsupported_claims", "5. Unsupported Claims (5=No hallucinations/safe, 1=Major false claims)"),
    ("human_escalation_appropriateness", "6. Escalation Appropriateness (1-5, matches auto_handle/escalate)"),
]


def load_annotations(annotations_path: Path) -> List[Dict[str, Any]]:
    """Load annotations file."""
    records = []
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def save_annotations(annotations_path: Path, records: List[Dict[str, Any]]) -> None:
    """Atomically write all records to disk."""
    temp_path = annotations_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    temp_path.replace(annotations_path)


def prompt_score(prompt_text: str, default: Optional[int] = None) -> int:
    """Prompt user for integer score between 1 and 5."""
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"  {prompt_text}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if raw.lower() == "q":
            raise KeyboardInterrupt("User requested exit")
        try:
            val = int(raw)
            if 1 <= val <= 5:
                return val
            print("    Error: Please enter an integer from 1 to 5.")
        except ValueError:
            print("    Error: Invalid number. Enter 1 to 5, or 'q' to quit.")


def prompt_decision(calculated_decision: str) -> str:
    """Prompt user for decision with default recommendation."""
    valid = {"pass", "borderline", "fail", "p", "b", "f"}
    mapping = {"p": "pass", "b": "borderline", "f": "fail"}
    while True:
        raw = input(f"  Decision (pass/borderline/fail) [{calculated_decision}]: ").strip().lower()
        if not raw:
            return calculated_decision
        if raw == "q":
            raise KeyboardInterrupt("User requested exit")
        if raw in valid:
            return mapping.get(raw, raw)
        print("    Error: Enter 'pass', 'borderline', or 'fail'.")


def annotate_interactive(annotations_path: Path) -> None:
    """Run interactive annotation loop."""
    sys.stdout.reconfigure(encoding="utf-8")
    records = load_annotations(annotations_path)
    total = len(records)
    completed = sum(1 for r in records if r.get("human_overall_score") is not None)

    print("=" * 75)
    print("      HUMAN REPLY EVALUATION CLI (BLINDED PROTOCOL)")
    print("=" * 75)
    print(f" Loaded: {total} examples from {annotations_path.name}")
    print(f" Progress: {completed} / {total} completed ({completed/total*100:.1f}%)")
    print(" Enter 'q' at any prompt to save and exit.")
    print("=" * 75)

    if completed == total:
        print("\nAll 50 examples have already been annotated!")
        return

    for idx, r in enumerate(records, 1):
        if r.get("human_overall_score") is not None:
            continue

        gid = r["golden_id"]
        print(f"\n[{idx}/{total}] Golden ID: {gid}")
        print("-" * 75)
        print(f"CUSTOMER MESSAGE:")
        print(f"  \"{r.get('customer_text')}\"")
        if r.get("conversation_context") and r.get("conversation_context") != "None":
            print(f"CONTEXT:")
            print(f"  {r.get('conversation_context')}")
        print(f"GOLD LABELS:")
        print(f"  Intent: {r.get('gold_intent')} | Escalation: {r.get('gold_escalation')}")

        if r.get("support_historical_response"):
            print(f"REFERENCE HISTORICAL RESPONSE (Reference only, do not require exact match):")
            print(f"  \"{r.get('support_historical_response')}\"")

        print(f"\nAI GENERATED REPLY TO EVALUATE:")
        print(f"  \"{r.get('pred_reply')}\"")
        print("-" * 75)

        try:
            scores = {}
            for dim_key, dim_desc in DIMENSIONS:
                scores[dim_key] = prompt_score(dim_desc)

            # Auto calculate overall and recommended decision
            avg_score = round(sum(scores.values()) / 6.0, 2)
            has_crit_fail = scores["human_correctness"] <= 1 or scores["human_unsupported_claims"] <= 1
            has_low_dim = any(v < 3 for v in scores.values())

            if has_crit_fail or avg_score < 3.0:
                rec_decision = "fail"
            elif avg_score >= 4.0 and not has_low_dim:
                rec_decision = "pass"
            else:
                rec_decision = "borderline"

            print(f"\n  Calculated Overall Average: {avg_score} / 5.0")
            overall_val_str = input(f"  Confirm/Edit Overall Score (1.0 - 5.0) [{avg_score}]: ").strip()
            overall_val = float(overall_val_str) if overall_val_str else avg_score

            decision_val = prompt_decision(rec_decision)
            notes = input("  Optional Notes (press Enter to skip): ").strip()

            # Record human annotations
            r.update(scores)
            r["human_overall_score"] = overall_val
            r["human_decision"] = decision_val
            r["human_notes"] = notes if notes else None

            # Save after every record
            save_annotations(annotations_path, records)
            print(f"  [Saved {gid}]")

        except KeyboardInterrupt:
            print("\nExiting and saving progress...")
            save_annotations(annotations_path, records)
            print("Progress saved. You can resume anytime.")
            return

    print("\nAnnotation complete! All 50 examples scored.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human reply evaluation CLI.")
    parser.add_argument(
        "--annotations_file",
        type=Path,
        default=Path("data/golden/human_reply_annotations.jsonl"),
        help="Path to annotations JSONL",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display completion status and exit",
    )
    args = parser.parse_args()

    if args.status:
        records = load_annotations(args.annotations_file)
        completed = sum(1 for r in records if r.get("human_overall_score") is not None)
        print(f"Human Annotations: {completed} / {len(records)} completed ({completed/len(records)*100:.1f}%)")
        return

    annotate_interactive(args.annotations_file)


if __name__ == "__main__":
    main()
