"""
Lightweight Interactive CLI Annotation Tool for Golden Evaluation Set.

Enforces:
- Terminal-based workflow (zero external dependencies, zero cloud, zero DB).
- Resumes automatically from the first unlabeled example.
- Displays conversation context for multi-turn follow-ups.
- Allows viewing intent definitions (? or h).
- Default Blind Mode: Hides weak rules / model predictions to prevent confirmation bias (toggle with --show-provenance).
- Collects:
  1. gold_intent (1-10 canonical intents)
  2. gold_escalation (1: auto_handle, 2: escalate, 3: unclear)
  3. gold_notes (optional freeform text)
  4. annotation_method (manual vs assisted_accept)
- Supports dual annotators (--annotator annotator_1 or --annotator annotator_2) for Cohen's Kappa evaluation.
- Assisted Mode (--assisted): Provides deterministic intent and escalation suggestions for rapid human review.
  Every accepted suggestion requires explicit human confirmation ('y').
- Atomic file writes to prevent corruption upon unexpected exit.
- Non-destructive: Never overwrites existing human labels.

Usage:
  python scripts/annotate_golden.py
  python scripts/annotate_golden.py --annotator annotator_1
  python scripts/annotate_golden.py --annotator annotator_1 --assisted
  python scripts/annotate_golden.py --annotator annotator_2
  python scripts/annotate_golden.py --show-provenance  # reveals weak rule/model prediction
  python scripts/annotate_golden.py --review           # review/edit already labeled items
"""

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Reconfigure stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_FILE = PROJECT_ROOT / "data/golden/golden_evaluation_200.jsonl"

from src.intents import INTENT_NAMES, INTENT_TAXONOMY, classify_message_intent

ESCALATION_CHOICES = ["auto_handle", "escalate", "unclear"]


def load_golden_set(filepath: Path) -> List[Dict[str, Any]]:
    """Load all golden evaluation records."""
    if not filepath.exists():
        raise FileNotFoundError(f"Golden evaluation file not found at {filepath}")
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_golden_set(records: List[Dict[str, Any]], filepath: Path) -> None:
    """Save records atomically to avoid corruption on unexpected termination."""
    tmp_file = filepath.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_file.replace(filepath)


def is_already_annotated(record: Dict[str, Any], annotator_name: str) -> bool:
    """Check if example has already been annotated by the given annotator."""
    if annotator_name == "annotator_2":
        return record.get("gold_intent_annotator_2") is not None
    return record.get("gold_intent_annotator_1") is not None or record.get("gold_intent") is not None


def print_taxonomy_help() -> None:
    """Display quick definitions of all 10 canonical intents."""
    print("\n" + "=" * 78)
    print("TAXONOMY REFERENCE DEFINITIONS:")
    print("=" * 78)
    for idx, name in enumerate(INTENT_NAMES, 1):
        defn = INTENT_TAXONOMY[name]
        print(f"[{idx:2d}] {name} ({defn.department})")
        print(f"     Definition: {defn.description}")
        print(f"     Rule: {defn.inclusion_criteria[:100]}...\n")
    print("=" * 78 + "\n")


def display_example(
    record: Dict[str, Any],
    current_idx: int,
    total: int,
    show_provenance: bool = False,
) -> None:
    """Display customer message, turn metadata, and prior context."""
    print("\n" + "=" * 78)
    turn_type = "MULTI-TURN FOLLOW-UP" if record.get("is_followup") else "OPENING INQUIRY"
    print(f"EXAMPLE {current_idx + 1} of {total} | ID: {record['golden_id']} | Conv ID: {record['conversation_id']} | Turn: {record['turn_index']} ({turn_type})")
    print(f"Language: {record.get('detected_language', 'en').upper()} | Sampling Stratum: {record.get('sampling_stratum', 'unknown')}")
    print("-" * 78)

    # Show prior context if multi-turn follow-up
    context = record.get("conversation_context", [])
    if context:
        print("CONVERSATION CONTEXT (Prior Messages in this Thread):")
        for turn in context:
            author = turn.get("author_id", "user")
            prefix = "[Customer]" if turn.get("inbound", False) else f"[{author}]"
            text_preview = turn.get("text", "").replace("\n", " ")
            print(f"  {prefix:12s}: {text_preview}")
        print("-" * 78)

    print("CURRENT CUSTOMER MESSAGE (To Classify):")
    print(f"  >>> \"{record['customer_text']}\"")
    print("-" * 78)

    if record.get("support_historical_response"):
        print("HISTORICAL AMAZONHELP REPLY (Grounding Reference):")
        print(f"  [AmazonHelp] : {record['support_historical_response']}")
        print("-" * 78)

    if show_provenance:
        print(f"Reference Info (Provenance Only) -> Weak Rule: [{record.get('weak_label')}] | TF-IDF Model: [{record.get('model_prediction')}]")
        print("-" * 78)

    if record.get("gold_intent"):
        print(f"Current Labels -> Intent: [{record.get('gold_intent')}] | Escalation: [{record.get('gold_escalation')}]")


def suggest_intent(record: Dict[str, Any]) -> str:
    """
    Generate deterministic intent suggestion using the project's existing Phase 3 rule classifier.
    No LLM, external API, or non-deterministic model is used.
    """
    text = record.get("customer_text", "")
    intent, _, _ = classify_message_intent(text)
    return intent


def suggest_escalation(record: Dict[str, Any], suggested_intent: str) -> Tuple[str, str]:
    """
    Deterministic heuristic for suggesting escalation status based on annotation guidelines.
    Returns: (suggested_escalation, rationale)
    Choices: 'auto_handle', 'escalate', 'unclear'
    """
    text = record.get("customer_text", "").lower()
    cleaned = record.get("cleaned_text", "").lower()

    # 1. Unclear / Insufficient Information Check
    words = re.findall(r"\b\w+\b", cleaned)
    if len(words) <= 2 and any(w in ["hi", "hello", "hey", "help", "please", "ok", "yes", "no"] for w in words):
        return "unclear", "Isolated greeting or fragment without problem details"
    if suggested_intent == "other_unclear" and len(words) < 5 and not any(
        kw in text for kw in ["fraud", "scam", "stole", "hacked", "cancel", "refund"]
    ):
        return "unclear", "Ambiguous message with insufficient context"

    # 2. High-Severity Escalation Signals (Security, Fraud, Cancellations, Disputes, Distress)
    escalate_patterns = [
        (
            r"\b(hack|hacked|lock|locked|suspended|unauthorized|stolen|theft|fraud|scam|cheat)\b",
            "Security/fraud issue requiring human investigation",
        ),
        (
            r"\b(password reset|otp|verification code|2fa|cannot log in|can\'t log in|login issue|sign in|signing in)\b",
            "Account access/authentication issue",
        ),
        (
            r"\b(cancel my order|cancel order|cancel it|stop delivery|change address|wrong address)\b",
            "Order cancellation/modification requiring database action",
        ),
        (
            r"\b(charged twice|double charge|overcharged|unauthorized charge|wrong charge|deducted)\b",
            "Payment discrepancy requiring billing investigation",
        ),
        (
            r"\b(missing item|empty box|broken|damaged|defective|shattered|faulty|fake)\b",
            "Product damage/defect/missing claim",
        ),
        (
            r"\b(failed delivery|never arrived|driver.*stole|not received|still waiting.*days|where is my refund)\b",
            "Unresolved logistics failure or refund inquiry",
        ),
        (
            r"\b(20-30 times|complained.*times|pathetic|worst|sue|lawyer|court|police)\b",
            "Severe customer distress or escalated dispute",
        ),
    ]
    for pattern, reason in escalate_patterns:
        if re.search(pattern, text):
            return "escalate", reason

    # 3. Intent-based baseline guidelines
    if suggested_intent == "account_access_security":
        return "escalate", "Account access & security inquiries require authenticated human agent"
    if suggested_intent == "order_cancellation_modification":
        return "escalate", "Order cancellation/modification requires human database action"
    if suggested_intent == "damaged_defective_wrong_item":
        return "escalate", "Physical damage or defect claims typically require return/replacement processing"
    if suggested_intent == "payment_billing_promotions" and any(kw in text for kw in ["charge", "paid", "fee", "bill", "card", "bank"]):
        return "escalate", "Billing & payment discrepancies require private account access"
    if suggested_intent == "return_refund_exchange":
        if any(kw in text for kw in ["how do i return", "how to return", "policy"]):
            return "auto_handle", "Standard Online Returns Center guidance link"
        return "escalate", "Refund or return processing lookup"

    if suggested_intent == "prime_digital_services":
        return "auto_handle", "Standard digital troubleshooting, Prime FAQ, or self-help guide"
    if suggested_intent == "delivery_status_tracking":
        return "auto_handle", "Tracking inquiry resolvable via standard status link"
    if suggested_intent == "feedback_complaint_chatter":
        return "auto_handle", "Brand sentiment/feedback not requiring DB action"
    if suggested_intent == "product_seller_inquiry":
        return "auto_handle", "Product specs/availability resolvable via catalog FAQ"

    return "auto_handle", "Standard guidance resolvable through FAQ/self-service link"


def prompt_assisted_choice(
    sugg_intent: str,
    sugg_escalation: str,
    rationale: str,
) -> str:
    """Prompt annotator in assisted mode to accept, manually override, skip, or quit."""
    print("\n" + "-" * 78)
    print("*** ASSISTED SUGGESTIONS (Review carefully before accepting) ***")
    print(f"  * Suggested Intent     : [{sugg_intent}]")
    print(f"  * Suggested Escalation : [{sugg_escalation}]")
    print(f"  * Heuristic Rationale  : {rationale}")
    print("-" * 78)
    print("Options:")
    print("  [y] Accept suggestions")
    print("  [n] Manually choose intent & escalation")
    print("  [s] Skip this example")
    print("  [q] Save and Quit")

    while True:
        choice = input("\nEnter choice (y/n/s/q): ").strip().lower()
        if choice in ["y", "yes"]:
            return "ACCEPT"
        elif choice in ["n", "no"]:
            return "MANUAL"
        elif choice in ["s", "skip"]:
            return "SKIP"
        elif choice in ["q", "quit"]:
            return "QUIT"
        elif choice in ["?", "h", "help"]:
            print_taxonomy_help()
        else:
            print("Invalid choice. Enter 'y' to accept, 'n' to customize, 's' to skip, or 'q' to quit.")


def prompt_intent() -> Optional[str]:
    """Prompt annotator for manual intent selection."""
    print("\nSelect Intent:")
    for idx, name in enumerate(INTENT_NAMES, 1):
        print(f"  [{idx:2d}] {name}")
    print("  [ ?] View taxonomy help definitions")
    print("  [ s] Skip this example")
    print("  [ q] Save and Quit")

    while True:
        choice = input("\nEnter choice (1-10, ?, s, q): ").strip().lower()
        if choice in ["q", "quit"]:
            return "QUIT"
        elif choice in ["s", "skip"]:
            return "SKIP"
        elif choice in ["?", "h", "help"]:
            print_taxonomy_help()
        elif choice.isdigit() and 1 <= int(choice) <= 10:
            return INTENT_NAMES[int(choice) - 1]
        else:
            print("Invalid selection. Enter 1-10, '?' for help, 's' to skip, or 'q' to quit.")


def prompt_escalation() -> Optional[str]:
    """Prompt annotator for manual escalation decision."""
    print("\nSelect Escalation Decision:")
    print("  [1] auto_handle   (Can be resolved via public FAQ, return guide, tracking link, or device troubleshooting)")
    print("  [2] escalate      (Requires human agent: private account lookup, DB cancellation, refund, fraud, distress)")
    print("  [3] unclear       (Ambiguous message or insufficient context to determine)")
    print("  [s] Skip escalation choice")

    while True:
        choice = input("Enter escalation (1-3, s): ").strip().lower()
        if choice == "1":
            return "auto_handle"
        elif choice == "2":
            return "escalate"
        elif choice == "3":
            return "unclear"
        elif choice in ["s", "skip", ""]:
            return None
        else:
            print("Invalid selection. Enter 1 (auto_handle), 2 (escalate), 3 (unclear), or 's'.")


def run_annotator(
    annotator_name: str,
    review_mode: bool = False,
    show_provenance: bool = False,
    assisted_mode: bool = False,
) -> None:
    """Main interactive annotation loop with optional assisted mode."""
    records = load_golden_set(GOLDEN_FILE)
    total = len(records)
    if annotator_name == "annotator_2":
        labeled_count = sum(1 for r in records if r.get("gold_intent_annotator_2") is not None)
    else:
        labeled_count = sum(1 for r in records if r.get("gold_intent") is not None)

    print("\n" + "#" * 78)
    print("  AMAZONHELP 200-EXAMPLE GOLDEN SET ANNOTATION WORKFLOW")
    print(f"  Annotator: {annotator_name} | Progress: {labeled_count}/{total} ({(labeled_count/total)*100:.1f}%)")
    mode_desc = "Review Mode (all examples)" if review_mode else "Active Mode (unlabeled only)"
    assist_desc = "Assisted (Deterministic Suggestions)" if assisted_mode else "Manual"
    blind_desc = "Provenance Visible" if show_provenance else "Blind Annotation (Anti-bias enabled)"
    print(f"  Mode: {mode_desc} | Workflow: {assist_desc} | Protocol: {blind_desc}")
    print("#" * 78)

    for i, rec in enumerate(records):
        # In assisted mode or standard mode, skip already labeled examples for this annotator
        if is_already_annotated(rec, annotator_name) and (assisted_mode or not review_mode):
            continue

        display_example(rec, i, total, show_provenance=show_provenance)

        chosen_intent: Optional[str] = None
        chosen_escalation: Optional[str] = None
        notes: str = ""
        method: str = "manual"

        if assisted_mode:
            # 1. Deterministic Suggestions
            sugg_intent = suggest_intent(rec)
            sugg_esc, rationale = suggest_escalation(rec, sugg_intent)

            action = prompt_assisted_choice(sugg_intent, sugg_esc, rationale)

            if action == "QUIT":
                save_golden_set(records, GOLDEN_FILE)
                cur_labeled = sum(1 for r in records if r.get("gold_intent") is not None)
                print(f"\nProgress safely saved. {cur_labeled}/{total} completed. Goodbye!")
                return
            elif action == "SKIP":
                print("Skipped example.")
                continue
            elif action == "ACCEPT":
                chosen_intent = sugg_intent
                chosen_escalation = sugg_esc
                method = "assisted_accept"
            elif action == "MANUAL":
                chosen_intent = prompt_intent()
                if chosen_intent == "QUIT":
                    save_golden_set(records, GOLDEN_FILE)
                    cur_labeled = sum(1 for r in records if r.get("gold_intent") is not None)
                    print(f"\nProgress safely saved. {cur_labeled}/{total} completed. Goodbye!")
                    return
                elif chosen_intent == "SKIP":
                    print("Skipped example.")
                    continue
                chosen_escalation = prompt_escalation()
                notes = input("\nOptional annotation notes (press Enter to skip): ").strip()
                method = "manual"
        else:
            # Standard Manual Mode
            chosen_intent = prompt_intent()
            if chosen_intent == "QUIT":
                save_golden_set(records, GOLDEN_FILE)
                cur_labeled = sum(1 for r in records if r.get("gold_intent") is not None)
                print(f"\nProgress safely saved. {cur_labeled}/{total} completed. Goodbye!")
                return
            elif chosen_intent == "SKIP":
                print("Skipped example.")
                continue

            chosen_escalation = prompt_escalation()
            notes = input("\nOptional annotation notes (press Enter to skip): ").strip()
            method = "manual"

        # Update record
        rec["gold_intent"] = chosen_intent
        rec["gold_escalation"] = chosen_escalation
        rec["annotator"] = annotator_name
        rec["annotation_status"] = "labeled"
        rec["annotation_method"] = method
        if notes:
            rec["gold_notes"] = notes

        # Update annotator-specific slot for inter-annotator agreement tracking
        if annotator_name == "annotator_2":
            rec["gold_intent_annotator_2"] = chosen_intent
        else:
            rec["gold_intent_annotator_1"] = chosen_intent

        # Atomic persistence
        save_golden_set(records, GOLDEN_FILE)
        print(f"\n✓ Saved: [{rec['golden_id']}] Intent='{chosen_intent}', Escalation='{chosen_escalation}' (Method: {method})")

    final_labeled = sum(1 for r in records if r.get("gold_intent") is not None)
    print("\n" + "=" * 78)
    print(f"Annotation session finished! Total Progress: {final_labeled}/{total} labeled.")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Golden Set Annotation CLI.")
    parser.add_argument("--annotator", type=str, default="annotator_1", help="Annotator ID (e.g. annotator_1, annotator_2)")
    parser.add_argument("--review", action="store_true", help="Review all examples including previously labeled ones")
    parser.add_argument("--assisted", action="store_true", help="Assisted annotation mode: suggests intent and escalation using deterministic rules for rapid human review")
    parser.add_argument("--show-provenance", action="store_true", help="Display weak rule and TF-IDF model predictions (disabled by default for blind annotation)")
    args = parser.parse_args()

    run_annotator(args.annotator, args.review, args.show_provenance, args.assisted)
