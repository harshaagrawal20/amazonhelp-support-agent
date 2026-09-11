"""
Sampling and Construction Script for the 200-Example Golden Evaluation Set.

Enforces:
1. Provenance: HELD-OUT TEST CONVERSATIONS ONLY (amazonhelp_test.jsonl).
2. One example per conversation (exactly 200 unique conversation_ids).
3. Stratified + targeted sampling:
   - Model / weak-rule disagreements (stress-testing confusion boundaries)
   - Rare operational intents (boosting tail-intent representation)
   - Multilingual inquiries (Japanese, German, French, Spanish, Italian, Portuguese)
   - Core operational intents (high-volume delivery, refund, payment, prime)
   - Social chatter & complaints (brand sentiment, quizzes, praise)
   - Ambiguous / unclassified opening queries
   - Context-dependent follow-up turns (testing multi-turn conversational context)
4. Determinism: Fixed random seed (42).
5. Unlabeled schema: gold_intent, gold_reply_quality, gold_escalation initialized to null.
"""

from collections import Counter
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Reconfigure stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intents import (
    INTENT_NAMES,
    clean_tweet_text,
    classify_message_intent,
)


def detect_language_hint(text: str) -> str:
    """Lightweight language detector for stratification."""
    # CJK / Japanese check
    if re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text):
        return "ja"
    # German markers
    if re.search(r"\b(ich|nicht|eine|einen|bitte|danke|lieferung|bestellung|versand|kann|habe|wurde)\b", text, re.I):
        return "de"
    # French markers
    if re.search(r"\b(bonjour|merci|commande|colis|livraison|pourquoi|avec|votre|cette|faire)\b", text, re.I):
        return "fr"
    # Spanish markers
    if re.search(r"\b(hola|gracias|pedido|paquete|entrega|reembolso|cuando|por favor|donde|puedo)\b", text, re.I):
        return "es"
    # Italian markers
    if re.search(r"\b(ciao|grazie|pacco|ordine|consegna|rimborso|perché|posso|quando|spedizione)\b", text, re.I):
        return "it"
    # Portuguese markers
    if re.search(r"\b(olá|obrigado|obrigada|pedido|entrega|você|minha|comprei|chegar)\b", text, re.I):
        return "pt"
    return "en"


def fit_tfidf_baseline() -> Tuple[TfidfVectorizer, LogisticRegression]:
    """Fit TF-IDF + Logistic Regression model on training split to produce model predictions."""
    print("Fitting TF-IDF + Logistic Regression baseline on train split...")
    X_train, y_train = [], []
    with open(PROJECT_ROOT / "data/processed/amazonhelp_dev_train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            X_train.append(r["cleaned_text"])
            y_train.append(r["intent"])

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)
    X_vec = vec.fit_transform(X_train)
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_vec, y_train)
    return vec, clf


def extract_test_candidate_pool(
    test_convs_path: Path,
    vec: TfidfVectorizer,
    clf: LogisticRegression,
) -> List[Dict[str, Any]]:
    """
    Extract all candidate interactions from test conversations with metadata,
    weak labels, model predictions, and language tags.
    """
    print(f"Extracting candidates from {test_convs_path.name}...")
    candidates = []
    with open(test_convs_path, "r", encoding="utf-8") as f:
        for line in f:
            conv = json.loads(line)
            cid = conv["conversation_id"]
            root_id = conv["root_id"]
            messages = conv["messages"]

            for pair in conv.get("interaction_pairs", []):
                cust_text = pair["customer_text"]
                cleaned = clean_tweet_text(cust_text)
                weak_lbl, conf, method = classify_message_intent(cust_text)
                lang = detect_language_hint(cust_text)

                # Prior conversation context
                prior = pair.get("prior_context", [])

                candidates.append({
                    "conversation_id": cid,
                    "root_id": root_id,
                    "turn_index": pair["turn_index"],
                    "is_followup": (pair["turn_index"] > 1),
                    "customer_tweet_id": pair["customer_tweet_id"],
                    "customer_author_id": pair["customer_author_id"],
                    "customer_text": cust_text,
                    "cleaned_text": cleaned,
                    "support_tweet_id": pair["support_tweet_id"],
                    "support_response": pair["support_text"],
                    "conversation_context": prior,
                    "weak_label": weak_lbl,
                    "weak_confidence": conf,
                    "weak_method": method,
                    "language": lang,
                })

    # Batch model predictions for performance
    cleaned_texts = [c["cleaned_text"] for c in candidates]
    preds = clf.predict(vec.transform(cleaned_texts))
    for c, pred in zip(candidates, preds):
        c["model_prediction"] = pred
        c["is_disagreement"] = (c["weak_label"] != pred)

    print(f"Total candidate interaction turns extracted: {len(candidates)}")
    return candidates


def sample_golden_dataset(
    candidates: List[Dict[str, Any]],
    seed: int = 42,
    target_count: int = 200,
) -> List[Dict[str, Any]]:
    """
    Deterministically sample exactly 200 examples across balanced, diverse strata.
    Enforces strictly 1 example per unique conversation_id.
    """
    rng = random.Random(seed)

    # Group candidates by conversation_id to ensure conversation-level selection
    used_cids: Set[int] = set()
    selected: List[Dict[str, Any]] = []

    def select_from_pool(pool: List[Dict[str, Any]], target_k: int, stratum_name: str) -> int:
        nonlocal selected, used_cids
        eligible = [c for c in pool if c["conversation_id"] not in used_cids]
        rng.shuffle(eligible)
        picked = eligible[:target_k]
        for item in picked:
            item["sampling_stratum"] = stratum_name
            used_cids.add(item["conversation_id"])
            selected.append(item)
        return len(picked)

    # --------------------------------------------------------------------------
    # Stratum 1: Model vs Weak-Rule Disagreements (35 examples)
    # High-ambiguity / boundary-case stress test
    # --------------------------------------------------------------------------
    disagreements = [c for c in candidates if c["is_disagreement"] and c["turn_index"] == 1]
    count1 = select_from_pool(disagreements, 35, "model_rule_disagreement")
    print(f"Stratum 1 (Model/Rule Disagreements): selected {count1}")

    # --------------------------------------------------------------------------
    # Stratum 2: Rare Operational Intents (40 examples: ~10 per rare class)
    # --------------------------------------------------------------------------
    rare_classes = [
        "product_seller_inquiry",
        "account_access_security",
        "order_cancellation_modification",
        "damaged_defective_wrong_item",
    ]
    count2 = 0
    for rc in rare_classes:
        rc_pool = [c for c in candidates if c["turn_index"] == 1 and c["weak_label"] == rc]
        c_picked = select_from_pool(rc_pool, 10, f"rare_intent_{rc}")
        count2 += c_picked
    print(f"Stratum 2 (Rare Operational Intents): selected {count2}")

    # --------------------------------------------------------------------------
    # Stratum 3: Multilingual Inquiries (25 examples)
    # Non-English queries (Japanese, German, French, Spanish, Italian, Portuguese)
    # --------------------------------------------------------------------------
    non_en = [c for c in candidates if c["turn_index"] == 1 and c["language"] != "en"]
    count3 = select_from_pool(non_en, 25, "multilingual_inquiry")
    print(f"Stratum 3 (Multilingual Inquiries): selected {count3}")

    # --------------------------------------------------------------------------
    # Stratum 4: Core Operational Intents (40 examples: 10 per core class)
    # --------------------------------------------------------------------------
    core_classes = [
        "delivery_status_tracking",
        "return_refund_exchange",
        "payment_billing_promotions",
        "prime_digital_services",
    ]
    count4 = 0
    for cc in core_classes:
        cc_pool = [c for c in candidates if c["turn_index"] == 1 and c["weak_label"] == cc]
        c_picked = select_from_pool(cc_pool, 10, f"core_intent_{cc}")
        count4 += c_picked
    print(f"Stratum 4 (Core Operational Intents): selected {count4}")

    # --------------------------------------------------------------------------
    # Stratum 5: Social Chatter & Complaints (15 examples)
    # --------------------------------------------------------------------------
    chatter_pool = [c for c in candidates if c["turn_index"] == 1 and c["weak_label"] == "feedback_complaint_chatter"]
    count5 = select_from_pool(chatter_pool, 15, "feedback_complaint_chatter")
    print(f"Stratum 5 (Feedback / Chatter): selected {count5}")

    # --------------------------------------------------------------------------
    # Stratum 6: Ambiguous / Unclear Opening Queries (20 examples)
    # --------------------------------------------------------------------------
    unclear_pool = [c for c in candidates if c["turn_index"] == 1 and c["weak_label"] == "other_unclear"]
    count6 = select_from_pool(unclear_pool, 20, "ambiguous_other_unclear")
    print(f"Stratum 6 (Ambiguous / Other Unclear): selected {count6}")

    # --------------------------------------------------------------------------
    # Stratum 7: Context-Dependent Follow-Up Turns (25 examples)
    # Multi-turn turns (turn_index >= 2) with non-empty conversation_context
    # --------------------------------------------------------------------------
    followup_pool = [c for c in candidates if c["turn_index"] > 1 and len(c["conversation_context"]) > 0]
    count7 = select_from_pool(followup_pool, 25, "context_dependent_followup")
    print(f"Stratum 7 (Context-Dependent Follow-ups): selected {count7}")

    # Fill any remaining slot up to target_count (200) deterministically from unused candidates
    remainder = target_count - len(selected)
    if remainder > 0:
        remaining_pool = [c for c in candidates if c["conversation_id"] not in used_cids]
        count_rem = select_from_pool(remaining_pool, remainder, "balanced_general_pool")
        print(f"Remainder filled: {count_rem}")

    assert len(selected) == target_count, f"Expected {target_count} examples, got {len(selected)}"
    assert len(used_cids) == target_count, f"Expected {target_count} unique conversations, got {len(used_cids)}"

    # Sort deterministically by conversation_id for reproducible golden_id numbering
    selected.sort(key=lambda x: x["conversation_id"])

    # Build final golden evaluation records with null human-annotation placeholders
    golden_records = []
    for idx, item in enumerate(selected, 1):
        record = {
            "golden_id": f"gold_{idx:03d}",
            "conversation_id": item["conversation_id"],
            "root_id": item["root_id"],
            "tweet_id": item["customer_tweet_id"],
            "turn_index": item["turn_index"],
            "is_followup": item["is_followup"],
            "customer_text": item["customer_text"],
            "cleaned_text": item["cleaned_text"],
            "conversation_context": item["conversation_context"],
            "support_historical_response": item["support_response"],
            "sampling_stratum": item["sampling_stratum"],
            "detected_language": item["language"],
            "weak_label": item["weak_label"],
            "model_prediction": item["model_prediction"],
            # Explicit null human annotation fields (Principle: Zero fabricated labels)
            "gold_intent": None,
            "gold_intent_annotator_1": None,
            "gold_intent_annotator_2": None,
            "gold_reply_quality": None,
            "gold_escalation": None,
            "gold_notes": "",
            "annotator": "",
            "annotation_status": "unlabeled",
        }
        golden_records.append(record)

    return golden_records


def generate_sampling_report(
    golden_records: List[Dict[str, Any]],
    output_path: Path,
    seed: int,
) -> None:
    """Generate comprehensive markdown report on the 200 Golden evaluation examples."""
    total = len(golden_records)
    strata_counts = Counter(r["sampling_stratum"] for r in golden_records)
    weak_counts = Counter(r["weak_label"] for r in golden_records)
    model_counts = Counter(r["model_prediction"] for r in golden_records)
    lang_counts = Counter(r["detected_language"] for r in golden_records)
    followup_count = sum(1 for r in golden_records if r["is_followup"])
    disagreement_count = sum(1 for r in golden_records if r["weak_label"] != r["model_prediction"])

    md = []
    md.append("# Phase 4: Golden Evaluation Set (200 Examples) Sampling Report\n")
    md.append("> **METHODOLOGICAL INTEGRITY STATEMENT**: All human annotation fields in this evaluation set are currently initialized to `null`. Weak labels and TF-IDF predictions are included purely for sampling provenance and disagreement analysis. They will **NOT** be used as evaluation ground truth.\n")

    md.append("## 1. Executive Summary & Provenance\n")
    md.append(f"- **Total Golden Examples**: `{total}`")
    md.append(f"- **Target Count Satisfied**: Exactly {total} examples")
    md.append(f"- **Source Partition**: Held-Out Test Set ONLY (`amazonhelp_test.jsonl`)")
    md.append(f"- **Conversation-Level Sampling**: `{len(set(r['conversation_id'] for r in golden_records))}` unique conversations (1 example per conversation, 0 duplicates)")
    md.append(f"- **Random Seed**: `{seed}` (completely deterministic)")
    md.append(f"- **Turn 1 Opening Inquiries**: `{total - followup_count}` ({((total - followup_count)/total)*100:.1f}%)")
    md.append(f"- **Context-Dependent Follow-Up Inquiries**: `{followup_count}` ({(followup_count/total)*100:.1f}%)")
    md.append(f"- **Model vs Rule Disagreement Cases**: `{disagreement_count}` ({(disagreement_count/total)*100:.1f}%)\n")

    md.append("## 2. Sampling Strata Breakdown\n")
    md.append("| Sampling Stratum | Count | % of Golden Set | Purpose / Design Objective |")
    md.append("|---|---:|---:|---|")
    strata_descriptions = {
        "model_rule_disagreement": "Disagreement between weak regex rules and TF-IDF classifier (boundary cases)",
        "rare_intent_product_seller_inquiry": "Boost representation of rare seller/pre-purchase inquiries (tail intent)",
        "rare_intent_account_access_security": "Boost representation of account security and login lockouts (tail intent)",
        "rare_intent_order_cancellation_modification": "Boost representation of order cancellation and address updates (tail intent)",
        "rare_intent_damaged_defective_wrong_item": "Boost representation of product defect and damage reports (tail intent)",
        "multilingual_inquiry": "Non-English queries (Japanese, German, French, Spanish, Italian, Portuguese)",
        "core_intent_delivery_status_tracking": "Unambiguous tracking and shipping inquiries (core high-volume intent)",
        "core_intent_return_refund_exchange": "Unambiguous returns and refund inquiries (core high-volume intent)",
        "core_intent_payment_billing_promotions": "Unambiguous payment, promo code, and gift card queries (core intent)",
        "core_intent_prime_digital_services": "Unambiguous Prime Video, Kindle, and Alexa inquiries (core intent)",
        "feedback_complaint_chatter": "Brand sentiment, rants, gratitude, and social media banter",
        "ambiguous_other_unclear": "Ambiguous/unclassified opening queries ('hi need help', isolated noise)",
        "context_dependent_followup": "Multi-turn follow-ups with prior conversation context ('Here is my email', '123-456')",
    }
    for stratum, cnt in strata_counts.most_common():
        desc = strata_descriptions.get(stratum, "Stratified sample")
        md.append(f"| `{stratum}` | {cnt} | {(cnt/total)*100:.1f}% | {desc} |")
    md.append("\n")

    md.append("## 3. Coverage by Weak Label & Model Prediction\n")
    md.append("| Intent | Weak Label Count | Weak % | Model Pred Count | Model % |")
    md.append("|---|---:|---:|---:|---:|")
    for name in INTENT_NAMES:
        w_c = weak_counts.get(name, 0)
        m_c = model_counts.get(name, 0)
        md.append(f"| `{name}` | {w_c} | {(w_c/total)*100:.1f}% | {m_c} | {(m_c/total)*100:.1f}% |")
    md.append("\n")

    md.append("## 4. Multilingual & Script Diversity\n")
    md.append("| Detected Language | Count | Percentage | Representative Sample |")
    md.append("|---|---:|---:|---|")
    lang_samples = {
        "en": "English (US / UK / IN)",
        "ja": "Japanese (Kanji / Hiragana / Katakana)",
        "de": "German (DE / AT)",
        "fr": "French (FR)",
        "es": "Spanish (ES / MX)",
        "it": "Italian (IT)",
        "pt": "Portuguese (BR)",
    }
    for lang, cnt in lang_counts.most_common():
        name = lang_samples.get(lang, lang)
        md.append(f"| `{lang}` | {cnt} | {(cnt/total)*100:.1f}% | {name} |")
    md.append("\n")

    md.append("## 5. Current Annotation Completion Status\n")
    md.append("- **Labeled**: `0 / 200` (0.0%)")
    md.append("- **Unlabeled**: `200 / 200` (100.0%)")
    md.append("- **Tooling Available**: `python scripts/annotate_golden.py` (CLI interactive annotator)")
    md.append("- **Validation Script**: `python scripts/validate_golden.py`\n")

    md.append("## 6. First 20 Golden Set Examples Preview\n")
    md.append("| Golden ID | Conv ID | Turn | Lang | Customer Message | Weak Label | Model Pred | Stratum |")
    md.append("|---|---:|---:|---|---|---|---|---|")
    for r in golden_records[:20]:
        msg_snip = r["customer_text"].replace("\n", " ").replace("|", "/")[:50]
        md.append(f"| `{r['golden_id']}` | `{r['conversation_id']}` | {r['turn_index']} | `{r['detected_language']}` | \"{msg_snip}...\" | `{r['weak_label']}` | `{r['model_prediction']}` | `{r['sampling_stratum']}` |")
    md.append("\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main() -> None:
    test_convs_path = PROJECT_ROOT / "data/processed/amazonhelp_test.jsonl"
    golden_output_path = PROJECT_ROOT / "data/golden/golden_evaluation_200.jsonl"
    report_output_path = PROJECT_ROOT / "results/golden_sampling_report.md"

    golden_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)

    vec, clf = fit_tfidf_baseline()
    candidates = extract_test_candidate_pool(test_convs_path, vec, clf)

    golden_records = sample_golden_dataset(candidates, seed=42, target_count=200)

    print(f"\nWriting exactly {len(golden_records)} records to {golden_output_path.name}...")
    with open(golden_output_path, "w", encoding="utf-8") as f:
        for r in golden_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generating sampling report at {report_output_path.name}...")
    generate_sampling_report(golden_records, report_output_path, seed=42)

    print("\nGolden set construction complete!")


if __name__ == "__main__":
    main()
