#!/usr/bin/env python3
"""Evaluate trained intent classifier models on the FROZEN 200-example Golden Set.

STRICT EVALUATION INTEGRITY RULES:
1. data/golden/golden_evaluation_200.jsonl is STRICTLY FROZEN ground truth.
   Never train on it, alter labels, remove records, or fit features on it.
2. All vectorizers and models are fit STRICTLY on data/processed/amazonhelp_dev_train.jsonl (57,760 examples).
3. Evaluates:
   - Baseline: TF-IDF Word (1,2) + LogisticRegression (C=1.0)
   - Candidate A1: Word+Char TF-IDF + LogisticRegression (C=1.0)
   - Candidate A2: Word+Char TF-IDF + LogisticRegression Balanced (C=1.0)
   - Candidate A3: Word+Char TF-IDF + LinearSVC Balanced (C=0.5) [Winning Pure ML Model]
   - Candidate D: Hybrid (Regex Cascade >= 0.85 + Word+Char LinearSVC) [Winning Overall Model]
4. Outputs:
   - results/intent_golden_evaluation.json
   - results/intent_golden_evaluation.md
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.intents import (
    INTENT_NAMES,
    clean_tweet_text,
    classify_message_intent,
)

DEV_TRAIN_FILE = PROJECT_ROOT / "data/processed/amazonhelp_dev_train.jsonl"
GOLDEN_FILE = PROJECT_ROOT / "data/golden/golden_evaluation_200.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_classification(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4),
            "support": int(s[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def main():
    print("=" * 78, flush=True)
    print("  EVALUATING INTENT CLASSIFIER MODELS ON FROZEN 200 GOLDEN SET", flush=True)
    print("=" * 78, flush=True)

    # 1. Load Golden Set
    print("\n[1/4] Loading Frozen Golden Set ground truth...", flush=True)
    golden_records = load_jsonl(GOLDEN_FILE)
    assert len(golden_records) == 200, f"Expected 200 golden examples, got {len(golden_records)}"
    
    # Sort golden examples by ID
    golden_map = {r["golden_id"]: r for r in golden_records}
    sorted_golden_ids = sorted(
        golden_map.keys(),
        key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else x
    )
    sorted_golden = [golden_map[gid] for gid in sorted_golden_ids]
    y_golden_true = [r["gold_intent"] for r in sorted_golden]
    golden_raw_texts = [r["customer_text"] for r in sorted_golden]
    golden_cleaned_texts = [clean_tweet_text(r["customer_text"]) for r in sorted_golden]
    print(f"  [+] Verified 200 frozen human ground truth records loaded.", flush=True)

    # 2. Load Dev Train
    print("\n[2/4] Loading dev train dataset...", flush=True)
    train_records = load_jsonl(DEV_TRAIN_FILE)
    train_texts = [r["cleaned_text"] for r in train_records]
    train_labels = [r["intent"] for r in train_records]
    print(f"  [+] Loaded {len(train_records):,} training interactions strictly from dev train.", flush=True)

    # 3. Fit Feature Representations
    print("\n[3/4] Fitting feature representations and training models strictly on dev train...", flush=True)
    
    # Feature 1: Word-only TF-IDF (Baseline)
    t0 = time.time()
    vec_word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)
    X_tr_word = vec_word.fit_transform(train_texts)
    X_gold_word = vec_word.transform(golden_cleaned_texts)
    print(f"  [+] Fitted Baseline Word TF-IDF in {time.time()-t0:.2f}s", flush=True)

    # Feature 2: Word + Char Subword FeatureUnion
    t0 = time.time()
    union_iso = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)),
        ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=5, max_features=25000, sublinear_tf=True)),
    ])
    X_tr_union = union_iso.fit_transform(train_texts)
    X_gold_union = union_iso.transform(golden_cleaned_texts)
    print(f"  [+] Fitted Word+Char FeatureUnion in {time.time()-t0:.2f}s", flush=True)

    # Model 1: Baseline Logistic Regression (C=1.0)
    print("  -> Training Baseline (TF-IDF Word + LogisticRegression C=1.0)...", flush=True)
    clf_base = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf_base.fit(X_tr_word, train_labels)
    preds_base = clf_base.predict(X_gold_word).tolist()

    # Model 2: Candidate A1 (Word+Char + LR C=1.0)
    print("  -> Training Candidate A1 (Word+Char + LogisticRegression C=1.0)...", flush=True)
    clf_a1 = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf_a1.fit(X_tr_union, train_labels)
    preds_a1 = clf_a1.predict(X_gold_union).tolist()

    # Model 3: Candidate A2 (Word+Char + LR Balanced C=1.0)
    print("  -> Training Candidate A2 (Word+Char + LogisticRegression Balanced C=1.0)...", flush=True)
    clf_a2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=42)
    clf_a2.fit(X_tr_union, train_labels)
    preds_a2 = clf_a2.predict(X_gold_union).tolist()

    # Model 4: Candidate A3 (Word+Char + LinearSVC Balanced C=0.5)
    print("  -> Training Candidate A3 (Word+Char + LinearSVC Balanced C=0.5)...", flush=True)
    clf_a3 = LinearSVC(C=0.5, class_weight="balanced", dual=False, random_state=42, max_iter=2000)
    clf_a3.fit(X_tr_union, train_labels)
    preds_a3 = clf_a3.predict(X_gold_union).tolist()

    # Model 5: Candidate D (Hybrid Regex >= 0.85 + Word+Char LinearSVC)
    print("  -> Evaluating Candidate D (Hybrid Regex >= 0.85 + Word+Char LinearSVC)...", flush=True)
    preds_d = []
    for raw_text, svc_pred in zip(golden_raw_texts, preds_a3):
        rule_intent, rule_conf, _ = classify_message_intent(raw_text)
        if rule_conf >= 0.85 and rule_intent != "other_unclear":
            preds_d.append(rule_intent)
        else:
            preds_d.append(svc_pred)

    # 4. Evaluate all models on Human Golden Set
    print("\n[4/4] Computing evaluation metrics against Frozen 200 Human Golden Set...", flush=True)
    models_to_eval = [
        ("Baseline (TF-IDF Word + LogisticRegression C=1.0)", preds_base),
        ("Candidate A1 (Word+Char TF-IDF + LogisticRegression C=1.0)", preds_a1),
        ("Candidate A2 (Word+Char TF-IDF + LogisticRegression Balanced C=1.0)", preds_a2),
        ("Candidate A3 (Word+Char TF-IDF + LinearSVC Balanced C=0.5)", preds_a3),
        ("Candidate D (Hybrid: Regex >= 0.85 Cascade + Word+Char LinearSVC)", preds_d),
    ]

    results_table = []
    full_eval_dict = {}

    print("\n" + "=" * 78, flush=True)
    print(f"{'Model Name':<55} | {'Acc':<6} | {'Macro-F1':<8} | {'Weighted-F1':<11}", flush=True)
    print("-" * 78, flush=True)

    for model_name, preds in models_to_eval:
        metrics = evaluate_classification(y_golden_true, preds, INTENT_NAMES)
        full_eval_dict[model_name] = metrics
        results_table.append({
            "model_name": model_name,
            "golden_accuracy": metrics["accuracy"],
            "golden_macro_f1": metrics["macro_f1"],
            "golden_weighted_f1": metrics["weighted_f1"],
        })
        print(
            f"{model_name:<55} | {metrics['accuracy']:<6.4f} | {metrics['macro_f1']:<8.4f} | {metrics['weighted_f1']:<11.4f}",
            flush=True
        )
    print("=" * 78, flush=True)

    # Save outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_DIR / "intent_golden_evaluation.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_set": "data/golden/golden_evaluation_200.jsonl (200 records)",
            "models": results_table,
            "eval_details": full_eval_dict,
        }, f, indent=2)
    print(f"\nSaved golden evaluation results to {output_json.name}", flush=True)

    # Save Markdown report
    md_lines = [
        "# Intent Classification Evaluation on FROZEN 200 Human Golden Set",
        "",
        "Evaluated on 200 stratified, human-annotated ground-truth conversations from `data/golden/golden_evaluation_200.jsonl`.",
        "All models trained strictly on `amazonhelp_dev_train.jsonl` (57,760 examples).",
        "",
        "| Model Architecture | Golden Accuracy | Golden Macro-F1 | Golden Weighted-F1 | Delta Macro-F1 vs Baseline |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    base_macro = next(r["golden_macro_f1"] for r in results_table if "Baseline" in r["model_name"])
    for row in results_table:
        delta = row["golden_macro_f1"] - base_macro
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        md_lines.append(
            f"| {row['model_name']} | {row['golden_accuracy']:.4f} | {row['golden_macro_f1']:.4f} | {row['golden_weighted_f1']:.4f} | {delta_str} |"
        )
    md_lines.append("")

    # Per-intent comparison for Baseline vs Candidate A3 vs Candidate D
    md_lines.append("## Per-Intent F1 Comparison (Human Golden Ground Truth)")
    md_lines.append("")
    md_lines.append("| Intent Name | Golden Support | Baseline F1 | Candidate A3 F1 (LinearSVC) | Candidate D F1 (Hybrid) |")
    md_lines.append("|---|:---:|:---:|:---:|:---:|")

    base_per_class = full_eval_dict["Baseline (TF-IDF Word + LogisticRegression C=1.0)"]["per_class"]
    a3_per_class = full_eval_dict["Candidate A3 (Word+Char TF-IDF + LinearSVC Balanced C=0.5)"]["per_class"]
    d_per_class = full_eval_dict["Candidate D (Hybrid: Regex >= 0.85 Cascade + Word+Char LinearSVC)"]["per_class"]

    for intent in INTENT_NAMES:
        supp = base_per_class[intent]["support"]
        f1_base = base_per_class[intent]["f1"]
        f1_a3 = a3_per_class[intent]["f1"]
        f1_d = d_per_class[intent]["f1"]
        md_lines.append(f"| `{intent}` | {supp} | {f1_base:.4f} | {f1_a3:.4f} | {f1_d:.4f} |")

    output_md = RESULTS_DIR / "intent_golden_evaluation.md"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved Markdown report to {output_md.name}", flush=True)


if __name__ == "__main__":
    main()
