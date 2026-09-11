#!/usr/bin/env python3
"""Benchmark and compare candidate intent classification models on AmazonHelp.

Fits models strictly on: data/processed/amazonhelp_dev_train.jsonl (57,760 examples)
Evaluates models strictly on: data/processed/amazonhelp_dev_val.jsonl (12,378 examples)

Evaluates:
1. Baseline: TF-IDF Word (1,2) + Logistic Regression (C=1.0)
2. Candidate A1: Word+Char TF-IDF + Logistic Regression (Default Weights, C=1.0)
3. Candidate A2: Word+Char TF-IDF + Logistic Regression (Balanced, C=1.0)
4. Candidate A3: Word+Char TF-IDF + LinearSVC (Balanced, C=0.5, dual=False)
5. Candidate B: Dense LSA Embeddings (TruncatedSVD 100) + Logistic Regression (Balanced)
6. Candidate C: Context-Aware Word+Char TF-IDF + LinearSVC (Balanced, C=0.5)
7. Candidate D: Hybrid (High-Precision Regex Rules >= 0.85 Cascade + ML Fallback)

Selects the best model strictly using Validation Macro-F1 (with Accuracy as secondary tie-breaker).
Outputs:
- results/intent_model_comparison.json
- results/intent_model_comparison.md
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
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
DEV_VAL_FILE = PROJECT_ROOT / "data/processed/amazonhelp_dev_val.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_text_isolated(record: Dict[str, Any]) -> str:
    """Extract and clean isolated customer message."""
    if "cleaned_text" in record and record["cleaned_text"]:
        return record["cleaned_text"]
    raw = record.get("customer_message") or record.get("customer_text") or ""
    return clean_tweet_text(raw)


def get_text_with_context(record: Dict[str, Any]) -> str:
    """Extract current customer message augmented with preceding dialogue turns."""
    curr = get_text_isolated(record)
    prior = record.get("prior_context") or record.get("conversation_context") or []
    if not prior:
        return curr
    context_snippets = []
    for turn in prior[-3:]:
        text = turn.get("text", "").strip()
        if text:
            cleaned = clean_tweet_text(text)
            if cleaned:
                context_snippets.append(cleaned)
    if not context_snippets:
        return curr
    return " ".join(context_snippets) + " [SEP] " + curr


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = INTENT_NAMES,
) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for i, name in enumerate(labels):
        per_class[name] = {
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
    print("  AMAZONHELP INTENT CLASSIFIER BENCHMARK & MODEL SELECTION", flush=True)
    print("=" * 78, flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading dev train and dev val datasets...", flush=True)
    train_records = load_jsonl(DEV_TRAIN_FILE)
    val_records = load_jsonl(DEV_VAL_FILE)
    print(f"  [+] Train size: {len(train_records):,} records", flush=True)
    print(f"  [+] Val size:   {len(val_records):,} records", flush=True)

    y_train = [r["intent"] for r in train_records]
    y_val = [r["intent"] for r in val_records]

    train_texts_iso = [get_text_isolated(r) for r in train_records]
    val_texts_iso = [get_text_isolated(r) for r in val_records]

    train_texts_ctx = [get_text_with_context(r) for r in train_records]
    val_texts_ctx = [get_text_with_context(r) for r in val_records]

    print("\n[2/4] Precomputing Feature Representations...", flush=True)
    
    # 1. Baseline word-only representation
    t0_b = time.time()
    vec_word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)
    X_tr_word = vec_word.fit_transform(train_texts_iso)
    X_va_word = vec_word.transform(val_texts_iso)
    print(f"  [+] Baseline Word TF-IDF fitted in {time.time()-t0_b:.2f}s | Shape: {X_tr_word.shape}", flush=True)

    # 2. Shared Word + Char FeatureUnion for isolated text
    t0_u = time.time()
    union_iso = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)),
        ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=5, max_features=25000, sublinear_tf=True)),
    ])
    X_tr_union_iso = union_iso.fit_transform(train_texts_iso)
    X_va_union_iso = union_iso.transform(val_texts_iso)
    print(f"  [+] Word+Char FeatureUnion (Isolated) fitted in {time.time()-t0_u:.2f}s | Shape: {X_tr_union_iso.shape}", flush=True)

    # 3. Dense LSA representation (100 components)
    t0_svd = time.time()
    svd = TruncatedSVD(n_components=100, n_iter=5, random_state=42)
    X_tr_lsa = svd.fit_transform(X_tr_union_iso)
    X_va_lsa = svd.transform(X_va_union_iso)
    print(f"  [+] Dense LSA (100 components) fitted in {time.time()-t0_svd:.2f}s | Shape: {X_tr_lsa.shape}", flush=True)

    # 4. Context-aware Word + Char FeatureUnion
    t0_ctx = time.time()
    union_ctx = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)),
        ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=5, max_features=25000, sublinear_tf=True)),
    ])
    X_tr_union_ctx = union_ctx.fit_transform(train_texts_ctx)
    X_va_union_ctx = union_ctx.transform(val_texts_ctx)
    print(f"  [+] Word+Char FeatureUnion (Context-Aware) fitted in {time.time()-t0_ctx:.2f}s | Shape: {X_tr_union_ctx.shape}", flush=True)

    print("\n[3/4] Training and evaluating candidate models on Validation Set...", flush=True)

    results_table = []
    full_eval_data = {}

    def log_result(name: str, preds: List[str], elapsed: float):
        eval_dict = evaluate_predictions(y_val, preds)
        print(f"  -> {name}", flush=True)
        print(f"     Val Acc: {eval_dict['accuracy']:.4f} | Macro-F1: {eval_dict['macro_f1']:.4f} | Weighted-F1: {eval_dict['weighted_f1']:.4f} | Time: {elapsed:.2f}s", flush=True)
        results_table.append({
            "model_name": name,
            "val_accuracy": eval_dict["accuracy"],
            "val_macro_f1": eval_dict["macro_f1"],
            "val_weighted_f1": eval_dict["weighted_f1"],
            "train_time_sec": round(elapsed, 2),
        })
        full_eval_data[name] = eval_dict

    # 1. Baseline
    t0 = time.time()
    clf_base = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf_base.fit(X_tr_word, y_train)
    preds_base = clf_base.predict(X_va_word).tolist()
    log_result("Baseline (TF-IDF Word + LogisticRegression C=1.0)", preds_base, time.time() - t0)

    # 2. Candidate A1: Word+Char + LR (C=1.0, default weights)
    t0 = time.time()
    clf_a1 = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf_a1.fit(X_tr_union_iso, y_train)
    preds_a1 = clf_a1.predict(X_va_union_iso).tolist()
    log_result("Candidate A1 (Word+Char TF-IDF + LogisticRegression C=1.0)", preds_a1, time.time() - t0)

    # 3. Candidate A2: Word+Char + LR (Balanced, C=1.0)
    t0 = time.time()
    clf_a2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=42)
    clf_a2.fit(X_tr_union_iso, y_train)
    preds_a2 = clf_a2.predict(X_va_union_iso).tolist()
    log_result("Candidate A2 (Word+Char TF-IDF + LogisticRegression Balanced C=1.0)", preds_a2, time.time() - t0)

    # 4. Candidate A3: Word+Char + LinearSVC (Balanced, C=0.5, dual=False)
    t0 = time.time()
    clf_a3 = LinearSVC(C=0.5, class_weight="balanced", dual=False, random_state=42, max_iter=2000)
    clf_a3.fit(X_tr_union_iso, y_train)
    preds_a3 = clf_a3.predict(X_va_union_iso).tolist()
    log_result("Candidate A3 (Word+Char TF-IDF + LinearSVC Balanced C=0.5)", preds_a3, time.time() - t0)

    # 5. Candidate B: Dense LSA Embeddings (100 components) + LogisticRegression Balanced
    t0 = time.time()
    scaler = StandardScaler()
    X_tr_lsa_scaled = scaler.fit_transform(X_tr_lsa)
    X_va_lsa_scaled = scaler.transform(X_va_lsa)
    clf_b = LogisticRegression(C=1.0, class_weight="balanced", max_iter=200, random_state=42)
    clf_b.fit(X_tr_lsa_scaled, y_train)
    preds_b = clf_b.predict(X_va_lsa_scaled).tolist()
    log_result("Candidate B (Dense LSA 100-dim + LogisticRegression Balanced)", preds_b, time.time() - t0)

    # 6. Candidate C: Context-Aware Word+Char TF-IDF + LinearSVC (Balanced, C=0.5)
    t0 = time.time()
    clf_c = LinearSVC(C=0.5, class_weight="balanced", dual=False, random_state=42, max_iter=2000)
    clf_c.fit(X_tr_union_ctx, y_train)
    preds_c = clf_c.predict(X_va_union_ctx).tolist()
    log_result("Candidate C (Context-Aware Word+Char TF-IDF + LinearSVC Balanced)", preds_c, time.time() - t0)

    # 7. Candidate D: Hybrid (Regex Rules >= 0.85 Cascade + Word+Char LinearSVC)
    t0 = time.time()
    preds_d = []
    for r, svc_pred in zip(val_records, preds_a3):
        raw_text = r.get("customer_message") or r.get("customer_text") or ""
        rule_intent, rule_conf, _ = classify_message_intent(raw_text)
        if rule_conf >= 0.85 and rule_intent != "other_unclear":
            preds_d.append(rule_intent)
        else:
            preds_d.append(svc_pred)
    log_result("Candidate D (Hybrid: Regex >= 0.85 Cascade + Word+Char LinearSVC)", preds_d, time.time() - t0)

    # 8. Candidate D2: Hybrid (Regex Rules >= 0.85 Cascade + Context-Aware LinearSVC)
    t0 = time.time()
    preds_d2 = []
    for r, svc_pred in zip(val_records, preds_c):
        raw_text = r.get("customer_message") or r.get("customer_text") or ""
        rule_intent, rule_conf, _ = classify_message_intent(raw_text)
        if rule_conf >= 0.85 and rule_intent != "other_unclear":
            preds_d2.append(rule_intent)
        else:
            preds_d2.append(svc_pred)
    log_result("Candidate D2 (Hybrid: Regex >= 0.85 Cascade + Context-Aware LinearSVC)", preds_d2, time.time() - t0)

    # Selection based primarily on Validation Macro-F1
    best_model_info = max(results_table, key=lambda x: (x["val_macro_f1"], x["val_accuracy"]))
    print("\n" + "=" * 78, flush=True)
    print("  BEST MODEL SELECTED (by Validation Macro-F1):", flush=True)
    print(f"  {best_model_info['model_name']}", flush=True)
    print(f"  Val Macro-F1: {best_model_info['val_macro_f1']:.4f} | Val Accuracy: {best_model_info['val_accuracy']:.4f}", flush=True)
    print("=" * 78, flush=True)

    output_json = RESULTS_DIR / "intent_model_comparison.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": best_model_info,
            "comparison": results_table,
            "eval_details": full_eval_data,
        }, f, indent=2)
    print(f"\n[4/4] Saved comparison metrics to {output_json.name}", flush=True)

    md_lines = [
        "# AmazonHelp Intent Model Comparison (Validation Set)",
        "",
        "Models fit strictly on `amazonhelp_dev_train.jsonl` (57,760 examples) and evaluated on `amazonhelp_dev_val.jsonl` (12,378 examples).",
        "",
        "| Model Architecture | Validation Accuracy | Validation Macro-F1 | Validation Weighted-F1 | Train Time (s) |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    for row in sorted(results_table, key=lambda x: x["val_macro_f1"], reverse=True):
        is_best = (row["model_name"] == best_model_info["model_name"])
        prefix = "**" if is_best else ""
        suffix = "** (Selected)" if is_best else ""
        md_lines.append(
            f"| {prefix}{row['model_name']}{suffix} | {row['val_accuracy']:.4f} | {row['val_macro_f1']:.4f} | {row['val_weighted_f1']:.4f} | {row['train_time_sec']}s |"
        )
    md_lines.append("")
    md_lines.append("**Selection Criterion**: Model selection is determined strictly by **Validation Macro-F1** on held-out validation data.")

    output_md = RESULTS_DIR / "intent_model_comparison.md"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved Markdown report to {output_md.name}", flush=True)


if __name__ == "__main__":
    main()
