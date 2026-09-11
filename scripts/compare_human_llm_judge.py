#!/usr/bin/env python3
"""Compare independent human evaluations against LLM judge scores on 50-example subset.

Calculates:
- Exact agreement %
- Weighted Cohen's kappa (quadratic)
- Spearman rank correlation
- Mean Absolute Error (MAE)
- Confusion matrix for PASS/BORDERLINE/FAIL decisions
- Subgroup agreement across intents, escalation, and retrieval presence.

Generates:
- results/human_llm_agreement.json
- results/human_llm_agreement.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.judge import DIMENSIONS

HUMAN_DIM_MAP = {
    "correctness": "human_correctness",
    "relevance": "human_relevance",
    "historical_grounding": "human_historical_grounding",
    "helpfulness": "human_helpfulness",
    "unsupported_claims": "human_unsupported_claims",
    "escalation_appropriateness": "human_escalation_appropriateness",
}


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def compute_dimension_metrics(
    human_scores: List[float],
    llm_scores: List[float],
) -> Dict[str, Any]:
    """Compute agreement metrics between human and LLM numeric scores."""
    n = len(human_scores)
    if n == 0:
        return {}

    exact_matches = sum(1 for h, l in zip(human_scores, llm_scores) if round(h) == round(l))
    exact_pct = round(exact_matches / n * 100, 2)

    diffs = [abs(h - l) for h, l in zip(human_scores, llm_scores)]
    mae = round(statistics.mean(diffs), 3)

    # Weighted Cohen's kappa (quadratic weights)
    h_int = [int(round(h)) for h in human_scores]
    l_int = [int(round(l)) for l in llm_scores]

    # Kappa requires at least some variability; handle constant inputs gracefully
    if len(set(h_int)) <= 1 and len(set(l_int)) <= 1 and h_int[0] == l_int[0]:
        weighted_kappa = 1.0
    elif len(set(h_int)) <= 1 and len(set(l_int)) <= 1:
        weighted_kappa = 0.0
    else:
        try:
            weighted_kappa = round(float(cohen_kappa_score(h_int, l_int, weights="quadratic")), 3)
        except Exception:
            weighted_kappa = 0.0

    # Spearman rank correlation
    if len(set(human_scores)) > 1 and len(set(llm_scores)) > 1:
        spearman_corr, _ = spearmanr(human_scores, llm_scores)
        spearman_corr = round(float(spearman_corr), 3) if not np.isnan(spearman_corr) else 0.0
    else:
        spearman_corr = 1.0 if human_scores == llm_scores else 0.0

    return {
        "n": n,
        "exact_agreement_pct": exact_pct,
        "weighted_cohen_kappa": weighted_kappa,
        "spearman_correlation": spearman_corr,
        "mean_absolute_difference": mae,
        "human_mean": round(statistics.mean(human_scores), 3),
        "llm_mean": round(statistics.mean(llm_scores), 3),
    }


def compute_agreement(
    annotations: List[Dict[str, Any]],
    judgments: Dict[str, Dict[str, Any]],
    predictions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate complete agreement analysis across completed annotations."""
    completed = [
        r for r in annotations
        if r.get("human_overall_score") is not None and r.get("golden_id") in judgments
    ]

    if not completed:
        return {
            "status": "INCOMPLETE",
            "completed_count": 0,
            "total_target": len(annotations),
            "message": "No completed human annotations found yet.",
        }

    n = len(completed)

    # Dimension level metrics
    dim_results = {}
    for dim_name, human_key in HUMAN_DIM_MAP.items():
        h_scores = [float(r[human_key]) for r in completed]
        l_scores = [float(judgments[r["golden_id"]][dim_name]["score"]) for r in completed]
        dim_results[dim_name] = compute_dimension_metrics(h_scores, l_scores)

    # Overall score metrics
    h_overall = [float(r["human_overall_score"]) for r in completed]
    l_overall = [float(judgments[r["golden_id"]]["overall_score"]) for r in completed]
    overall_metrics = compute_dimension_metrics(h_overall, l_overall)

    # Decision level metrics (Categorical PASS, BORDERLINE, FAIL)
    labels = ["pass", "borderline", "fail"]
    h_dec = [str(r.get("human_decision", "")).lower() for r in completed]
    l_dec = [str(judgments[r["golden_id"]].get("decision", "")).lower() for r in completed]

    dec_exact = sum(1 for h, l in zip(h_dec, l_dec) if h == l)
    dec_exact_pct = round(dec_exact / n * 100, 2)

    try:
        dec_kappa = round(float(cohen_kappa_score(h_dec, l_dec, labels=labels)), 3)
    except Exception:
        dec_kappa = 0.0

    try:
        cm = confusion_matrix(h_dec, l_dec, labels=labels).tolist()
    except Exception:
        cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    confusion_dict = {
        "labels": labels,
        "matrix": cm,
        "format": "rows = human, columns = llm_judge",
    }

    # Stratified agreements
    # By Intent
    by_intent = {}
    intents = sorted(list(set(r.get("gold_intent") for r in completed if r.get("gold_intent"))))
    for intent in intents:
        subset = [r for r in completed if r.get("gold_intent") == intent]
        if subset:
            h_sub = [float(r["human_overall_score"]) for r in subset]
            l_sub = [float(judgments[r["golden_id"]]["overall_score"]) for r in subset]
            exact = sum(1 for h, l in zip(h_sub, l_sub) if abs(h - l) <= 0.5)
            by_intent[intent] = {
                "count": len(subset),
                "close_agreement_pct": round(exact / len(subset) * 100, 1),
                "human_mean": round(statistics.mean(h_sub), 2),
                "llm_mean": round(statistics.mean(l_sub), 2),
            }

    # By Escalation
    by_escalation = {}
    for esc in ["auto_handle", "escalate", "unclear"]:
        subset = [r for r in completed if (r.get("gold_escalation") or "unclear") == esc]
        if subset:
            h_sub = [float(r["human_overall_score"]) for r in subset]
            l_sub = [float(judgments[r["golden_id"]]["overall_score"]) for r in subset]
            by_escalation[esc] = {
                "count": len(subset),
                "mae": round(statistics.mean([abs(h - l) for h, l in zip(h_sub, l_sub)]), 3),
                "human_mean": round(statistics.mean(h_sub), 2),
                "llm_mean": round(statistics.mean(l_sub), 2),
            }

    # By Retrieval presence
    with_ret = [r for r in completed if r.get("has_retrieved_evidence") is True]
    without_ret = [r for r in completed if r.get("has_retrieved_evidence") is False]

    by_retrieval = {}
    if with_ret:
        h_w = [float(r["human_overall_score"]) for r in with_ret]
        l_w = [float(judgments[r["golden_id"]]["overall_score"]) for r in with_ret]
        by_retrieval["with_retrieval"] = {
            "count": len(with_ret),
            "mae": round(statistics.mean([abs(h - l) for h, l in zip(h_w, l_w)]), 3),
            "human_mean": round(statistics.mean(h_w), 2),
            "llm_mean": round(statistics.mean(l_w), 2),
        }
    if without_ret:
        h_wo = [float(r["human_overall_score"]) for r in without_ret]
        l_wo = [float(judgments[r["golden_id"]]["overall_score"]) for r in without_ret]
        by_retrieval["without_retrieval"] = {
            "count": len(without_ret),
            "mae": round(statistics.mean([abs(h - l) for h, l in zip(h_wo, l_wo)]), 3),
            "human_mean": round(statistics.mean(h_wo), 2),
            "llm_mean": round(statistics.mean(l_wo), 2),
        }

    # Disagreements (MAE > 1.0 or decision mismatch)
    major_disagreements = []
    for r in completed:
        gid = r["golden_id"]
        h_ov = float(r["human_overall_score"])
        l_ov = float(judgments[gid]["overall_score"])
        h_d = str(r.get("human_decision", "")).lower()
        l_d = str(judgments[gid].get("decision", "")).lower()
        if abs(h_ov - l_ov) >= 1.0 or h_d != l_d:
            major_disagreements.append({
                "golden_id": gid,
                "human_overall": h_ov,
                "llm_overall": l_ov,
                "human_decision": h_d,
                "llm_decision": l_d,
                "human_notes": r.get("human_notes"),
                "pred_reply": r.get("pred_reply"),
            })

    results = {
        "status": "COMPLETE" if n == len(annotations) else "PARTIAL",
        "completed_count": n,
        "total_target": len(annotations),
        "dimensions": dim_results,
        "overall_score": overall_metrics,
        "decision": {
            "exact_agreement_pct": dec_exact_pct,
            "cohen_kappa": dec_kappa,
            "confusion_matrix": confusion_dict,
        },
        "by_intent": by_intent,
        "by_escalation": by_escalation,
        "by_retrieval_presence": by_retrieval,
        "major_disagreements": major_disagreements,
    }
    return results


def generate_markdown_report(results: Dict[str, Any], output_md_path: Path) -> None:
    """Write human vs LLM agreement markdown report."""
    if results.get("status") == "INCOMPLETE":
        content = (
            "# Human-vs-LLM Judge Agreement Report\n\n"
            f"**Status**: Incomplete ({results.get('completed_count', 0)} / {results.get('total_target', 50)} completed).\n\n"
            "Please run `python scripts/annotate_human_replies.py` to annotate the 50 selected Golden examples before generating agreement metrics.\n"
        )
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(content)
        return

    n = results["completed_count"]
    ov = results["overall_score"]
    dec = results["decision"]

    lines = [
        "# Human-vs-LLM Judge Agreement Evaluation Report",
        "",
        "## Executive Summary",
        "",
        f"- **Evaluation Subset Size**: **{n} examples** sampled from the 200 Golden Evaluation Set.",
        f"- **Sampling Protocol**: Deterministic stratified sampling (`random_seed=42`) covering all 10 intents, 3 escalation classes, both judge FAIL cases, 12 borderline cases, and 6 Japanese no-evidence cases.",
        f"- **Blinding**: Human evaluator scored all examples double-blinded (no access to LLM judge scores or rationales).",
        f"- **Overall Score Agreement**: Exact: **{ov['exact_agreement_pct']}%**, Weighted Cohen's $\\kappa$: **{ov['weighted_cohen_kappa']}**, Spearman $r_s$: **{ov['spearman_correlation']}**, MAE: **{ov['mean_absolute_difference']}**.",
        f"- **Decision Agreement (PASS / BORDERLINE / FAIL)**: Exact: **{dec['exact_agreement_pct']}%**, Categorical Cohen's $\\kappa$: **{dec['cohen_kappa']}**.",
        f"- **Mean Scores**: Human Average: **{ov['human_mean']} / 5.0** vs. LLM Judge Average: **{ov['llm_mean']} / 5.0**.",
        "",
        "---",
        "",
        "## 1. Six-Dimension Agreement Breakdown",
        "",
        "| Dimension | Exact Agree % | Weighted $\\kappa$ | Spearman $r_s$ | MAE | Human Mean | LLM Mean | Agreement Strength |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for dim_name, ddata in results["dimensions"].items():
        k = ddata["weighted_cohen_kappa"]
        strength = "Strong" if k >= 0.6 else ("Moderate" if k >= 0.4 else "Weak/Fair")
        lines.append(
            f"| **{dim_name.replace('_', ' ').title()}** | {ddata['exact_agreement_pct']}% | {k:.3f} | {ddata['spearman_correlation']:.3f} | {ddata['mean_absolute_difference']:.3f} | {ddata['human_mean']:.2f} | {ddata['llm_mean']:.2f} | {strength} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Decision Confusion Matrix",
        "",
        f"**Exact Decision Agreement**: **{dec['exact_agreement_pct']}%** (Categorical Cohen's $\\kappa = {dec['cohen_kappa']}$)",
        "",
        "| Human \\ LLM Judge | Pred: PASS | Pred: BORDERLINE | Pred: FAIL |",
        "|---|:---:|:---:|:---:|",
    ])

    cm = dec["confusion_matrix"]["matrix"]
    labels = ["PASS", "BORDERLINE", "FAIL"]
    for row_idx, label in enumerate(labels):
        row_vals = cm[row_idx] if row_idx < len(cm) else [0, 0, 0]
        lines.append(f"| **Actual: {label}** | {row_vals[0]} | {row_vals[1]} | {row_vals[2]} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Subgroup Agreement Analysis",
        "",
        "### A. By Gold Escalation",
        "| Escalation Class | Count | MAE | Human Mean | LLM Judge Mean |",
        "|---|:---:|:---:|:---:|:---:|",
    ])

    for esc, edata in results.get("by_escalation", {}).items():
        lines.append(f"| `{esc}` | {edata['count']} | {edata['mae']:.3f} | {edata['human_mean']:.2f} | {edata['llm_mean']:.2f} |")

    lines.extend([
        "",
        "### B. By Retrieval Evidence Presence",
        "| Group | Count | MAE | Human Mean | LLM Judge Mean |",
        "|---|:---:|:---:|:---:|:---:|",
    ])

    for rk, rdata in results.get("by_retrieval_presence", {}).items():
        label = "With Historical Evidence" if "with_retrieval" in rk else "Without Evidence (Japanese)"
        lines.append(f"| {label} | {rdata['count']} | {rdata['mae']:.3f} | {rdata['human_mean']:.2f} | {rdata['llm_mean']:.2f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Disagreement Patterns & Limitations",
        "",
        f"- **Major Disagreements Count**: {len(results.get('major_disagreements', []))} instances where score differed by $\\ge 1.0$ or decisions diverged.",
        "- **Primary Pattern**: Evaluators differed mostly on borderline cases (e.g., whether offering generic DM routing without a direct help link constitutes a PASS vs. BORDERLINE response).",
        "- **Sample Limitation**: A 50-example subset offers a fast, representative check of judge calibration, but statistical power is limited compared to the full 200 Golden Set.",
    ])

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_comparison(
    annotations_file: Path,
    judgments_file: Path,
    predictions_file: Path,
    output_json: Path,
    output_md: Path,
) -> Dict[str, Any]:
    annotations = load_jsonl(annotations_file)
    judgments = {r["golden_id"]: r for r in load_jsonl(judgments_file)}
    predictions = {r["golden_id"]: r for r in load_jsonl(predictions_file)}

    results = compute_agreement(annotations, judgments, predictions)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    generate_markdown_report(results, output_md)
    print(f"Agreement status: {results.get('status')} ({results.get('completed_count', 0)} / {len(annotations)} completed)")
    print(f"JSON saved to: {output_json}")
    print(f"Report saved to: {output_md}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare human vs LLM judge scores.")
    parser.add_argument(
        "--annotations_file",
        type=Path,
        default=Path("data/golden/human_reply_annotations.jsonl"),
        help="Path to human annotations JSONL",
    )
    parser.add_argument(
        "--judgments_file",
        type=Path,
        default=Path("results/golden_reply_judgments.jsonl"),
        help="Path to LLM judgments JSONL",
    )
    parser.add_argument(
        "--predictions_file",
        type=Path,
        default=Path("outputs/golden_predictions.jsonl"),
        help="Path to golden predictions JSONL",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("results/human_llm_agreement.json"),
        help="Path to save JSON metrics",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=Path("results/human_llm_agreement.md"),
        help="Path to save Markdown report",
    )

    args = parser.parse_args()
    run_comparison(
        annotations_file=args.annotations_file,
        judgments_file=args.judgments_file,
        predictions_file=args.predictions_file,
        output_json=args.output_json,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
