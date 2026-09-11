#!/usr/bin/env python3
"""Run LLM-as-a-judge evaluation across Golden Set generated replies.

Scores each generated reply across 6 rubric dimensions:
1. correctness
2. relevance
3. historical_grounding
4. helpfulness
5. unsupported_claims
6. escalation_appropriateness

Generates:
- results/golden_reply_judgments.jsonl
- results/golden_reply_judge_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.judge import DIMENSIONS, ResponseJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(predictions_path: Path) -> List[Dict[str, Any]]:
    """Load records from golden predictions JSONL."""
    records = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def load_existing_judgments(judgments_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load existing judgments to allow atomic resume."""
    if not judgments_path.exists():
        return {}

    existing = {}
    with open(judgments_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    data = json.loads(line_str)
                    gid = data.get("golden_id")
                    if gid and "overall_score" in data and "decision" in data:
                        existing[gid] = data
                except json.JSONDecodeError:
                    continue
    return existing


def append_judgment(judgments_path: Path, judgment: Dict[str, Any]) -> None:
    """Atomically append a single judgment line to file."""
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    with open(judgments_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(judgment, ensure_ascii=False) + "\n")


def compute_aggregate_summary(
    judgments: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate evaluation statistics grouped by rubric, intent, and escalation."""
    n = len(judgments)
    if n == 0:
        return {"n": 0}

    rec_by_id = {r.get("golden_id"): r for r in records}

    dimension_stats: Dict[str, Dict[str, float]] = {}
    for dim in DIMENSIONS:
        scores = [j[dim]["score"] for j in judgments if dim in j and "score" in j[dim]]
        if scores:
            dimension_stats[dim] = {
                "mean": round(statistics.mean(scores), 3),
                "median": round(statistics.median(scores), 3),
                "std": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores),
            }

    overall_scores = [j.get("overall_score", 0.0) for j in judgments]
    decisions = [j.get("decision", "unknown") for j in judgments]

    pass_count = decisions.count("pass")
    borderline_count = decisions.count("borderline")
    fail_count = decisions.count("fail")

    # Score distributions (e.g. buckets)
    distribution = {
        "score_5": sum(1 for s in overall_scores if s >= 4.8),
        "score_4_to_4_79": sum(1 for s in overall_scores if 4.0 <= s < 4.8),
        "score_3_to_3_99": sum(1 for s in overall_scores if 3.0 <= s < 4.0),
        "score_below_3": sum(1 for s in overall_scores if s < 3.0),
    }

    # Grouping by gold escalation
    by_escalation: Dict[str, Dict[str, Any]] = {}
    for esc_label in ["auto_handle", "escalate", "unclear"]:
        matching = [
            j for j in judgments
            if rec_by_id.get(j.get("golden_id"), {}).get("gold_escalation") == esc_label
        ]
        if matching:
            esc_scores = [m.get("overall_score", 0.0) for m in matching]
            esc_decisions = [m.get("decision") for m in matching]
            by_escalation[esc_label] = {
                "count": len(matching),
                "mean_overall": round(statistics.mean(esc_scores), 3),
                "median_overall": round(statistics.median(esc_scores), 3),
                "pass_rate": round(esc_decisions.count("pass") / len(matching), 3),
                "fail_rate": round(esc_decisions.count("fail") / len(matching), 3),
            }

    # Grouping by gold intent
    by_intent: Dict[str, Dict[str, Any]] = {}
    intents = sorted(list(set(
        rec_by_id.get(j.get("golden_id"), {}).get("gold_intent") or "other_unclear"
        for j in judgments
    )))
    for intent in intents:
        matching = [
            j for j in judgments
            if rec_by_id.get(j.get("golden_id"), {}).get("gold_intent") == intent
        ]
        if matching:
            int_scores = [m.get("overall_score", 0.0) for m in matching]
            int_decisions = [m.get("decision") for m in matching]
            by_intent[intent] = {
                "count": len(matching),
                "mean_overall": round(statistics.mean(int_scores), 3),
                "pass_count": int_decisions.count("pass"),
                "pass_rate": round(int_decisions.count("pass") / len(matching), 3),
            }

    # Grouping by retrieval presence (177 with evidence vs 23 with empty evidence)
    with_retrieval = [
        j for j in judgments
        if len(rec_by_id.get(j.get("golden_id"), {}).get("retrieved_examples") or []) > 0
    ]
    without_retrieval = [
        j for j in judgments
        if len(rec_by_id.get(j.get("golden_id"), {}).get("retrieved_examples") or []) == 0
    ]

    by_retrieval: Dict[str, Any] = {}
    if with_retrieval:
        w_scores = [m.get("overall_score", 0.0) for m in with_retrieval]
        w_decisions = [m.get("decision") for m in with_retrieval]
        by_retrieval["with_retrieval_evidence"] = {
            "count": len(with_retrieval),
            "mean_overall": round(statistics.mean(w_scores), 3),
            "median_overall": round(statistics.median(w_scores), 3),
            "pass_count": w_decisions.count("pass"),
            "pass_rate": round(w_decisions.count("pass") / len(with_retrieval), 3),
            "fail_rate": round(w_decisions.count("fail") / len(with_retrieval), 3),
        }
    if without_retrieval:
        wo_scores = [m.get("overall_score", 0.0) for m in without_retrieval]
        wo_decisions = [m.get("decision") for m in without_retrieval]
        by_retrieval["without_retrieval_evidence"] = {
            "count": len(without_retrieval),
            "mean_overall": round(statistics.mean(wo_scores), 3),
            "median_overall": round(statistics.median(wo_scores), 3),
            "pass_count": wo_decisions.count("pass"),
            "pass_rate": round(wo_decisions.count("pass") / len(without_retrieval), 3),
            "fail_rate": round(wo_decisions.count("fail") / len(without_retrieval), 3),
        }

    summary = {
        "n": n,
        "mean_overall_score": round(statistics.mean(overall_scores), 3),
        "median_overall_score": round(statistics.median(overall_scores), 3),
        "overall_score_std": round(statistics.stdev(overall_scores), 3) if len(overall_scores) > 1 else 0.0,
        "decisions": {
            "pass_count": pass_count,
            "borderline_count": borderline_count,
            "fail_count": fail_count,
            "pass_rate": round(pass_count / n, 4),
            "borderline_rate": round(borderline_count / n, 4),
            "fail_rate": round(fail_count / n, 4),
        },
        "dimension_statistics": dimension_stats,
        "score_distribution": distribution,
        "by_gold_escalation": by_escalation,
        "by_gold_intent": by_intent,
        "by_retrieval_presence": by_retrieval,
    }
    return summary


def print_evaluation_report(summary: Dict[str, Any], judge_model: str) -> None:
    """Print clean formatted report to terminal."""
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n" + "=" * 75)
    print("           LLM-AS-A-JUDGE EVALUATION REPORT")
    print("=" * 75)
    print(f" Judge Provider / Model : Groq ({judge_model})")
    print(f" Total Examples Judged  : {summary.get('n', 0)}")
    print(f" Overall Mean Score     : {summary.get('mean_overall_score', 0.0)} / 5.0")
    print(f" Overall Median Score   : {summary.get('median_overall_score', 0.0)} / 5.0")
    
    decs = summary.get("decisions", {})
    print(f"\n--- DECISION BREAKDOWN ---")
    print(f" PASS       : {decs.get('pass_count', 0)} ({decs.get('pass_rate', 0)*100:.1f}%)")
    print(f" BORDERLINE : {decs.get('borderline_count', 0)} ({decs.get('borderline_rate', 0)*100:.1f}%)")
    print(f" FAIL       : {decs.get('fail_count', 0)} ({decs.get('fail_rate', 0)*100:.1f}%)")

    print(f"\n--- 6 EVALUATION DIMENSIONS (Mean / Median / Std) ---")
    dim_stats = summary.get("dimension_statistics", {})
    for dim, ddata in dim_stats.items():
        print(f" {dim:<27}: Mean {ddata['mean']:.2f} | Median {ddata['median']:.2f} | Std {ddata['std']:.2f}")

    print(f"\n--- PERFORMANCE BY RETRIEVAL EVIDENCE ---")
    ret_stats = summary.get("by_retrieval_presence", {})
    for rkey, rdata in ret_stats.items():
        label = "With Evidence (177)" if "with_retrieval" in rkey else "Without Evidence (23 Japanese)"
        print(f" {label:<32}: Mean {rdata['mean_overall']:.2f} | Pass Rate: {rdata['pass_rate']*100:.1f}% (n={rdata['count']})")

    print(f"\n--- PERFORMANCE BY GOLD ESCALATION ---")
    esc_stats = summary.get("by_gold_escalation", {})
    for ekey, edata in esc_stats.items():
        print(f" {ekey:<15}: Mean {edata['mean_overall']:.2f} | Pass Rate: {edata['pass_rate']*100:.1f}% (n={edata['count']})")

    print("=" * 75 + "\n")


def run_judge_pipeline(
    predictions_file: Path,
    output_file: Path,
    summary_file: Path,
    limit: Optional[int] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    pacing_seconds: float = 0.35,
) -> Dict[str, Any]:
    """Execute the judging process over Golden predictions."""
    load_dotenv()
    records = load_predictions(predictions_file)
    if limit is not None and limit > 0:
        records = records[:limit]

    logger.info("Loaded %d target records from %s", len(records), predictions_file)

    judge = ResponseJudge(model=model, temperature=temperature)
    logger.info("Initialized ResponseJudge: Provider=%s, Model=%s, Temp=%.1f", judge.provider, judge.model, judge.temperature)

    existing_judgments = load_existing_judgments(output_file)
    logger.info("Found %d existing judgments in %s", len(existing_judgments), output_file)

    judged_records: List[Dict[str, Any]] = []
    newly_judged = 0
    failures = 0

    for idx, rec in enumerate(records, 1):
        gid = rec.get("golden_id")
        if gid in existing_judgments:
            judged_records.append(existing_judgments[gid])
            continue

        print(f"  [{idx}/{len(records)}] Judging {gid}...", end="", flush=True)
        start_t = time.time()

        try:
            judgment = judge.judge_example(rec)
            append_judgment(output_file, judgment)
            judged_records.append(judgment)
            newly_judged += 1
            duration = time.time() - start_t
            print(f" Done ({duration:.2f}s) | Score: {judgment.get('overall_score')} ({judgment.get('decision')})")
        except Exception as exc:
            failures += 1
            print(f" FAILED ({exc})")
            logger.error("Failed to judge %s: %s", gid, exc)

        if pacing_seconds > 0:
            time.sleep(pacing_seconds)

    logger.info("Judging completed: %d total, %d newly judged, %d failures", len(judged_records), newly_judged, failures)

    summary = compute_aggregate_summary(judged_records, records)
    summary["judge_model"] = judge.model
    summary["judge_provider"] = judge.provider
    summary["failures"] = failures
    summary["newly_judged"] = newly_judged

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_evaluation_report(summary, judge.model)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Golden Set replies using LLM-as-a-judge.")
    parser.add_argument(
        "--predictions_file",
        type=Path,
        default=Path("outputs/golden_predictions.jsonl"),
        help="Path to generated golden predictions",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("results/golden_reply_judgments.jsonl"),
        help="Path to save output judgments JSONL",
    )
    parser.add_argument(
        "--summary_file",
        type=Path,
        default=Path("results/golden_reply_judge_summary.json"),
        help="Path to save summary JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate (e.g. 5 for smoke test)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Groq judge model to use (default: env GROQ_JUDGE_MODEL or openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--pacing_seconds",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)",
    )

    args = parser.parse_args()

    run_judge_pipeline(
        predictions_file=args.predictions_file,
        output_file=args.output_file,
        summary_file=args.summary_file,
        limit=args.limit,
        model=args.model,
        temperature=args.temperature,
        pacing_seconds=args.pacing_seconds,
    )


if __name__ == "__main__":
    main()
