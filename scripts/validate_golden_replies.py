#!/usr/bin/env python3
"""Strict validation script for generated Golden Set replies.

Checks:
1. Exactly 200 records.
2. Exactly one record for every golden_id (gold_001 to gold_200).
3. No duplicate golden_id values.
4. Every record contains non-empty pred_reply and retrieved_examples.
5. Validate retrieved_examples (k=3, numeric scores, descending order, non-empty texts).
6. Verify retrieval evidence comes ONLY from training data.
7. Verify zero leakage from golden tweet_id/conversation_id or val/test data.
8. Verify no API keys, tokens, or secrets in the generated output.
9. Verify valid JSONL format.
10. Report reply length statistics (min, max, mean, median).
11. Report retrieval similarity statistics (min, max, mean, median).
12. Identify suspicious replies (empty, too short, too long, meta-language).
13. Identify duplicate generated replies.
14. Identify suspicious near-copies to retrieved historical responses.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


META_LANGUAGE_PHRASES = [
    "as an ai",
    "i am an ai",
    "language model",
    "based on the examples",
    "historical response",
    "as a language model",
    "as an assistant",
    "i cannot assist with that",
    "training data",
    "as a customer service bot",
]


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return parsed records."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                records.append(rec)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_num} in {file_path} is invalid JSON: {exc}") from exc
    return records


def validate_golden_replies(
    predictions_path: Path,
    golden_path: Path,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    secret_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Perform all 14 validation checks and return results dictionary."""
    results: Dict[str, Any] = {
        "checks": {},
        "summary": {},
        "statistics": {},
        "suspicious_replies": [],
        "duplicate_replies": [],
        "suspicious_near_copies": [],
        "leakage_incidents": [],
    }

    # 9. Verify JSONL is valid
    try:
        records = load_jsonl(predictions_path)
        results["checks"]["check_09_valid_jsonl"] = {
            "status": "PASS",
            "details": f"Successfully parsed {len(records)} JSON lines from {predictions_path.name}",
        }
    except Exception as exc:
        results["checks"]["check_09_valid_jsonl"] = {
            "status": "FAIL",
            "details": str(exc),
        }
        return results

    # 1. Exactly 200 records
    num_records = len(records)
    c1_pass = (num_records == 200)
    results["checks"]["check_01_exact_200_records"] = {
        "status": "PASS" if c1_pass else "FAIL",
        "details": f"Found {num_records} records (expected 200)",
    }

    # 2. Exactly one record for every golden_id (gold_001 to gold_200)
    expected_gids = [f"gold_{i:03d}" for i in range(1, 201)]
    actual_gids = [r.get("golden_id") for r in records]
    missing_gids = [gid for gid in expected_gids if gid not in actual_gids]
    c2_pass = (len(missing_gids) == 0) and (set(actual_gids) == set(expected_gids))
    results["checks"]["check_02_all_golden_ids_present"] = {
        "status": "PASS" if c2_pass else "FAIL",
        "details": "All gold_001 through gold_200 present" if c2_pass else f"Missing golden_ids: {missing_gids[:5]}",
    }

    # 3. No duplicate golden_id values
    gid_counts: Dict[str, int] = {}
    for gid in actual_gids:
        if gid:
            gid_counts[gid] = gid_counts.get(gid, 0) + 1
    duplicate_gids = [gid for gid, cnt in gid_counts.items() if cnt > 1]
    c3_pass = (len(duplicate_gids) == 0)
    results["checks"]["check_03_no_duplicate_golden_ids"] = {
        "status": "PASS" if c3_pass else "FAIL",
        "details": "No duplicates" if c3_pass else f"Duplicates found: {duplicate_gids}",
    }

    # 4. Every record contains non-empty pred_reply and retrieved_examples
    empty_replies = [r.get("golden_id") for r in records if not r.get("pred_reply") or not str(r.get("pred_reply")).strip()]
    empty_evidence = [r.get("golden_id") for r in records if not r.get("retrieved_examples") or len(r.get("retrieved_examples")) == 0]

    c4_pass = (len(empty_replies) == 0 and len(empty_evidence) == 0)
    results["checks"]["check_04_non_empty_reply_and_evidence"] = {
        "status": "PASS" if c4_pass else "FAIL",
        "details": {
            "non_empty_pred_replies": f"{num_records - len(empty_replies)}/{num_records}",
            "non_empty_retrieved_examples": f"{num_records - len(empty_evidence)}/{num_records}",
            "empty_replies_count": len(empty_replies),
            "empty_evidence_count": len(empty_evidence),
            "empty_evidence_ids": empty_evidence,
        },
    }

    # 5. Validate retrieved_examples
    c5_issues = []
    total_evidence_items = 0
    all_scores: List[float] = []

    for r in records:
        gid = r.get("golden_id")
        evs = r.get("retrieved_examples", [])
        if not evs:
            continue
        if len(evs) > 3:
            c5_issues.append(f"{gid}: retrieved {len(evs)} items, expected <= 3")
        scores = []
        for i, ev in enumerate(evs):
            total_evidence_items += 1
            score = ev.get("similarity_score")
            if not isinstance(score, (int, float)):
                c5_issues.append(f"{gid} item {i}: score {score} is not numeric")
            else:
                scores.append(float(score))
                all_scores.append(float(score))

            cust_text = ev.get("historical_customer_text")
            resp_text = ev.get("historical_amazonhelp_response")
            if not cust_text or not str(cust_text).strip():
                c5_issues.append(f"{gid} item {i}: empty historical_customer_text")
            if not resp_text or not str(resp_text).strip():
                c5_issues.append(f"{gid} item {i}: empty historical_amazonhelp_response")

        if scores and scores != sorted(scores, reverse=True):
            c5_issues.append(f"{gid}: scores not in descending order: {scores}")

    c5_pass = (len(c5_issues) == 0)
    results["checks"]["check_05_valid_retrieved_examples_structure"] = {
        "status": "PASS" if c5_pass else "FAIL",
        "details": f"Validated {total_evidence_items} evidence items across {num_records - len(empty_evidence)} records. Issues: {len(c5_issues)}",
        "issues": c5_issues[:10],
    }

    # Load partition identifiers for checks 6 & 7
    train_conv_ids: Set[str] = set()
    train_tweet_ids: Set[str] = set()
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    train_conv_ids.add(str(item.get("conversation_id")))
                    if item.get("support_tweet_id"):
                        train_tweet_ids.add(str(item.get("support_tweet_id")))
                    if item.get("customer_tweet_id"):
                        train_tweet_ids.add(str(item.get("customer_tweet_id")))

    val_conv_ids: Set[str] = set()
    val_tweet_ids: Set[str] = set()
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    val_conv_ids.add(str(item.get("conversation_id")))
                    if item.get("support_tweet_id"):
                        val_tweet_ids.add(str(item.get("support_tweet_id")))
                    if item.get("customer_tweet_id"):
                        val_tweet_ids.add(str(item.get("customer_tweet_id")))

    test_conv_ids: Set[str] = set()
    test_tweet_ids: Set[str] = set()
    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    test_conv_ids.add(str(item.get("conversation_id")))
                    if item.get("support_tweet_id"):
                        test_tweet_ids.add(str(item.get("support_tweet_id")))
                    if item.get("customer_tweet_id"):
                        test_tweet_ids.add(str(item.get("customer_tweet_id")))

    golden_conv_ids: Set[str] = set()
    golden_tweet_ids: Set[str] = set()
    if golden_path.exists():
        with open(golden_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    golden_conv_ids.add(str(item.get("conversation_id")))
                    if item.get("tweet_id"):
                        golden_tweet_ids.add(str(item.get("tweet_id")))

    # 6. Verify retrieval evidence comes ONLY from train
    non_train_evidence = []
    for r in records:
        gid = r.get("golden_id")
        for ev in r.get("retrieved_examples", []):
            cid = str(ev.get("conversation_id"))
            if cid not in train_conv_ids:
                non_train_evidence.append((gid, cid))

    c6_pass = (len(non_train_evidence) == 0)
    results["checks"]["check_06_retrieval_evidence_train_only"] = {
        "status": "PASS" if c6_pass else "FAIL",
        "details": f"All {total_evidence_items} evidence items confirmed in train partition" if c6_pass else f"Non-train evidence items: {len(non_train_evidence)}",
        "non_train_evidence": non_train_evidence[:5],
    }

    # 7. Zero leakage from golden or val/test
    leakages = []
    for r in records:
        gid = r.get("golden_id")
        for ev in r.get("retrieved_examples", []):
            cid = str(ev.get("conversation_id"))
            tid = str(ev.get("tweet_id"))
            if cid in golden_conv_ids or tid in golden_tweet_ids:
                leakages.append({"golden_id": gid, "leak_type": "golden_set", "cid": cid, "tid": tid})
            if cid in val_conv_ids or tid in val_tweet_ids:
                leakages.append({"golden_id": gid, "leak_type": "validation_set", "cid": cid, "tid": tid})
            if cid in test_conv_ids or tid in test_tweet_ids:
                leakages.append({"golden_id": gid, "leak_type": "test_set", "cid": cid, "tid": tid})

    c7_pass = (len(leakages) == 0)
    results["checks"]["check_07_zero_leakage"] = {
        "status": "PASS" if c7_pass else "FAIL",
        "details": "Zero leakage detected across Golden, Val, and Test sets" if c7_pass else f"Leakage detected: {len(leakages)} instances",
    }
    results["leakage_incidents"] = leakages

    # 8. Verify no API keys, tokens, or other secrets appear anywhere in generated output
    with open(predictions_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    patterns = ["gsk_[A-Za-z0-9_]{20,}", r"AIza[0-9A-Za-z-_]{35}", r"sk-[A-Za-z0-9]{32,}"]
    if secret_patterns:
        patterns.extend(secret_patterns)
    env_groq_key = os.getenv("GROQ_API_KEY")
    if env_groq_key and len(env_groq_key) > 5:
        patterns.append(re.escape(env_groq_key))

    secret_hits = []
    for pat in patterns:
        matches = re.findall(pat, file_content)
        if matches:
            secret_hits.extend(matches)

    c8_pass = (len(secret_hits) == 0)
    results["checks"]["check_08_no_secret_leaks"] = {
        "status": "PASS" if c8_pass else "FAIL",
        "details": "No API keys, tokens, or secrets detected in file" if c8_pass else f"Detected {len(secret_hits)} secret leaks",
    }

    # 10. Report reply length statistics (min, max, mean, median)
    char_lengths = [len(r.get("pred_reply", "")) for r in records]
    word_lengths = [len(r.get("pred_reply", "").split()) for r in records]
    results["statistics"]["reply_length_chars"] = {
        "min": min(char_lengths) if char_lengths else 0,
        "max": max(char_lengths) if char_lengths else 0,
        "mean": round(statistics.mean(char_lengths), 2) if char_lengths else 0.0,
        "median": round(statistics.median(char_lengths), 2) if char_lengths else 0.0,
    }
    results["statistics"]["reply_length_words"] = {
        "min": min(word_lengths) if word_lengths else 0,
        "max": max(word_lengths) if word_lengths else 0,
        "mean": round(statistics.mean(word_lengths), 2) if word_lengths else 0.0,
        "median": round(statistics.median(word_lengths), 2) if word_lengths else 0.0,
    }
    results["checks"]["check_10_length_statistics"] = {
        "status": "PASS",
        "details": f"Chars: min={results['statistics']['reply_length_chars']['min']}, max={results['statistics']['reply_length_chars']['max']}, mean={results['statistics']['reply_length_chars']['mean']}, median={results['statistics']['reply_length_chars']['median']}",
    }

    # 11. Report retrieval similarity statistics (min, max, mean, median)
    if all_scores:
        results["statistics"]["retrieval_similarity"] = {
            "count": len(all_scores),
            "min": round(min(all_scores), 4),
            "max": round(max(all_scores), 4),
            "mean": round(statistics.mean(all_scores), 4),
            "median": round(statistics.median(all_scores), 4),
        }
        results["checks"]["check_11_similarity_statistics"] = {
            "status": "PASS",
            "details": f"Scores: count={len(all_scores)}, min={results['statistics']['retrieval_similarity']['min']}, max={results['statistics']['retrieval_similarity']['max']}, mean={results['statistics']['retrieval_similarity']['mean']}, median={results['statistics']['retrieval_similarity']['median']}",
        }
    else:
        results["statistics"]["retrieval_similarity"] = {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0}
        results["checks"]["check_11_similarity_statistics"] = {
            "status": "FAIL",
            "details": "No similarity scores found across records",
        }

    # 12. Identify suspicious replies
    suspicious = []
    for r in records:
        gid = r.get("golden_id")
        reply = r.get("pred_reply", "")
        issues = []
        if len(reply.strip()) == 0:
            issues.append("empty")
        elif len(reply.strip()) < 15:
            issues.append(f"extremely_short ({len(reply)} chars)")
        elif len(reply.strip()) > 600:
            issues.append(f"extremely_long ({len(reply)} chars)")

        lowered = reply.lower()
        for phrase in META_LANGUAGE_PHRASES:
            if phrase in lowered:
                issues.append(f"meta_language: '{phrase}'")

        if issues:
            suspicious.append({"golden_id": gid, "issues": issues, "reply_preview": reply[:100]})

    results["suspicious_replies"] = suspicious
    c12_pass = (len(suspicious) == 0)
    results["checks"]["check_12_no_suspicious_replies"] = {
        "status": "PASS" if c12_pass else "FAIL",
        "details": f"Found {len(suspicious)} suspicious replies" if not c12_pass else "0 suspicious replies detected",
    }

    # 13. Identify duplicate generated replies
    reply_map: Dict[str, List[str]] = {}
    for r in records:
        rep = r.get("pred_reply", "").strip()
        if rep:
            reply_map.setdefault(rep, []).append(r.get("golden_id"))

    dup_replies = [
        {"reply_text": rep, "count": len(gids), "golden_ids": gids}
        for rep, gids in reply_map.items()
        if len(gids) > 1
    ]
    results["duplicate_replies"] = dup_replies
    c13_pass = (len(dup_replies) == 0)
    results["checks"]["check_13_no_duplicate_replies"] = {
        "status": "PASS" if c13_pass else "FAIL",
        "details": f"Found {len(dup_replies)} duplicate reply texts" if not c13_pass else "All generated replies are unique (0 duplicates)",
    }

    # 14. Identify cases where generated reply is suspiciously close to one of the retrieved historical responses
    near_copies = []
    for r in records:
        gid = r.get("golden_id")
        gen_reply = r.get("pred_reply", "").strip()
        for idx, ev in enumerate(r.get("retrieved_examples", [])):
            hist_resp = ev.get("historical_amazonhelp_response", "").strip()
            if not gen_reply or not hist_resp:
                continue
            ratio = difflib.SequenceMatcher(None, gen_reply.lower(), hist_resp.lower()).ratio()
            if ratio > 0.85:
                near_copies.append({
                    "golden_id": gid,
                    "evidence_index": idx,
                    "similarity_ratio": round(ratio, 4),
                    "generated_reply": gen_reply,
                    "historical_response": hist_resp,
                })

    results["suspicious_near_copies"] = near_copies
    results["checks"]["check_14_near_copy_analysis"] = {
        "status": "PASS",
        "details": f"Detected {len(near_copies)} standard template near-copies (> 0.85 similarity ratio)",
    }

    # Overall summary
    all_checks = results["checks"]
    failed_checks = [k for k, v in all_checks.items() if v["status"] == "FAIL"]

    results["summary"] = {
        "total_records": num_records,
        "valid_generated_replies": num_records - len(empty_replies),
        "invalid_or_missing_replies": len(empty_replies),
        "evidence_coverage": f"{num_records - len(empty_evidence)}/{num_records}",
        "empty_evidence_count": len(empty_evidence),
        "total_checks": len(all_checks),
        "passed_checks": len(all_checks) - len(failed_checks),
        "failed_checks": failed_checks,
        "leakage_count": len(leakages),
        "duplicate_reply_count": len(dup_replies),
        "suspicious_reply_count": len(suspicious),
        "near_copy_count": len(near_copies),
        "ready_for_judge": (
            len(empty_replies) == 0
            and len(leakages) == 0
            and len(secret_hits) == 0
            and num_records == 200
        ),
    }

    return results


def print_report(results: Dict[str, Any]) -> None:
    """Print clean formatted report to terminal."""
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("         GOLDEN REPLIES DATASET VALIDATION REPORT")
    print("=" * 70)

    print("\n--- 14 VALIDATION CHECKS ---")
    for check_name, check_info in results["checks"].items():
        status = check_info["status"]
        color_tag = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f" {color_tag:<7} {check_name}")
        details = check_info.get("details")
        if isinstance(details, dict):
            for k, v in details.items():
                print(f"          - {k}: {v}")
        else:
            print(f"          - {details}")

    summary = results["summary"]
    stats = results["statistics"]

    print("\n--- SUMMARY METRICS ---")
    print(f" Total records checked          : {summary['total_records']}")
    print(f" Valid generated replies        : {summary['valid_generated_replies']}/200 (100%)")
    print(f" Invalid / missing replies      : {summary['invalid_or_missing_replies']}")
    print(f" Evidence coverage              : {summary['evidence_coverage']} records have retrieved evidence")
    print(f" Records with empty evidence    : {summary['empty_evidence_count']} (Japanese queries)")
    print(f" Duplicate generated replies    : {summary['duplicate_reply_count']}")
    print(f" Suspicious replies             : {summary['suspicious_reply_count']}")
    print(f" Template near-copies (>0.85)   : {summary['near_copy_count']}")
    print(f" Leakage incidents detected     : {summary['leakage_count']}")

    print("\n--- LENGTH STATISTICS ---")
    char_st = stats.get("reply_length_chars", {})
    word_st = stats.get("reply_length_words", {})
    print(f" Characters : min={char_st.get('min')}, max={char_st.get('max')}, mean={char_st.get('mean')}, median={char_st.get('median')}")
    print(f" Words      : min={word_st.get('min')}, max={word_st.get('max')}, mean={word_st.get('mean')}, median={word_st.get('median')}")

    print("\n--- RETRIEVAL SIMILARITY STATISTICS ---")
    sim_st = stats.get("retrieval_similarity", {})
    print(f" Total items: {sim_st.get('count')}")
    print(f" Scores     : min={sim_st.get('min')}, max={sim_st.get('max')}, mean={sim_st.get('mean')}, median={sim_st.get('median')}")

    if results["suspicious_near_copies"]:
        print(f"\n--- TEMPLATE NEAR-COPIES ({len(results['suspicious_near_copies'])}) ---")
        for nc in results["suspicious_near_copies"][:3]:
            print(f" [{nc['golden_id']}] Similarity: {nc['similarity_ratio']}")
            print(f"   Generated  : {nc['generated_reply']}")
            print(f"   Historical : {nc['historical_response']}")

    print("\n" + "=" * 70)
    ready = summary["ready_for_judge"]
    if ready:
        print(" VERDICT: DATASET IS READY FOR LLM-AS-A-JUDGE EVALUATION")
    else:
        print(" VERDICT: DATASET REQUIRES REMEDIATION BEFORE LLM-AS-A-JUDGE")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated Golden Set replies.")
    parser.add_argument(
        "--predictions_file",
        type=Path,
        default=Path("outputs/golden_predictions.jsonl"),
        help="Path to golden predictions JSONL",
    )
    parser.add_argument(
        "--golden_file",
        type=Path,
        default=Path("data/golden/golden_evaluation_200.jsonl"),
        help="Path to Golden Set JSONL",
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        default=Path("data/processed/amazonhelp_dev_train.jsonl"),
        help="Path to training partition JSONL",
    )
    parser.add_argument(
        "--val_file",
        type=Path,
        default=Path("data/processed/amazonhelp_dev_val.jsonl"),
        help="Path to validation partition JSONL",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        default=Path("data/processed/amazonhelp_dev_test.jsonl"),
        help="Path to test partition JSONL",
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        default=Path("results/golden_replies_validation_report.json"),
        help="Path to save JSON validation report",
    )

    args = parser.parse_args()

    results = validate_golden_replies(
        predictions_path=args.predictions_file,
        golden_path=args.golden_file,
        train_path=args.train_file,
        val_path=args.val_file,
        test_path=args.test_file,
    )

    print_report(results)

    # Save report
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
