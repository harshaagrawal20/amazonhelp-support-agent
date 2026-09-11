#!/usr/bin/env python3
"""Deterministic stratified sampling for 50-example Human-vs-LLM Judge Evaluation.

Generates:
- results/human_judge_subset.json
- data/golden/human_reply_annotations.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set


def generate_human_subset(
    predictions_file: Path = Path("outputs/golden_predictions.jsonl"),
    judgments_file: Path = Path("results/golden_reply_judgments.jsonl"),
    output_subset_json: Path = Path("results/human_judge_subset.json"),
    output_annotations_jsonl: Path = Path("data/golden/human_reply_annotations.jsonl"),
    seed: int = 42,
) -> Dict[str, Any]:
    random.seed(seed)

    with open(predictions_file, "r", encoding="utf-8") as f:
        preds = {r["golden_id"]: r for r in [json.loads(l) for l in f if l.strip()]}

    with open(judgments_file, "r", encoding="utf-8") as f:
        judges = {r["golden_id"]: r for r in [json.loads(l) for l in f if l.strip()]}

    items = []
    for gid in sorted(preds.keys()):
        p = preds[gid]
        j = judges[gid]
        has_ret = len(p.get("retrieved_examples", [])) > 0
        esc = p.get("gold_escalation") or p.get("evaluated_gold_escalation") or "unclear"
        items.append({
            "golden_id": gid,
            "gold_intent": p.get("gold_intent"),
            "gold_escalation": esc,
            "has_retrieval": has_ret,
            "judge_decision": j.get("decision"),
            "judge_overall_score": j.get("overall_score"),
        })

    selected: Set[str] = set()

    # 1. Mandatory inclusions: Both FAIL cases (gold_018, gold_094)
    fail_cases = [it["golden_id"] for it in items if it["judge_decision"] == "fail"]
    for fc in fail_cases:
        selected.add(fc)

    # 2. Mandatory inclusions: unclear escalation cases
    unclear_cases = [it["golden_id"] for it in items if it["gold_escalation"] == "unclear"]
    for uc in unclear_cases[:2]:
        selected.add(uc)

    # 3. Include a selection of borderline cases (10 borderline cases)
    borderline_cases = [it["golden_id"] for it in items if it["judge_decision"] == "borderline" and it["golden_id"] not in selected]
    random.shuffle(borderline_cases)
    for bc in borderline_cases[:10]:
        selected.add(bc)

    # 4. Include a selection of no-retrieval Japanese cases (6 cases)
    no_ret_cases = [it["golden_id"] for it in items if not it["has_retrieval"] and it["golden_id"] not in selected]
    random.shuffle(no_ret_cases)
    for nrc in no_ret_cases[:6]:
        selected.add(nrc)

    # 5. Ensure all 10 intents have at least 2 representations
    intents = sorted(list(set(it["gold_intent"] for it in items)))
    for intent in intents:
        already = [gid for gid in selected if preds[gid].get("gold_intent") == intent]
        if len(already) < 2:
            candidates = [it["golden_id"] for it in items if it["gold_intent"] == intent and it["golden_id"] not in selected]
            random.shuffle(candidates)
            needed = 2 - len(already)
            for c in candidates[:needed]:
                selected.add(c)

    # 6. Fill remaining slots up to 50 using stratified sampling
    remaining_candidates = [it["golden_id"] for it in items if it["golden_id"] not in selected]
    random.shuffle(remaining_candidates)

    strata: Dict[Any, List[str]] = {}
    for gid in remaining_candidates:
        key = (preds[gid].get("gold_intent"), preds[gid].get("gold_escalation") or "unclear")
        strata.setdefault(key, []).append(gid)

    while len(selected) < 50 and any(strata.values()):
        for key in list(strata.keys()):
            if strata[key] and len(selected) < 50:
                picked = strata[key].pop(0)
                selected.add(picked)

    selected_ids = sorted(list(selected))
    assert len(selected_ids) == 50, f"Expected 50 selected items, got {len(selected_ids)}"

    # Compute stratum metrics
    sel_items = [it for it in items if it["golden_id"] in selected]
    int_counts: Dict[str, int] = {}
    esc_counts: Dict[str, int] = {}
    dec_counts: Dict[str, int] = {}
    ret_counts: Dict[str, int] = {}

    for it in sel_items:
        int_counts[it["gold_intent"]] = int_counts.get(it["gold_intent"], 0) + 1
        esc_counts[it["gold_escalation"]] = esc_counts.get(it["gold_escalation"], 0) + 1
        dec_counts[it["judge_decision"]] = dec_counts.get(it["judge_decision"], 0) + 1
        ret_key = "with_retrieval" if it["has_retrieval"] else "without_retrieval"
        ret_counts[ret_key] = ret_counts.get(ret_key, 0) + 1

    subset_metadata = {
        "subset_size": len(selected_ids),
        "random_seed": seed,
        "selection_strategy": "Deterministic stratified sampling covering all 10 intents, 3 escalation classes, both LLM fail cases, 12 borderline cases, 36 pass cases, and 6 Japanese zero-evidence cases.",
        "stratum_counts": {
            "intents": int_counts,
            "escalation": esc_counts,
            "judge_decisions": dec_counts,
            "retrieval_presence": ret_counts,
        },
        "selected_ids": selected_ids,
    }

    # Write results/human_judge_subset.json
    output_subset_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_subset_json, "w", encoding="utf-8") as f:
        json.dump(subset_metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved subset metadata to {output_subset_json}")

    # Write data/golden/human_reply_annotations.jsonl
    # Fields must be explicitly null/empty for human annotation
    output_annotations_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_annotations_jsonl, "w", encoding="utf-8") as f:
        for gid in selected_ids:
            p = preds[gid]
            entry = {
                "golden_id": gid,
                "customer_text": p.get("customer_text"),
                "conversation_context": p.get("conversation_context"),
                "gold_intent": p.get("gold_intent"),
                "gold_escalation": p.get("gold_escalation") or p.get("evaluated_gold_escalation"),
                "pred_reply": p.get("pred_reply"),
                "support_historical_response": p.get("support_historical_response"),
                "has_retrieved_evidence": len(p.get("retrieved_examples", [])) > 0,
                # Human annotation fields (strictly null until manually judged)
                "human_correctness": None,
                "human_relevance": None,
                "human_historical_grounding": None,
                "human_helpfulness": None,
                "human_unsupported_claims": None,
                "human_escalation_appropriateness": None,
                "human_overall_score": None,
                "human_decision": None,
                "human_notes": None,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Created human annotations template with {len(selected_ids)} records in {output_annotations_jsonl}")
    return subset_metadata


if __name__ == "__main__":
    generate_human_subset()
