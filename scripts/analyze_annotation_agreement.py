r"""
Inter-Annotator Agreement Evaluation Script for Golden Evaluation Set.

Calculates:
1. Raw observed percentage agreement ($P_o$).
2. Cohen's Kappa ($\kappa$) for categorical multi-class intent agreement.
3. Confusion matrix between Annotator 1 and Annotator 2.
4. Summary of disagreements to facilitate arbitration.

If dual annotations are incomplete, reports current progress and marks agreement evaluation as pending.

Usage:
  python scripts/analyze_annotation_agreement.py
"""

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Reconfigure stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_FILE = PROJECT_ROOT / "data/golden/golden_evaluation_200.jsonl"
REPORT_OUTPUT = PROJECT_ROOT / "results/annotation_agreement_report.md"

from src.intents import INTENT_NAMES


def load_golden_set(filepath: Path) -> List[Dict[str, Any]]:
    """Load all records from golden dataset."""
    if not filepath.exists():
        raise FileNotFoundError(f"Golden dataset file not found: {filepath}")
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_agreement(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate Cohen's kappa and raw agreement between Annotator 1 and Annotator 2."""
    total = len(records)
    ann1_list: List[Optional[str]] = [r.get("gold_intent_annotator_1") for r in records]
    ann2_list: List[Optional[str]] = [r.get("gold_intent_annotator_2") for r in records]

    # Count completion
    ann1_completed = sum(1 for x in ann1_list if x is not None)
    ann2_completed = sum(1 for x in ann2_list if x is not None)
    both_completed = [
        (r["golden_id"], r["customer_text"], a1, a2)
        for r, a1, a2 in zip(records, ann1_list, ann2_list)
        if a1 is not None and a2 is not None
    ]
    overlap_count = len(both_completed)

    print("=" * 78)
    print("INTER-ANNOTATOR AGREEMENT ANALYSIS (Golden Evaluation Set)")
    print("=" * 78)
    print(f"Total Golden Examples        : {total}")
    print(f"Annotator 1 Completed        : {ann1_completed} / {total} ({(ann1_completed/total)*100:.1f}%)")
    print(f"Annotator 2 Completed        : {ann2_completed} / {total} ({(ann2_completed/total)*100:.1f}%)")
    print(f"Dual-Annotated Overlap Cases : {overlap_count} / {total} ({(overlap_count/total)*100:.1f}%)")

    if overlap_count < 10:
        print("\n[STATUS: PENDING DUAL ANNOTATION]")
        print(f"Only {overlap_count} examples have been dual-annotated so far.")
        print("To run dual annotation:")
        print("  1. Annotator 1 runs: python scripts/annotate_golden.py --annotator annotator_1")
        print("  2. Annotator 2 runs: python scripts/annotate_golden.py --annotator annotator_2")
        print("Once both passes are complete, Cohen's Kappa and confusion matrix will be computed.")

        # Save placeholder markdown report
        md = [
            "# Golden Evaluation Set: Inter-Annotator Agreement Report\n",
            "> **STATUS**: **PENDING DUAL ANNOTATION** (Methodological integrity enforced: no synthetic agreement scores).\n",
            f"- **Total Examples**: `{total}`",
            f"- **Annotator 1 Progress**: `{ann1_completed} / {total}`",
            f"- **Annotator 2 Progress**: `{ann2_completed} / {total}`",
            f"- **Dual Annotated Overlap**: `{overlap_count} / {total}`",
            "\n### Agreement Evaluation Instructions\n",
            "1. Annotator 1 completes labeling via `python scripts/annotate_golden.py --annotator annotator_1`.",
            "2. Annotator 2 independently completes labeling via `python scripts/annotate_golden.py --annotator annotator_2`.",
            "3. Run `python scripts/analyze_annotation_agreement.py` to calculate observed agreement and Cohen's Kappa.",
        ]
        with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        print(f"\nReport written to {REPORT_OUTPUT.name}")
        return {
            "status": "pending",
            "overlap_count": overlap_count,
            "ann1_completed": ann1_completed,
            "ann2_completed": ann2_completed,
        }

    # If 10+ dual annotations exist, compute Cohen's Kappa and confusion
    y1 = [item[2] for item in both_completed]
    y2 = [item[3] for item in both_completed]

    raw_matches = sum(1 for a, b in zip(y1, y2) if a == b)
    raw_agreement = raw_matches / overlap_count
    kappa = cohen_kappa_score(y1, y2, labels=INTENT_NAMES)

    disagreements = [
        {"golden_id": gid, "customer_text": text, "ann1": a1, "ann2": a2}
        for gid, text, a1, a2 in both_completed
        if a1 != a2
    ]

    print("\nAgreement Results:")
    print(f"  Raw Observed Agreement: {raw_agreement:.4f} ({raw_matches}/{overlap_count})")
    print(f"  Cohen's Kappa (κ)     : {kappa:.4f}")
    print(f"  Disagreements         : {len(disagreements)}")

    # Confusion matrix
    cm = confusion_matrix(y1, y2, labels=INTENT_NAMES)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Ann1_{x}" for x in INTENT_NAMES],
        columns=[f"Ann2_{x}" for x in INTENT_NAMES],
    )

    # Save detailed report
    md = [
        "# Golden Evaluation Set: Inter-Annotator Agreement Report\n",
        f"- **Evaluated Overlap Count**: `{overlap_count}`",
        f"- **Raw Observed Agreement**: `{raw_agreement:.4f}` ({raw_matches}/{overlap_count})",
        f"- **Cohen's Kappa (κ)**: `{kappa:.4f}`",
        f"- **Total Disagreements**: `{len(disagreements)}`\n",
        "## Annotator Confusion Matrix\n",
        cm_df.to_markdown(),
        "\n## Disagreements for Arbitration\n",
    ]
    for d in disagreements[:20]:
        md.append(f"- **{d['golden_id']}**: *\"{d['customer_text'][:100]}\"*")
        md.append(f"  - Annotator 1: `{d['ann1']}` | Annotator 2: `{d['ann2']}`")

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nAgreement report written to {REPORT_OUTPUT.name}")

    return {
        "status": "completed",
        "overlap_count": overlap_count,
        "raw_agreement": round(raw_agreement, 4),
        "cohen_kappa": round(kappa, 4),
        "disagreements_count": len(disagreements),
    }


if __name__ == "__main__":
    records = load_golden_set(GOLDEN_FILE)
    analyze_agreement(records)
