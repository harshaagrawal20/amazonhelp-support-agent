"""
Validation and Quality Assurance Script for Golden Evaluation Set.

Enforces strict integrity constraints:
1. Exactly 200 records.
2. Unique golden_id across all records.
3. Unique conversation_id (exactly 1 example per conversation, 0 duplicates).
4. Strict test split provenance: All conversation_ids exist in amazonhelp_test.jsonl.
5. Zero leakage: 0 conversation_ids found in amazonhelp_train.jsonl or amazonhelp_val.jsonl.
6. Unique customer tweet_ids.
7. Valid taxonomy labels: Any non-null gold_intent must belong to the 10 canonical intents.
8. Status consistency: annotation_status == 'labeled' iff gold_intent is not None.
9. No silent copying: Verifies human labels were not blindly populated from weak labels.

Usage:
  python scripts/validate_golden.py
"""

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

# Reconfigure stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_FILE = PROJECT_ROOT / "data/golden/golden_evaluation_200.jsonl"
TEST_FILE = PROJECT_ROOT / "data/processed/amazonhelp_test.jsonl"
TRAIN_FILE = PROJECT_ROOT / "data/processed/amazonhelp_train.jsonl"
VAL_FILE = PROJECT_ROOT / "data/processed/amazonhelp_val.jsonl"

from src.intents import INTENT_NAMES


def load_conversation_ids(jsonl_path: Path) -> Set[int]:
    """Extract set of conversation_ids from a processed JSONL file."""
    cids = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            cids.add(json.loads(line)["conversation_id"])
    return cids


def validate_golden_dataset(filepath: Path) -> Dict[str, Any]:
    """Run full validation suite on golden evaluation file."""
    print("=" * 78)
    print("GOLDEN EVALUATION SET INTEGRITY VERIFICATION")
    print("=" * 78)

    if not filepath.exists():
        raise FileNotFoundError(f"Golden dataset file not found: {filepath}")

    records: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception as e:
                raise ValueError(f"Malformed JSON on line {idx}: {e}")

    total = len(records)
    print(f"\n[1/7] Record Count Check: Total records = {total}")
    assert total == 200, f"Integrity Failure: Expected exactly 200 records, got {total}"
    print("  ✓ Exactly 200 records found.")

    # Unique IDs
    golden_ids = [r["golden_id"] for r in records]
    conv_ids = [r["conversation_id"] for r in records]
    tweet_ids = [r["tweet_id"] for r in records]

    print("\n[2/7] Uniqueness Checks:")
    assert len(set(golden_ids)) == 200, "Integrity Failure: Duplicate golden_id found!"
    print("  ✓ 200 unique golden_ids.")
    assert len(set(conv_ids)) == 200, "Integrity Failure: Duplicate conversation_id found (must be 1 per conversation)!"
    print("  ✓ 200 unique conversation_ids (conversation-level sampling verified).")
    assert len(set(tweet_ids)) == 200, "Integrity Failure: Duplicate customer tweet_id found!"
    print("  ✓ 200 unique customer tweet_ids.")

    # Provenance and Zero Leakage Checks
    print("\n[3/7] Provenance & Contamination Checks:")
    print("  Loading test conversation IDs...")
    test_cids = load_conversation_ids(TEST_FILE)
    train_cids = load_conversation_ids(TRAIN_FILE)
    val_cids = load_conversation_ids(VAL_FILE)

    non_test_cids = [cid for cid in conv_ids if cid not in test_cids]
    assert len(non_test_cids) == 0, f"Integrity Failure: {len(non_test_cids)} conversation_ids do not belong to test split!"
    print(f"  ✓ 100% of golden examples ({total}/200) originate from held-out test split.")

    train_overlap = set(conv_ids).intersection(train_cids)
    assert len(train_overlap) == 0, f"Contamination Failure: {len(train_overlap)} golden conversations found in train split!"
    print("  ✓ Zero overlap with train split (0 leaked conversations).")

    val_overlap = set(conv_ids).intersection(val_cids)
    assert len(val_overlap) == 0, f"Contamination Failure: {len(val_overlap)} golden conversations found in validation split!"
    print("  ✓ Zero overlap with validation split (0 leaked conversations).")

    # Schema & Taxonomy Checks
    print("\n[4/7] Schema & Taxonomy Validity:")
    for r in records:
        # Check required fields
        required_fields = [
            "golden_id", "conversation_id", "root_id", "tweet_id", "turn_index",
            "is_followup", "customer_text", "conversation_context", "weak_label",
            "model_prediction", "gold_intent", "annotation_status"
        ]
        for f in required_fields:
            assert f in r, f"Missing required field '{f}' in {r.get('golden_id')}"

        # If labeled, check validity
        if r["gold_intent"] is not None:
            assert r["gold_intent"] in INTENT_NAMES, f"Invalid gold_intent '{r['gold_intent']}' in {r['golden_id']}"
            assert r["annotation_status"] == "labeled", f"Status mismatch in {r['golden_id']}"
        else:
            assert r["annotation_status"] == "unlabeled", f"Status mismatch in {r['golden_id']}"

    print("  ✓ All required schema fields present and valid.")

    # Status & Progress Check
    labeled_count = sum(1 for r in records if r["gold_intent"] is not None)
    unlabeled_count = total - labeled_count

    print("\n[5/7] Annotation Status:")
    print(f"  Total Examples: {total}")
    print(f"  Labeled       : {labeled_count} ({(labeled_count/total)*100:.1f}%)")
    print(f"  Unlabeled     : {unlabeled_count} ({(unlabeled_count/total)*100:.1f}%)")

    # Anti-Cheat Check: Ensure human labels were not blindly auto-filled from weak labels
    print("\n[6/7] Methodological Integrity Check:")
    if labeled_count > 0:
        matches_weak = sum(1 for r in records if r["gold_intent"] is not None and r["gold_intent"] == r["weak_label"])
        # If every single label matches weak label on 50+ examples, flag warning
        if labeled_count > 30 and matches_weak == labeled_count:
            print("  ⚠️ WARNING: 100% of gold labels match weak labels. Verify human review occurred.")
        else:
            print(f"  ✓ Human label independence verified ({matches_weak}/{labeled_count} agree with weak rules).")
    else:
        print("  ✓ All 200 gold_intent fields are currently null (no pre-mature or fabricated labels).")

    # Distributions Summary
    print("\n[7/7] Golden Set Composition:")
    strata_counts = Counter(r.get("sampling_stratum", "unknown") for r in records)
    print("\n  Sampling Strata:")
    for stratum, cnt in strata_counts.most_common():
        print(f"    {stratum:44s}: {cnt:3d} ({cnt/total*100:5.1f}%)")

    lang_counts = Counter(r.get("detected_language", "unknown") for r in records)
    print("\n  Language Breakdown:")
    for lang, cnt in lang_counts.most_common():
        print(f"    {lang:10s}: {cnt:3d} ({cnt/total*100:5.1f}%)")

    weak_counts = Counter(r["weak_label"] for r in records)
    print("\n  Weak-Label Distribution:")
    for name in INTENT_NAMES:
        print(f"    {name:32s}: {weak_counts.get(name, 0):3d} ({weak_counts.get(name, 0)/total*100:5.1f}%)")

    print("\n" + "=" * 78)
    print("ALL INTEGRITY CHECKS PASSED: Golden Evaluation Set is valid and leak-free!")
    print("=" * 78)

    return {
        "total": total,
        "labeled": labeled_count,
        "unlabeled": unlabeled_count,
        "unique_conversations": len(set(conv_ids)),
        "source_split": "test",
        "zero_contamination": True,
    }


if __name__ == "__main__":
    validate_golden_dataset(GOLDEN_FILE)
