"""
Grounded Reply Generation Script for Golden Evaluation Set.

Pipeline:
1. Loads 200 Golden Set evaluation records from data/golden/golden_evaluation_200.jsonl
   and existing predictions from outputs/golden_predictions.jsonl.
2. Fits HistoricalRetriever strictly on data/processed/amazonhelp_dev_train.jsonl (zero leakage).
3. For each example:
   - Retrieves top-3 historical AmazonHelp interaction pairs.
   - Prompts Groq (llama-3.3-70b-versatile) with grounded instructions.
   - Generates professional, Twitter-aligned customer-facing response.
4. Preserves all existing fields and appends:
   - pred_reply
   - retrieved_examples
   - reply_model
   - retrieval_method
   - retrieval_k
5. Supports --limit N for smoke-testing and atomic resume on interruption.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Set up project root and console encoding
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.retrieval import HistoricalRetriever
from src.agent import SupportAgent

GOLDEN_FILE = PROJECT_ROOT / "data/golden/golden_evaluation_200.jsonl"
DEV_TRAIN_FILE = PROJECT_ROOT / "data/processed/amazonhelp_dev_train.jsonl"
PREDICTIONS_FILE = PROJECT_ROOT / "outputs/golden_predictions.jsonl"


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load records from a jsonl file."""
    if not filepath.exists():
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl_atomic(records: List[Dict[str, Any]], filepath: Path) -> None:
    """Save records atomically via temporary file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_path.replace(filepath)


def main():
    parser = argparse.ArgumentParser(description="Generate grounded replies for Golden Set.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to generate (e.g. 3 for smoke test)")
    parser.add_argument("--k", type=int, default=3, help="Number of historical examples to retrieve (default: 3)")
    parser.add_argument("--force", action="store_true", help="Force regeneration of already generated replies")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for Groq (default: 0.2)")
    args = parser.parse_args()

    print("=" * 78)
    print("  AMAZONHELP HISTORICAL GROUNDED REPLY GENERATION")
    print("=" * 78)

    # 1. Load Golden Set and existing predictions
    print(f"\n[1/4] Loading Golden Set and existing predictions...")
    golden_records = load_jsonl(GOLDEN_FILE)
    if len(golden_records) != 200:
        raise ValueError(f"Expected exactly 200 records in {GOLDEN_FILE}, found {len(golden_records)}")

    existing_preds = load_jsonl(PREDICTIONS_FILE)
    preds_by_id = {r["golden_id"]: r for r in existing_preds}
    print(f"  ✓ Loaded {len(golden_records)} golden records and {len(existing_preds)} existing predictions.")

    # 2. Build / Fit Historical Retriever strictly on Train Split
    print(f"\n[2/4] Initializing and fitting HistoricalRetriever on training partition only...")
    t0_ret = time.time()
    retriever = HistoricalRetriever(
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    retriever.fit(DEV_TRAIN_FILE)
    ret_time = time.time() - t0_ret
    print(f"  ✓ Indexed {len(retriever.records):,} training pairs in {ret_time:.2f}s (Zero test/golden leakage).")

    # 3. Initialize Agent
    print(f"\n[3/4] Initializing SupportAgent with Groq client...")
    agent = SupportAgent(
        retriever=retriever,
        temperature=args.temperature,
    )
    print(f"  ✓ Agent initialized with model: {agent.model_name} (Temperature: {args.temperature})")

    # 4. Generate Replies
    print(f"\n[4/4] Generating replies (k={args.k}, limit={args.limit or 'ALL 200'})...")
    
    # Merge existing predictions with golden records
    merged_records = []
    for g_rec in golden_records:
        gid = g_rec["golden_id"]
        if gid in preds_by_id:
            record = preds_by_id[gid]
        else:
            record = dict(g_rec)
        # Ensure context is present
        if "conversation_context" not in record:
            record["conversation_context"] = g_rec.get("conversation_context", [])
        merged_records.append(record)

    generated_count = 0
    skipped_count = 0
    failed_count = 0

    target_records = merged_records if args.limit is None else merged_records[:args.limit]

    for idx, rec in enumerate(target_records):
        gid = rec["golden_id"]
        has_reply = bool(rec.get("pred_reply"))

        if has_reply and not args.force:
            skipped_count += 1
            continue

        print(f"  [{idx+1}/{len(target_records)}] Generating for {gid}...", end="", flush=True)

        cust_text = rec["customer_text"]
        context = rec.get("conversation_context", [])
        pred_intent = rec.get("pred_intent_tfidf_lr")
        pred_esc = rec.get("pred_escalation")
        esc_rationale = rec.get("escalation_rationale")

        # Retrieve top-k
        retrieved = retriever.retrieve(cust_text, k=args.k)

        try:
            t0_gen = time.time()
            reply, _ = agent.generate_reply(
                customer_text=cust_text,
                conversation_context=context,
                pred_intent=pred_intent,
                pred_escalation=pred_esc,
                escalation_rationale=esc_rationale,
                retrieved_examples=retrieved,
                k=args.k,
            )
            gen_time = time.time() - t0_gen

            # Update record
            rec["pred_reply"] = reply
            rec["retrieved_examples"] = retrieved
            rec["reply_model"] = agent.model_name
            rec["retrieval_method"] = "tfidf_cosine"
            rec["retrieval_k"] = args.k

            generated_count += 1
            print(f" Done ({gen_time:.2f}s) | Reply len: {len(reply)} chars")

            # Atomic save on each generation to prevent lost work
            save_jsonl_atomic(merged_records, PREDICTIONS_FILE)

            # Polite pacing for Groq rate limits
            time.sleep(0.35)

        except Exception as e:
            failed_count += 1
            print(f" FAILED: {e}")

    print("\n" + "=" * 78)
    print("  REPLY GENERATION SUMMARY")
    print("=" * 78)
    print(f"Total Target Examples : {len(target_records)}")
    print(f"Newly Generated       : {generated_count}")
    print(f"Skipped (Already Done): {skipped_count}")
    print(f"Failures              : {failed_count}")
    print(f"Predictions File      : {PREDICTIONS_FILE}")
    print("=" * 78)


if __name__ == "__main__":
    main()
