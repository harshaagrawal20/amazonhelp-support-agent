# Hiver AI Customer Support Agent (SDE Intern Take-Home)

## Project Overview

This repository contains the engineering implementation for the Hiver SDE Intern take-home assignment: building an evaluation-first, production-grade AI customer-support agent based on real-world Twitter customer service interactions.

The primary dataset is the Kaggle Customer Support on Twitter dataset (`thoughtvector/customer-support-on-twitter`), containing ~2.81 million tweets across 108 brands.

### System Objectives (Overall Project)
1. **Intent Classification**: Classify incoming customer messages into distinct, data-grounded operational intents.
2. **Reply Drafting**: Generate replies grounded in historical brand resolution patterns.
3. **Escalation Decision**: Decide whether an incoming issue can be safely auto-handled or must be escalated to a human agent, with an explicit stated rationale.
4. **Rigorous Evaluation Harness**: Evaluate intent classification, RAG retrieval accuracy, escalation F1, and response quality using an LLM-as-a-judge validated against a hand-labelled Golden Evaluation Set (150–250 examples).

> **Current Phase Status**: Complete end-to-end implementation including Intent Classification, 200-example Golden Ground Truth evaluation, Historical RAG Reply Drafting, Independent LLM-as-a-Judge evaluation, and verified 50-example Human-vs-LLM Judge Agreement validation.

---

## Repository Structure

```
hiver-support-agent/
│
├── data/
│   ├── raw/                  # Reference pointer to KaggleHub cache (gitignored)
│   ├── processed/            # Single-brand splits & filtered conversations (upcoming)
│   └── golden/               # 150-250 hand-labelled golden evaluation set (upcoming)
│
├── scripts/
│   ├── download_data.py      # Programmatic dataset download & discovery via KaggleHub
│   └── explore_data.py       # Full dataset exploration & candidate brand benchmarking
│
├── src/                      # Core agent, intent, retrieval, and evaluation modules (upcoming)
├── tests/                    # Pytest test suite for utilities and data pipeline
├── notebooks/                # Exploratory notebooks (optional)
├── results/
│   ├── brand_summary.csv     # Quantitative comparison of candidate brands
│   └── data_exploration.md  # Comprehensive data quality & brand selection report
│
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusions (data, venv, secrets, caches)
├── pytest.ini                # Pytest configuration (safe temp dirs on Windows)
├── requirements.txt          # Python dependencies
├── DECISION_LOG.md           # Engineering decision records with rationale
├── REPORT.md                 # 6-page comprehensive project report (wip)
└── README.md                 # Project documentation & reproduction instructions
```

---

## Environment Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Kaggle account (optional; public dataset download does not require credentials)

### 1. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

---

## ⚡ Quick Reproduction Guide: Headline Results in < 15 Minutes

Reviewers can independently reproduce and verify all core claims, benchmarks, and agreement statistics in **under 15 minutes** using the included frozen evaluation artifacts:

```bash
# Step 1: Run all automated unit tests & pipeline integrity invariants (~30s)
pytest

# Step 2: Reproduce Candidate D's Frozen 200 Golden Set Benchmark (~2m)
# Validates 57.00% accuracy & 0.5971 Macro-F1 (+18.50% over baseline)
python scripts/evaluate_intent_models_golden.py

# Step 3: Reproduce Human-vs-LLM Judge Agreement Analysis (~5s)
# Validates 50/50 manual provenance, 0.700 Weighted Kappa, 76% decision agreement
python scripts/compare_human_llm_judge.py

# Step 4: Verify 14 Invariant Schema & Quality Checks on 200 Generated Replies (~5s)
python scripts/validate_golden_replies.py
```

*Note: To rebuild the full multi-turn conversation dataset from scratch using raw Kaggle tweets (optional, ~10-12 mins), follow the data pipeline steps below.*

---

## How to Download the Dataset

The raw dataset is downloaded programmatically using `kagglehub`. It caches the file locally in the standard KaggleHub cache and saves an absolute path pointer to `data/raw/dataset_location.txt`, keeping the Git repository lightweight.

Run:
```bash
python scripts/download_data.py
```

Expected output:
- `sample.csv` (16.95 KB)
- `twcs.csv` (492.58 MB, 2,811,774 rows)
- Pointer recorded at `data/raw/dataset_location.txt`

---

## How to Run the Data Exploration

To execute the two-pass dataset analysis across all 2.81M rows and reconstruct conversation graphs:
```bash
python scripts/explore_data.py
```

### Outputs Generated:
- `results/brand_summary.csv`: Computed metrics for top 15 candidate brands (usable conversation counts, average conversation length, canned response frequency, escalation phrase rates, noise percentages).
- `results/data_exploration.md`: In-depth analysis of schema, null distributions, conversation tree dynamics, and brand selection rationale.

---

## Running Tests

Run the test suite via `pytest`:
```bash
pytest
```

---

## Selected Brand: AmazonHelp

Based on empirical analysis of all 108 brands in `twcs.csv`, **`AmazonHelp`** was chosen as the operational target:
- **82,534 complete multi-turn conversation trees** (highest in the dataset).
- **Average conversation length of 4.49 tweets**, exhibiting rich back-and-forth diagnostic and resolution context.
- **Natural, distinct retail intents**: Delivery Delays, Shipping Tracking, Refunds/Returns, Defective Items, Cancellations, Prime Video, Account Access.
- **Realistic escalation boundary**: Clear operational distinction between public advice (order policies, help links) and escalation requiring private order lookups (DMs).

---

## AmazonHelp Dataset Processing Pipeline

The dataset preparation pipeline (`scripts/build_amazonhelp_dataset.py`) converts raw, fragmented tweets into structured, chronologically sorted, leak-free conversation datasets.

### Running the Dataset Build Pipeline
```bash
python scripts/build_amazonhelp_dataset.py
```
*(Executes in under 60 seconds)*

### Key Engineering Steps:
1. **Conversation Graph Traversal**: Recursively traces `in_response_to_tweet_id` to establish a stable `root_id` for every conversation tree using path compression.
2. **Chronological Message Ordering**: Converts raw Twitter timestamps to UTC datetimes to order messages chronologically, preserving turn causality.
3. **Quality & Usability Filtering**: Validates that each conversation contains at least one customer query, at least one AmazonHelp reply, and a direct response link, while discarding anomalous viral threads (>100 tweets). 82,515 conversations (99.98%) qualify as usable.
4. **Interaction Pair Extraction**: Extracts 167,699 direct customer $\rightarrow$ AmazonHelp interaction pairs with accumulated conversational history.
5. **Leakage-Safe Partitioning**:
   - Assigns splits strictly at the conversation `root_id` level using seed `42`:
     - **Train (70%)**: 57,760 conversations (117,077 interaction pairs)
     - **Validation (15%)**: 12,377 conversations (25,203 interaction pairs)
     - **Test (15%)**: 12,378 conversations (25,419 interaction pairs)
   - Zero root or tweet ID overlap is enforced with an automated verifier.

### Processed Artifacts Generated:
- `data/processed/amazonhelp_conversations.jsonl` (All 82,534 reconstructed conversations)
- `data/processed/amazonhelp_train.jsonl` (57,760 training conversations)
- `data/processed/amazonhelp_val.jsonl` (12,377 validation conversations)
- `data/processed/amazonhelp_test.jsonl` (12,378 test conversations)
- `results/amazonhelp_intent_exploration.csv` (Empirical problem theme distribution)
- `results/amazonhelp_conversation_analysis.md` (Detailed quantitative report with real conversation examples)

> **Note**: All `data/processed/` JSONL files are excluded from git due to their size (9–350 MB each). Regenerate them by running `python scripts/build_amazonhelp_dataset.py` followed by `python scripts/benchmark_intents.py`.

---

## Phase 3: Intent Classification Benchmark

In Phase 3, we established the initial intent classification benchmark answering:
> *"How well can we classify AmazonHelp customer problems using simple, transparent methods?"*

### 1. Intent Taxonomy (10 Distinguishable Categories)
Empirical analysis of customer inquiries resolved the previous ~50.8% broad `general_inquiry_other` bucket into actionable operational categories:
1. `delivery_status_tracking` (29.50%): Logistics, shipment tracking, delays, carrier transit issues.
2. `return_refund_exchange` (6.48%): Return labels, refund processing, product exchanges.
3. `damaged_defective_wrong_item` (3.23%): Smashed packages, defective items, missing parts, counterfeit.
4. `order_cancellation_modification` (3.17%): Canceling accidental orders, address updates before dispatch.
5. `payment_billing_promotions` (6.10%): Double charges, payment failures, gift cards, promo codes.
6. `prime_digital_services` (6.55%): Prime Video streaming, Kindle ebooks, Fire TV, Echo/Alexa devices.
7. `account_access_security` (1.36%): Login/password reset, OTP/2FA, suspended/compromised accounts.
8. `product_seller_inquiry` (0.92%): Stock availability, seller questions, pre-purchase inquiries.
9. `feedback_complaint_chatter` (7.60%): Brand rants, social chatter, cat box photos, app quiz contests.
10. `other_unclear` (35.10%): Context-dependent follow-up fragments ('__email__', '123-456'), ambiguous greetings.

### 2. Development Labeling Strategy
Deterministic priority regex rules in `src/intents.py` provide consistent, interpretable development labels.
> **Note**: These are explicitly documented as **weak development labels** to establish baseline feasibility, debug modeling pipelines, and uncover failure modes. They will be complemented by a hand-labeled Golden Evaluation Set in subsequent phases.

### 3. Baseline Performance Comparison

| Model | Val Accuracy | Val Macro F1 | Val Weighted F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Majority Class Baseline** | 0.3522 | 0.0521 | 0.1835 | 0.3475 | 0.0516 | 0.1792 |
| **TF-IDF + Logistic Regression** | 0.8581 | 0.7743 | 0.8535 | 0.8612 | 0.7860 | 0.8573 |

*Both models fit strictly on the training partition (57,760 examples).*

### 4. Running the Intent Benchmark
```bash
python scripts/benchmark_intents.py
```

### Artifacts Generated:
- `data/processed/amazonhelp_dev_train.jsonl` (57,760 labeled Turn-1 interaction records with preserved context)
- `data/processed/amazonhelp_dev_val.jsonl` (12,377 labeled validation interaction records)
- `data/processed/amazonhelp_dev_test.jsonl` (12,378 labeled test interaction records)
- `results/intent_baseline_results.json` (Full metrics, per-class reports, error samples)
- `results/intent_baseline_report.md` (Formatted evaluation and error analysis report)
- `results/intent_confusion_matrix.csv` (10x10 confusion matrix)

> **Note**: The `data/processed/` JSONL files are excluded from git (see `.gitignore`). They are deterministically reproducible by running `python scripts/build_amazonhelp_dataset.py` then `python scripts/benchmark_intents.py` with `random_state=42`.

---

## Phase 4: Golden Evaluation Set (200 Examples)

> **CRITICAL EVALUATION DISTINCTION**:
> Phase 3 metrics (86.12% accuracy) are **weak-label development benchmarks** reflecting rule alignment. The **200-example Golden Evaluation Set** is the authoritative, human-grounded source of truth for all final evaluations.

### 1. Sampling & Stratification Methodology
The 200 Golden examples were sampled deterministically (`seed=42`) **strictly from held-out test conversations (`amazonhelp_test.jsonl`)**, enforcing exactly 1 example per conversation (200 unique `conversation_id`s, 0 duplicates, 0 leakage from train/val):
- **Model vs Weak-Rule Disagreements (35 examples / 17.5%)**: Stress-tests high-ambiguity boundary cases where TF-IDF and regex rules disagree.
- **Rare Operational Intents (40 examples / 20.0%)**: Guarantees strong representation of tail intents (`product_seller_inquiry`, `account_access_security`, `order_cancellation_modification`, `damaged_defective_wrong_item`).
- **Multilingual Inquiries (25 examples / 12.5%)**: Covers Japanese, German, French, Spanish, Italian, and Portuguese tweets.
- **Core High-Volume Intents (40 examples / 20.0%)**: Unambiguous tracking, return, billing, and prime queries.
- **Social Chatter & Complaints (15 examples / 7.5%)**: Brand sentiment, praise, and social banter.
- **Ambiguous Opening Queries (20 examples / 10.0%)**: Generic greetings and unclassified queries.
- **Context-Dependent Follow-Up Turns (25 examples / 12.5%)**: Multi-turn interactions (Turn $\ge 2$) with full conversation history for context-ablation experiments.

### 2. Annotation Workflow & Tooling
- **Annotation Guide**: [`data/golden/ANNOTATION_GUIDE.md`](file:///d:/hiver/data/golden/ANNOTATION_GUIDE.md) provides comprehensive operational definitions, tie-breaking rules, and multi-turn context protocols.
- **Interactive CLI Annotator**:
  ```bash
  # Annotator 1
  python scripts/annotate_golden.py --annotator annotator_1

  # Annotator 2 (for inter-annotator agreement)
  python scripts/annotate_golden.py --annotator annotator_2
  ```
- **Integrity Validation**:
  ```bash
  python scripts/validate_golden.py
  ```
  *(Validates 200 count, ID uniqueness, test-only provenance, zero leakage, valid taxonomy keys, and null anti-cheating check).*
- **Inter-Annotator Agreement Evaluation**:
  ```bash
  python scripts/analyze_annotation_agreement.py
  ```
  *(Calculates raw agreement and Cohen's Kappa once dual passes are completed).*

### Artifacts Generated:
- `data/golden/golden_evaluation_200.jsonl` (Authoritative 200-example Golden Set with preserved context)
- `data/golden/ANNOTATION_GUIDE.md` (Operational annotation guide with tie-breaking rules)
- `results/golden_sampling_report.md` (Comprehensive sampling and composition report)
- `scripts/annotate_golden.py` (CLI interactive annotation tool)
- `scripts/validate_golden.py` (Quality assurance and leakage verifier)
- `scripts/analyze_annotation_agreement.py` (Cohen's Kappa agreement calculator)

---

## The Evaluation Hierarchy

To avoid conflating synthetic development checks with true operational quality, our evaluation harness is structured into five distinct tiers:

1. **Weak-Label Baseline (Tier 1)**: Regex-derived intent classification across 12,378 held-out dev test tweets.
2. **Human-Labeled Golden Set Evaluation (Tier 2)**: Authoritative 200-case hand-labeled test set (`golden_evaluation_200.jsonl`) measuring true human-level accuracy.
3. **Reply Generation with Historical Retrieval (Tier 3)**: Historical retrieval indexed over 57,760 training pairs from `amazonhelp_dev_train.jsonl`, validated against 14 schema and quality invariants.
4. **Independent LLM-as-a-Judge Evaluation (Tier 4)**: 200 Golden replies evaluated across six rubrics by an independent model (`openai/gpt-oss-120b`).
5. **Human-vs-LLM Judge Agreement Validation (Tier 5)**: Double-blind manual human evaluation of a 50-example stratified subset auditing judge calibration.

---

## Empirical Classification Benchmark: Weak Labels vs. Human Ground Truth

### Comparison Results

| Evaluation Tier | Dataset | Accuracy | Macro F1 | Weighted F1 |
|---|---|:---:|:---:|:---:|
| **Weak-Label Baseline** | 12,378 held-out dev test tweets | **86.12%** | **0.7860** | **0.8573** |
| **Human Golden Set Baseline** | 200 hand-labeled Golden cases | **38.50%** | **0.4034** | **0.4278** |
| *Majority Class Baseline* | 200 Golden cases | 5.00% | 0.0095 | 0.0048 |

> [!IMPORTANT]
> **Why the 86.12% Result Must Not Be Presented as True Human-Level Accuracy:**
> The 86.12% test accuracy achieved by the TF-IDF baseline reflects alignment with deterministic regex rules. It is a benchmark against **weak pseudo-labels**, not human truth.
>
> On the human-labeled Golden Set—featuring high-ambiguity boundary cases, multi-turn contexts, and non-English inquiries—accuracy is **38.50%**. Real-world customer queries involve sarcasm, mixed intents, and informal phrasing that simple n-gram models cannot resolve.

### Iterative Model Improvement & Selection (Validation Benchmark)

To advance beyond the 38.50% baseline without overfitting to the test set, eight model architectures were benchmarked strictly on held-out development validation data (`amazonhelp_dev_val.jsonl`, 12,377 interactions) with selection determined **strictly by Validation Macro-F1**:

| Model Architecture | Val Accuracy | Val Macro-F1 | Val Weighted-F1 | Status |
|---|:---:|:---:|:---:|---|
| **Candidate D (Hybrid: Regex >= 0.85 Cascade + Word+Char LinearSVC)** | **0.9920** | **0.9906** | **0.9920** | **Selected Overall Architecture** |
| Candidate D2 (Hybrid: Regex >= 0.85 Cascade + Context-Aware LinearSVC) | 0.9800 | 0.9776 | 0.9798 | Candidate |
| **Candidate A3 (Word+Char TF-IDF + LinearSVC Balanced C=0.5)** | **0.9512** | **0.9283** | **0.9510** | **Selected Pure ML Architecture** |
| Candidate C (Context-Aware Word+Char TF-IDF + LinearSVC Balanced) | 0.9289 | 0.9012 | 0.9289 | Candidate |
| Candidate A1 (Word+Char TF-IDF + LogisticRegression C=1.0) | 0.9221 | 0.8700 | 0.9207 | Candidate |
| Candidate A2 (Word+Char TF-IDF + LogisticRegression Balanced C=1.0) | 0.9208 | 0.8921 | 0.9211 | Candidate |
| Baseline (TF-IDF Word (1,2) + LogisticRegression C=1.0) | 0.8581 | 0.7743 | 0.8535 | Baseline Floor |
| Candidate B (Dense LSA 100-dim + LogisticRegression Balanced) | 0.6048 | 0.5388 | 0.6373 | Candidate |

> [!IMPORTANT]
> **Weak-Label Ceiling Effect for Hybrid Models**: Candidate D's validation accuracy (99.20%) is elevated because the validation labels were generated by the same deterministic regex rules that power Stage 1 of the hybrid classifier. When the hybrid's regex cascade fires (conf ≥ 0.85), it will always agree with the regex-derived validation label — inflating agreement on the development set. This is expected and documented. The authoritative performance measure is the **frozen 200-example Human Golden Set**, where Candidate D achieves **57.00% accuracy and 0.5971 Macro-F1** — real human-annotated ground truth.

### Final Performance on FROZEN 200 Human Golden Set

Evaluating the selected frozen architectures once on `data/golden/golden_evaluation_200.jsonl`:

| Model Architecture | Golden Accuracy | Golden Macro-F1 | Golden Weighted-F1 | Delta vs Baseline |
|---|:---:|:---:|:---:|:---:|
| Baseline (TF-IDF Word + LogisticRegression C=1.0) | 38.50% | 0.4034 | 0.4278 | Baseline Floor |
| Candidate A1 (Word+Char TF-IDF + LogisticRegression C=1.0) | 50.00% | 0.5148 | 0.5293 | +11.50% Acc, +0.1114 Macro-F1 |
| Candidate A2 (Word+Char TF-IDF + LogisticRegression Balanced C=1.0) | 52.50% | 0.5489 | 0.5444 | +14.00% Acc, +0.1455 Macro-F1 |
| **Candidate A3 (Word+Char TF-IDF + LinearSVC Balanced C=0.5)** | **54.00%** | **0.5623** | **0.5612** | **+15.50% Acc, +0.1589 Macro-F1** |
| **Candidate D (Hybrid: Regex >= 0.85 Cascade + Word+Char LinearSVC)** | **57.00%** | **0.5971** | **0.5915** | **+18.50% Acc, +0.1937 Macro-F1** |

To reproduce the multi-model validation benchmark and frozen Golden evaluation:
```bash
# 1. Multi-model candidate validation selection
python scripts/benchmark_intent_models.py

# 2. Final frozen Golden Set evaluation
python scripts/evaluate_intent_models_golden.py
```

---

## Phase 5: Historical Retrieval & Reply Drafting (RAG)

- **Retrieval Engine**: Historical retrieval was indexed over 57,760 training pairs from `amazonhelp_dev_train.jsonl` (representing Turn-1 customer-support resolutions from the 57,760 training conversation trees, which encompass 117,077 total interaction pairs in the broader training split `amazonhelp_train.jsonl`), using a leakage-free TF-IDF retriever combining lexical similarity with historical resolution templates.
- **Automated Validation**: All 200 generated replies in `outputs/golden_predictions.jsonl` pass 14/14 automated validation checks (`scripts/validate_golden_replies.py`), guaranteeing zero empty outputs, zero unrendered template tokens, exact ID matching, and authentic multilingual handling.

---

## Phase 6: Independent LLM-as-a-Judge Evaluation (200 Golden Set)

All 200 generated replies were scored by an independent judge (`openai/gpt-oss-120b` via Groq at `temperature=0.0`) across six rubrics:

| Metric | Value (200 Golden Set) |
|---|:---:|
| **Mean Overall Score** | **4.746 / 5.0** (Median: 5.0) |
| **PASS Rate** | **176 / 200 (88.0%)** |
| **BORDERLINE Rate** | **22 / 200 (11.0%)** |
| **FAIL Rate** | **2 / 200 (1.0%)** |

---

## Phase 7: Human-vs-LLM Judge Agreement Validation (50-Example Stratified Subset)

To audit and validate the reliability of the LLM judge, a representative 50-example stratified subset was selected (`random_seed=42`) spanning all 10 intents, all 3 escalation labels, both failure cases, 12 borderline cases, and 6 Japanese no-evidence cases.

### Annotation Protocol & Provenance
- **Double-Blind Independence**: Human evaluation was performed independently. Annotators did not use or have access to the LLM judge's scores, category rationales, or decision labels while scoring.
- **Verified Manual Provenance**: All 50 records in `data/golden/human_reply_annotations.jsonl` were manually reviewed and scored via CLI (`scripts/annotate_human_replies.py`). No synthetic or copied scores were used.

### Verified Agreement Metrics (50-Example Human-Reviewed Subset)

| Agreement Metric | Result | Interpretation |
|---|:---:|---|
| **Completed Annotations** | **50 / 50 (100.0%)** | Verified manual provenance |
| **Overall Score Exact Agreement** | **72.0%** | Exact score match within rounding |
| **Weighted Cohen's $\kappa$** | **0.700** | **Substantial agreement on the ordinal overall-quality scale** |
| **Spearman Rank Correlation ($r_s$)** | **0.594** | Positive monotonic ranking alignment |
| **Mean Absolute Error (MAE)** | **0.294** | Average deviation is $< 0.3$ points on a 5-point scale |
| **Decision Exact Agreement** | **76.0%** | Exact match on `pass`, `borderline`, or `fail` |
| **Unsupported Claims Agreement** | **100.0%** ($\kappa = 1.000$) | Perfect mutual consensus on hallucination detection |
| **Human Mean Score** | **4.583 / 5.0** | Independent human ratings |
| **LLM Judge Mean Score** | **4.537 / 5.0** | Calibrated with slight conservative tendency |

### Decision Confusion Matrix

```
                      LLM Judge Prediction
                   PASS    BORDERLINE    FAIL
Human    PASS        34        10          0
Actual   BORDERLINE   2         2          0
         FAIL         0         0          2
```

### Consensus on Critical Failure Cases
On the 50-example human-reviewed subset, the LLM judge and human evaluator identified the same two failures, with no disagreement on failure detection:
- **`gold_018` (French)**: The response incorrectly claimed that a DM had already been sent (*"Bonjour, nous vous avons répondu par DM. ^ARC"*). Both human and LLM gave scores of $1.0$ for Unsupported Claims and issued a consensus **FAIL**.
- **`gold_094` (German)**: The response contradicted Amazon billing policy by claiming advance-charge configuration was possible (*"Du kannst die Zahlungsmethode in deinem Konto so einstellen, dass die Bestellung vorab belastet wird..."*). Both human and LLM flagged this major policy hallucination and issued a consensus **FAIL**.

> [!NOTE]
> **Scope Boundary**: These findings are derived strictly from the **50-example human-reviewed subset** and must not be described as universal or population-level validation. Within this representative subset, they demonstrate that the LLM judge is calibrated and achieves 100% agreement on critical failure detection.




