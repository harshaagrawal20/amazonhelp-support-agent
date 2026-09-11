# Human-vs-LLM Judge Agreement Evaluation Report

## Executive Summary

- **Evaluation Subset Size**: **50 examples** sampled from the 200 Golden Evaluation Set.
- **Sampling Protocol**: Deterministic stratified sampling (`random_seed=42`) covering all 10 intents, 3 escalation classes, both judge FAIL cases, 12 borderline cases, and 6 Japanese no-evidence cases.
- **Blinding**: Human evaluator scored all examples double-blinded (no access to LLM judge scores or rationales).
- **Overall Score Agreement**: Exact: **72.0%**, Weighted Cohen's $\kappa$: **0.7**, Spearman $r_s$: **0.594**, MAE: **0.294**.
- **Decision Agreement (PASS / BORDERLINE / FAIL)**: Exact: **76.0%**, Categorical Cohen's $\kappa$: **0.306**.
- **Mean Scores**: Human Average: **4.583 / 5.0** vs. LLM Judge Average: **4.537 / 5.0**.

---

## 1. Six-Dimension Agreement Breakdown

| Dimension | Exact Agree % | Weighted $\kappa$ | Spearman $r_s$ | MAE | Human Mean | LLM Mean | Agreement Strength |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Correctness** | 70.0% | 0.615 | 0.532 | 0.460 | 4.50 | 4.28 | Strong |
| **Relevance** | 78.0% | 0.330 | 0.363 | 0.380 | 4.78 | 4.60 | Weak/Fair |
| **Historical Grounding** | 58.0% | 0.474 | 0.346 | 0.440 | 4.46 | 4.78 | Moderate |
| **Helpfulness** | 64.0% | 0.562 | 0.487 | 0.460 | 4.36 | 4.38 | Moderate |
| **Unsupported Claims** | 100.0% | 1.000 | 1.000 | 0.000 | 4.78 | 4.78 | Strong |
| **Escalation Appropriateness** | 70.0% | 0.501 | 0.722 | 0.460 | 4.62 | 4.40 | Moderate |

---

## 2. Decision Confusion Matrix

**Exact Decision Agreement**: **76.0%** (Categorical Cohen's $\kappa = 0.306$)

| Human \ LLM Judge | Pred: PASS | Pred: BORDERLINE | Pred: FAIL |
|---|:---:|:---:|:---:|
| **Actual: PASS** | 34 | 10 | 0 |
| **Actual: BORDERLINE** | 2 | 2 | 0 |
| **Actual: FAIL** | 0 | 0 | 2 |

---

## 3. Subgroup Agreement Analysis

### A. By Gold Escalation
| Escalation Class | Count | MAE | Human Mean | LLM Judge Mean |
|---|:---:|:---:|:---:|:---:|
| `auto_handle` | 26 | 0.320 | 4.51 | 4.47 |
| `escalate` | 21 | 0.240 | 4.69 | 4.58 |
| `unclear` | 3 | 0.447 | 4.50 | 4.83 |

### B. By Retrieval Evidence Presence
| Group | Count | MAE | Human Mean | LLM Judge Mean |
|---|:---:|:---:|:---:|:---:|
| With Historical Evidence | 44 | 0.319 | 4.54 | 4.47 |
| Without Evidence (Japanese) | 6 | 0.112 | 4.89 | 5.00 |

---

## 4. Disagreement Patterns & Limitations

- **Major Disagreements Count**: 12 instances where score differed by $\ge 1.0$ or decisions diverged.
- **Primary Pattern**: Evaluators differed mostly on borderline cases (e.g., whether offering generic DM routing without a direct help link constitutes a PASS vs. BORDERLINE response).
- **Sample Limitation**: A 50-example subset offers a fast, representative check of judge calibration, but statistical power is limited compared to the full 200 Golden Set.
