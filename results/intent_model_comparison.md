# AmazonHelp Intent Model Comparison (Validation Set)

Models fit strictly on `amazonhelp_dev_train.jsonl` (57,760 examples) and evaluated on `amazonhelp_dev_val.jsonl` (12,378 examples).

| Model Architecture | Validation Accuracy | Validation Macro-F1 | Validation Weighted-F1 | Train Time (s) |
|---|:---:|:---:|:---:|:---:|
| **Candidate D (Hybrid: Regex >= 0.85 Cascade + Word+Char LinearSVC)** (Selected) | 0.9920 | 0.9906 | 0.9920 | 3.09s |
| Candidate D2 (Hybrid: Regex >= 0.85 Cascade + Context-Aware LinearSVC) | 0.9800 | 0.9776 | 0.9798 | 2.88s |
| Candidate A3 (Word+Char TF-IDF + LinearSVC Balanced C=0.5) | 0.9512 | 0.9283 | 0.9510 | 29.37s |
| Candidate C (Context-Aware Word+Char TF-IDF + LinearSVC Balanced) | 0.9289 | 0.9012 | 0.9289 | 34.15s |
| Candidate A2 (Word+Char TF-IDF + LogisticRegression Balanced C=1.0) | 0.9208 | 0.8921 | 0.9211 | 52.12s |
| Candidate A1 (Word+Char TF-IDF + LogisticRegression C=1.0) | 0.9221 | 0.8700 | 0.9207 | 59.04s |
| Baseline (TF-IDF Word + LogisticRegression C=1.0) | 0.8581 | 0.7743 | 0.8535 | 20.71s |
| Candidate B (Dense LSA 100-dim + LogisticRegression Balanced) | 0.6048 | 0.5388 | 0.6373 | 23.14s |

**Selection Criterion**: Model selection is determined strictly by **Validation Macro-F1** on held-out validation data.
