# Intent Classification Evaluation on FROZEN 200 Human Golden Set

Evaluated on 200 stratified, human-annotated ground-truth conversations from `data/golden/golden_evaluation_200.jsonl`.
All models trained strictly on `amazonhelp_dev_train.jsonl` (57,760 examples).

| Model Architecture | Golden Accuracy | Golden Macro-F1 | Golden Weighted-F1 | Delta Macro-F1 vs Baseline |
|---|:---:|:---:|:---:|:---:|
| Baseline (TF-IDF Word + LogisticRegression C=1.0) | 0.3850 | 0.4034 | 0.4278 | 0.0000 |
| Candidate A1 (Word+Char TF-IDF + LogisticRegression C=1.0) | 0.5000 | 0.5148 | 0.5293 | +0.1114 |
| Candidate A2 (Word+Char TF-IDF + LogisticRegression Balanced C=1.0) | 0.5250 | 0.5489 | 0.5444 | +0.1455 |
| Candidate A3 (Word+Char TF-IDF + LinearSVC Balanced C=0.5) | 0.5400 | 0.5623 | 0.5612 | +0.1589 |
| Candidate D (Hybrid: Regex >= 0.85 Cascade + Word+Char LinearSVC) | 0.5700 | 0.5971 | 0.5915 | +0.1937 |

## Per-Intent F1 Comparison (Human Golden Ground Truth)

| Intent Name | Golden Support | Baseline F1 | Candidate A3 F1 (LinearSVC) | Candidate D F1 (Hybrid) |
|---|:---:|:---:|:---:|:---:|
| `delivery_status_tracking` | 46 | 0.5570 | 0.5676 | 0.5946 |
| `return_refund_exchange` | 19 | 0.5161 | 0.6829 | 0.6667 |
| `damaged_defective_wrong_item` | 15 | 0.1111 | 0.3077 | 0.4138 |
| `order_cancellation_modification` | 9 | 0.6250 | 0.9000 | 0.9000 |
| `payment_billing_promotions` | 23 | 0.3750 | 0.5854 | 0.5854 |
| `prime_digital_services` | 21 | 0.3125 | 0.6316 | 0.6842 |
| `account_access_security` | 14 | 0.6364 | 0.7407 | 0.7407 |
| `product_seller_inquiry` | 10 | 0.2857 | 0.4211 | 0.5455 |
| `feedback_complaint_chatter` | 33 | 0.4231 | 0.4906 | 0.5185 |
| `other_unclear` | 10 | 0.1923 | 0.2951 | 0.3214 |
