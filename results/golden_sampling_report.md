# Phase 4: Golden Evaluation Set (200 Examples) Sampling Report

> **METHODOLOGICAL INTEGRITY STATEMENT**: All human annotation fields in this evaluation set are currently initialized to `null`. Weak labels and TF-IDF predictions are included purely for sampling provenance and disagreement analysis. They will **NOT** be used as evaluation ground truth.

## 1. Executive Summary & Provenance

- **Total Golden Examples**: `200`
- **Target Count Satisfied**: Exactly 200 examples
- **Source Partition**: Held-Out Test Set ONLY (`amazonhelp_test.jsonl`)
- **Conversation-Level Sampling**: `200` unique conversations (1 example per conversation, 0 duplicates)
- **Random Seed**: `42` (completely deterministic)
- **Turn 1 Opening Inquiries**: `175` (87.5%)
- **Context-Dependent Follow-Up Inquiries**: `25` (12.5%)
- **Model vs Rule Disagreement Cases**: `76` (38.0%)

## 2. Sampling Strata Breakdown

| Sampling Stratum | Count | % of Golden Set | Purpose / Design Objective |
|---|---:|---:|---|
| `model_rule_disagreement` | 35 | 17.5% | Disagreement between weak regex rules and TF-IDF classifier (boundary cases) |
| `context_dependent_followup` | 25 | 12.5% | Multi-turn follow-ups with prior conversation context ('Here is my email', '123-456') |
| `multilingual_inquiry` | 25 | 12.5% | Non-English queries (Japanese, German, French, Spanish, Italian, Portuguese) |
| `ambiguous_other_unclear` | 20 | 10.0% | Ambiguous/unclassified opening queries ('hi need help', isolated noise) |
| `feedback_complaint_chatter` | 15 | 7.5% | Brand sentiment, rants, gratitude, and social media banter |
| `rare_intent_account_access_security` | 10 | 5.0% | Boost representation of account security and login lockouts (tail intent) |
| `core_intent_prime_digital_services` | 10 | 5.0% | Unambiguous Prime Video, Kindle, and Alexa inquiries (core intent) |
| `core_intent_delivery_status_tracking` | 10 | 5.0% | Unambiguous tracking and shipping inquiries (core high-volume intent) |
| `rare_intent_product_seller_inquiry` | 10 | 5.0% | Boost representation of rare seller/pre-purchase inquiries (tail intent) |
| `core_intent_return_refund_exchange` | 10 | 5.0% | Unambiguous returns and refund inquiries (core high-volume intent) |
| `rare_intent_order_cancellation_modification` | 10 | 5.0% | Boost representation of order cancellation and address updates (tail intent) |
| `rare_intent_damaged_defective_wrong_item` | 10 | 5.0% | Boost representation of product defect and damage reports (tail intent) |
| `core_intent_payment_billing_promotions` | 10 | 5.0% | Unambiguous payment, promo code, and gift card queries (core intent) |


## 3. Coverage by Weak Label & Model Prediction

| Intent | Weak Label Count | Weak % | Model Pred Count | Model % |
|---|---:|---:|---:|---:|
| `delivery_status_tracking` | 27 | 13.5% | 33 | 16.5% |
| `return_refund_exchange` | 20 | 10.0% | 12 | 6.0% |
| `damaged_defective_wrong_item` | 14 | 7.0% | 3 | 1.5% |
| `order_cancellation_modification` | 11 | 5.5% | 7 | 3.5% |
| `payment_billing_promotions` | 18 | 9.0% | 9 | 4.5% |
| `prime_digital_services` | 17 | 8.5% | 11 | 5.5% |
| `account_access_security` | 13 | 6.5% | 8 | 4.0% |
| `product_seller_inquiry` | 12 | 6.0% | 4 | 2.0% |
| `feedback_complaint_chatter` | 21 | 10.5% | 19 | 9.5% |
| `other_unclear` | 47 | 23.5% | 94 | 47.0% |


## 4. Multilingual & Script Diversity

| Detected Language | Count | Percentage | Representative Sample |
|---|---:|---:|---|
| `en` | 142 | 71.0% | English (US / UK / IN) |
| `ja` | 27 | 13.5% | Japanese (Kanji / Hiragana / Katakana) |
| `es` | 12 | 6.0% | Spanish (ES / MX) |
| `de` | 8 | 4.0% | German (DE / AT) |
| `fr` | 6 | 3.0% | French (FR) |
| `it` | 4 | 2.0% | Italian (IT) |
| `pt` | 1 | 0.5% | Portuguese (BR) |


## 5. Current Annotation Completion Status

- **Labeled**: `0 / 200` (0.0%)
- **Unlabeled**: `200 / 200` (100.0%)
- **Tooling Available**: `python scripts/annotate_golden.py` (CLI interactive annotator)
- **Validation Script**: `python scripts/validate_golden.py`

## 6. First 20 Golden Set Examples Preview

| Golden ID | Conv ID | Turn | Lang | Customer Message | Weak Label | Model Pred | Stratum |
|---|---:|---:|---|---|---|---|---|
| `gold_001` | `9129` | 1 | `en` | "@AmazonHelp It looks as though someone has changed..." | `account_access_security` | `account_access_security` | `rare_intent_account_access_security` |
| `gold_002` | `60254` | 1 | `en` | "@117804 Preciso de mais para me acalmar nesses dia..." | `prime_digital_services` | `prime_digital_services` | `core_intent_prime_digital_services` |
| `gold_003` | `67541` | 2 | `en` | "@AmazonHelp All been sorted. Thank you. My refund ..." | `return_refund_exchange` | `return_refund_exchange` | `context_dependent_followup` |
| `gold_004` | `77537` | 1 | `en` | "@AmazonHelp Trying to reset my password because I ..." | `account_access_security` | `account_access_security` | `rare_intent_account_access_security` |
| `gold_005` | `98812` | 6 | `en` | "@AmazonHelp  Hv u taken any action on my previous ..." | `other_unclear` | `other_unclear` | `context_dependent_followup` |
| `gold_006` | `102988` | 1 | `en` | "@AmazonHelp 2 failed delivery attempts before the ..." | `return_refund_exchange` | `delivery_status_tracking` | `model_rule_disagreement` |
| `gold_007` | `125326` | 1 | `es` | "@AmazonHelp Hola amazon, aún no tengo información ..." | `other_unclear` | `other_unclear` | `ambiguous_other_unclear` |
| `gold_008` | `133762` | 1 | `de` | "@AmazonHelp Bekomme morgen das erste Mal eine Lief..." | `delivery_status_tracking` | `other_unclear` | `multilingual_inquiry` |
| `gold_009` | `161968` | 2 | `en` | "@AmazonHelp Not sure what I can do, it told us it ..." | `delivery_status_tracking` | `delivery_status_tracking` | `context_dependent_followup` |
| `gold_010` | `162050` | 1 | `es` | "@AmazonHelp Hola. ¿Por qué un pedido que he realiz..." | `other_unclear` | `other_unclear` | `multilingual_inquiry` |
| `gold_011` | `239342` | 1 | `en` | "@116928 como es posible que compre una bandera y m..." | `other_unclear` | `other_unclear` | `ambiguous_other_unclear` |
| `gold_012` | `246422` | 20 | `en` | "@AmazonHelp But it should be not more than mrp.act..." | `other_unclear` | `other_unclear` | `context_dependent_followup` |
| `gold_013` | `248675` | 1 | `en` | "@116928 los canarios no existimos en la web? no po..." | `other_unclear` | `other_unclear` | `ambiguous_other_unclear` |
| `gold_014` | `255025` | 1 | `en` | "arrogant @115821 does not accept my review for a d..." | `delivery_status_tracking` | `delivery_status_tracking` | `core_intent_delivery_status_tracking` |
| `gold_015` | `264656` | 1 | `en` | "Bought new dell laptop from https://t.co/rxdjNIgvj..." | `product_seller_inquiry` | `other_unclear` | `rare_intent_product_seller_inquiry` |
| `gold_016` | `275834` | 1 | `en` | "@115850 Your interns at online support are useless..." | `return_refund_exchange` | `return_refund_exchange` | `core_intent_return_refund_exchange` |
| `gold_017` | `284366` | 1 | `ja` | "某Amazonでミニスーファミを予約注文したところ、配送ラベルのエラーだか何だかで、勝手に注文キャン..." | `order_cancellation_modification` | `other_unclear` | `rare_intent_order_cancellation_modification` |
| `gold_018` | `294711` | 1 | `fr` | "@AmazonHelp Bonjour on peut avoir du support en Fr..." | `other_unclear` | `other_unclear` | `multilingual_inquiry` |
| `gold_019` | `297368` | 1 | `en` | "@AmazonHelp what is the latest time you deliver? B..." | `delivery_status_tracking` | `delivery_status_tracking` | `core_intent_delivery_status_tracking` |
| `gold_020` | `297399` | 1 | `en` | "@AmazonHelp os itens que tem "Elegível Frete Gráti..." | `other_unclear` | `other_unclear` | `ambiguous_other_unclear` |

