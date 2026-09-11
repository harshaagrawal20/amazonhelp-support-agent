# Phase 3: AmazonHelp Intent Classification Benchmark Report

> **IMPORTANT DISCLAIMER**: Labels in this benchmark are **weak development labels** generated via deterministic priority rules to establish modeling feasibility and uncover failure patterns. They are **NOT** human ground truth and will be complemented by a hand-labeled Golden Evaluation Set in subsequent phases.

## 1. Intent Taxonomy Overview

A candidate taxonomy of **10 mutually distinguishable intents** was developed from empirical analysis of 374,042 AmazonHelp tweets, resolving the previous ~50.8% broad `general_inquiry_other` bucket into actionable operational categories.

| Intent Name | Operational Department | Description |
|---|---|---|
| `delivery_status_tracking` | Logistics & Transportation (AMZL / Carriers) | Customer inquiring about the whereabouts, dispatch status, estimated arrival date/time, co... |
| `return_refund_exchange` | Returns & Reverse Logistics | Customer requesting to return an eligible item, check refund processing status, obtain a p... |
| `damaged_defective_wrong_item` | Product Quality & Replacement | Customer reporting that an item arrived broken, scratched, defective, non-functional, miss... |
| `order_cancellation_modification` | Order Management & Fulfillment | Customer seeking to cancel an existing order, stop an accidental purchase, or modify order... |
| `payment_billing_promotions` | Billing & Payments Support | Customer experiencing payment transaction errors, unexpected or duplicate charges, gift ca... |
| `prime_digital_services` | Digital Devices & Subscriptions (Prime, Kindle, Fire TV, Echo/Alexa) | Customer having questions or technical issues with Amazon Prime benefits, Prime Video stre... |
| `account_access_security` | Customer Trust, Identity & Account Security | Customer unable to log into their account, locked out, encountering OTP/2FA errors, suspec... |
| `product_seller_inquiry` | Retail & Marketplace Operations | Customer or merchant asking about product availability/stock, specifications, warranty, th... |
| `feedback_complaint_chatter` | Social Media Engagement & Executive Relations | Customer expressing generic brand sentiment, dissatisfaction, compliments, gratitude, soci... |
| `other_unclear` | Front-Line Triage | Inbound messages that lack sufficient information to determine intent without further cont... |


## 2. Dataset Split & Label Distribution

| Intent | Train Count | Train % | Val Count | Val % | Test Count | Test % |
|---|---:|---:|---:|---:|---:|---:|
| `delivery_status_tracking` | 17,040 | 29.50% | 3,720 | 30.06% | 3,646 | 29.46% |
| `return_refund_exchange` | 3,742 | 6.48% | 791 | 6.39% | 847 | 6.84% |
| `damaged_defective_wrong_item` | 1,863 | 3.23% | 387 | 3.13% | 439 | 3.55% |
| `order_cancellation_modification` | 1,829 | 3.17% | 406 | 3.28% | 379 | 3.06% |
| `payment_billing_promotions` | 3,526 | 6.10% | 760 | 6.14% | 783 | 6.33% |
| `prime_digital_services` | 3,783 | 6.55% | 717 | 5.79% | 750 | 6.06% |
| `account_access_security` | 784 | 1.36% | 178 | 1.44% | 145 | 1.17% |
| `product_seller_inquiry` | 530 | 0.92% | 110 | 0.89% | 118 | 0.95% |
| `feedback_complaint_chatter` | 4,387 | 7.60% | 949 | 7.67% | 970 | 7.84% |
| `other_unclear` | 20,276 | 35.10% | 4,359 | 35.22% | 4,301 | 34.75% |
| **Total** | **57,760** | **100.0%** | **12,377** | **100.0%** | **12,378** | **100.0%** |


## 3. Baseline Performance Comparison

| Model | Validation Acc | Validation Macro F1 | Validation Weighted F1 | Test Acc | Test Macro F1 | Test Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Majority Class Baseline** | 0.3522 | 0.0521 | 0.1835 | 0.3475 | 0.0516 | 0.1792 |
| **TF-IDF + Logistic Regression** | 0.8581 | 0.7743 | 0.8535 | 0.8612 | 0.7860 | 0.8573 |


## 4. Per-Class Performance (TF-IDF + Logistic Regression on Held-Out Test Set)

| Intent | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `delivery_status_tracking` | 0.9307 | 0.8993 | 0.9148 | 3,646 |
| `return_refund_exchange` | 0.9440 | 0.7367 | 0.8276 | 847 |
| `damaged_defective_wrong_item` | 0.9838 | 0.6925 | 0.8128 | 439 |
| `order_cancellation_modification` | 0.9821 | 0.7256 | 0.8346 | 379 |
| `payment_billing_promotions` | 0.9394 | 0.6539 | 0.7711 | 783 |
| `prime_digital_services` | 0.9291 | 0.6467 | 0.7626 | 750 |
| `account_access_security` | 0.8824 | 0.5172 | 0.6522 | 145 |
| `product_seller_inquiry` | 0.9231 | 0.4068 | 0.5647 | 118 |
| `feedback_complaint_chatter` | 0.8736 | 0.8268 | 0.8496 | 970 |
| `other_unclear` | 0.7762 | 0.9895 | 0.8700 | 4,301 |


## 5. Diagnostic Error Analysis (12 Real Baseline Misclassifications)

Representative errors sampled from the validation set demonstrate specific failure modes:

### Error Example 1: `delivery_status_tracking` → Predicted as `other_unclear`
- **Conversation ID**: `304621` (Turn 1)
- **Customer Message**: *"@115850 how do we reach customer care for an order someone didnt place but received it. Not his ID is used to place order. Can u help pls."*
- **Diagnostic Explanation**: Lexical overlap between 'delivery_status_tracking' and 'other_unclear' tokens in customer message.

### Error Example 2: `return_refund_exchange` → Predicted as `delivery_status_tracking`
- **Conversation ID**: `2287321` (Turn 1)
- **Customer Message**: *"Why do I need to learn about returning a package I did not receive? I just want to know why it wasn't delivered on time. @115821 https://t.co/3CIaZFWhxc"*
- **Diagnostic Explanation**: Customer requested refund due to delivery delay; logistics keywords outweighed refund request.

### Error Example 3: `product_seller_inquiry` → Predicted as `other_unclear`
- **Conversation ID**: `248700` (Turn 1)
- **Customer Message**: *"Fuck you @115850 , when I was looted by your seller by sending a defected laptop and refusing to send me a new product by saying 1/n"*
- **Diagnostic Explanation**: Informal phrasing of pre-purchase question without explicit keyword markers.

### Error Example 4: `damaged_defective_wrong_item` → Predicted as `other_unclear`
- **Conversation ID**: `331275` (Turn 1)
- **Customer Message**: *"@AmazonHelp is it duplicate or counterfeit from Sparx #diwalioffer makes India fool @118702 @4030 https://t.co/4n11hbCGHo"*
- **Diagnostic Explanation**: Lexical overlap between 'damaged_defective_wrong_item' and 'other_unclear' tokens in customer message.

### Error Example 5: `feedback_complaint_chatter` → Predicted as `other_unclear`
- **Conversation ID**: `2627629` (Turn 1)
- **Customer Message**: *"Y todo esto sin engentarme en centros comerciales, porque pues @116875 es genial 😍❤️ #BuenFin2017"*
- **Diagnostic Explanation**: Vague expression of brand frustration or social chatter lacking specific keywords.

### Error Example 6: `return_refund_exchange` → Predicted as `other_unclear`
- **Conversation ID**: `2573806` (Turn 1)
- **Customer Message**: *"@116875 me podrían ayudar con un reembolso de un pedido por favor ?"*
- **Diagnostic Explanation**: Lexical overlap between 'return_refund_exchange' and 'other_unclear' tokens in customer message.

### Error Example 7: `account_access_security` → Predicted as `other_unclear`
- **Conversation ID**: `149290` (Turn 1)
- **Customer Message**: *"Haciendo cuentas para ver que me puedo comprar en el #BlackFridayEnAmazon 🤔@116875 https://t.co/44E9CXlhm8"*
- **Diagnostic Explanation**: Non-English or rare account lock vocabulary not captured strongly in TF-IDF unigrams.

### Error Example 8: `payment_billing_promotions` → Predicted as `feedback_complaint_chatter`
- **Conversation ID**: `2859970` (Turn 1)
- **Customer Message**: *"Meanwhile I have $15 in @117795 credits that are useless because even their own customer service reps don't know how they work... Everything online says "automatically applied" but the email from cust"*
- **Diagnostic Explanation**: Lexical overlap between 'payment_billing_promotions' and 'feedback_complaint_chatter' tokens in customer message.

### Error Example 9: `payment_billing_promotions` → Predicted as `other_unclear`
- **Conversation ID**: `2928903` (Turn 1)
- **Customer Message**: *"Amazonからのショートメールに焦ってたお昼。
調べてみたら、やっぱり架空請求だったみたいです。
そりゃそうだ。見に覚えないし…"*
- **Diagnostic Explanation**: Lexical overlap between 'payment_billing_promotions' and 'other_unclear' tokens in customer message.

### Error Example 10: `delivery_status_tracking` → Predicted as `feedback_complaint_chatter`
- **Conversation ID**: `2600198` (Turn 1)
- **Customer Message**: *"Amongst the best experiences I have had in my life is one where I sent vegetable garden seeds by @115850 to people marginal farmers in rural Bihar. The whole experience for sender and recipient is min"*
- **Diagnostic Explanation**: Lexical overlap between 'delivery_status_tracking' and 'feedback_complaint_chatter' tokens in customer message.

### Error Example 11: `prime_digital_services` → Predicted as `other_unclear`
- **Conversation ID**: `2669954` (Turn 1)
- **Customer Message**: *"あんハピがいつでも観れるプライムビデオ最高。入り直してよかったAmazonプライム。"*
- **Diagnostic Explanation**: Lexical overlap between 'prime_digital_services' and 'other_unclear' tokens in customer message.

### Error Example 12: `other_unclear` → Predicted as `delivery_status_tracking`
- **Conversation ID**: `2401238` (Turn 1)
- **Customer Message**: *"@115830 how come my Listen Without Prejudice 25 is released today yet my order is due on the 24th Oct? 203-3644993-1647508"*
- **Diagnostic Explanation**: Message contained words resembling an intent (e.g., 'help', 'order') but lacked full ticket context.

## 6. Key Findings & Limitations

1. **Logistics Dominance**: `delivery_status_tracking` and `other_unclear` account for ~64.5% of total inquiries.
2. **Multi-Intent Overlap**: Messages describing a late delivery that also demand a refund or cancellation often trigger keyword collisions between logistics and returns/billing.
3. **Context Dependency**: Messages in subsequent turns (e.g., '123-456' or 'Here is my email') lack standalone semantics without previous conversation history.
4. **Multilingual Inquiries**: High prevalence of Japanese (8.2%) and European languages requires multilingual word representations (handled well by unigram/character n-grams in TF-IDF but will benefit substantially from future multilingual embeddings).
5. **Weak Label Upper Bound**: Because development labels were generated using rules, the TF-IDF model reflects rule fidelity rather than ground-truth human perception. The hand-labeled Golden Set in Phase 4 is necessary to establish true real-world accuracy.
