# AmazonHelp Intent Annotation Guide (Golden Evaluation Set)

## Purpose & Overview

This guide establishes the authoritative annotation protocol for the **200-example Golden Evaluation Set** for the AmazonHelp AI customer-support agent.

The goal of annotation is **high consistency, empirical accuracy, and scientific reproducibility**. Every annotation decision must be grounded in what the customer is actually communicating, using conversational context where necessary.

---

## The 10-Intent Taxonomy

Below is the complete reference specification for each of the 10 canonical intents.

---

### 1. `delivery_status_tracking`
- **Definition**: Customer inquiring about the physical whereabouts, tracking progress, estimated delivery date/time, courier hand-off, transit delays, or unreceived packages.
- **Inclusion Rules**:
  - Inquiries asking where an order or package is.
  - Complaints that tracking is stuck or out for delivery but hasn't arrived.
  - Inquiries about courier delivery attempts, carrier contact numbers, or delivery time slots.
- **Exclusion Rules**:
  - Reports that the package arrived smashed, open, or with missing items (use `damaged_defective_wrong_item`).
  - Demands to cancel an order because delivery is taking too long (use `order_cancellation_modification`).
- **Representative Real Examples**:
  1. *"why is my order at my local courier for the last 6 days and still hasn’t been delivered to me?? Over 1 week late"*
  2. *"Item has not been delivered but tracking says it was handed to me over an hour ago... 2nd time this has happened."*
  3. *"発送された商品の追跡番号を教えてください。"* (Please tell me the tracking number for the shipped goods)
- **Common Confusions**:
  - `damaged_defective_wrong_item`: Confused when courier damages the parcel.
  - `return_refund_exchange`: Confused when customer demands refund due to late delivery.
- **Tie-Breaking Rule**: If the customer primarily wants to know **where their package is or why it is delayed**, assign `delivery_status_tracking`. If they state they **no longer want it and demand a refund/cancellation**, assign `return_refund_exchange` or `order_cancellation_modification`.

---

### 2. `return_refund_exchange`
- **Definition**: Customer requesting to return an eligible retail product, check refund status, obtain a prepaid return shipping label, or initiate an exchange or monetary reimbursement.
- **Inclusion Rules**:
  - Asking how to return an item or where to drop off a return package.
  - Asking when a refund will be credited to their account or card.
  - Requesting an exchange for a different size, variant, or replacement.
- **Exclusion Rules**:
  - Reporting that an item arrived defective or broken without explicitly demanding a return/refund yet (use `damaged_defective_wrong_item`).
  - Inquiring about unauthorized credit card charges or payment gateway errors (use `payment_billing_promotions`).
- **Representative Real Examples**:
  1. *"I called customer service and was told my membership wouldn't be renewed. How do I get a refund?"*
  2. *"Can I get a return shipping label? The item is still unopened."*
  3. *"Je souhaite retourner cet article et être remboursé au plus vite."* (I wish to return this item and be refunded as soon as possible)
- **Common Confusions**:
  - `damaged_defective_wrong_item`: Customer received broken goods and requests refund.
  - `payment_billing_promotions`: Customer waiting for credit to appear on bank statement.
- **Tie-Breaking Rule**: If an explicit return, refund, or reimbursement request is made, label as `return_refund_exchange` even if the initial reason was delivery delay or dissatisfaction.

---

### 3. `damaged_defective_wrong_item`
- **Definition**: Customer reporting that an order arrived physically broken, shattered, non-functional, missing parts/accessories, inside an opened/empty box, or completely different from what was ordered (including counterfeit/fake items).
- **Inclusion Rules**:
  - Physical damage: cracked screens, torn clothing, dented appliances.
  - Functional defect: electronic device won't turn on, faulty hardware.
  - Wrong item received: ordered size 10 shoes, received size 7; received different product.
  - Missing contents: box arrived empty, or sealed package was missing components.
- **Exclusion Rules**:
  - Packages delayed or lost by the carrier before delivery (use `delivery_status_tracking`).
  - Digital content playback glitches (e.g. Prime Video freezing, use `prime_digital_services`).
- **Representative Real Examples**:
  1. *"my package was ‘accidentally’ opened.. 4 items missing worth £97. You need better delivery drivers!!"*
  2. *"The screen on the monitor I received is cracked and shattered completely."*
  3. *"Ihr Flaschen! Erst habt ihr mir ein Paket mit falschem Inhalt geschickt..."* (Sent me a package with the wrong content)
- **Common Confusions**:
  - `delivery_status_tracking`: When damage occurred during courier transit.
  - `return_refund_exchange`: When reporting damage alongside asking for return.
- **Tie-Breaking Rule**: If the customer's message is fundamentally a **problem report about the physical condition or correctness of the item received**, label as `damaged_defective_wrong_item`.

---

### 4. `order_cancellation_modification`
- **Definition**: Customer seeking to cancel an active order, revoke an accidental purchase, or modify order parameters (such as delivery address or delivery speed) prior to fulfillment.
- **Inclusion Rules**:
  - "Please cancel order # 123..."
  - "I placed this order by mistake, how do I cancel?"
  - "Can I update my delivery address before it ships?"
- **Exclusion Rules**:
  - Requesting account closure or profile deletion (use `account_access_security`).
  - Canceling Amazon Prime recurring membership (use `prime_digital_services`).
- **Representative Real Examples**:
  1. *"Please cancel order # 123-4567890-1234567 I bought it by accident."*
  2. *"Can I change the delivery address for my order? It hasn't shipped yet."*
  3. *"注文をキャンセルしたいのですが、まだ間に合いますか？"* (Want to cancel my order, is there still time?)
- **Common Confusions**:
  - `prime_digital_services`: Canceling a Prime subscription.
  - `delivery_status_tracking`: Attempting to cancel an order already in transit.
- **Tie-Breaking Rule**: Only use `order_cancellation_modification` for **retail physical orders**. For subscriptions, memberships, and digital channels, route to `prime_digital_services`.

---

### 5. `payment_billing_promotions`
- **Definition**: Customer experiencing payment processing failures, duplicate charges, unexpected credit/debit card deductions, gift card redemption errors, promotional coupon failures, or invoice inquiries.
- **Inclusion Rules**:
  - "My card was charged twice for order #..."
  - "My gift card voucher says already redeemed, but my balance is £0."
  - "Promo code SAVE20 says invalid at checkout."
  - Inquiries asking when payment will be deducted or requesting a tax invoice.
- **Exclusion Rules**:
  - Refund tracking for returned goods (use `return_refund_exchange`).
  - Fraudulent account takeover where someone hacked into the account (use `account_access_security`).
- **Representative Real Examples**:
  1. *"where can I chat with a support member for a false charge"*
  2. *"when do you guys charge for the Xbox one X I preordered? It ships the 6th."*
  3. *"promotion code doesn’t work on checkout"*
- **Common Confusions**:
  - `return_refund_exchange`: Disputing missing refund credits.
  - `prime_digital_services`: Inquiring about annual/monthly Prime subscription charges.
- **Tie-Breaking Rule**: If the core inquiry is about **payment instruments, billing errors, discounts, or gift cards**, assign `payment_billing_promotions`.

---

### 6. `prime_digital_services`
- **Definition**: Inquiries or technical troubleshooting regarding Amazon Prime benefits, Prime Video streaming, Kindle e-readers and ebooks, Fire TV sticks, Echo/Alexa devices, or Amazon Music.
- **Inclusion Rules**:
  - Video buffering, audio sync issues, or missing episodes on Prime Video.
  - Fire TV app crashes or HDMI connectivity.
  - Kindle ebook formatting, page numbers, or syncing issues.
  - Questions regarding Prime membership renewal, cancellation, student discount, or benefits.
- **Exclusion Rules**:
  - Physical orders placed using Prime two-day shipping that are delayed (use `delivery_status_tracking`).
  - Defective physical retail products (use `damaged_defective_wrong_item`).
- **Representative Real Examples**:
  1. *"also, beim Addams Family-Film in Prime sind Bild und Ton nicht wirklich synchron. Wie kommt's?"*
  2. *"I'm watching SUITS on firetv. Whenever I play SUITS, it always go to SE03EP01 by default"*
  3. *"you need to notify new users that the page numbers do not match up with the paper copies on Kindle."*
- **Common Confusions**:
  - `delivery_status_tracking`: Customer complaining Prime 2-day delivery arrived late.
  - `payment_billing_promotions`: Dispute over Prime subscription membership billing.
- **Tie-Breaking Rule**: If the issue concerns **content consumption (streaming, reading, listening) or hardware setup (Fire TV, Alexa)**, assign `prime_digital_services`. If the customer is merely complaining about delivery speed of a retail package bought on Prime, assign `delivery_status_tracking`.

---

### 7. `account_access_security`
- **Definition**: Customer unable to authenticate, locked out of account, experiencing OTP/2FA SMS delivery failures, reporting hacked/compromised credentials, or requesting permanent account closure.
- **Inclusion Rules**:
  - "I cannot log into my Amazon account."
  - "Password reset email never arrived."
  - "Two-step verification code (OTP) not received on my phone."
  - "My account was hacked and someone changed my email address."
  - "I want to permanently delete/close my Amazon account."
- **Exclusion Rules**:
  - Changing shipping address inside a normal working account (use `order_cancellation_modification`).
  - Credit card disputes where account credentials remain secure (use `payment_billing_promotions`).
- **Representative Real Examples**:
  1. *"I want my amazon payments account CLOSED. dm me please."*
  2. *"Why would Amazon hack my account? Some employee must have been in there hacking people."*
  3. *"Bonjour, mon compte Amazon est bloque suite a une commande, toujours rien après 24h."*
- **Common Confusions**:
  - `feedback_complaint_chatter`: Customer angrily venting about being suspended without mentioning account recovery.
  - `payment_billing_promotions`: Suspicious transactions caused by unauthorized access.
- **Tie-Breaking Rule**: If the obstacle is **logging in, authentication, identity verification, account lockout, or account closure**, assign `account_access_security`.

---

### 8. `product_seller_inquiry`
- **Definition**: Pre-purchase questions regarding product availability, stock restock dates, technical specifications, third-party seller authenticity, marketplace policies (A-to-Z guarantee), or selling on Amazon.
- **Inclusion Rules**:
  - "When will the Nintendo Switch be back in stock?"
  - "Is this item sold by Amazon or a third-party marketplace seller?"
  - "I want to register as a seller to sell goods globally."
  - Product dimension, voltage, or compatibility questions prior to purchase.
- **Exclusion Rules**:
  - Inquiries about orders already placed (use `delivery_status_tracking` or `return_refund_exchange`).
  - Reporting a defective or fake product already received (use `damaged_defective_wrong_item`).
- **Representative Real Examples**:
  1. *"I want to sell globally on Amazon. Where do I register as a seller?"*
  2. *"When will the Nintendo Switch be back in stock on Amazon?"*
  3. *"Amazonとかで物買ったことある人いますか？いたら買い方教えてほしいです"* (Has anyone bought here, how to buy?)
- **Common Confusions**:
  - `damaged_defective_wrong_item`: Asking if a seller is selling fakes vs receiving a fake.
  - `other_unclear`: Very generic pre-sales inquiries ("How does Amazon work?").
- **Tie-Breaking Rule**: If the customer has **not yet purchased or is asking general catalog/seller questions**, assign `product_seller_inquiry`.

---

### 9. `feedback_complaint_chatter`
- **Definition**: Inbound tweets expressing general brand sentiment, customer service complaints/rants without a specific actionable ticket, compliments/gratitude, casual banter, social media photos (e.g. cats in boxes), or promotional quiz contests.
- **Inclusion Rules**:
  - Broad rants: *"Amazon is the worst company in the world! Horrible customer service!"*
  - Gratitude/Praise: *"Thank you @AmazonHelp for sorting this out so fast!"*
  - Social media banter: Photos of pets playing in Amazon boxes, funny courier interactions.
  - Contests & Quizzes: Inquiries about #AmazonAppQuiz winner lists.
- **Exclusion Rules**:
  - Actionable customer support issues where an order number, tracking number, or specific problem is reported (route to operational intent).
  - Isolated meaningless fragments without sentiment (use `other_unclear`).
- **Representative Real Examples**:
  1. *"thank you for your help. Got my thing done within a day. really appreciate it 👍"*
  2. *"Au lieu de jeter un carton il vaut mieux adopter un ou deux chats ! C’est bien plus utile !"*
  3. *"#OnePlus5TAppQuiz @AmazonHelp where is a OnePlus5t winner list"*
- **Common Confusions**:
  - `other_unclear`: Vague rants vs completely uninterpretable messages.
- **Tie-Breaking Rule**: If the tweet contains **discernible human sentiment, social interaction, brand feedback, or contest inquiry** but lacks a specific transactional operational problem, assign `feedback_complaint_chatter`.

---

### 10. `other_unclear`
- **Definition**: Messages where intent cannot be reliably determined without additional information, including standalone greetings, isolated follow-up fragments, or uninterpretable noise.
- **Inclusion Rules**:
  - Standalone greetings: *"Hi", "Hello", "Need help"*.
  - Isolated follow-up fragments: *"__email__"*, *"123-456789"*, *"DM sent"*, *"Yes please"*, *"OK"*.
  - Meaningless noise, broken URLs, or uninterpretable gibberish.
- **Exclusion Rules**:
  - Any message expressing clear operational intent or clear brand sentiment/feedback.
- **Representative Real Examples**:
  1. *"hi I need Help"*
  2. *"__email__"*
  3. *"Yes, I already replied to your DM."*
- **Common Confusions**:
  - `feedback_complaint_chatter`: Generic complaints vs context-less fragments.
- **Tie-Breaking Rule**: Use `other_unclear` as the **honest admission of ambiguity**. If an annotator would have to guess the intent without evidence, assign `other_unclear`.

---

## Special Annotation Protocols

### A. Multi-Intent Messages
- Customers often combine multiple problems (e.g., *"My package arrived late, the box was damaged, and I want a refund"*).
- **Rule**: Identify the **primary actionable resolution** requested:
  - If demanding money back: $\rightarrow$ `return_refund_exchange`
  - If demanding order cancellation: $\rightarrow$ `order_cancellation_modification`
  - If reporting item damage without stating remedy: $\rightarrow$ `damaged_defective_wrong_item`
  - If asking where the order is: $\rightarrow$ `delivery_status_tracking`
- Document secondary intents in `gold_notes`.

### B. Vague Follow-up Turns (Multi-Turn Messages)
- In multi-turn dialogues (where `is_followup = True`), the customer's message may be an isolated fragment (e.g., *"My order number is 203-12345"* or *"I already sent a DM"*).
- **Rule**:
  - Review the `conversation_context`.
  - If the previous turn requested the order number for a **delayed package**, and the customer provides it, the conversational intent is `delivery_status_tracking`.
  - If the previous turn requested the order number for a **refund**, the conversational intent is `return_refund_exchange`.
  - If the message is purely conversational close (e.g. *"Thanks, that helped!"*), assign `feedback_complaint_chatter`.
  - If even with context the intent remains ambiguous, assign `other_unclear`.

### C. Multilingual Messages
- AmazonHelp receives tweets in Japanese, German, French, Spanish, Italian, and Portuguese.
- **Rule**:
  - Do NOT classify a message as `other_unclear` simply because it is in a foreign language.
  - Translate the message (or use provided translation tools).
  - Map to the appropriate operational intent based on the translated semantics.

### D. What to Do When None of the 10 Intents Fits
- If an authentic customer issue genuinely does not belong to any of the 9 operational categories and is not ambiguous noise, assign `other_unclear` and explicitly document the anomaly in `gold_notes` (e.g., *"Customer asking about Amazon AWS cloud infrastructure"*).

### E. Blind Annotation Protocol (Anti-Bias Rule)
- Annotators must make their decisions **blindly** based solely on the `customer_text` and `conversation_context`.
- Do **NOT** look at the weak regex label or the TF-IDF model prediction prior to deciding your annotation.
- Model predictions and regex labels are historical artifacts included strictly for provenance and disagreement tracking; treating them as hints will corrupt inter-annotator independence.

---

## Escalation Annotation Protocol (`gold_escalation`)

For each incoming customer message, determine whether an AI agent can safely resolve the inquiry or whether it must be escalated to a human customer service agent:

### 1. `auto_handle`
- The inquiry can be fully addressed using public information, FAQs, self-service URLs, or standard troubleshooting steps.
- **Examples**:
  - *"How do I return an unopened item?"* (Provide link to Online Returns Center)
  - *"Where do I find my tracking number?"* (Explain Your Orders $\rightarrow$ Track Package)
  - *"Fire TV video buffering troubleshooting"* (Restart device, clear cache)
  - *"What is your return policy for holiday orders?"* (State policy dates)

### 2. `escalate`
- The inquiry requires private customer authentication, accessing internal order databases, executing financial transactions (refunds/cancellations), or addressing security incidents.
- **Examples**:
  - *"Why was my credit card charged twice for order 112-984...?"* (Requires billing lookup)
  - *"My account was hacked and my email was changed."* (Security escalation)
  - *"The delivery driver opened my package and stole my phone."* (Incident investigation)
  - *"Cancel my order right now!"* (Order fulfillment cancellation in DB)

### 3. `unclear`
- The message is too ambiguous, fragmented, or context-less to determine whether escalation is warranted (e.g., *"hi"*, *"__email__"* without preceding context).

---

## Reply Quality Annotation Protocol (`gold_reply_quality`)

> **IMPORTANT TIMING NOTE**: Candidate AI replies do not yet exist in this phase. `gold_reply_quality` will remain `null` until Phase 6, when our RAG agent generates candidate responses.

When candidate replies are generated, they will be rated on the following 5-point Likert scale:
- **1 — Poor / Unsafe / Irrelevant**: Hallucinates policies, provides dead links, ignores the customer's problem, or leaks confidential guidance.
- **2 — Partially Useful**: Addresses part of the inquiry but is materially incomplete, misses essential next steps, or uses an inappropriate tone.
- **3 — Acceptable**: Accurately addresses the core question with correct policy, but is generic or canned.
- **4 — Strong & Grounded**: Empathetic, accurate, grounded in historical Amazon resolution patterns, with clear self-help guidance.
- **5 — Excellent / Exemplary**: Highly personalized, perfectly empathetic, fully grounded, provides proactive guidance while respecting security boundaries.

---

## Assisted Human Annotation Protocol (`--assisted`)

To accelerate annotation while maintaining strict human-in-the-loop evaluation integrity, `scripts/annotate_golden.py` includes an assisted annotation mode:

```bash
python scripts/annotate_golden.py --annotator annotator_1 --assisted
```

### Key Methodological Standards:
1. **Deterministic Rule Suggestions**:
   - Intent suggestions are generated strictly using the existing Phase 3 deterministic priority regex rules (`src.intents.classify_message_intent`).
   - Escalation suggestions are generated via deterministic heuristics derived directly from the escalation criteria above.
2. **Zero LLM / External API Usage**:
   - Suggestions do not use any LLM, cloud API, or non-deterministic model.
3. **Explicit Human Confirmation Required**:
   - Suggestions are clearly displayed as candidates. They are **never** automatically applied without human review.
   - The annotator reviews the message, context, and historical response, then decides:
     - `[y]`: Explicitly accept suggestions (recorded as `annotation_method="assisted_accept"`).
     - `[n]`: Manually choose intent and escalation (recorded as `annotation_method="manual"`).
     - `[s]`: Skip the example (remains unlabeled).
     - `[q]`: Save and quit safely.
4. **Legitimate Human-Reviewed Ground Truth**:
   - Because every example is vetted and explicitly approved by the human annotator, accepted suggestions are genuine human ground-truth labels.
5. **Non-Destructive Protection**:
   - Examples already annotated by the active annotator are automatically skipped and protected from modification.


