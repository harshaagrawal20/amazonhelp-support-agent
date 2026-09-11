# Human Evaluation Annotation Guide: Support Reply Quality & Grounding

This guide establishes the standardized rubric and protocol for human evaluation of AI-generated customer support replies on the held-out **AmazonHelp 50-example Golden Evaluation Subset**.

---

## 1. Core Evaluation Principles

1. **Independent Evaluation (Double-Blind)**:
   - Evaluators must assess the customer issue and generated reply **without seeing the LLM judge's scores or rationales**.
2. **Customer Issue as Primary Truth**:
   - The customer's message and contextual complaint define what needs resolution.
3. **Historical Grounding vs. Exact Verbatim Matching**:
   - Retrieved historical interactions provide evidence of standard policies, routing options, and official tone.
   - **Do NOT penalize a response simply because its wording differs from the reference historical reply**. Customer service inquiries frequently allow multiple valid, empathetic resolutions.
4. **Conciseness & Twitter Constraints**:
   - Customer support tweets are naturally concise. **Do not reward verbosity** or penalize short answers if they give clear, correct guidance.
5. **Strict Anti-Hallucination / Unsupported Claims Policy**:
   - Support agents on public social channels cannot directly inspect private accounts or process refunds/cancellations without moving to a private channel (e.g., DM or secure customer service link).
   - Heavily penalize any reply that claims an account-specific action has already been executed.
6. **Multilingual Evaluation**:
   - Non-English tweets (Spanish, French, Portuguese, Japanese) must be evaluated on **semantic appropriateness and polite customer service register**, not penalized for language.

---

## 2. Six Quality Dimensions (Scored 1 to 5)

### Dimension 1: Correctness
*Does the reply accurately address the customer's real problem without factual errors or false assumptions?*

- **5 (Excellent)**: Completely accurate, factually sound, and addresses the root problem.
- **4 (Good)**: Mostly correct; minor technical omission that does not mislead the customer.
- **3 (Acceptable)**: Addresses only part of a multi-issue complaint, or makes an unverified benign assumption.
- **2 (Poor)**: Substantially misses the core problem or misinterprets key facts (e.g., advises tracking when package was already delivered damaged).
- **1 (Unacceptable)**: Factually contradictory, blatantly wrong, or contradicts Amazon corporate policy.

---

### Dimension 2: Relevance
*Is the response directly and specifically relevant to the customer's issue and conversation context?*

- **5 (Excellent)**: Highly targeted to the exact issue raised by the user.
- **4 (Good)**: Relevant, with slight generic phrasing.
- **3 (Acceptable)**: Generic response that applies broadly but lacks specificity to the user's details.
- **2 (Poor)**: Tangentially related or responds to an unrelated part of the message.
- **1 (Unacceptable)**: Completely irrelevant to what the customer asked.

---

### Dimension 3: Historical Grounding
*Does the reply follow authentic AmazonHelp resolution practices, policies, and conventions demonstrated in training evidence?*

- **5 (Excellent)**: Fully aligns with AmazonHelp resolution patterns (proper referral to DM, official help URLs `https://t.co/...`, agent sign-off tag `^XX`).
- **4 (Good)**: Aligns with Amazon policy and tone; may omit minor conventional elements (e.g., missing agent tag).
- **3 (Acceptable)**: Plausible customer service response, but deviates slightly from standard Amazon communication practices.
- **2 (Poor)**: Noticeably conflicts with typical Amazon resolution workflows (e.g., directing customer to external competitor or incorrect portal).
- **1 (Unacceptable)**: Completely ungrounded; invents non-existent policies or mechanisms.

---

### Dimension 4: Helpfulness
*Does the reply provide actionable, clear next steps to resolve the customer's problem?*

- **5 (Excellent)**: Actionable, frictionless next step (e.g., direct secure link, specific instructions on what info to send via DM).
- **4 (Good)**: Helpful direction provided, but customer must infer minor steps.
- **3 (Acceptable)**: Offers generic assistance ("please let us know how we can help") without concrete next steps.
- **2 (Poor)**: Minimal utility; asks the customer for information they already provided.
- **1 (Unacceptable)**: Unhelpful, dismissive, or dead-end response.

---

### Dimension 5: Unsupported Claims (Safety / Hallucination)
*Does the reply avoid inventing actions, refunds, credits, guarantees, delivery dates, or private account status?*

- **5 (Fully Safe / Zero Hallucination)**: Completely safe. No false claims, no invented refunds, no fake tracking numbers, no claims that account actions were already completed.
- **4 (Minor Assumption)**: Assumes benign context without making dangerous promises.
- **3 (Overconfident / Ambiguous)**: Ambiguous guarantee or slightly overconfident claim about carrier or order timeline.
- **2 (Significant Unsupported Claim)**: Promises a refund, credit, or specific delivery date without system access.
- **1 (Severe Hallucination / Critical Risk)**: Falsely asserts an account action was completed (e.g., *"We have refunded your $50"*) or invents non-existent policies.

---

### Dimension 6: Escalation Appropriateness
*Is the response appropriately calibrated to the customer's issue and gold escalation requirement?*

- **When `auto_handle`**: The reply should address or resolve the inquiry directly without unnecessary human escalation.
- **When `escalate`**: The reply should recognize that private account details / human intervention are needed and route the user to DM or official secure support without false assurances.
- **When `unclear`**: The reply should safely request clarification without jumping to conclusions.

- **5 (Excellent)**: Perfectly calibrated escalation behavior.
- **4 (Good)**: Appropriate routing with slightly suboptimal phrasing.
- **3 (Acceptable)**: Borderline routing (e.g., escalates a simple FAQ or attempts to self-handle a complex billing issue).
- **2 (Poor)**: Inappropriate handling (e.g., publicly asks for sensitive credentials or refuses support for an urgent issue).
- **1 (Unacceptable)**: Dangerous escalation failure.

---

## 3. Overall Score and Decision Rule

### Overall Score (1.0 to 5.0)
- Calculated as the arithmetic average of the six dimensions:
  $$\text{Overall} = \frac{\text{Correctness} + \text{Relevance} + \text{Grounding} + \text{Helpfulness} + \text{Safety} + \text{Escalation}}{6}$$

### Overall Decision:
1. **`pass`**:
   - Overall score $\ge 4.0$, AND
   - No critical dimension score $< 3$ (all dimensions $\ge 3.0$).
2. **`borderline`**:
   - Overall score $\ge 3.0$, but does not meet all criteria for `pass` (e.g. minor omission or one dimension scored 2).
3. **`fail`**:
   - Overall score $< 3.0$, OR
   - Any critical safety/correctness violation ($\le 1$ on Unsupported Claims or Correctness).
