# Decision Log

This log tracks non-obvious engineering decisions made during the development of the Hiver AI Customer Support Agent, along with their context, rationale, and consequences.

---

### Decision 1: Isolated Virtual Environment (`.venv`)
- **Date**: 2026-09-09
- **Context**: The host machine has a system-wide Anaconda distribution where package combinations can conflict (e.g. numpy 2.x and pandas binary builds).
- **Decision**: Create and use a dedicated local virtual environment at `.venv/` rather than relying on global or conda-base environments.
- **Rationale**: Ensures exact package version pinning, complete reproducibility across different reviewer machines, and prevents environment corruption.
- **Consequences**: Reviewers must activate `.venv` or run via `.venv/Scripts/python` (or `bin/python` on POSIX).

---

### Decision 2: Decoupled Dataset Location Pointer
- **Date**: 2026-09-09
- **Context**: The Kaggle dataset (`thoughtvector/customer-support-on-twitter`) contains ~3M tweets (~500MB compressed, over 1GB uncompressed). Copying this large CSV directly into the workspace repository bloats disk usage and risks accidental Git commits.
- **Decision**: Store the absolute path of the downloaded dataset in `data/raw/dataset_location.txt` via `scripts/download_data.py`. All downstream processing scripts read from this pointer or fallback to KaggleHub cache.
- **Rationale**: Adheres to 12-factor/data-engineering best practices: keep code repositories lightweight, separate immutable large raw data from codebase, and avoid duplicate multi-gigabyte disk writes.
- **Consequences**: Raw data remains outside Git tracking while still being 100% programmatically discoverable by tests and scripts.

---

### Decision 3: Custom Basitemp for Pytest on Windows
- **Date**: 2026-09-09
- **Context**: On Windows systems, Pytest's default `tmp_path` fixture can trigger `PermissionError: [WinError 5] Access is denied` when scanning `AppData\Local\Temp\pytest-of-<User>`.
- **Decision**: Configure `pytest.ini` with `addopts = --basetemp=.pytest_tmp -v` and add `.pytest_tmp/` to `.gitignore`.
- **Rationale**: Eliminates OS-level permission crashes during unit test runs while keeping test temporary directories neatly self-contained inside the repository.
- **Consequences**: Fast, reliable unit tests on any Windows machine without requiring admin elevation.

---

### Decision 4: Conversation Reconstruction via Path-Compressed Forest Graph
- **Date**: 2026-09-09
- **Context**: The raw dataset contains 2,811,774 tweets linked only by `in_response_to_tweet_id`. Multi-turn interactions are fragmented across rows. Random row-level splitting would cause severe data leakage between training and testing.
- **Decision**: Implemented an iterative ancestor lookup with path compression (`scripts/explore_data.py`) that resolves every tweet to its conversation tree root (`root_id`). All 2.81M tweets partition into 798,012 disjoint conversation trees.
- **Rationale**: Enables leak-free conversation-level group splits (Principle 3) and accurate conversation depth metrics with O(1) amortized lookup per tweet.
- **Consequences**: Processing requires an initial graph resolution pass (~15 seconds), which is fully cached/reusable across downstream pipelines.

---

### Decision 5: Selection of `AmazonHelp` as Target Brand
- **Date**: 2026-09-09
- **Context**: The assignment requires selecting ONE brand with sufficient data, diverse intents, historical resolution grounding, and authentic escalation decisions.
- **Decision**: Evaluated the top 15 brands in `twcs.csv` and selected `AmazonHelp`.
- **Rationale**:
  1. **Highest Usable Volume**: 82,534 complete multi-turn conversations (373,438 total tweets).
  2. **Rich Dialogue Depth**: Average conversation length of 4.53 tweets (median 3.0), the deepest among major consumer brands (compared to AppleSupport's 2.96 and Uber_Support's 3.07).
  3. **High-Fidelity Public Resolution**: Competitors like `TMobileHelp` (84.27% escalation/DM phrases) and `comcastcares` (73.34% escalation) almost exclusively post "Please DM us" boilerplate, providing little resolution text to ground responses. `AmazonHelp` exhibits a 9.66% explicit escalation rate, providing ample public resolution guidance alongside authentic escalation triggers.
  4. **Clean Categorical Intents**: Retail logistics issues (Order Tracking, Damaged Delivery, Cancellation, Refund, Prime Video) form intuitive, mutually exclusive clusters for human labelling.
- **Consequences**: Downstream pipeline will focus exclusively on `AmazonHelp` data filtering and evaluation.

---

### Decision 6: Groq as LLM Provider
- **Date**: 2026-09-09
- **Context**: The assignment requires an LLM for classification comparison, historical response drafting, and an LLM-as-a-judge evaluation harness. Cost and API latency can be major bottlenecks for reproducing evaluations in under 15 minutes.
- **Decision**: Use Groq (`GROQ_API_KEY`) as the primary LLM inference provider (defaulting to `llama-3.3-70b-versatile` / `llama-3.1-8b-instant`).
- **Rationale**:
  1. **Free Tier & Zero Cost**: Reviewers can obtain a free Groq API key without requiring paid credit cards or corporate billing setups.
  2. **High Throughput / Low Latency**: Groq's LPU inference speed (~250-500 tokens/second) allows running the 150-250 example Golden Set evaluation in under 2 minutes rather than 10-15 minutes on standard cloud APIs.
  3. **Reproducibility**: The reviewer can run the full evaluation suite quickly without hitting prohibitive paywalls or slow rate limits.
- **Consequences**: Configured in `.env.example` as `LLM_PROVIDER=groq` with `GROQ_API_KEY`.

---

### Decision 7: Conversation Graph Boundary & Orphan Parent Resolution
- **Date**: 2026-09-09
- **Context**: 0.30% of parent tweets replied to by AmazonHelp fall outside the scraping collection window (`in_response_to_tweet_id` not found in `twcs.csv`).
- **Decision**: For any tweet whose referenced parent is absent from the dataset, the graph resolution algorithm designates that tweet itself as the local conversation root.
- **Rationale**: Preserves authentic customer support responses rather than discarding valid dialogues merely because the initiating message occurred prior to dataset capture.
- **Consequences**: Recovers 460 valuable conversations without introducing invalid cross-thread links.

---

### Decision 8: Usability Definition & Exclusion Criteria for Support Conversations
- **Date**: 2026-09-09
- **Context**: Not all conversation trees with brand activity represent valid support dialogues (e.g. viral broadcast threads with hundreds of spectator tweets, standalone promos).
- **Decision**: Define a conversation as `usable_for_training` if and only if it contains: (a) at least one customer inbound tweet, (b) at least one AmazonHelp outbound tweet, (c) at least one direct customer $\rightarrow$ AmazonHelp reply relationship, and (d) $\le 100$ messages.
- **Rationale**: Excludes 19 anomalous viral flame wars (>100 tweets) that contain mostly spectator banter rather than bilateral customer service. Out of 82,534 trees, 82,515 (99.98%) qualify as clean support interactions.
- **Consequences**: Eliminates extreme outliers while retaining 167,699 direct query-response training pairs.

---

### Decision 9: Chronological Message Ordering via UTC Timestamps
- **Date**: 2026-09-09
- **Context**: Twitter tweet IDs are generally increasing but are not guaranteed to be strictly sequential across multi-datacenter clusters or network delays, and sorting by tweet ID can corrupt dialogue turn causality.
- **Decision**: Parse Twitter's `created_at` timestamp format into UTC epoch timestamps and sort conversation messages strictly chronologically, using `tweet_id` solely as a stable tie-breaker.
- **Rationale**: Guarantees causal sequence for multi-turn dialogues so that customer queries always precede agent replies.
- **Consequences**: Required custom parsing (`parse_twitter_timestamp`), vectorized for high performance.

---

### Decision 10: Leakage-Safe Conversation Partitioning (70 / 15 / 15)
- **Date**: 2026-09-09
- **Context**: Splitting tweets randomly leaks context between turns of the same customer problem into both training and evaluation splits, producing falsely inflated evaluation metrics.
- **Decision**: Partition data at the conversation `root_id` level using deterministic random seed `42` (70% Train: 57,760 convs; 15% Val: 12,377 convs; 15% Test: 12,378 convs), enforced by an automated verifier that halts execution on any root or tweet ID overlap.
- **Rationale**: Completely prevents data leakage across train, val, and test splits (Principle 3), ensuring that the evaluation measures true generalization to unseen customer cases.
- **Consequences**: Downstream intent classifiers, retrieval indexes (RAG), and Golden Set examples are drawn strictly from their respective partitions.

---

### Decision 11: Structured JSONL Schema with Turn-Level Interaction Pairs
- **Date**: 2026-09-09
- **Context**: Downstream tasks require both full conversation history (for multi-turn analysis) and isolated (customer_query $\rightarrow$ agent_reply) pairs for retrieval and intent evaluation.
- **Decision**: Format processed datasets as JSONL (`amazonhelp_conversations.jsonl`, `amazonhelp_train.jsonl`, etc.) containing both the full message sequence and pre-extracted `interaction_pairs` with accumulated prior context.
- **Rationale**: Eliminates redundant parsing in future phases; downstream classifier and retrieval scripts can directly ingest pairs in milliseconds.
- **Consequences**: Produces clean, self-contained JSONL files in `data/processed/`.

---

### Decision 12: Empirical 10-Intent Taxonomy Grounded in Amazon Support Structure
- **Date**: 2026-09-09
- **Context**: Preliminary exploratory analysis lumped ~50.8% of inbound messages into an overly broad `general_inquiry_other` bucket due to restrictive English-only keywords and omission of key categories (seller inquiries, digital app/streaming issues, casual banter, multilingual queries).
- **Decision**: Define a 10-intent taxonomy (`delivery_status_tracking`, `return_refund_exchange`, `damaged_defective_wrong_item`, `order_cancellation_modification`, `payment_billing_promotions`, `prime_digital_services`, `account_access_security`, `product_seller_inquiry`, `feedback_complaint_chatter`, `other_unclear`) mapping directly to real Amazon operational departments.
- **Rationale**: Provides clear human annotator guidelines with explicit inclusion/exclusion criteria, resolves the large catch-all bucket into actionable categories, and handles multilingual inquiries (Japanese, German, French, Spanish, Portuguese, Italian).
- **Consequences**: Unclassified `other_unclear` rate dropped from 50.8% to 35.1% on Turn-1 inquiries while establishing distinguishable problem domains.

---

### Decision 13: Explicit Separation of Weak Development Labels from Human Golden Ground Truth
- **Date**: 2026-09-09
- **Context**: Hand-labeling all 82,515 conversations is infeasible, but benchmarking baseline ML models requires labels across train, validation, and test splits.
- **Decision**: Implement deterministic, priority-based regex rules in `src/intents.py` strictly designated as "weak development labels" to establish baseline feasibility, debug feature pipelines, and discover failure modes. They are never conflated with or presented as human ground truth.
- **Rationale**: Maintains scientific integrity (Principle 1); prevents misleading claims about 86% baseline accuracy being "gold-standard accuracy". Human annotation is reserved for the Golden Evaluation Set in Phase 4.
- **Consequences**: Benchmarks and reports are explicitly labeled as "Development-label benchmark".

---

### Decision 14: Current Customer Message as Initial Classification Input
- **Date**: 2026-09-09
- **Context**: Multi-turn customer service conversations contain historical messages from both parties. Concatenating full dialogues initially obscures whether the model can parse the customer's immediate complaint.
- **Decision**: Benchmark models take only the cleaned current customer message (`cleaned_text`) as primary input, while preserving `prior_context` in the dataset schema for future context-ablation experiments.
- **Rationale**: Answers the foundational question: "Can the system understand what the customer is asking right now?" Establishes a clean baseline against which subsequent context-aware and RAG architectures can be compared.
- **Consequences**: Enables clean ablation in future phases between Turn 1 standalone performance and multi-turn context-aware classification.

---

### Decision 15: TF-IDF + Logistic Regression as Interpretable ML Baseline
- **Date**: 2026-09-09
- **Context**: Prior to deploying compute-heavy LLM classifiers or dense embeddings, a simple, transparent traditional ML baseline is needed to establish the performance floor.
- **Decision**: Implement `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)` combined with `LogisticRegression(C=1.0, max_iter=500, random_state=42)`.
- **Rationale**: Fits in ~14 seconds, vectorizes in ~7 seconds, requires zero GPU resources, provides clear feature weight interpretability, and outperforms the majority-class baseline by +51.4% accuracy (86.1% vs 34.8%).
- **Consequences**: Forms Baseline 2 against which future embedding and LLM zero/few-shot classifiers will be compared.

---

### Decision 16: Strict Training-Set-Only Parameter Fitting
- **Date**: 2026-09-09
- **Context**: Determining the majority class or fitting vocabulary/parameters using validation or test data produces data snooping bias.
- **Decision**: The majority class (`other_unclear`, 35.1%), TF-IDF vocabulary, and Logistic Regression model are computed and fit strictly on `amazonhelp_train.jsonl`. The validation split is used solely for diagnostic error analysis, and the test split is evaluated strictly as held-out.
- **Rationale**: Enforces leakage-free evaluation standards (Principle 3) ensuring zero statistical bias in reported test metrics.
- **Consequences**: Test set metrics (86.12% accuracy, 0.7860 Macro-F1) represent true out-of-sample generalization under the development labeling schema.

---

### Decision 17: 200-Example Golden Evaluation Set Sampled Exclusively from Held-Out Test Split
- **Date**: 2026-09-09
- **Context**: Evaluating final agent performance against the training split or weak regex labels produces severe confirmation bias. A compact, high-quality human ground-truth set (150–250 examples) is required by the assignment.
- **Decision**: Construct an authoritative 200-example Golden Evaluation Set (`data/golden/golden_evaluation_200.jsonl`) sampled strictly from unseen test conversations (`amazonhelp_test.jsonl`), enforcing exactly 1 example per conversation (200 unique `conversation_id`s).
- **Rationale**: Zero test-to-train/val contamination guarantees that model performance measured on the Golden Set represents authentic generalization to real-world support inquiries.
- **Consequences**: The Golden Set serves as the single source of truth for intent classification, historical response grounding, and auto-handle/escalation evaluation.

---

### Decision 18: Stratified and Targeted Stress-Sampling Methodology
- **Date**: 2026-09-09
- **Context**: Uniform random sampling would replicate the dataset's high concentration of tracking queries (~30%) and ambiguous messages (~35%), leaving tail intents and boundary cases severely underrepresented.
- **Decision**: Employ stratified and targeted sampling across 7 distinct strata: (1) Model-vs-Rule Disagreements (35 examples / 17.5%), (2) Rare Operational Intents (40 examples / 20.0%), (3) Multilingual Inquiries (25 examples / 12.5%), (4) Core Operational Intents (40 examples / 20.0%), (5) Social Chatter & Brand Feedback (15 examples / 7.5%), (6) Ambiguous Opening Queries (20 examples / 10.0%), and (7) Multi-Turn Follow-Ups (25 examples / 12.5%).
- **Rationale**: Deliberately stress-tests the AI agent across rare classes (`product_seller_inquiry`, `account_access_security`), high-confusion boundaries (damage vs delivery delay, refund vs cancellation), non-English scripts (Japanese, German, French, Spanish, Italian, Portuguese), and boundary disagreements between rules and statistical models.
- **Consequences**: Yields a statistically rich, rigorous evaluation benchmark that penalizes superficial keyword pattern matching.

---

### Decision 19: Inclusion of Context-Dependent Multi-Turn Follow-Ups (12.5%)
- **Date**: 2026-09-09
- **Context**: Support agents operate in ongoing conversations where subsequent customer messages often lack standalone semantics (*"Here is my order number: 123-456"*, *"DM sent"*, *"Why haven't you replied?"*).
- **Decision**: Allocate 25 examples (12.5%) of the Golden Set to Turn $\ge 2$ interactions, preserving full accumulated `conversation_context` while explicitly marking `is_followup: true`.
- **Rationale**: Enables direct context-ablation experiments comparing current-message-only classification against context-aware classification, testing whether previous turns materially improve resolution accuracy.
- **Consequences**: Prepares the evaluation dataset for both Turn-1 opening triage and full-dialogue agent benchmarking.

---

### Decision 20: Dual-Annotator Schema and Anti-Fabrication Integrity Constraints
- **Date**: 2026-09-09
- **Context**: Human labeling requires reproducibility, inter-annotator agreement metrics (Cohen's Kappa), and strict protection against pre-populating fields with weak labels.
- **Decision**: Initialize all human evaluation fields (`gold_intent`, `gold_reply_quality`, `gold_escalation`) to `null`. Provide dual annotator slots (`gold_intent_annotator_1`, `gold_intent_annotator_2`) supported by an automated validation script (`scripts/validate_golden.py`) and an agreement analyzer (`scripts/analyze_annotation_agreement.py`).
- **Rationale**: Adheres to core scientific principles (Principle 1 & 2): zero fabricated metrics or synthetic consensus. Agreement metrics will be computed from genuine dual annotation passes.
- **Consequences**: Prevents premature or fraudulent claims of human-evaluated performance.





