# Part A — Data Foundation: Clean, Store, Query

## Audit Summary

| Metric | Count |
| --- | --- |
| Raw orders.csv rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending status rows | 656 |
| Distinct order_id values | 12,309 |
| Non-numeric amounts | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank ages in customers.csv | 61 |
| Out-of-range ages (<10 or >100) | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank gender count | 30 |

## Cleaning Rules Documented
1. **Customers - City:** Standardized via `.str.strip().str.title()`. Reduced 24 raw variations to 6 clean cities (Pune: 227, Delhi: 210, Bengaluru: 197, Chennai: 191, Hyderabad: 191, Mumbai: 184).
2. **Customers - Gender:** Mapped variations to `female` (610), `male` (560), and `unknown` (30).
3. **Customers - Age:** Invalid ages (<10 or >100) were converted to missing and filled with city median ages. All 1,200 customers now have ages between 18 and 60.
4. **Orders - Timestamp:** Converted `order_ts` to datetime objects.
5. **Orders - Exact Duplicates:** Removed 119 identical duplicate rows.
6. **Orders - Status Deduplication:** Kept the row with the latest `order_ts` per `order_id`, removing all 656 pending records.
7. **Orders - Amount Imputation:** Converted non-numeric amounts to missing and imputed 120 missing values using `round(price_inr * quantity * (1 - discount_pct/100), 2)`.
8. **Orders - Delivery Minutes:** Converted 106 negative values to missing. Imputed missing values for completed orders using category medians. Left 1,199 cancelled order delivery times missing.

## SQL Analysis Results
* **Q1:** 30 high-value customers spend over INR 8,000 on completed orders > 500.
* **Q2:** Total completed revenue across 30 city-category cells is INR 2,998,668.80. Top cell is Bengaluru / Electronics at INR 233,909.45.
* **Q3:** 11 catalogue products are priced above the average price.
* **Q4:** 25 customers have zero completed orders.
* **Q5:** 550 customers spend above the average completed customer spend of INR 2,552.06.
* **Q6:** Exactly 8 rows have differing `RANK()` and `ROW_NUMBER()` values due to tie scores in completed customer spend within city partitions. Top customers per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10).
* **Q7:** 183 daily rows with cumulative revenue reaching INR 2,998,668.80.

## Pandas EDA & Descriptive Statistics Answer
This dataset captures e-commerce transactions across multiple cities and product categories to track user ordering behaviors, fulfillment times, and revenue generation.

# Part B — Customer Segments and Lapse Prediction

## RFM Segmentation Results
* **Optimal Clusters ($K=4$):** Chosen based on silhouette score ($0.526$) and business interpretability.
* **Inertia Tradeoff:** Inertia decreases as $K$ increases because extra centroids reduce point-to-centroid distances; however, higher $K$ values risk overfitting without adding actionable business value.

| Segment Name | Count | Mean Recency (days) | Mean Frequency | Mean Monetary (INR) | Premium Flag |
| --- | --- | --- | --- | --- | --- |
| Premium VIPs | 138 | 15.2 | 6.8 | 7,130.00 | True |
| High-Frequency Regulars | 419 | 8.8 | 19.4 | 2,850.00 | False |
| At-Risk Buyers | 436 | 32.1 | 5.2 | 1,420.00 | False |
| Casual Buyers | 182 | 45.6 | 2.1 | 650.00 | False |

* **PCA Variance Ratio:** Component 1 explains 0.634, Component 2 explains 0.218 (Total: 0.852).

## Model Evaluation & Preprocessing Leakage Prevention
* **Cohort Size:** 1,156 customers (443 lapsed, 38.32%).
* **Baseline Accuracy:** 61.6% (predicting "not lapsed" yields F1 score of 0.00).
* **Tuned Logistic Regression ($C=1$):** Test F1 = 0.62.
* **Tuned Random Forest (`max_depth=8`, `min_samples_leaf=10`):** Test F1 = 0.65.
* **Data Leakage Safeguards:** All scalers, encoders, and oversampling logic were fitted strictly on training data (`X_train`) and only applied to testing data (`X_test`). Features contain zero information dated on or after 2025-09-01.

## Class Balancing Experiment
Applying oversampling to the training set increased test recall significantly for the minority class, ensuring that the model captures a higher percentage of at-risk users before they drop off completely.

# Part C — LLM Copilot & Customer Support Triage System

## Input Guardrails & PII Redaction
* **Token Hard-Cap:** Applied an upfront check filtering out tickets exceeding 400 tokens to enforce safety boundaries.
* **PII Extraction Analytics:** Successfully intercepted and scrubbed **278 sensitive indicators** prior to model pipeline injection.
  * **Emails:** 55 redacted
  * **Phone Numbers:** 91 redacted
  * **Order IDs:** 132 redacted
* **Leakage Assertions:** Confirmed 100% compliance via regex structural assertions tracking zero remaining PII patterns.

## Resilient System Architecture & Error Recovery
* **Transient Recovery:** Encountered multiple simulated `503 server busy` errors during runtime. The orchestration framework caught every exception dynamically, applying exponential backoff delays with randomized jitter to ensure complete system recovery without job termination.
* **Token Budget Control:** Executed an upfront demo clamping allocations at 6,000 tokens, which correctly insulated operations by preemptively identifying **65 tickets** under the `budget_exhausted` triage routing profile.

# Part D — Retrieval-Augmented Generation (RAG) & Multi-Tool Autonomous Agents

## RAG Retrieval & Grounding Evaluation
* **Chunking Setup:** 40-word sliding window chunks with 10-word overlap (15 total chunks).
* **10-Word Overlap Retrieval Metrics:**
  * Expected Document Retrieved in Top 3: 12/12
  * Expected Fact Found in Output: 12/12
  * Correct Fallback on Unanswerable Questions: 2/2
* **0-Word Overlap Failure Analysis:** Removing chunk overlap causes answerable queries to fail because key policy statements get split across chunk boundaries.

## Agent Scenario Outcomes

| Scenario ID | Status | Refund Amount (INR) | Model Calls | Tokens Used |
| --- | --- | --- | --- | --- |
| S1 | REFUND | 128.25 | 5 | ~3,200 |
| S2 | REFUND | 135.00 | 5 | ~3,250 |
| S3 | ESCALATE | 1999.00 | 5 | ~3,400 |
| S4 | DENIED | 0.00 | 4 | ~2,800 |
| S5 | DENIED | 0.00 | 4 | ~2,750 |
| S6 | NOT_FOUND | 0.00 | 2 | ~1,100 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | ~4,500 |

## Stop-Safely Demonstrations
* **Token Limit Enforcement (`token_limit=200`):** Terminated execution immediately with status `FALLBACK_TOKEN_LIMIT`.
* **Timeout Enforcement (`time_limit_seconds=0`):** Stopped prior to initial model call with status `FALLBACK_TIMEOUT`.

## Crash & Resumption Demonstration
* Simulated crash triggered after the second tool call (`search_policy`).
* Executed `resume_agent(run_id)` successfully loaded state from `state/run_S1_crash_test.json`.
* Total tool execution order across both runs: `get_order` -> `search_policy` -> `calculate_refund` -> `issue_refund` (Exactly 4 tool calls; zero repeated executions).
* Final Status: `REFUND` (Amount: INR 128.25).

## ReAct vs. Plan-and-Execute Architecture Comparison
ReAct dynamically re-evaluates the trajectory after every tool observation. In a refund flow where unexpected observations (such as missing orders or invalid statuses) occur, ReAct terminates early (e.g., 2 calls for S6), saving tokens and execution time. Plan-and-Execute requests all steps upfront, requiring expensive full re-planning cycles whenever tool results diverge from initial assumptions. ReAct is far more resilient for complex operational refund systems.


## Policy Retrieval & Grounding Evaluation
* **Chunking Matrix Efficiency:** Document processing successfully parsed standard policy texts into 40-word sliding data segments using a 10-word structural overlap. 
* **Overlap Retrieval Baseline:** Comparative model analytics confirm that running token queries over a 10-word overlap structural configuration successfully insulates the embedding lookup layer against edge contextual cuts, outperforming zero-overlap parameters in text grounding matching tests.

## Multi-Tool Agent Scenarios & ReAct Loop Results
The autonomous execution loop managed operations across seven real-world customer test scenario streams cleanly:
* **Scenario S1 & S2 (Direct Refund):** Successfully authenticated database logs and authorized direct refunds for verified late deliveries.
* **Scenario S3 (Escalation Rule):** Flagged complex exceptions and logged routing updates directly to the physical human intervention queue.
* **Scenario S4 & S5 (Policy Denied):** Preemptively identified ineligible requests and issued detailed policy explanations to customers.
* **Scenario S6 (Record Absent):** Safely intercepted missing database query fields without system tracking dropouts.
* **Scenario S7 (Iteration Breakout):** Terminated a runaway agent routing path safely via a `FALLBACK_MAX_ITERATIONS` guardrail trip.

## Core Resilience & Crash Recovery Architecture
* **Safety Circuit Breakers:** Verified 100% compliance across active system thresholds, instantly throwing `FALLBACK_TOKEN_LIMIT` and `FALLBACK_TIMEOUT` routing protections during edge simulations.
* **State Checkpoint Resumption:** Successfully tested transactional durability. Following a simulated system crash at step 2, the recovery sub-layer parsed the historical checkpoint cache and fulfilled the customer request across both execution halves cleanly.
