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
