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
