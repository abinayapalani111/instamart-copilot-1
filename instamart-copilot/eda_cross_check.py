import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")

# Load tables
orders = pd.read_sql_query("SELECT * FROM orders", conn)
customers = pd.read_sql_query("SELECT * FROM customers", conn)
products = pd.read_sql_query("SELECT * FROM products", conn)

completed = orders[orders["status"] == "completed"].merge(customers, on="customer_id").merge(products, on="product_id")

# (a) Pivot Table Cross-Check for Q2
pivot_df = pd.pivot_table(completed, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("Pivot Table Grand Total:", pivot_df.loc["All", "All"])

# City aggregations
city_stats = completed.groupby("city")["order_amount"].agg(["sum", "mean", "count"])
print("\nCity Aggregation Stats:\n", city_stats)

# (b) Descriptive statistics for order_amount (completed orders)
amt = completed["order_amount"]
mean_val = amt.mean()
median_val = amt.median()
skew_val = amt.skew()
q1 = amt.quantile(0.25)
q3 = amt.quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
lower_fence = q1 - 1.5 * iqr
outliers_above = (amt > upper_fence).sum()
outliers_below = (amt < lower_fence).sum()

print("\n--- ORDER AMOUNT EDA STATISTICS ---")
print(f"Mean: {mean_val:.2f}, Median: {median_val:.2f}, Skewness: {skew_val:.2f}")
print(f"Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
print(f"Upper Fence: {upper_fence:.2f}, Outliers Above: {outliers_above}")
print(f"Lower Fence: {lower_fence:.2f}, Outliers Below: {outliers_below}")

# Correlation matrix
corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr_matrix = completed[corr_cols].corr()
print("\nCorrelation Matrix:\n", corr_matrix)