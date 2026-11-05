import os
import sqlite3
import numpy as np
import pandas as pd

# Define paths
RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
DB_PATH = os.path.join("data", "instamart.db")
os.makedirs(CLEAN_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. AUDIT RAW DATA
# ---------------------------------------------------------
# Note: keep_default_na=False ensures empty string tickets stay empty strings
customers_raw = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
products_raw = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
orders_raw = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
tickets_raw = pd.read_csv(
    os.path.join(RAW_DIR, "tickets.csv"), keep_default_na=False
)

# Raw Audit Calculations
raw_orders_rows = len(orders_raw)
exact_duplicates = orders_raw.duplicated().sum()
pending_rows = (orders_raw["status"] == "pending").sum()
distinct_order_ids = orders_raw["order_id"].nunique()

non_numeric_amounts = pd.to_numeric(
    orders_raw["order_amount"], errors="coerce"
).isna().sum()
deliv_numeric = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
negative_deliv = (deliv_numeric < 0).sum()
blank_deliv = orders_raw["delivery_minutes"].isna().sum()

blank_ages = customers_raw["age"].isna().sum()
cust_age_num = pd.to_numeric(customers_raw["age"], errors="coerce")
out_of_range_ages = ((cust_age_num < 10) | (cust_age_num > 100)).sum()
distinct_city_spellings = customers_raw["city"].nunique()

# Gender audit
non_blank_genders = customers_raw["gender"].dropna()
distinct_gender_spellings = non_blank_genders.nunique()
blank_genders = customers_raw["gender"].isna().sum()

print("--- RAW AUDIT CHECKPOINTS ---")
print(f"raw orders.csv rows: {raw_orders_rows}")
print(f"exact duplicate rows: {exact_duplicates}")
print(f"pending status rows: {pending_rows}")
print(f"distinct order_id values: {distinct_order_ids}")
print(f"non-numeric amounts: {non_numeric_amounts}")
print(f"negative delivery_minutes: {negative_deliv}")
print(f"blank delivery_minutes: {blank_deliv}")
print(f"blank age count: {blank_ages}")
print(f"ages outside 10-100: {out_of_range_ages}")
print(f"distinct raw city spellings: {distinct_city_spellings}")
print(f"distinct non-blank raw gender spellings: {distinct_gender_spellings}")
print(f"blank gender count: {blank_genders}")

# ---------------------------------------------------------
# 2. CLEAN CUSTOMERS
# ---------------------------------------------------------
customers_clean = customers_raw.copy()

# (a) Standardize city: strip whitespace and title case
customers_clean["city"] = customers_clean["city"].astype(str).str.strip().str.title()

# (b) Map gender
def map_gender(g):
    if pd.isna(g) or str(g).strip() == "":
        return "unknown"
    g_str = str(g).strip().lower()
    if g_str in ["male", "m"]:
        return "male"
    elif g_str in ["female", "f"]:
        return "female"
    return "unknown"

customers_clean["gender"] = customers_clean["gender"].apply(map_gender)

# (c) Handle ages: Treat outside 10-100 as missing, fill missing with city median
customers_clean["age"] = pd.to_numeric(customers_clean["age"], errors="coerce")
customers_clean.loc[
    (customers_clean["age"] < 10) | (customers_clean["age"] > 100), "age"
] = np.nan
customers_clean["age"] = customers_clean.groupby("city")["age"].transform(
    lambda x: x.fillna(x.median())
)

# (d) Convert signup_date to datetime
customers_clean["signup_date"] = pd.to_datetime(customers_clean["signup_date"])

# ---------------------------------------------------------
# 3. CLEAN ORDERS
# ---------------------------------------------------------
orders_clean = orders_raw.copy()

# Step 1: Convert order_ts to datetime
orders_clean["order_ts"] = pd.to_datetime(orders_clean["order_ts"])

# Step 2: Remove exact duplicate rows
orders_clean = orders_clean.drop_duplicates()

# Step 3: Deduplicate order_id keeping latest order_ts
orders_clean = orders_clean.sort_values("order_ts").groupby("order_id", as_index=False).last()

# Step 4: Clean order_amount and fill missing from catalogue price
orders_clean["order_amount"] = pd.to_numeric(orders_clean["order_amount"], errors="coerce")

# Merge product prices for imputation
orders_clean = orders_clean.merge(products_raw[["product_id", "category", "price_inr"]], on="product_id", how="left")

missing_amt_mask = orders_clean["order_amount"].isna()
orders_clean.loc[missing_amt_mask, "order_amount"] = (
    orders_clean.loc[missing_amt_mask, "price_inr"] * 
    orders_clean.loc[missing_amt_mask, "quantity"] * 
    (1 - orders_clean.loc[missing_amt_mask, "discount_pct"] / 100.0)
).round(2)

# Step 5: Clean delivery_minutes
orders_clean["delivery_minutes"] = pd.to_numeric(orders_clean["delivery_minutes"], errors="coerce")

# Set negative values to missing
orders_clean.loc[orders_clean["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

# Impute completed order missing delivery_minutes with category median among completed orders
completed_mask = orders_clean["status"] == "completed"
category_medians = orders_clean[completed_mask].groupby("category")["delivery_minutes"].transform("median")

orders_clean.loc[completed_mask, "delivery_minutes"] = orders_clean.loc[completed_mask, "delivery_minutes"].fillna(category_medians)
# Cancelled orders are deliberately left missing (NaN)

# ---------------------------------------------------------
# 4. STORE TO CSV AND SQLITE
# ---------------------------------------------------------
# Drop merged temporary column
products_clean = products_raw.copy()
orders_clean_export = orders_clean.drop(columns=["category", "price_inr"])

# Format dates to string for CSV/SQLite export
customers_clean_export = customers_clean.copy()
customers_clean_export["signup_date"] = customers_clean_export["signup_date"].dt.strftime("%Y-%m-%d")

orders_clean_sqlite = orders_clean_export.copy()
orders_clean_sqlite["order_ts"] = orders_clean_sqlite["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Export CSVs
customers_clean_export.to_csv(os.path.join(CLEAN_DIR, "customers.csv"), index=False)
products_clean.to_csv(os.path.join(CLEAN_DIR, "products.csv"), index=False)
orders_clean_sqlite.to_csv(os.path.join(CLEAN_DIR, "orders.csv"), index=False)

# Export to SQLite database
conn = sqlite3.connect(DB_PATH)
customers_clean_export.to_sql("customers", conn, if_exists="replace", index=False)
products_clean.to_sql("products", conn, if_exists="replace", index=False)
orders_clean_sqlite.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()

# ---------------------------------------------------------
# 5. VALIDATION SCRIPT ASSERTIONS
# ---------------------------------------------------------
assert len(orders_clean_export) == 12309
assert (orders_clean_export["status"] == "completed").sum() == 11110
assert (orders_clean_export["status"] == "cancelled").sum() == 1199
assert (orders_clean_export["status"] == "pending").sum() == 0
assert orders_clean_export["order_amount"].isna().sum() == 0
assert orders_clean_export[orders_clean_export["status"] == "completed"]["delivery_minutes"].isna().sum() == 0
assert orders_clean_export[orders_clean_export["status"] == "cancelled"]["delivery_minutes"].isna().sum() == 1199

completed_rev = orders_clean_export[orders_clean_export["status"] == "completed"]["order_amount"].sum()
assert round(completed_rev, 2) == 2998668.80

print("\nALL CHECKPOINTS PASSED")