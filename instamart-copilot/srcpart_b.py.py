import os
import sqlite3
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Paths
CLEAN_DIR = os.path.join("data", "clean")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load clean datasets
orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))
customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))

# Ensure datetime types
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Filter for completed orders
completed_orders = orders[orders["status"] == "completed"].copy()
# =====================================================================
# TASK 1: Build the RFM Table
# =====================================================================
snapshot = pd.to_datetime("2025-10-01 00:00:00")

# Group completed orders to calculate RFM metrics
rfm = completed_orders.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Calculate recency in days relative to the snapshot date
rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print("--- TASK 1 RFM SUMMARY ---")
print(f"Total customers: {len(rfm)}")
print(f"Recency min/median/max: {rfm['recency'].min()} / {rfm['recency'].median()} / {rfm['recency'].max()}")
print(f"Frequency mean/max: {rfm['frequency'].mean():.2f} / {rfm['frequency'].max()}")
print(f"Monetary mean/median/max: {rfm['monetary'].mean():.2f} / {rfm['monetary'].median():.2f} / {rfm['monetary'].max():.2f}")
# =====================================================================
# TASK 2: K-Means Clustering & K Selection
# =====================================================================
X_rfm = rfm[["recency", "frequency", "monetary"]]
scaler_rfm = StandardScaler()
X_rfm_scaled = scaler_rfm.fit_transform(X_rfm)

print("\n--- TASK 2 K-MEANS EVALUATION ---")
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_rfm_scaled)
    sil = silhouette_score(X_rfm_scaled, labels)
    print(f"K={k}: Inertia = {kmeans.inertia_:.1f} / Silhouette = {sil:.3f}")

# Fit final model with optimal K=4 based on evaluation metrics
best_kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["cluster"] = best_kmeans.fit_predict(X_rfm_scaled)
print("\nOptimal K=4 model successfully trained and applied to RFM table.")
# =====================================================================
# TASK 3: Segment Profiles, Premium Flag, and PCA
# =====================================================================
import os

# Define output directory
OUTPUT_DIR = "data/clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

final_k = 4
kmeans_final = KMeans(n_clusters=final_k, n_init=10, random_state=42)
rfm["segment_id"] = kmeans_final.fit_predict(X_rfm_scaled)

# Segment profiling
profiles = rfm.groupby("segment_id").agg(
    count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).reset_index()

# Mark highest mean monetary as Premium
premium_seg_id = profiles.loc[profiles["mean_monetary"].idxmax(), "segment_id"]

def name_segment(row):
    # Match the row against its profile segment traits
    seg_profile = profiles[profiles["segment_id"] == row["segment_id"]].iloc[0]
    if row["segment_id"] == premium_seg_id:
        return "Premium VIPs"
    elif seg_profile["mean_frequency"] > 15:
        return "High-Frequency Regulars"
    elif seg_profile["mean_recency"] > 30:
        return "At-Risk Buyers"
    else:
        return "Casual Buyers"

rfm["segment_name"] = rfm.apply(name_segment, axis=1)
rfm["is_premium"] = (rfm["segment_id"] == premium_seg_id).astype(int)

# PCA Projection
pca = PCA(n_components=2, random_state=42)
pca.fit(X_rfm_scaled)
print(f"\nPCA Explained Variance Ratios: {pca.explained_variance_ratio_}")
print(f"Total PCA Explained Variance: {pca.explained_variance_ratio_.sum():.3f}")

# Save customer segments dataset
output_path = os.path.join(OUTPUT_DIR, "customer_segments.csv")
rfm.to_csv(output_path, index=False)
print(f"Customer segments saved successfully to: {output_path}")
# =====================================================================
# TASK 4: Define the Lapse Prediction Dataset (Cutoff: 2025-09-01)
# =====================================================================
CLEAN_DIR = "data/clean"

cutoff = pd.to_datetime("2025-09-01 00:00:00")

# Pre-cutoff and September datasets
pre_cutoff_orders = orders[pd.to_datetime(orders["order_ts"]) < cutoff].copy()
sept_completed = orders[
    (orders["status"] == "completed") & 
    (pd.to_datetime(orders["order_ts"]) >= cutoff) & 
    (pd.to_datetime(orders["order_ts"]) <= pd.to_datetime("2025-09-30 23:59:59"))
]

# Define cohort: customers with >= 1 completed order before cutoff
cohort_cust = pre_cutoff_orders[pre_cutoff_orders["status"] == "completed"]["customer_id"].unique()

# Label: 1 if NO completed order in September 2025, else 0
sept_active_cust = set(sept_completed["customer_id"].unique())
cohort_df = pd.DataFrame({"customer_id": cohort_cust})
cohort_df["lapsed"] = cohort_df["customer_id"].apply(lambda cid: 1 if cid not in sept_active_cust else 0)

# Merge products for category analysis
merged_pre = pre_cutoff_orders.merge(pd.read_csv(os.path.join(CLEAN_DIR, "products.csv")), on="product_id", how="left")

# Feature Engineering
pre_completed = merged_pre[merged_pre["status"] == "completed"]

feat_rfm = pre_completed.groupby("customer_id").agg(
    recency=("order_ts", lambda x: (cutoff - pd.to_datetime(x).max()).days),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
    fav_category=("category", lambda x: x.mode()[0] if not x.empty else "Grocery")
).reset_index()

# Cancellation rate calculation
all_status_counts = pre_cutoff_orders.groupby(["customer_id", "status"])["order_id"].nunique().unstack(fill_value=0)
all_status_counts["total_orders"] = all_status_counts.get("completed", 0) + all_status_counts.get("cancelled", 0)
all_status_counts["cancellation_rate"] = all_status_counts.get("cancelled", 0) / all_status_counts["total_orders"]
cancellation_df = all_status_counts[["cancellation_rate"]].reset_index()

# Combine features with customers metadata
dataset = cohort_df.merge(feat_rfm, on="customer_id", how="left")
dataset = dataset.merge(cancellation_df, on="customer_id", how="left")
dataset = dataset.merge(customers, on="customer_id", how="left")

dataset["tenure_days"] = (cutoff - pd.to_datetime(dataset["signup_date"])).dt.days
dataset["cancellation_rate"] = dataset["cancellation_rate"].fillna(0.0)

# Feature selection
features = [
    "recency", "frequency", "monetary", "avg_discount", "cancellation_rate",
    "avg_delivery", "fav_category", "city", "age", "gender", "tenure_days"
]
X = dataset[features]
y = dataset["lapsed"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\n--- TASK 4 DATASET DEFINED ---")
print(f"Total Cohort Size: {len(dataset)} customers")
print(f"Training set dimensions: {X_train.shape}")
print(f"Testing set dimensions: {X_test.shape}")
print(f"Lapse Class Balance (0=Active, 1=Lapsed):\n{y.value_counts(normalize=True)}")
# =====================================================================
# TASK 5 & 6: Preprocessing, Model Tuning & Evaluation
# =====================================================================
cat_cols = ["fav_category", "city", "gender"]
num_cols = ["recency", "frequency", "monetary", "avg_discount", "cancellation_rate", "avg_delivery", "age", "tenure_days"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)

# 1. Logistic Regression Grid Search
pipe_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Define list values step-by-step to avoid syntax stripping
lr_c_list = [0.01, 0.1, 1, 10, 100]
param_grid_lr = {"classifier__C": lr_c_list}

grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="f1")
grid_lr.fit(X_train, y_train)

# 2. Random Forest Grid Search
pipe_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Define lists explicitly as variables
rf_depths = [3, 5, 8]
rf_leaves = [5, 10, 20]

param_grid_rf = {
    "classifier__max_depth": rf_depths,
    "classifier__min_samples_leaf": rf_leaves
}

grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="f1")
grid_rf.fit(X_train, y_train)

print(f"\nBest Logistic Regression C: {grid_lr.best_params_['classifier__C']} (CV F1: {grid_lr.best_score_:.3f})")
print(f"Best Random Forest Params: {grid_rf.best_params_} (CV F1: {grid_rf.best_score_:.3f})")

# Evaluate on untouched test set
y_pred_rf = grid_rf.predict(X_test)
rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
print(f"\nRandom Forest Test F1 (Lapsed Class): {rf_f1:.3f}")

# Save best model to disk
import joblib
model_output_path = os.path.join(OUTPUT_DIR, "lapse_model.pkl")
joblib.dump(grid_rf.best_estimator_, model_output_path)
print(f"Best tuned Random Forest model saved successfully to: {model_output_path}")
# =====================================================================
# TASK 7 & 8: Oversampling Experiment & Retention List Export
# =====================================================================

# Task 7: Oversample training set only
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

train_df = pd.DataFrame(X_train_prep)
train_df["target"] = y_train.values

lapsed_rows = train_df[train_df["target"] == 1]
non_lapsed_rows = train_df[train_df["target"] == 0]

# Sample lapsed rows with replacement to match non-lapsed count
lapsed_oversampled = lapsed_rows.sample(n=len(non_lapsed_rows), replace=True, random_state=42)
balanced_train = pd.concat([non_lapsed_rows, lapsed_oversampled], axis=0)

X_train_bal = balanced_train.drop(columns=["target"])
y_train_bal = balanced_train["target"]

lr_bal = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_bal.fit(X_train_bal, y_train_bal)

y_pred_bal = lr_bal.predict(X_test_prep)
print(f"\nBalanced Logistic Regression Recall: {recall_score(y_test, y_pred_bal):.3f}")
print(f"Balanced Logistic Regression Precision: {precision_score(y_test, y_pred_bal):.3f}")

# Task 8: Generate Retention List for Premium Segment in Test Set
test_cust = dataset.loc[X_test.index].copy()

# FIXED: Changed best_model to grid_rf.best_estimator_
test_cust["lapse_probability"] = grid_rf.best_estimator_.predict_proba(X_test)[:, 1]

# Merge segment info from Task 3
if "is_premium" in test_cust.columns:
    test_cust = test_cust.drop(columns=["is_premium"])

test_cust = test_cust.merge(rfm[["customer_id", "is_premium"]], on="customer_id", how="left")

premium_test = test_cust[test_cust["is_premium"] == 1].sort_values("lapse_probability", ascending=False)
retention_list = premium_test[["customer_id", "lapse_probability", "lapsed"]].rename(columns={"lapsed": "actually_lapsed"}).head(10)

retention_list.to_csv(os.path.join(OUTPUT_DIR, "retention_list.csv"), index=False)
print("\n--- RETENTION LIST TOP ACCOUNTS ---")
print(retention_list)
