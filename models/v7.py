# %%
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from xgboost import XGBRegressor

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

RANDOM_STATE = 42
TARGET = "trips_per_dock"

# %%
# --------------------------------------------------
# Load data
# --------------------------------------------------
total_df = pd.read_csv("../data/cleaned/combined_datasets/v7/combined_dataset_v7.csv")
df = pd.read_csv("../data/cleaned/combined_datasets/v7/ml_dataset_v7.csv")

print("combined_dataset_v7 shape:", total_df.shape)
print("ml_dataset_v7 shape      :", df.shape)

# %%
# --------------------------------------------------
# Separate features / target
# --------------------------------------------------
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

# Keep station metadata for later inspection
meta_cols = [
    col for col in ["id", "name", "district", TARGET] if col in total_df.columns
]
meta = total_df.loc[df.index, meta_cols].copy()

# %%
# --------------------------------------------------
# Drop duplicate feature columns (same values, different names)
# --------------------------------------------------
duplicate_mask = X.T.duplicated()
duplicate_cols = X.columns[duplicate_mask].tolist()

if duplicate_cols:
    print("\nDropping exact duplicate feature columns:")
    for col in duplicate_cols:
        print(f"  - {col}")

X = X.loc[:, ~duplicate_mask].copy()

# Drop constant columns
constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
if constant_cols:
    print("\nDropping constant columns:")
    for col in constant_cols:
        print(f"  - {col}")
    X = X.drop(columns=constant_cols)

# Make sure everything is numeric
X = X.apply(pd.to_numeric, errors="coerce")

print("\nFinal feature matrix shape:", X.shape)
print("Target shape:", y.shape)

# %%
# --------------------------------------------------
# Log-transform target
# log1p is safer than plain log in case of zeros
# --------------------------------------------------
y_log = np.log1p(y)

# %%
# --------------------------------------------------
# Train / validation / test split
# --------------------------------------------------
(
    X_train_full,
    X_test,
    y_train_full_log,
    y_test_log,
    y_train_full_orig,
    y_test_orig,
    meta_train_full,
    meta_test,
) = train_test_split(
    X,
    y_log,
    y,
    meta,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

(
    X_train,
    X_val,
    y_train_log,
    y_val_log,
) = train_test_split(
    X_train_full,
    y_train_full_log,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

print("\nTrain shape:", X_train.shape)
print("Val shape  :", X_val.shape)
print("Test shape :", X_test.shape)

# %%
# --------------------------------------------------
# XGBoost regressor
# Train on log target
# --------------------------------------------------
model = XGBRegressor(
    n_estimators=2000,
    max_depth=4,
    learning_rate=0.03,
    min_child_weight=3,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_alpha=0.10,
    reg_lambda=1.00,
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=50,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
)

model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
    verbose=False,
)

# %%
# --------------------------------------------------
# Test-set evaluation
# Predict on log scale, convert back to original scale
# --------------------------------------------------
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_pred = np.clip(y_pred, a_min=0, a_max=None)

r2 = r2_score(y_test_orig, y_pred)
mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))

print("\nXGBoost Test Results (trained on log target, evaluated on original scale)")
print("-" * 70)
print(f"R²   : {r2:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"Best iteration used: {model.best_iteration}")

# Also show log-scale test R²
test_r2_log = r2_score(y_test_log, y_pred_log)
print(f"Test R² on log scale: {test_r2_log:.3f}")

# %%
# --------------------------------------------------
# 5-fold cross-validation on the log target
# --------------------------------------------------
cv_model = XGBRegressor(
    n_estimators=max(model.best_iteration, 100)
    if model.best_iteration is not None
    else 300,
    max_depth=4,
    learning_rate=0.03,
    min_child_weight=3,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_alpha=0.10,
    reg_lambda=1.00,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = cross_validate(
    cv_model,
    X,
    y_log,
    cv=kf,
    scoring={
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    },
    n_jobs=-1,
)

cv_r2 = cv_results["test_r2"]
cv_mae = -cv_results["test_mae"]
cv_rmse = -cv_results["test_rmse"]

print("\n5-Fold CV Results (log target scale)")
print("-" * 70)
print("R² scores   :", np.round(cv_r2, 3))
print(f"Mean CV R²  : {cv_r2.mean():.3f}")
print(f"Std CV R²   : {cv_r2.std():.3f}")
print(f"Mean CV MAE : {cv_mae.mean():.3f}")
print(f"Mean CV RMSE: {cv_rmse.mean():.3f}")

# %%
# --------------------------------------------------
# Feature importance
# --------------------------------------------------
importance_df = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importances_,
    }
).sort_values("importance", ascending=False)

print("\nTop 40 Features")
print("-" * 70)
print(importance_df.head(40).to_string(index=False))

# %%
# --------------------------------------------------
# Test predictions by station
# --------------------------------------------------
test_results = meta_test.copy()
test_results["actual_trips_per_dock"] = y_test_orig.values
test_results["predicted_trips_per_dock"] = y_pred
test_results["predicted_log_trips_per_dock"] = y_pred_log
test_results["residual"] = (
    test_results["actual_trips_per_dock"] - test_results["predicted_trips_per_dock"]
)
test_results["abs_error"] = np.abs(test_results["residual"])
test_results = test_results.sort_values("abs_error", ascending=False)

print("\nWorst-predicted test stations")
print("-" * 70)
print(test_results.head(15).to_string(index=False))

# %%
# --------------------------------------------------
# Plot: actual vs predicted on original scale
# --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_pred, alpha=0.8)
min_val = min(y_test_orig.min(), y_pred.min())
max_val = max(y_test_orig.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Actual trips_per_dock")
plt.ylabel("Predicted trips_per_dock")
plt.title("XGBoost (log target): Actual vs Predicted")
plt.tight_layout()
plt.show()

# %%
# --------------------------------------------------
# Plot: actual vs predicted on log scale
# --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(np.log1p(y_test_orig), y_pred_log, alpha=0.8)
min_val = min(np.log1p(y_test_orig).min(), y_pred_log.min())
max_val = max(np.log1p(y_test_orig).max(), y_pred_log.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Actual log1p(trips_per_dock)")
plt.ylabel("Predicted log1p(trips_per_dock)")
plt.title("XGBoost (log target): Actual vs Predicted on Log Scale")
plt.tight_layout()
plt.show()

# %%
# --------------------------------------------------
# Plot: top 20 feature importances
# --------------------------------------------------
plot_df = importance_df.head(20).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(plot_df["feature"], plot_df["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 XGBoost Feature Importances (log target)")
plt.tight_layout()
plt.show()

# %%
# --------------------------------------------------
# Save outputs
# --------------------------------------------------
importance_df.to_csv("xgb_log_feature_importance_trips_per_dock.csv", index=False)
test_results.to_csv("xgb_log_test_predictions_trips_per_dock.csv", index=False)

print("\nSaved:")
print("- xgb_log_feature_importance_trips_per_dock.csv")
print("- xgb_log_test_predictions_trips_per_dock.csv")
