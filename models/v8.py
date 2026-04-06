# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# %%
# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("../data/g_transit/clean/stations.csv")

# keep station names for the residual table
station_names = df["name"]

# model dataset
ml_df = df.drop(
    columns=[
        "name",
        "trips",
        "lat",
        "lon",
        "north_campus_area_within_275m",
        "north_campus_area_within_550m",
        "west_campus_area_within_825m",
        "north_campus_area_within_550m",
        "undergrad",
        "grad",
    ]
)

# target on original scale
y = ml_df["trips_per_dock"]

# features
X = ml_df.drop(columns=["trips_per_dock"])

# log target for training
y_log = np.log1p(y)

# %%
# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train_log, y_test_log, names_train, names_test = train_test_split(
    X, y_log, station_names, test_size=0.2, random_state=21
)

# original-scale y for evaluation
y_train_orig = np.expm1(y_train_log)
y_test_orig = np.expm1(y_test_log)

# %%
# -----------------------------
# Fit XGBoost model
# -----------------------------
model = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
)

model.fit(X_train, y_train_log)

# %%
# -----------------------------
# Predict
# -----------------------------
y_pred_log = model.predict(X_test)
y_pred_orig = np.expm1(y_pred_log)

# %%
# -----------------------------
# Metrics on original scale
# -----------------------------
r2 = r2_score(y_test_orig, y_pred_orig)
mae = mean_absolute_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

print("XGBoost Test Results (trained on log target, evaluated on original scale)")
print("-" * 70)
print(f"R²   : {r2:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")

# %%
# -----------------------------
# 5-fold CV on log target
# -----------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X, y_log, cv=cv, scoring="r2")

print("\n5-Fold CV R² Scores (log target scale)")
print("-" * 40)
print(np.round(cv_scores, 3))
print(f"Mean CV R²: {cv_scores.mean():.3f}")
print(f"Std CV R² : {cv_scores.std():.3f}")

# %%
# -----------------------------
# Residual table for evaluation set
# residual = actual - predicted
# -----------------------------
residual_table = pd.DataFrame(
    {
        "station": names_test.values,
        "actual_trips_per_dock": y_test_orig.values,
        "predicted_trips_per_dock": y_pred_orig,
        "actual_log1p": y_test_log.values,
        "predicted_log1p": y_pred_log,
    }
)

residual_table["residual"] = (
    residual_table["actual_trips_per_dock"] - residual_table["predicted_trips_per_dock"]
)

residual_table["abs_residual"] = residual_table["residual"].abs()

residual_table["percent_error"] = np.where(
    residual_table["actual_trips_per_dock"] != 0,
    residual_table["residual"] / residual_table["actual_trips_per_dock"] * 100,
    np.nan,
)

residual_table = residual_table.sort_values("abs_residual", ascending=False)

print("\nResidual Table (Evaluation Set)")
print("-" * 70)
print(residual_table.round(3).to_string(index=False))

# %%
# -----------------------------
# Most important features
# -----------------------------
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

print("\nMost Important Features")
print("-" * 40)
print(feature_importance.to_string(index=False))
