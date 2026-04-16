# %%
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# %%
# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("../../data/g_transit/clean/stations.csv")

ut = [
    "dean_keeton_park_place",
    "deen_keeton_whitis",
    "dean_keeton_robert_dedman_dr",
    "dean_keeton_whitisdean_keeton_speedway",
    "dean_keeton_whitis",
    "e_21st_speedway_at_pcl",
    "e_23rd_san_jacinto_at_dkr_stadium",
    "guadalupe_west_mall_at_university_co-op",
    "w_21st_guadalupe",
    "w_21st_university",
    "w_225_rio_grande",
    "w_22nd_pearl",
    "w_23rd_san_gabriel",
    "w_26th_nueces",
    "w_28th_rio_grande",
]

# keep station names for the residual table
# station_names = df.loc[~df["name"].isin(ut), "name"]
station_names = df["name"]
# df = df[~df["name"].isin(ut)]


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
        "count_undergrad",
        "count_grad",
        "bikeable_infrastructure",
        # "median_age",
        # "median_income",
        # "count_population",
        # "count_undergrad",
        # "count_grad",
        # "population_density",
        # "undergrad_percentage",
        # "grad_percentage",
        "west_campus_area_within_275m",
        "west_campus_area_within_550m",
        "distance_to_west_campus_m",
        # "distance_to_ut_m",
        # "ut_area_within_275m",
        # "ut_area_within_550m",
        # "nearest_dining_hall_m",
        # "west_campus_area_within_275m",
        # "west_campus_area_within_550m",
        "nearest_dining_hall_m",
        # "nearest_retail_m",
        # "count_amenities_275m",
        "nearest_bikeshare_station_m",
        # "jobs_count_within_550m",
        # "jobs_count_within_275m"
    ]
)

ml_df.columns

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
    X, y_log, station_names, test_size=0.2, random_state=56
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
# manual loop so feature_weights are applied in every fold
# -----------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in cv.split(X):
    X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
    y_cv_train, y_cv_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

    cv_model = clone(model)
    cv_model.fit(X_cv_train, y_cv_train)

    cv_score = cv_model.score(X_cv_val, y_cv_val)  # R² on log scale
    cv_scores.append(cv_score)

cv_scores = np.array(cv_scores)

print("\n5-Fold CV R² Scores (log target scale)")
print("-" * 40)
print(np.round(cv_scores, 3))
print(f"Mean CV R²: {cv_scores.mean():.3f}")
print(f"Std CV R² : {cv_scores.std():.3f}")

# %%
# %%
# -----------------------------
# Residual table for test set
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

print("\nResidual Table (Test Set)")
print("-" * 90)
print(residual_table.to_string(index=False))


# %%
# save trained model
joblib.dump(model, "v8_general.pkl")

# save exact training column order
with open("v8_general.json", "w") as f:
    json.dump(X.columns.tolist(), f)
