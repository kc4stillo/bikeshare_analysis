# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

# %%
# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("../../data/cleaned/combined_datasets/v6/ml_dataset_v6.csv")

df_with_names = pd.read_csv(
    "../../data/cleaned/combined_datasets/v6/combined_dataset_v6.csv"
)
# %%
# ----------------------------
# Define features and target
# ----------------------------
feature_cols = [
    "total_docks",
    "ebs_station",
    "is_ut",
    "lat",
    "lon",
    "transit_nearby",
    "nearest_transit_stop_dist_m",
    "avg_dist_3_nearest_transit_stops_m",
    "jobs_nearby_275m",
    "housing_nearby_275m",
    "housing_nearby_1000m",
    "job_housing_ratio_275m",
    "low_income_access_score",
    "amenities_nearby",
    "avg_dist_3_nearest_amenities_m",
    "park_area_nearby",
    "nearest_park_dist_m",
    "bike_infra_score",
    "retail_nearby",
    "avg_dist_3_nearest_retail_m",
    "entertainment_nearby",
    "avg_dist_3_nearest_entertainment_m",
    "tourism_nearby",
    "avg_dist_3_nearest_tourism_m",
    "nearest_station_dist_m",
    "stations_within_500m",
    "stations_within_1000m",
    "avg_stations_dist_3_nearest_m",
    "nearest_dining_hall_dist_m",
    "nearest_dorm_dist_m",
    "nearest_dorm_pop",
    "dorm_pop_within_500m",
    "dist_to_west_campus_center_m",
    "in_ut_shape",
    "dist_to_ut_shape_m",
    "ut_shape_area_within_275m",
    "ut_shape_share_of_275m_buffer",
    "ut_shape_touches_275m_buffer",
    "ut_shape_area_within_500m",
    "ut_shape_share_of_500m_buffer",
    "ut_shape_touches_500m_buffer",
    "in_west_campus_shape",
    "dist_to_west_campus_shape_m",
    "west_campus_shape_area_within_275m",
    "west_campus_shape_share_of_275m_buffer",
    "west_campus_shape_touches_275m_buffer",
    "west_campus_shape_area_within_500m",
    "west_campus_shape_area_within_1000m",
    "west_campus_shape_share_of_500m_buffer",
    "west_campus_shape_touches_500m_buffer",
    "in_north_campus_shape",
    "dist_to_north_campus_shape_m",
    "north_campus_shape_area_within_275m",
    "north_campus_shape_share_of_275m_buffer",
    "north_campus_shape_touches_275m_buffer",
    "north_campus_shape_area_within_500m",
    "north_campus_shape_share_of_500m_buffer",
    "north_campus_shape_touches_500m_buffer",
    "ut_x_dorm_pop_500m",
    "ut_x_dining_dist",
    "ut_x_transit",
    "ut_x_housing_275m",
    "ut_x_in_ut_shape",
    "ut_x_in_west_campus_shape",
    "ut_x_in_north_campus_shape",
    "ut_x_ut_shape_share_275m",
    "ut_x_west_campus_share_275m",
    "ut_x_north_campus_share_275m",
]

target_col = "trips_per_dock"

X = df[feature_cols].copy()
y = df[target_col].copy()

# %%
# ----------------------------
# Clean weird values
# ----------------------------
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# %%
# ----------------------------
# Log-transform target
# ----------------------------
y_log = np.log1p(y)

# %%
# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train_log, y_test_log, y_train_orig, y_test_orig = train_test_split(
    X, y_log, y, test_size=0.2, random_state=42
)

# %%
# ----------------------------
# Build XGBoost model
# ----------------------------
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
)

# %%
# ----------------------------
# Fit on log target
# ----------------------------
model.fit(X_train, y_train_log)

# %%
# ----------------------------
# Predict on log scale
# ----------------------------
y_pred_log = model.predict(X_test)

# Convert predictions back to original scale
y_pred_orig = np.expm1(y_pred_log)

# %%
# ----------------------------
# Evaluate on original scale
# ----------------------------
r2 = r2_score(y_test_orig, y_pred_orig)
mae = mean_absolute_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

print("XGBoost Test Results (trained on log target, evaluated on original scale)")
print("-" * 70)
print(f"R²   : {r2:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")

# %%
# ----------------------------
# Cross-validation on log scale
# ----------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2_log = cross_val_score(model, X, y_log, cv=cv, scoring="r2")

print("\n5-Fold CV R² Scores (log target scale)")
print("-" * 40)
print(np.round(cv_r2_log, 3))
print(f"Mean CV R²: {cv_r2_log.mean():.3f}")
print(f"Std CV R² : {cv_r2_log.std():.3f}")

# %%
# ----------------------------
# Feature importance
# ----------------------------
importance_df = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

print("\nTop Features")
print("-" * 40)
print(importance_df.to_string(index=False))

# %%
# ----------------------------
# Station-level prediction accuracy on test set
# ----------------------------

# keep the test-set row indices so we can recover station names
X_train, X_test, y_train_log, y_test_log, y_train_orig, y_test_orig = train_test_split(
    X, y_log, y, test_size=0.5, random_state=21
)

# Fit model
model.fit(X_train, y_train_log)

# Predict
y_pred_log = model.predict(X_test)
y_pred_orig = np.expm1(y_pred_log)

# Build results dataframe
results_df = df_with_names.loc[
    X_test.index, ["id", "name", "district", "trips_per_dock", "is_ut"]
].copy()
results_df["actual"] = y_test_orig.values
results_df["predicted"] = y_pred_orig
results_df["error"] = results_df["predicted"] - results_df["actual"]
results_df["abs_error"] = np.abs(results_df["error"])

# optional: percent error
results_df["pct_error"] = np.where(
    results_df["actual"] != 0,
    results_df["abs_error"] / results_df["actual"] * 100,
    np.nan,
)

# Most accurate predictions = smallest absolute error
most_accurate = results_df.sort_values("abs_error", ascending=True)

# Least accurate predictions = largest absolute error
least_accurate = results_df.sort_values("abs_error", ascending=False)

print("\nMost Accurate Predictions")
print("-" * 60)

most_accurate[
    [
        "name",
        "district",
        "actual",
        "predicted",
        "error",
        "abs_error",
        "pct_error",
        "is_ut",
    ]
].sort_values("abs_error")
