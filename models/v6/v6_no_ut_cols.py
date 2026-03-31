# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

# %%
# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("../../data/cleaned/combined_datasets/v6/ml_dataset_v6.csv")

df_with_names = pd.read_csv(
    "../../data/cleaned/combined_datasets/v6/combined_dataset_v6.csv"
)
# id	name	district	total_docks	trips_per_dock	ebs_station	is_ut	lat	lon	transit_nearby	...	ut_x_dorm_pop_500m	ut_x_dining_dist	ut_x_transit	ut_x_housing_275m	ut_x_in_ut_shape	ut_x_in_west_campus_shape	ut_x_in_north_campus_shape	ut_x_ut_shape_share_275m	ut_x_west_campus_share_275m	ut_x_north_campus_share_275m
# 0	15	w_28th_rio_grande	9	7	3299.714286	0	1	30.293330	-97.744120	3	...	588.0	521.29	3	811.0	0	1	0	0.0000	0.9430	0.0770
# 1	10	w_22_5_rio_grande	9	5	2588.400000	0	1	30.286200	-97.745160	1	...	980.0	704.37	1	1654.0	0	1	0	0.0102	1.0000	0.0000
# 2	8	e_21st_speedway_pcl	9	17	2579.470588	0	1	30.283000	-97.737500	2	...	5103.0	64.92	2	0.0	1	0	0	0.9866	0.0517	0.0000
# 3	14	w_26th_nueces	9	13	2532.307692	0	1	30.290680	-97.742920	5	...	2172.0	302.53	5	1053.0	0	1	0	0.1929	0.8171	0.0241
# 4	12	w_23rd_san_gabriel	9	11	2505.454545	0	1	30.287400	-97.747800	2	...	0.0	844.95	2	508.0	0	1	0	0.0000	1.0000	0.0000
# ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
# 67	67	rosewood_angelina	1	11	104.818182	0	0	30.268880	-97.724310	3	...	0.0	0.00	0	0.0	0	0	0	0.0000	0.0000	0.0000
# 68	4	e_11th_san_jacinto	9	11	100.545455	0	0	30.271930	-97.738540	3	...	0.0	0.00	0	0.0	0	0	0	0.0000	0.0000	0.0000
# 69	5	e_12th_san_jacinto_state_cap_visitors_garage	9	11	99.363636	0	0	30.273499	-97.738097	4	...	0.0	0.00	0	0.0	0	0	0	0.0000	0.0000	0.0000
# 70	34	e_8th_trinity	9	15	95.666667	0	0	30.268956	-97.738686	6	...	0.0	0.00	0	0.0	0	0	0	0.0000	0.0000	0.0000
# 71	46	e_11th_waller	1	11	84.636364	0	0	30.268998	-97.728434	3	...	0.0	0.00	0	0.0	0	0	0	0.0000	0.0000	0.0000
# 72 rows × 71 columns

# %%
# ----------------------------
# Define features and target
# ----------------------------
feature_cols = [
    "total_docks",
    "ebs_station",
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
    X, y_log, y, test_size=0.5, random_state=21
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
print(
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
    ]
    .head(10)
    .to_string(index=False)
)

print("\nLeast Accurate Predictions")
print("-" * 60)
print(
    least_accurate[
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
    ]
    .head(10)
    .to_string(index=False)
)

# Most Accurate Predictions
# ------------------------------------------------------------
#                              name  district     actual  predicted      error  abs_error  pct_error  is_ut
#                    w_4th_congress         9 360.333333 358.074585  -2.258748   2.258748   0.626850      0
#                   e_8th_red_river         9 186.909091 192.386337   5.477246   5.477246   2.930433      0
# electric_drive_pfluger_ped_bridge         9 409.315789 421.811310  12.495520  12.495520   3.052782      0
#                      e_6th_chicon         3 171.636364 151.270432 -20.365932  20.365932  11.865744      0
# w_11th_congress_the_texas_capitol         9 283.090909 249.955765 -33.135144  33.135144  11.704772      0
#         hollow_creek_barton_hills         5 277.000000 241.629166 -35.370834  35.370834  12.769254      0
#                      w_6th_lavaca         9 259.428571 216.094543 -43.334028  43.334028  16.703645      0
#             cesar_chavez_congress         9 335.909091 281.517151 -54.391940  54.391940  16.192458      0
#              south_congress_james         9 222.285714 149.512360 -72.773355  72.773355  32.738656      0
#     south_congress_barton_springs         9 363.818182 443.222229  79.404047  79.404047  21.825200      0

# Least Accurate Predictions
# ------------------------------------------------------------
#                                 name  district      actual   predicted        error   abs_error  pct_error  is_ut
#                    w_28th_rio_grande         9 3299.714286  598.945679 -2700.768607 2700.768607  81.848559      1
#                  e_21st_speedway_pcl         9 2579.470588  578.769653 -2000.700935 2000.700935  77.562464      1
#                        w_26th_nueces         9 2532.307692 1508.296387 -1024.011306 1024.011306  40.437871      1
#                         w_22nd_pearl         9 1062.130435 1998.931519   936.801084  936.801084  88.200192      1
#                    w_21st_university         9 1332.315789  507.670074  -824.645715  824.645715  61.895665      1
# guadalupe_west_mall_university_co_op         9 1244.933333 1876.377319   631.443986  631.443986  50.721108      1
#                       e_5th_broadway         3  239.533333  805.844116   566.310783  566.310783 236.422537      0
#                      rainey_cummings         9  800.263158  254.810013  -545.453145  545.453145  68.159222      0
#                    w_22_5_rio_grande         9 2588.400000 2058.618896  -529.781104  529.781104  20.467513      1
#                  south_congress_mary         9  271.727273  761.518127   489.790855  489.790855 180.250900      0
