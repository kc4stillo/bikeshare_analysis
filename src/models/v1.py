# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from xgboost import XGBRegressor

# %%
df = pd.read_csv("../../data/cleaned/combined_datasets/v1/ml_dataset_v1.csv")

feature_cols = [
    "total_docks",
    "ebs_station",
    "transit_nearby",
    "jobs_nearby",
    "housing_nearby",
    "low_income_access_score",
    "amenities_nearby",
    "park_area_nearby",
    "bike_infra_score",
    "retail_nearby",
    "nearest_station_dist_m",
    "stations_within_500m",
    "is_ut",
]

X = df[feature_cols]
y = df["trips_per_dock"]

# %%
kf = KFold(n_splits=10, shuffle=True, random_state=42)

xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}

grid = GridSearchCV(
    estimator=xgb, param_grid=param_grid, cv=kf, scoring="r2", n_jobs=-1
)

grid.fit(X, y)
best_xgb = grid.best_estimator_

print("Best Parameters:")
print(grid.best_params_)
print(f"Best CV R²: {grid.best_score_:.3f}")

# %%
r2_scores = cross_val_score(best_xgb, X, y, cv=kf, scoring="r2")
mae_scores = -cross_val_score(best_xgb, X, y, cv=kf, scoring="neg_mean_absolute_error")
rmse_scores = np.sqrt(
    -cross_val_score(best_xgb, X, y, cv=kf, scoring="neg_mean_squared_error")
)

print("\n10-Fold XGBoost Results")
print("-" * 40)
print("R² scores by fold:")
print(np.round(r2_scores, 3))
print(f"Mean R²: {r2_scores.mean():.3f}")
print(f"Std R²:  {r2_scores.std():.3f}")
print()

print("MAE by fold:")
print(np.round(mae_scores, 3))
print(f"Mean MAE: {mae_scores.mean():.3f}")
print(f"Std MAE:  {mae_scores.std():.3f}")
print()

print("RMSE by fold:")
print(np.round(rmse_scores, 3))
print(f"Mean RMSE: {rmse_scores.mean():.3f}")
print(f"Std RMSE:  {rmse_scores.std():.3f}")

# %%
best_xgb.fit(X, y)

importance_df = pd.DataFrame(
    {"feature": X.columns, "importance": best_xgb.feature_importances_}
).sort_values("importance", ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Features to add:
# dorm proximity
# campus building density
# student-serving retail/food
# major academic/building clusters
# Land-use mix
# not just counts of jobs/housing/retail, but whether an area has a balanced mix
# Transit quality
# not just stops nearby, but number of routes / high-frequency service
# Walkability / connectivity
# intersection density
# street connectivity
# block size
# Temporal/context features
# semester vs break
# weekday/weekend
# season
# event areas

# INSTEAD OF ACTIVE DATE, PULL DATA FROM PUBLICLY AVALIABLE CSV TO CALCULATE TOTALS

# %%
# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
importance_df = pd.DataFrame(
    {"feature": X.columns, "importance": best_xgb.feature_importances_}
).sort_values("importance", ascending=True)

# %%
plt.figure(figsize=(8, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()

# %%
df = pd.read_csv("../../data/cleaned/combined_datasets/v1/ml_dataset_v1.csv")

# only do this if station names are in the file
# if not, ignore this section

X = df[feature_cols]
y = df["trips_per_dock"]
names = df["name"]  # only if this column exists

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, names, test_size=0.2, random_state=42
)

best_xgb.fit(X_train, y_train)
y_pred = best_xgb.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)

for actual, pred, name in zip(y_test, y_pred, names_test):
    plt.annotate(name, (actual, pred), fontsize=8, alpha=0.8)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual trips_per_dock")
plt.ylabel("Predicted trips_per_dock")
plt.title("Actual vs Predicted Trips per Dock (XGBoost)")
plt.tight_layout()
plt.show()
