# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

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
]

X = df[feature_cols]
y = df["trips_per_dock"]

# %%
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# %%
rf = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring="r2", n_jobs=-1)

grid.fit(X, y)

best_rf = grid.best_estimator_

print("Best Parameters:")
print(grid.best_params_)
print(f"Best CV R²: {grid.best_score_:.3f}")

# %%
r2_scores = cross_val_score(best_rf, X, y, cv=kf, scoring="r2")
mae_scores = -cross_val_score(best_rf, X, y, cv=kf, scoring="neg_mean_absolute_error")
rmse_scores = np.sqrt(
    -cross_val_score(best_rf, X, y, cv=kf, scoring="neg_mean_squared_error")
)

print("\n10-Fold Random Forest Results")
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
# fit on full dataset for feature importance
best_rf.fit(X, y)

importance_df = pd.DataFrame(
    {"feature": X.columns, "importance": best_rf.feature_importances_}
).sort_values("importance", ascending=False)

print("\nFeature Importances:")
print(importance_df)
