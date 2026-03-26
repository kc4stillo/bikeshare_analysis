# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from xgboost import XGBRegressor

# %%
df = pd.read_csv("../data/cleaned/combined_datasets/v4/ml_dataset_v4.csv")

# %%
# Features and target
X = df.drop(columns=["trips_per_dock", "name", "lat", "lon"])
y = np.log1p(df["trips_per_dock"])

print("X shape:", X.shape)
print("y shape:", y.shape)
print(X.columns.tolist())

# %%
# XGBoost model
model = XGBRegressor(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    objective="reg:squarederror",
    random_state=42,
)

# %%
# 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
    return_train_score=False,
)

cv_r2 = cv_results["test_r2"]
cv_mae = -cv_results["test_neg_mean_absolute_error"]
cv_rmse = -cv_results["test_neg_root_mean_squared_error"]

print("Log-target XGBoost CV R² scores:", np.round(cv_r2, 3))
print("Mean CV R²:", round(cv_r2.mean(), 3))
print("Std CV R²:", round(cv_r2.std(), 3))
print("Mean MAE:", round(cv_mae.mean(), 3))
print("Mean RMSE:", round(cv_rmse.mean(), 3))

# %%
# Final train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21
)

model.fit(X_train, y_train)

# %%
# Predictions
y_pred_log = model.predict(X_test)

# Convert back to original units
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred_log)

test_r2 = r2_score(y_test_orig, y_pred_orig)
test_mae = mean_absolute_error(y_test_orig, y_pred_orig)
test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

print("\nFinal Log-Target XGBoost Test Results (original units)")
print("--------------------------------------------------")
print("R²   :", round(test_r2, 3))
print("MAE  :", round(test_mae, 3))
print("RMSE :", round(test_rmse, 3))

# %%
# Feature importance
importance_df = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

print("\nTop 25 Features")
print(importance_df.head)

# %%
# Plot top feature importances
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["feature"].head(15)[::-1], importance_df["importance"].head(15)[::-1]
)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 XGBoost Feature Importances")
plt.tight_layout()
plt.show()

# %%
# Results dataframe
results = pd.DataFrame(
    {"actual_trips_per_dock": y_test_orig, "predicted_trips_per_dock": y_pred_orig},
    index=X_test.index,
)

results["name"] = df.loc[X_test.index, "name"]
results["residual"] = (
    results["actual_trips_per_dock"] - results["predicted_trips_per_dock"]
)
results["abs_residual"] = np.abs(results["residual"])

print("\nWorst predictions")
print(
    results.sort_values("abs_residual", ascending=False)[
        [
            "name",
            "actual_trips_per_dock",
            "predicted_trips_per_dock",
            "residual",
            "abs_residual",
        ]
    ].head(10)
)

# %%
# Actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(results["actual_trips_per_dock"], results["predicted_trips_per_dock"], s=70)

min_val = min(
    results["actual_trips_per_dock"].min(), results["predicted_trips_per_dock"].min()
)
max_val = max(
    results["actual_trips_per_dock"].max(), results["predicted_trips_per_dock"].max()
)

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Actual Trips per Dock")
plt.ylabel("Predicted Trips per Dock")
plt.title("Actual vs Predicted Trips per Dock")
plt.tight_layout()
plt.show()

# %%
# Labeled actual vs predicted
plt.figure(figsize=(10, 8))
plt.scatter(results["actual_trips_per_dock"], results["predicted_trips_per_dock"], s=80)

for _, row in results.iterrows():
    plt.annotate(
        row["name"],
        (row["actual_trips_per_dock"], row["predicted_trips_per_dock"]),
        fontsize=9,
    )

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Actual Trips per Dock")
plt.ylabel("Predicted Trips per Dock")
plt.title("Actual vs Predicted Trips per Dock (Labeled)")
plt.tight_layout()
plt.show()
# %%
