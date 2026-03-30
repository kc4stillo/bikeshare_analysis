# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# %%
# ----------------------------------------
# Load datasets
# df_with_names = for station names / interpretation
# df = ML-ready dataset
# ----------------------------------------
df_with_names = pd.read_csv(
    "../data/cleaned/combined_datasets/v5/combined_dataset_v5.csv"
)
df = pd.read_csv("../data/cleaned/combined_datasets/v5/ml_dataset_v5.csv")

# %%
# ----------------------------------------
# Define target + features
# ----------------------------------------
target = "trips_per_dock"

X = df.drop(columns=[target])
y = df[target]

# %%
# ----------------------------------------
# Train/test split
# Keep indices so we can reconnect station names later
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_idx = X_train.index
test_idx = X_test.index

# %%
# ----------------------------------------
# Log-transform target
# log1p handles zeros safely
# ----------------------------------------
y_train_log = np.log1p(y_train)

# %%
# ----------------------------------------
# XGBoost model
# These are close to the params you were using before
# ----------------------------------------
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=5,
    subsample=1.0,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

# %%
# ----------------------------------------
# Fit model on log target
# ----------------------------------------
xgb_model.fit(X_train, y_train_log)

# %%
# ----------------------------------------
# Predict on test set
# Convert predictions back to original units
# ----------------------------------------
y_pred_log = xgb_model.predict(X_test)
y_pred = np.expm1(y_pred_log)

# guard against tiny negative values after expm1
y_pred = np.maximum(y_pred, 0)

# %%
# ----------------------------------------
# Evaluate in original units
# ----------------------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Final Log-Target XGBoost Test Results (original units)")
print("--------------------------------------------------")
print(f"R²   : {r2:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")

# %%
# ----------------------------------------
# Feature importance
# ----------------------------------------
feature_importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": xgb_model.feature_importances_,
    }
).sort_values("importance", ascending=False)

print("\nTop 15 Features")
print("------------------------------")
print(feature_importance.head(15))

# %%
# ----------------------------------------
# Plot feature importance
# ----------------------------------------
top_n = 15
top_features = feature_importance.head(top_n).sort_values("importance")

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 XGBoost Feature Importances")
plt.tight_layout()
plt.show()

# %%
# ----------------------------------------
# Actual vs predicted table with names
# Use df_with_names to recover station names
# ----------------------------------------
results = df_with_names.loc[test_idx, ["name"]].copy()
results["actual_trips_per_dock"] = y_test.values
results["predicted_trips_per_dock"] = y_pred
results["residual"] = (
    results["actual_trips_per_dock"] - results["predicted_trips_per_dock"]
)
results["abs_residual"] = results["residual"].abs()

print("\nPredictions with station names")
print("--------------------------------")
print(results.sort_values("abs_residual", ascending=False).head(10))

# %%
# ----------------------------------------
# Worst predicted stations
# ----------------------------------------
worst_predictions = results.sort_values("abs_residual", ascending=False).head(10)

print("\nWorst predictions")
print("--------------------------------")
print(worst_predictions)

# %%
# ----------------------------------------
# Actual vs predicted scatterplot
# ----------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual Trips per Dock")
plt.ylabel("Predicted Trips per Dock")
plt.title("Actual vs Predicted Trips per Dock")
plt.tight_layout()
plt.show()

# %%
# ----------------------------------------
# Log-log version of actual vs predicted
# This helps if the bottom-left cluster is too squished
# ----------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(np.log1p(y_test), np.log1p(y_pred))

min_log = min(np.log1p(y_test).min(), np.log1p(y_pred).min())
max_log = max(np.log1p(y_test).max(), np.log1p(y_pred).max())
plt.plot([min_log, max_log], [min_log, max_log], linestyle="--")

plt.xlabel("log(1 + Actual Trips per Dock)")
plt.ylabel("log(1 + Predicted Trips per Dock)")
plt.title("Log-Scale Actual vs Predicted Trips per Dock")
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------
# Build feature importance dataframe
# ----------------------------------------
feature_importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": xgb_model.feature_importances_,
    }
).sort_values("importance", ascending=False)

# ----------------------------------------
# Plot all feature importances
# ----------------------------------------
plt.figure(figsize=(12, 10))
plt.barh(feature_importance["feature"], feature_importance["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  # most important at top
plt.tight_layout()
plt.show()
