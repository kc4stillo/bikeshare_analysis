# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

# %%
df = pd.read_csv("../data/cleaned/combined_datasets/v2/ml_dataset_v2.csv")

X = df.drop(columns=["trips_per_dock", "name"])
y = np.log1p(df["trips_per_dock"])

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
)

# %%
scores = cross_validate(
    rf_model,
    X,
    y,
    cv=kf,
    scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
)

print("Log-target RF CV R² scores:", np.round(scores["test_r2"], 3))
print("Mean CV R²:", scores["test_r2"].mean().round(3))
print("Std CV R²:", scores["test_r2"].std().round(3))
print("Mean MAE:", (-scores["test_neg_mean_absolute_error"].mean()).round(3))
print("Mean RMSE:", (-scores["test_neg_root_mean_squared_error"].mean()).round(3))

# %%
# Fit once on a train/test split so we can make prediction plots
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_log = rf_model.predict(X_test)

# back-transform to original scale
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred_log)

print("\nFinal Log-Target RF Test Results (original units)")
print("-" * 50)
print(f"R²   : {r2_score(y_test_orig, y_pred_orig):.3f}")
print(f"MAE  : {mean_absolute_error(y_test_orig, y_pred_orig):.3f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)):.3f}")

# %%
# 1. Cross-validation R² by fold
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(scores["test_r2"]) + 1), scores["test_r2"])
plt.xlabel("Fold")
plt.ylabel("R²")
plt.title("Log-Target Random Forest CV R² by Fold")
plt.xticks(range(1, len(scores["test_r2"]) + 1))
plt.tight_layout()
plt.show()

# %%
# 2. Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_pred_orig, s=80, alpha=0.8)

min_val = min(y_test_orig.min(), y_pred_orig.min())
max_val = max(y_test_orig.max(), y_pred_orig.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual trips_per_dock")
plt.ylabel("Predicted trips_per_dock")
plt.title("Actual vs Predicted Trips per Dock")
plt.tight_layout()
plt.show()

# %%
# 3. Residual plot
residuals = y_test_orig - y_pred_orig

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_orig, residuals, s=80, alpha=0.8)
plt.axhline(0)

plt.xlabel("Predicted trips_per_dock")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()

# %%
# 4. Feature importance
importance = pd.Series(rf_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=True)

plt.figure(figsize=(10, 8))
importance.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# %%
# 5. Distribution of residuals
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=15)
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Count")
plt.title("Distribution of Residuals")
plt.tight_layout()
plt.show()

# %%
# 6. Biggest errors table
results = pd.DataFrame(
    {
        "name": df.loc[X_test.index, "name"],
        "actual": y_test_orig,
        "predicted": y_pred_orig,
        "residual": y_test_orig - y_pred_orig,
        "abs_error": np.abs(y_test_orig - y_pred_orig),
    }
).sort_values("abs_error", ascending=False)

print("\nTop 10 Biggest Errors")
print(results.head(10))

# %%
# 7. Top 15 feature importances only
top_importance = importance.sort_values(ascending=False).head(15).sort_values()

plt.figure(figsize=(10, 7))
top_importance.plot(kind="barh")
plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
