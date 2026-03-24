# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from xgboost import XGBRegressor

# %%
df = pd.read_csv("../data/cleaned/combined_datasets/v2/ml_dataset_v2.csv")

X = df.drop(columns=["trips_per_dock", "name", "lat", "lon"])
y = np.log1p(df["trips_per_dock"])

X.head()

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    min_child_weight=5,
    subsample=1.0,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
)

# %%
scores = cross_validate(
    xgb_model,
    X,
    y,
    cv=kf,
    scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
)

print("Log-target XGBoost CV R² scores:", np.round(scores["test_r2"], 3))
print("Mean CV R²:", scores["test_r2"].mean().round(3))
print("Std CV R²:", scores["test_r2"].std().round(3))
print("Mean MAE:", (-scores["test_neg_mean_absolute_error"].mean()).round(3))
print("Mean RMSE:", (-scores["test_neg_root_mean_squared_error"].mean()).round(3))

# %%
# Fit once on a train/test split so we can make prediction plots
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_log = xgb_model.predict(X_test)

# back-transform to original scale
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred_log)

print("\nFinal Log-Target XGBoost Test Results (original units)")
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
plt.title("Log-Target XGBoost CV R² by Fold")
plt.xticks(range(1, len(scores["test_r2"]) + 1))
plt.tight_layout()
plt.show()

# %%
# Biggest errors table
results = pd.DataFrame(
    {
        "name": df.loc[X_test.index, "name"],
        "actual": y_test_orig,
        "predicted": y_pred_orig,
        "residual": y_test_orig - y_pred_orig,
        "abs_error": np.abs(y_test_orig - y_pred_orig),
    }
).sort_values("abs_error", ascending=False)

print("\nTop 5 Biggest Errors")
print(results.head())

# %%
# Smallest errors table
results = pd.DataFrame(
    {
        "name": df.loc[X_test.index, "name"],
        "actual": y_test_orig,
        "predicted": y_pred_orig,
        "residual": y_test_orig - y_pred_orig,
        "abs_error": np.abs(y_test_orig - y_pred_orig),
    }
).sort_values("abs_error", ascending=True)

print("\nTop 5 Smallest Errors")
print(results.head())

# %%
# WITHOUT UT STATIONS
df = pd.read_csv("../data/cleaned/combined_datasets/v2/ml_dataset_v2.csv")

df = df[df["is_ut"] == 0]

X = df.drop(columns=["trips_per_dock", "name", "lat", "lon"])
y = np.log1p(df["trips_per_dock"])

X.head()

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    min_child_weight=5,
    subsample=1.0,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
)

# %%
scores = cross_validate(
    xgb_model,
    X,
    y,
    cv=kf,
    scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
)

print("Log-target XGBoost CV R² scores:", np.round(scores["test_r2"], 3))
print("Mean CV R²:", scores["test_r2"].mean().round(3))
print("Std CV R²:", scores["test_r2"].std().round(3))
print("Mean MAE:", (-scores["test_neg_mean_absolute_error"].mean()).round(3))
print("Mean RMSE:", (-scores["test_neg_root_mean_squared_error"].mean()).round(3))

# %%
# Fit once on a train/test split so we can make prediction plots
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_log = xgb_model.predict(X_test)

# back-transform to original scale
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred_log)

print("\nFinal Log-Target XGBoost Test Results (original units)")
print("-" * 50)
print(f"R²   : {r2_score(y_test_orig, y_pred_orig):.3f}")
print(f"MAE  : {mean_absolute_error(y_test_orig, y_pred_orig):.3f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)):.3f}")

# %%
# ONLY UT STATIONS
df = pd.read_csv("../data/cleaned/combined_datasets/v2/ml_dataset_v2.csv")

df = df[df["is_ut"] == 1]

X = df.drop(columns=["trips_per_dock", "name", "lat", "lon"])
y = np.log1p(df["trips_per_dock"])

X.head()

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    min_child_weight=5,
    subsample=1.0,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
)

# %%
scores = cross_validate(
    xgb_model,
    X,
    y,
    cv=kf,
    scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
)

print("Log-target XGBoost CV R² scores:", np.round(scores["test_r2"], 3))
print("Mean CV R²:", scores["test_r2"].mean().round(3))
print("Std CV R²:", scores["test_r2"].std().round(3))
print("Mean MAE:", (-scores["test_neg_mean_absolute_error"].mean()).round(3))
print("Mean RMSE:", (-scores["test_neg_root_mean_squared_error"].mean()).round(3))

# %%
# Fit once on a train/test split so we can make prediction plots
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_log = xgb_model.predict(X_test)

# back-transform to original scale
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred_log)

print("\nFinal Log-Target XGBoost Test Results (original units)")
print("-" * 50)
print(f"R²   : {r2_score(y_test_orig, y_pred_orig):.3f}")
print(f"MAE  : {mean_absolute_error(y_test_orig, y_pred_orig):.3f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)):.3f}")

X.head()
