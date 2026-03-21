import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Load data
df = pd.read_csv("../data/cleaned/combined_datasets/v2/ml_dataset_v2.csv")

X = df.drop(columns=["trips_per_dock", "name"])
y = df["trips_per_dock"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Linear Regression": Pipeline([("model", LinearRegression())]),
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1))]),
    "Elastic Net": Pipeline(
        [("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.1, l1_ratio=0.5))]
    ),
    "Random Forest": Pipeline(
        [
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=42,
                ),
            )
        ]
    ),
    "SVR": Pipeline(
        [("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1))]
    ),
}

results = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
        return_train_score=False,
    )

    results.append(
        {
            "model": name,
            "mean_r2": scores["test_r2"].mean(),
            "std_r2": scores["test_r2"].std(),
            "mean_mae": -scores["test_neg_mean_absolute_error"].mean(),
            "mean_rmse": -scores["test_neg_root_mean_squared_error"].mean(),
        }
    )

results_df = pd.DataFrame(results).sort_values("mean_r2", ascending=False)
print(results_df.round(3))
