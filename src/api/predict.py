from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from src.api.feature_builder import build_features_for_point

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "v8" / "v8_general.pkl"
REFERENCE_X_PATH = PROJECT_ROOT / "models" / "v8" / "v8_general_training.csv"
reference_X = pd.read_csv(REFERENCE_X_PATH)

model = joblib.load(MODEL_PATH)


def percentile_rank(series, value):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float((s <= value).mean() * 100)


def build_prediction_summary(X_new, reference_X, feature_cols, model, pred_log):
    rows = []

    for col in feature_cols:
        value = X_new.iloc[0][col]
        s = pd.to_numeric(reference_X[col], errors="coerce").dropna()

        rows.append(
            {
                "feature": col,
                "value": float(value),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "percentile_rank": percentile_rank(s, value),
            }
        )

    percentile_df = pd.DataFrame(rows)

    percentile_df["relative_to_median"] = np.where(
        percentile_df["value"] > percentile_df["median"],
        "above median",
        np.where(
            percentile_df["value"] < percentile_df["median"],
            "below median",
            "at median",
        ),
    )

    dmatrix_new = xgb.DMatrix(X_new, feature_names=feature_cols)
    contribs = model.get_booster().predict(dmatrix_new, pred_contribs=True)

    feature_contribs = contribs[0][:-1]
    base_value = float(contribs[0][-1])

    contrib_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "shap_value": feature_contribs,
        }
    )

    summary_df = percentile_df.merge(contrib_df, on="feature", how="left")
    summary_df["abs_shap"] = summary_df["shap_value"].abs()
    summary_df = summary_df.sort_values("abs_shap", ascending=False)

    top_summary = summary_df[
        [
            "feature",
            "value",
            "shap_value",
            "percentile_rank",
            "median",
            "relative_to_median",
        ]
    ].head(8)

    return {
        "base_value_log": base_value,
        "contribution_check_log": float(base_value + feature_contribs.sum()),
        "prediction_check_log": float(pred_log),
        "top_feature_summary": top_summary.to_dict(orient="records"),
    }


def predict_for_point(lat: float, lon: float, docks: int = 19) -> dict:
    features = build_features_for_point(lat, lon, docks=docks)

    X = pd.DataFrame([features])

    pred_log = model.predict(X)[0]

    # If your model was trained on log1p target:
    predicted_trips_per_dock = np.expm1(pred_log)

    summary = build_prediction_summary(
        X_new=X,
        reference_X=reference_X,
        feature_cols=list(X.columns),
        model=model,
        pred_log=pred_log,
    )

    return {
        "predicted_trips_per_dock": float(predicted_trips_per_dock),
        "predicted_log": float(pred_log),
        "features": features,
        **summary,
    }
