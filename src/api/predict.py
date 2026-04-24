from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from src.api.feature_builder import build_features_for_point

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "v8" / "v8_general.pkl"
REFERENCE_X_PATH = PROJECT_ROOT / "models" / "v8" / "v8_general_training.csv"

STATIONS_PATH = PROJECT_ROOT / "data" / "a_stations" / "stations.csv"
stations_reference = pd.read_csv(STATIONS_PATH)

reference_X = pd.read_csv(REFERENCE_X_PATH)

model = joblib.load(MODEL_PATH)


def build_station_comparison(
    pred_trips_per_dock: float,
    stations_df: pd.DataFrame,
    candidate_name: str = "Your selected location",
) -> dict:
    """
    Compare the predicted candidate station against all existing stations.

    Returns:
    - rank position
    - percentile
    - all stations + candidate sorted by trips_per_dock
    - nearby context around the candidate
    """

    df = stations_df.copy()

    possible_name_cols = ["name", "station", "station_name"]
    possible_trip_cols = ["trips_per_dock", "actual_trips_per_dock"]

    name_col = next((col for col in possible_name_cols if col in df.columns), None)
    trips_col = next((col for col in possible_trip_cols if col in df.columns), None)

    if trips_col is None:
        raise ValueError(
            "Could not find a trips_per_dock column in stations dataframe."
        )

    if name_col is None:
        df["name"] = [f"Station {i + 1}" for i in range(len(df))]
        name_col = "name"

    df = df[[name_col, trips_col]].copy()
    df = df.rename(columns={name_col: "name", trips_col: "trips_per_dock"})

    df["trips_per_dock"] = pd.to_numeric(df["trips_per_dock"], errors="coerce")
    df = df.dropna(subset=["trips_per_dock"])

    df["is_candidate"] = False

    candidate_row = pd.DataFrame(
        {
            "name": [candidate_name],
            "trips_per_dock": [float(pred_trips_per_dock)],
            "is_candidate": [True],
        }
    )

    combined = pd.concat([df, candidate_row], ignore_index=True)

    # Highest trips_per_dock gets rank 1
    combined = combined.sort_values("trips_per_dock", ascending=False).reset_index(
        drop=True
    )

    combined["rank"] = combined.index + 1

    candidate_idx = combined.index[combined["is_candidate"]].tolist()[0]

    rank_position = int(candidate_idx + 1)
    total_stations = int(len(combined))

    # Percentile: percent of existing stations at or below candidate value
    rank_percentile = float((df["trips_per_dock"] <= pred_trips_per_dock).mean() * 100)

    # Nearby context: 3 above and 3 below candidate
    start_idx = max(candidate_idx - 3, 0)
    end_idx = min(candidate_idx + 4, len(combined))
    context_df = combined.iloc[start_idx:end_idx].copy()

    def clean_rows(rows_df):
        rows = []
        for _, row in rows_df.iterrows():
            rows.append(
                {
                    "rank": int(row["rank"]),
                    "name": str(row["name"]),
                    "trips_per_dock": float(row["trips_per_dock"]),
                    "is_candidate": bool(row["is_candidate"]),
                }
            )
        return rows

    return {
        "rank_percentile": rank_percentile,
        "rank_position": rank_position,
        "total_stations_plus_candidate": total_stations,
        "nearby_rank_context": clean_rows(context_df),
        "all_station_rankings": clean_rows(combined),
    }


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

    station_comparison = build_station_comparison(
        pred_trips_per_dock=predicted_trips_per_dock,
        stations_df=stations_reference,
    )

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
        "station_comparison": station_comparison,
        **summary,
    }
