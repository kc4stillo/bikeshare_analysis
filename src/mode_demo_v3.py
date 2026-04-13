import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt

sys.path.insert(0, str(Path.cwd().parents[0]))

from utilities.lat_lon import (
    lat_lon_area_occupied_within_radius,
    lat_lon_average_distance_to_3_nearest,
    lat_lon_count_points_within_radius,
    lat_lon_find_nearest_geometry_distance,
    lat_lon_find_nearest_point,
)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# %%
stations = pd.read_csv("../data/a_stations/stations.csv")
amenities = pd.read_csv("../data/b_amenities/clean/amenities.csv")
dining_halls = pd.read_csv("../data/b_amenities/clean/dining_halls.csv")
parks = pd.read_csv("../data/b_amenities/clean/parks.csv")
demographics = pd.read_csv("../data/c_demographics/clean/demographics.csv")
ut_shape = pd.read_csv("../data/d_ut_shapes/clean/ut_shape.csv")
west_campus = pd.read_csv("../data/d_ut_shapes/clean/west_campus.csv")
jobs = pd.read_csv("../data/e_jobs/clean/jobs.csv")
retail = pd.read_csv("../data/f_retail/clean/retail.csv")
transit = pd.read_csv("../data/g_transit/clean/transit.csv")

# %%
future_station_name = "nowhere"
lat = 24.286783805410494
lon = 173.95553835945773
docks = 9

# load trained model
model = joblib.load("../models/v8/v8_no_demo_no_ut.pkl")

# load exact column order used during training
with open("../models/v8/v8_no_demo_order_no_ut.json", "r") as f:
    feature_cols = json.load(f)

# %%
# convert pd to geopandas df
parks["geometry"] = parks["geometry"].apply(wkt.loads)
parks = gpd.GeoDataFrame(parks, geometry="geometry", crs="EPSG:4326")

demographics["geometry"] = demographics["geometry"].apply(wkt.loads)
demographics = gpd.GeoDataFrame(demographics, geometry="geometry", crs="EPSG:4326")

west_campus["geometry"] = west_campus["geometry"].apply(wkt.loads)
west_campus = gpd.GeoDataFrame(west_campus, geometry="geometry", crs="EPSG:4326")

ut_shape["geometry"] = ut_shape["geometry"].apply(wkt.loads)
ut_shape = gpd.GeoDataFrame(ut_shape, geometry="geometry", crs="EPSG:4326")


# %%
# MODEL INTERPRETABILITY
def align_features_to_training(X, feature_cols, fill_value=0):
    """
    Make sure prediction dataframe matches the exact training column order.
    Missing columns are added with fill_value.
    Extra columns are dropped.
    """
    X_aligned = X.copy()

    for col in feature_cols:
        if col not in X_aligned.columns:
            X_aligned[col] = fill_value

    X_aligned = X_aligned[feature_cols]
    return X_aligned


def get_model_feature_importance(model, feature_cols):
    """
    Return a dataframe of global feature importances if the model supports it.
    Works for tree-based models like RandomForest, GradientBoosting, XGBoost sklearn API, etc.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance_df.reset_index(drop=True)


def plot_global_feature_importance(model, feature_cols, top_n=15):
    """
    Plot top_n global feature importances.
    """
    importance_df = get_model_feature_importance(model, feature_cols)

    if importance_df is None:
        print(
            "Model does not expose feature_importances_. Skipping global importance plot."
        )
        return

    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Global Feature Importances")
    plt.tight_layout()
    plt.show()


def explain_single_prediction_shap(model, X_new, top_n=10):
    """
    Explain one prediction using SHAP, if shap is installed and compatible.

    Returns
    -------
    pd.DataFrame or None
        DataFrame of feature contributions for the single row.
    """
    if not SHAP_AVAILABLE:
        print("SHAP is not installed. Run: pip install shap")
        return None

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_new)

        contrib_df = pd.DataFrame(
            {
                "feature": X_new.columns,
                "value": X_new.iloc[0].values,
                "shap_value": shap_values.values[0],
            }
        )

        contrib_df["abs_shap_value"] = contrib_df["shap_value"].abs()
        contrib_df = contrib_df.sort_values("abs_shap_value", ascending=False)

        print("\nTop feature contributions for this prediction:")
        print(contrib_df.head(top_n)[["feature", "value", "shap_value"]])

        # waterfall plot
        try:
            shap.plots.waterfall(shap_values[0], max_display=top_n)
        except Exception:
            warnings.warn("Could not render SHAP waterfall plot in this environment.")

        return contrib_df

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return None


def summarize_local_effects(contrib_df, top_n=5):
    """
    Print the top positive and negative contributors from SHAP output.
    """
    if contrib_df is None or contrib_df.empty:
        return

    positive = contrib_df.sort_values("shap_value", ascending=False).head(top_n)
    negative = contrib_df.sort_values("shap_value", ascending=True).head(top_n)

    print("\nTop positive contributors (pushed prediction UP):")
    print(positive[["feature", "value", "shap_value"]])

    print("\nTop negative contributors (pushed prediction DOWN):")
    print(negative[["feature", "value", "shap_value"]])


# %%
def collect_feaures(lat, lon, docks):
    nearest_amenity_m = lat_lon_find_nearest_point(lat, lon, amenities)
    nearest_park_m = lat_lon_find_nearest_geometry_distance(lat, lon, parks)
    count_amenities_275m = lat_lon_count_points_within_radius(
        lat, lon, amenities, radius_m=275
    )
    avg_dist_3_amenities_m = lat_lon_average_distance_to_3_nearest(lat, lon, amenities)
    park_area_within_275m = lat_lon_area_occupied_within_radius(
        lat, lon, parks, radius_m=275
    )
    park_area_within_550m = lat_lon_area_occupied_within_radius(
        lat, lon, parks, radius_m=550
    )
    jobs_count_within_275m = lat_lon_count_points_within_radius(
        lat, lon, jobs, radius_m=275
    )
    jobs_count_within_550m = lat_lon_count_points_within_radius(
        lat, lon, jobs, radius_m=550
    )
    nearest_retail_m = lat_lon_find_nearest_point(lat, lon, retail)
    count_retail_275m = lat_lon_count_points_within_radius(
        lat, lon, retail, radius_m=275
    )
    count_retail_550m = lat_lon_count_points_within_radius(
        lat, lon, retail, radius_m=550
    )
    avg_dist_3_retail_m = lat_lon_average_distance_to_3_nearest(lat, lon, retail)
    nearest_bikeshare_station_m = lat_lon_find_nearest_point(lat, lon, stations)
    avg_dist_3_stations = lat_lon_average_distance_to_3_nearest(lat, lon, stations)
    bikeshare_station_count_within_275m = lat_lon_count_points_within_radius(
        lat, lon, stations, radius_m=275
    )
    bikeshare_station_count_within_550m = lat_lon_count_points_within_radius(
        lat, lon, stations, radius_m=550
    )
    nearest_transit_stop_distance_m = lat_lon_find_nearest_point(lat, lon, transit)
    count_transit_stop_275m = lat_lon_count_points_within_radius(
        lat, lon, transit, radius_m=275
    )
    count_transit_stop_550m = lat_lon_count_points_within_radius(
        lat, lon, transit, radius_m=550
    )

    return {
        "docks": docks,
        "nearest_amenity_m": nearest_amenity_m,
        "nearest_park_m": nearest_park_m,
        "count_amenities_275m": count_amenities_275m,
        "avg_dist_3_amenities_m": avg_dist_3_amenities_m,
        "park_area_within_275m": park_area_within_275m,
        "park_area_within_550m": park_area_within_550m,
        "jobs_count_within_275m": jobs_count_within_275m,
        "jobs_count_within_550m": jobs_count_within_550m,
        "nearest_retail_m": nearest_retail_m,
        "count_retail_275m": count_retail_275m,
        "count_retail_550m": count_retail_550m,
        "avg_dist_3_retail_m": avg_dist_3_retail_m,
        "nearest_bikeshare_station_m": nearest_bikeshare_station_m,
        "avg_dist_3_stations": avg_dist_3_stations,
        "bikeshare_station_count_within_275m": bikeshare_station_count_within_275m,
        "bikeshare_station_count_within_550m": bikeshare_station_count_within_550m,
        "nearest_transit_stop_distance_m": nearest_transit_stop_distance_m,
        "count_transit_stop_275m": count_transit_stop_275m,
        "count_transit_stop_550m": count_transit_stop_550m,
    }


features = collect_feaures(lat, lon, docks)

X_new = pd.DataFrame([features])
X_new = align_features_to_training(X_new, feature_cols)

pred_log = model.predict(X_new)[0]
pred_trips_per_dock = np.expm1(pred_log)

total_trips = pred_trips_per_dock * docks

print(f"\nPredicted log(trips_per_dock + 1): {pred_log:.4f}")
print(f"Predicted trips_per_dock: {pred_trips_per_dock:.4f}")
print(f"Predicted total trips: {total_trips:.2f}")

# -----------------------------
# Global interpretability
# -----------------------------
plot_global_feature_importance(model, feature_cols, top_n=15)

importance_df = get_model_feature_importance(model, feature_cols)
if importance_df is not None:
    print("\nTop global features:")
    print(importance_df.head(15))

# -----------------------------
# Local interpretability
# -----------------------------
contrib_df = explain_single_prediction_shap(model, X_new, top_n=21)
# summarize_local_effects(contrib_df, top_n=5)

# %%
import pandas as pd


def plot_temp_station_all(df, future_station_name, pred_trips_per_dock):
    """
    Plot all existing stations plus the temporary/future station,
    sorted by trips_per_dock, with the temporary station highlighted.
    """

    # existing stations
    existing = df[["name", "trips_per_dock"]].copy()

    # temp station row
    temp_row = pd.DataFrame(
        {
            "name": [future_station_name],
            "trips_per_dock": [pred_trips_per_dock],
        }
    )

    # combine and sort
    plot_df = pd.concat([existing, temp_row], ignore_index=True)
    plot_df = plot_df.sort_values("trips_per_dock").reset_index(drop=True)

    # color temp station differently
    colors = [
        "tomato" if name == future_station_name else "steelblue"
        for name in plot_df["name"]
    ]

    plt.figure(figsize=(18, 6))
    bars = plt.bar(plot_df["name"], plot_df["trips_per_dock"], color=colors)

    # optional: label the temp station
    temp_idx = plot_df.index[plot_df["name"] == future_station_name][0]
    temp_bar = bars[temp_idx]
    temp_height = temp_bar.get_height()

    plt.text(
        temp_bar.get_x() + temp_bar.get_width() / 2,
        temp_height + 5,
        f"{future_station_name}\n{temp_height:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    plt.xlabel("Station")
    plt.ylabel("Trips per Dock")
    plt.title(f"All Stations + Predicted Position for {future_station_name}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_temp_station_context(df, future_station_name, pred_trips_per_dock):
    """
    Plot a temporary station in context with the 3 nearest lower
    and 3 nearest higher existing stations by trips_per_dock.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'name', 'trips_per_dock'
    future_station_name : str
        Name of the temporary/future station
    pred_trips_per_dock : float
        Predicted trips_per_dock for the temporary station
    """

    # keep only needed columns
    existing = df[["name", "trips_per_dock"]].copy()

    # sort all existing stations by trips_per_dock
    existing = existing.sort_values("trips_per_dock").reset_index(drop=True)

    # 3 stations below
    lower = existing[existing["trips_per_dock"] < pred_trips_per_dock].tail(3)

    # 3 stations above
    upper = existing[existing["trips_per_dock"] >= pred_trips_per_dock].head(3)

    # temp station row
    temp_row = pd.DataFrame(
        {
            "name": [future_station_name],
            "trips_per_dock": [pred_trips_per_dock],
        }
    )

    # combine so temp station is in the middle
    plot_df = pd.concat([lower, temp_row, upper], ignore_index=True)

    # color temp station differently
    colors = [
        "steelblue" if name != future_station_name else "tomato"
        for name in plot_df["name"]
    ]

    # plot
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["name"], plot_df["trips_per_dock"], color=colors)

    plt.xlabel("Station")
    plt.ylabel("Trips per Dock")
    plt.title(f"Predicted Context for {future_station_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


plot_temp_station_all(stations, future_station_name, pred_trips_per_dock)
plot_temp_station_context(stations, future_station_name, pred_trips_per_dock)
