# DOES NOT INCLUDE ANY UT RELATED FEATURES
import json
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from shapely import wkt

sys.path.insert(0, str(Path.cwd().parents[0]))

from utilities.lat_lon import (
    lat_lon_area_occupied_within_radius,
    lat_lon_average_distance_to_3_nearest,
    lat_lon_count_points_within_radius,
    lat_lon_find_nearest_geometry_distance,
    lat_lon_find_nearest_point,
    lat_lon_get_polygon_attributes_with_nearest_fill,
)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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
future_station_name = "park"
lat = 30.270059847101038
lon = -97.76224416804668
docks = 9

# load trained model
model = joblib.load("../models/v8/v8_general.pkl")

# load exact column order used during training
with open("../models/v8/v8_general.json", "r") as f:
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
def collect_feaures(lat, lon, docks):
    demo = lat_lon_get_polygon_attributes_with_nearest_fill(lat, lon, demographics)
    median_age = demo["median_age"]
    median_income = demo["median_income"]
    count_population = demo["count_population"]
    population_density = demo["count_population"] / demo["area_m2"]
    undergrad_percentage = demo["count_undergrad"] / demo["count_population"]
    grad_percentage = demo["count_grad"] / demo["count_population"]
    west_campus_area_within_275m = lat_lon_area_occupied_within_radius(
        lat, lon, west_campus, radius_m=275
    )
    west_campus_area_within_550m = lat_lon_area_occupied_within_radius(
        lat, lon, west_campus, radius_m=550
    )
    distance_to_ut_m = lat_lon_find_nearest_geometry_distance(lat, lon, ut_shape)
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
    nearest_dining_hall_m = lat_lon_find_nearest_point(lat, lon, dining_halls)
    distance_to_west_campus_m = lat_lon_find_nearest_geometry_distance(
        lat, lon, west_campus
    )
    ut_area_within_275m = lat_lon_area_occupied_within_radius(
        lat, lon, ut_shape, radius_m=275
    )
    ut_area_within_550m = lat_lon_area_occupied_within_radius(
        lat, lon, ut_shape, radius_m=550
    )

    return {
        "docks": docks,
        "nearest_amenity_m": nearest_amenity_m,
        "nearest_park_m": nearest_park_m,
        "count_amenities_275m": count_amenities_275m,
        # "nearest_dining_hall_m": nearest_dining_hall_m,
        "avg_dist_3_amenities_m": avg_dist_3_amenities_m,
        "park_area_within_275m": park_area_within_275m,
        "park_area_within_550m": park_area_within_550m,
        "median_age": median_age,
        "median_income": median_income,
        # "count_population": count_population,
        "population_density": population_density,
        "undergrad_percentage": undergrad_percentage,
        "grad_percentage": grad_percentage,
        # "west_campus_area_within_275m": west_campus_area_within_275m,
        "west_campus_area_within_550m": west_campus_area_within_550m,
        "distance_to_west_campus_m": distance_to_west_campus_m,
        "distance_to_ut_m": distance_to_ut_m,
        # "ut_area_within_275m": ut_area_within_275m,
        # "ut_area_within_550m": ut_area_within_550m,
        "jobs_count_within_275m": jobs_count_within_275m,
        "jobs_count_within_550m": jobs_count_within_550m,
        "nearest_retail_m": nearest_retail_m,
        # "count_retail_275m": count_retail_275m,
        # "count_retail_550m": count_retail_550m,
        "avg_dist_3_retail_m": avg_dist_3_retail_m,
        "avg_dist_3_stations": avg_dist_3_stations,
        "nearest_bikeshare_station_m": nearest_bikeshare_station_m,
        "bikeshare_station_count_within_275m": bikeshare_station_count_within_275m,
        "bikeshare_station_count_within_550m": bikeshare_station_count_within_550m,
        "nearest_transit_stop_distance_m": nearest_transit_stop_distance_m,
        "count_transit_stop_275m": count_transit_stop_275m,
        "count_transit_stop_550m": count_transit_stop_550m,
    }


# %%
features = collect_feaures(lat, lon, docks)

X_new = pd.DataFrame([features])

# make sure column order matches training
X_new = X_new[feature_cols]

pred_log = model.predict(X_new)[0]
pred_trips_per_dock = np.expm1(pred_log)
total_trips = pred_trips_per_dock * docks

print(f"\nPredicted log(trips_per_dock + 1): {pred_log:.4f}")
print(f"Predicted trips_per_dock: {pred_trips_per_dock:.4f}")
print(f"Predicted total trips: {total_trips:.2f}")

# %%
# -----------------------------
# Reference training feature matrix
# -----------------------------
training_df = pd.read_csv("../data/g_transit/clean/stations.csv")
reference_X = training_df[feature_cols].copy()


# %%
def percentile_rank(series, value):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return (s <= value).mean() * 100


# %%
# -----------------------------
# Build percentile summary
# -----------------------------
rows = []

for col in feature_cols:
    value = X_new.iloc[0][col]
    s = pd.to_numeric(reference_X[col], errors="coerce").dropna()

    rows.append(
        {
            "feature": col,
            "value": value,
            "p25": s.quantile(0.25),
            "median": s.median(),
            "p75": s.quantile(0.75),
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

# %%
# -----------------------------
# XGBoost built-in feature contributions
# These are SHAP-like values on the model output scale
# Last column is the bias/base value
# -----------------------------
dmatrix_new = xgb.DMatrix(X_new, feature_names=feature_cols)
contribs = model.get_booster().predict(dmatrix_new, pred_contribs=True)

# feature contributions only (drop final bias term)
feature_contribs = contribs[0][:-1]
base_value = contribs[0][-1]

contrib_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "shap_value": feature_contribs,
    }
)

# merge percentile info with contribution info
summary_df = percentile_df.merge(contrib_df, on="feature", how="left")
summary_df["abs_shap"] = summary_df["shap_value"].abs()
summary_df = summary_df.sort_values("abs_shap", ascending=False)

print(f"\nBase value (log scale): {base_value:.4f}")
print(f"Prediction from contributions check: {base_value + feature_contribs.sum():.4f}")
print(f"Model prediction check: {pred_log:.4f}")

print("\nTop feature contributions for this prediction:")
print("-" * 120)
print(
    summary_df[
        [
            "feature",
            "value",
            "shap_value",
            "percentile_rank",
            "median",
            "relative_to_median",
        ]
    ]
    .round(3)
    .to_string(index=False)
)

# %%
# import pandas as pd


# def plot_temp_station_all(df, future_station_name, pred_trips_per_dock):
#     """
#     Plot all existing stations plus the temporary/future station,
#     sorted by trips_per_dock, with the temporary station highlighted.
#     """

#     # existing stations
#     existing = df[["name", "trips_per_dock"]].copy()

#     # temp station row
#     temp_row = pd.DataFrame(
#         {
#             "name": [future_station_name],
#             "trips_per_dock": [pred_trips_per_dock],
#         }
#     )

#     # combine and sort
#     plot_df = pd.concat([existing, temp_row], ignore_index=True)
#     plot_df = plot_df.sort_values("trips_per_dock").reset_index(drop=True)

#     # color temp station differently
#     colors = [
#         "tomato" if name == future_station_name else "steelblue"
#         for name in plot_df["name"]
#     ]

#     plt.figure(figsize=(18, 6))
#     bars = plt.bar(plot_df["name"], plot_df["trips_per_dock"], color=colors)

#     # optional: label the temp station
#     temp_idx = plot_df.index[plot_df["name"] == future_station_name][0]
#     temp_bar = bars[temp_idx]
#     temp_height = temp_bar.get_height()

#     plt.text(
#         temp_bar.get_x() + temp_bar.get_width() / 2,
#         temp_height + 5,
#         f"{future_station_name}\n{temp_height:.1f}",
#         ha="center",
#         va="bottom",
#         fontsize=9,
#         fontweight="bold",
#     )

#     plt.xlabel("Station")
#     plt.ylabel("Trips per Dock")
#     plt.title(f"All Stations + Predicted Position for {future_station_name}")
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     plt.show()


# def plot_temp_station_context(df, future_station_name, pred_trips_per_dock):
#     """
#     Plot a temporary station in context with the 3 nearest lower
#     and 3 nearest higher existing stations by trips_per_dock.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must contain columns: 'name', 'trips_per_dock'
#     future_station_name : str
#         Name of the temporary/future station
#     pred_trips_per_dock : float
#         Predicted trips_per_dock for the temporary station
#     """

#     # keep only needed columns
#     existing = df[["name", "trips_per_dock"]].copy()

#     # sort all existing stations by trips_per_dock
#     existing = existing.sort_values("trips_per_dock").reset_index(drop=True)

#     # 3 stations below
#     lower = existing[existing["trips_per_dock"] < pred_trips_per_dock].tail(3)

#     # 3 stations above
#     upper = existing[existing["trips_per_dock"] >= pred_trips_per_dock].head(3)

#     # temp station row
#     temp_row = pd.DataFrame(
#         {
#             "name": [future_station_name],
#             "trips_per_dock": [pred_trips_per_dock],
#         }
#     )

#     # combine so temp station is in the middle
#     plot_df = pd.concat([lower, temp_row, upper], ignore_index=True)

#     # color temp station differently
#     colors = [
#         "steelblue" if name != future_station_name else "tomato"
#         for name in plot_df["name"]
#     ]

#     # plot
#     plt.figure(figsize=(12, 6))
#     plt.bar(plot_df["name"], plot_df["trips_per_dock"], color=colors)

#     plt.xlabel("Station")
#     plt.ylabel("Trips per Dock")
#     plt.title(f"Predicted Context for {future_station_name}")
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     plt.show()


# plot_temp_station_all(stations, future_station_name, pred_trips_per_dock)
# plot_temp_station_context(stations, future_station_name, pred_trips_per_dock)
