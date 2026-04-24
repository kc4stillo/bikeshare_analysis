# DOES NOT INCLUDE ANY UT RELATED FEATURES
import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt

sys.path.insert(0, str(Path.cwd().parents[1]))

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

stations = pd.read_csv(DATA_DIR / "a_stations/stations.csv")
amenities = pd.read_csv(DATA_DIR / "b_amenities/clean/amenities.csv")
dining_halls = pd.read_csv(DATA_DIR / "b_amenities/clean/dining_halls.csv")
parks = pd.read_csv(DATA_DIR / "b_amenities/clean/parks.csv")
demographics = pd.read_csv(DATA_DIR / "c_demographics/clean/demographics.csv")
ut_shape = pd.read_csv(DATA_DIR / "d_ut_shapes/clean/ut_shape.csv")
west_campus = pd.read_csv(DATA_DIR / "d_ut_shapes/clean/west_campus.csv")
jobs = pd.read_csv(DATA_DIR / "e_jobs/clean/jobs.csv")
retail = pd.read_csv(DATA_DIR / "f_retail/clean/retail.csv")
transit = pd.read_csv(DATA_DIR / "g_transit/clean/transit.csv")

with open(MODELS_DIR / "v8/v8_general.json", "r") as f:
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
def collect_features(lat, lon, docks=13):
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
        # "west_campus_area_within_550m": west_campus_area_within_550m,
        # "distance_to_west_campus_m": distance_to_west_campus_m,
        "distance_to_ut_m": distance_to_ut_m,
        "ut_area_within_275m": ut_area_within_275m,
        "ut_area_within_550m": ut_area_within_550m,
        "jobs_count_within_275m": jobs_count_within_275m,
        "jobs_count_within_550m": jobs_count_within_550m,
        "nearest_retail_m": nearest_retail_m,
        "count_retail_275m": count_retail_275m,
        "count_retail_550m": count_retail_550m,
        "avg_dist_3_retail_m": avg_dist_3_retail_m,
        "avg_dist_3_stations": avg_dist_3_stations,
        "nearest_bikeshare_station_m": nearest_bikeshare_station_m,
        "bikeshare_station_count_within_275m": bikeshare_station_count_within_275m,
        "bikeshare_station_count_within_550m": bikeshare_station_count_within_550m,
        "nearest_transit_stop_distance_m": nearest_transit_stop_distance_m,
        "count_transit_stop_275m": count_transit_stop_275m,
        "count_transit_stop_550m": count_transit_stop_550m,
    }


def build_features_for_point(lat: float, lon: float, docks: int = 19) -> dict:
    features = collect_features(lat, lon, docks=docks)
    X = pd.DataFrame([features])
    X_aligned = align_features_to_training(X, feature_cols)

    return X_aligned.iloc[0].to_dict()
