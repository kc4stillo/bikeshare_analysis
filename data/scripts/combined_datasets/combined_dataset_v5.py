# %%
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

# %%
cleaned_prefix = "../../cleaned/"
output_prefix = "../../cleaned/combined_datasets/v5/"

# %%
# -----------------------------
# Load datasets
# -----------------------------
amenities = pd.read_csv(cleaned_prefix + "amenities/amenities.csv")
coords = pd.read_csv(cleaned_prefix + "coords/coords.csv")
housing = pd.read_csv(cleaned_prefix + "housing/housing.csv")
jobs = pd.read_csv(cleaned_prefix + "jobs/jobs.csv")
retail = pd.read_csv(cleaned_prefix + "retail/retail.csv")
scores = pd.read_csv(cleaned_prefix + "scoring/current_stations.csv")
transit = pd.read_csv(cleaned_prefix + "transit/transit.csv")
parks = pd.read_csv(cleaned_prefix + "amenities/parks.csv")
dining_halls = pd.read_csv(cleaned_prefix + "amenities/dining_halls.csv")
dorms = pd.read_csv(cleaned_prefix + "housing/dorms.csv")
ut_hotspots = pd.read_csv(cleaned_prefix + "amenities/ut_hotspots.csv")
wampus_hotspots = pd.read_csv(cleaned_prefix + "housing/wampus_hotspots.csv")

# %%
# -----------------------------
# Merge scores + coordinates
# -----------------------------
scores_and_coords = scores.merge(
    coords[["scoring_name", "lat", "lon"]],
    left_on="name",
    right_on="scoring_name",
    how="left",
)

scores_and_coords.drop(columns=["scoring_name"], inplace=True, errors="ignore")
scores_and_coords = scores_and_coords.dropna(subset=["lat", "lon"]).copy()

# %%
# -----------------------------
# Optional cleanup / rename
# -----------------------------
rename_map = {
    "transit_access_score": "transit_access_score",
    "jobs_access_score": "jobs_access_score",
    "households_access_score": "households_access_score",
    "public_amenities_access_score": "public_amenities_access_score",
    "retail_entertainment_access_score": "retail_entertainment_access_score",
    "existing_bikeshare_access_score": "existing_bikeshare_access_score",
}
scores_and_coords.rename(columns=rename_map, inplace=True)


# %%
# -----------------------------
# Helper functions
# -----------------------------
def make_points_gdf(df, lat_col="lat", lon_col="lon", crs="EPSG:4326"):
    out = df.dropna(subset=[lat_col, lon_col]).copy()
    return gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out[lon_col], out[lat_col]),
        crs=crs,
    )


def add_count_within_buffer(
    base_df,
    source_df,
    out_col,
    buffer_m=275,
    source_lat_col="lat",
    source_lon_col="lon",
    source_filter_col=None,
    source_filter_values=None,
):
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(epsg=3857)

    src = source_df.copy()
    if source_filter_col is not None and source_filter_values is not None:
        src = src[src[source_filter_col].isin(source_filter_values)].copy()

    if src.empty:
        out = base_df.copy()
        out[out_col] = 0
        return out

    source_gdf = make_points_gdf(src, source_lat_col, source_lon_col).to_crs(epsg=3857)

    stations_buffer = stations_gdf[["id", "geometry"]].copy()
    stations_buffer["geometry"] = stations_buffer.geometry.buffer(buffer_m)

    joined = gpd.sjoin(source_gdf, stations_buffer, how="inner", predicate="within")
    counts = joined.groupby("id").size().rename(out_col).reset_index()

    out = base_df.merge(counts, on="id", how="left")
    out[out_col] = out[out_col].fillna(0).astype(int)
    return out


def add_sum_within_buffer(
    base_df,
    source_df,
    value_col,
    out_col,
    buffer_m=275,
    source_lat_col="lat",
    source_lon_col="lon",
    source_filter_col=None,
    source_filter_values=None,
):
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(epsg=3857)

    src = source_df.copy()
    if source_filter_col is not None and source_filter_values is not None:
        src = src[src[source_filter_col].isin(source_filter_values)].copy()

    if src.empty:
        out = base_df.copy()
        out[out_col] = 0
        return out

    source_gdf = make_points_gdf(src, source_lat_col, source_lon_col).to_crs(epsg=3857)

    stations_buffer = stations_gdf[["id", "geometry"]].copy()
    stations_buffer["geometry"] = stations_buffer.geometry.buffer(buffer_m)

    joined = gpd.sjoin(source_gdf, stations_buffer, how="inner", predicate="within")
    sums = joined.groupby("id")[value_col].sum().rename(out_col).reset_index()

    out = base_df.merge(sums, on="id", how="left")
    out[out_col] = out[out_col].fillna(0)
    return out


def add_nearest_distance(
    base_df,
    source_df,
    out_col,
    source_lat_col="lat",
    source_lon_col="lon",
    source_filter_col=None,
    source_filter_values=None,
):
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(epsg=3857)

    src = source_df.copy()
    if source_filter_col is not None and source_filter_values is not None:
        src = src[src[source_filter_col].isin(source_filter_values)].copy()

    if src.empty:
        out = base_df.copy()
        out[out_col] = np.nan
        return out

    source_gdf = make_points_gdf(src, source_lat_col, source_lon_col).to_crs(epsg=3857)

    stations_gdf[out_col] = stations_gdf.geometry.apply(
        lambda station_geom: source_gdf.distance(station_geom).min()
    )

    out = base_df.merge(stations_gdf[["id", out_col]], on="id", how="left")
    out[out_col] = out[out_col].round(2)
    return out


def add_avg_k_nearest_distance(
    base_df,
    source_df,
    out_col,
    k=3,
    source_lat_col="lat",
    source_lon_col="lon",
    source_filter_col=None,
    source_filter_values=None,
):
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(epsg=3857)

    src = source_df.copy()
    if source_filter_col is not None and source_filter_values is not None:
        src = src[src[source_filter_col].isin(source_filter_values)].copy()

    if src.empty:
        out = base_df.copy()
        out[out_col] = np.nan
        return out

    source_gdf = make_points_gdf(src, source_lat_col, source_lon_col).to_crs(epsg=3857)

    def avg_k_dist(station_geom):
        dists = source_gdf.geometry.distance(station_geom).sort_values().values
        return dists[: min(k, len(dists))].mean()

    stations_gdf[out_col] = stations_gdf.geometry.apply(avg_k_dist)

    out = base_df.merge(stations_gdf[["id", out_col]], on="id", how="left")
    out[out_col] = out[out_col].round(2)
    return out


def add_nearest_dorm_info(base_df, dorms_df):
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(epsg=3857)
    dorms_gdf = make_points_gdf(dorms_df, "lat", "lon").to_crs(epsg=3857)

    if dorms_gdf.empty:
        out = base_df.copy()
        out["nearest_dorm_dist_m"] = np.nan
        out["nearest_dorm_pop"] = np.nan
        return out

    def nearest_dorm(station_geom):
        dists = dorms_gdf.geometry.distance(station_geom)
        nearest_idx = dists.idxmin()
        return pd.Series(
            {
                "nearest_dorm_dist_m": dists.loc[nearest_idx],
                "nearest_dorm_pop": dorms_gdf.loc[nearest_idx, "population"],
            }
        )

    stations_gdf[["nearest_dorm_dist_m", "nearest_dorm_pop"]] = (
        stations_gdf.geometry.apply(nearest_dorm)
    )

    out = base_df.merge(
        stations_gdf[["id", "nearest_dorm_dist_m", "nearest_dorm_pop"]],
        on="id",
        how="left",
    )

    out["nearest_dorm_dist_m"] = out["nearest_dorm_dist_m"].round(2)
    return out


def add_network_features(base_df):
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(epsg=3857)

    def station_network_metrics(row):
        dists = stations_gdf.geometry.distance(row.geometry)
        dists = dists[dists > 0].sort_values()

        nearest_station_dist_m = dists.iloc[0] if len(dists) >= 1 else np.nan
        stations_within_500m = int((dists <= 500).sum())
        stations_within_1000m = int((dists <= 1000).sum())
        avg_stations_dist_3_nearest_m = (
            dists.iloc[:3].mean() if len(dists) >= 3 else np.nan
        )

        return pd.Series(
            {
                "nearest_station_dist_m": nearest_station_dist_m,
                "stations_within_500m": stations_within_500m,
                "stations_within_1000m": stations_within_1000m,
                "avg_stations_dist_3_nearest_m": avg_stations_dist_3_nearest_m,
            }
        )

    stations_gdf[
        [
            "nearest_station_dist_m",
            "stations_within_500m",
            "stations_within_1000m",
            "avg_stations_dist_3_nearest_m",
        ]
    ] = stations_gdf.apply(station_network_metrics, axis=1)

    out = base_df.merge(
        stations_gdf[
            [
                "id",
                "nearest_station_dist_m",
                "stations_within_500m",
                "stations_within_1000m",
                "avg_stations_dist_3_nearest_m",
            ]
        ],
        on="id",
        how="left",
    )

    out["nearest_station_dist_m"] = out["nearest_station_dist_m"].round(2)
    out["avg_stations_dist_3_nearest_m"] = out["avg_stations_dist_3_nearest_m"].round(2)

    return out


def add_park_area_within_buffer(base_df, parks_df, buffer_m=275):
    stations_gdf = make_points_gdf(base_df, "lat", "lon")

    parks_polys = parks_df.copy()
    parks_polys["geometry"] = parks_polys["geometry"].apply(wkt.loads)
    parks_gdf = gpd.GeoDataFrame(parks_polys, geometry="geometry", crs="EPSG:26914")

    stations_gdf = stations_gdf.to_crs(parks_gdf.crs)

    stations_buffer = stations_gdf[["id", "geometry"]].copy()
    stations_buffer["geometry"] = stations_buffer.geometry.buffer(buffer_m)

    park_intersections = gpd.overlay(parks_gdf, stations_buffer, how="intersection")
    park_intersections["park_area_part"] = park_intersections.geometry.area

    park_area = (
        park_intersections.groupby("id")["park_area_part"]
        .sum()
        .rename("park_area_nearby")
        .reset_index()
    )

    out = base_df.merge(park_area, on="id", how="left")
    out["park_area_nearby"] = out["park_area_nearby"].fillna(0).round().astype(int)
    return out


def add_nearest_park_distance(base_df, parks_df):
    stations_gdf = make_points_gdf(base_df, "lat", "lon")

    parks_polys = parks_df.copy()
    parks_polys["geometry"] = parks_polys["geometry"].apply(wkt.loads)
    parks_gdf = gpd.GeoDataFrame(parks_polys, geometry="geometry", crs="EPSG:26914")

    stations_gdf = stations_gdf.to_crs(parks_gdf.crs)

    stations_gdf["nearest_park_dist_m"] = stations_gdf.geometry.apply(
        lambda station_geom: parks_gdf.distance(station_geom).min()
        if len(parks_gdf) > 0
        else np.nan
    )

    out = base_df.merge(
        stations_gdf[["id", "nearest_park_dist_m"]],
        on="id",
        how="left",
    )
    out["nearest_park_dist_m"] = out["nearest_park_dist_m"].round(2)
    return out


def add_manual_point_distance(base_df, out_col, point_lat, point_lon):
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(epsg=3857)

    point_gdf = gpd.GeoDataFrame(
        {"name": [out_col]},
        geometry=[Point(point_lon, point_lat)],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    target_geom = point_gdf.geometry.iloc[0]
    stations_gdf[out_col] = stations_gdf.geometry.distance(target_geom)

    out = base_df.merge(stations_gdf[["id", out_col]], on="id", how="left")
    out[out_col] = out[out_col].round(2)
    return out


def add_hotspot_summary_features(
    base_df,
    source_df,
    prefix,
    buffer_m_list=(300, 500),
    k=3,
    source_lat_col="lat",
    source_lon_col="lon",
):
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(epsg=3857)
    source_gdf = make_points_gdf(source_df, source_lat_col, source_lon_col).to_crs(
        epsg=3857
    )

    if source_gdf.empty:
        out = base_df.copy()
        out[f"min_dist_to_{prefix}_m"] = np.nan
        out[f"avg_dist_{k}_nearest_{prefix}_m"] = np.nan
        for buf in buffer_m_list:
            out[f"{prefix}_within_{buf}m"] = 0
        return out

    def calc_metrics(station_geom):
        dists = source_gdf.geometry.distance(station_geom).sort_values().values
        metrics = {
            f"min_dist_to_{prefix}_m": dists[0] if len(dists) >= 1 else np.nan,
            f"avg_dist_{k}_nearest_{prefix}_m": dists[: min(k, len(dists))].mean()
            if len(dists) >= 1
            else np.nan,
        }
        for buf in buffer_m_list:
            metrics[f"{prefix}_within_{buf}m"] = int((dists <= buf).sum())
        return pd.Series(metrics)

    feature_cols = [
        f"min_dist_to_{prefix}_m",
        f"avg_dist_{k}_nearest_{prefix}_m",
        *[f"{prefix}_within_{buf}m" for buf in buffer_m_list],
    ]

    stations_gdf[feature_cols] = stations_gdf.geometry.apply(calc_metrics)

    out = base_df.merge(stations_gdf[["id"] + feature_cols], on="id", how="left")

    for col in feature_cols:
        if "dist" in col:
            out[col] = out[col].round(2)
        else:
            out[col] = out[col].fillna(0).astype(int)

    return out


# %%
# -----------------------------
# Station network features
# -----------------------------
scores_and_coords = add_network_features(scores_and_coords)

# %%
# -----------------------------
# Transit
# -----------------------------
scores_and_coords = add_count_within_buffer(
    base_df=scores_and_coords,
    source_df=transit,
    out_col="transit_nearby",
    buffer_m=275,
)

scores_and_coords = add_nearest_distance(
    base_df=scores_and_coords,
    source_df=transit,
    out_col="nearest_transit_stop_dist_m",
)

scores_and_coords = add_avg_k_nearest_distance(
    base_df=scores_and_coords,
    source_df=transit,
    out_col="avg_dist_3_nearest_transit_stops_m",
    k=3,
)

# %%
# -----------------------------
# Jobs + housing
# -----------------------------
scores_and_coords = add_sum_within_buffer(
    base_df=scores_and_coords,
    source_df=jobs,
    value_col="job_count",
    out_col="jobs_nearby_275m",
    buffer_m=275,
)

scores_and_coords = add_sum_within_buffer(
    base_df=scores_and_coords,
    source_df=housing,
    value_col="count",
    out_col="housing_nearby_275m",
    buffer_m=275,
)

scores_and_coords = add_sum_within_buffer(
    base_df=scores_and_coords,
    source_df=housing,
    value_col="count",
    out_col="housing_nearby_1000m",
    buffer_m=1000,
)

scores_and_coords["job_housing_ratio_275m"] = np.where(
    scores_and_coords["housing_nearby_275m"] > 0,
    scores_and_coords["jobs_nearby_275m"] / scores_and_coords["housing_nearby_275m"],
    scores_and_coords["jobs_nearby_275m"],
)
scores_and_coords["job_housing_ratio_275m"] = (
    scores_and_coords["job_housing_ratio_275m"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

# %%
# -----------------------------
# Amenities
# -----------------------------
scores_and_coords = add_count_within_buffer(
    base_df=scores_and_coords,
    source_df=amenities,
    out_col="amenities_nearby",
    buffer_m=275,
)

scores_and_coords = add_avg_k_nearest_distance(
    base_df=scores_and_coords,
    source_df=amenities,
    out_col="avg_dist_3_nearest_amenities_m",
    k=3,
)

# %%
# -----------------------------
# Parks
# -----------------------------
scores_and_coords = add_park_area_within_buffer(scores_and_coords, parks, buffer_m=275)
scores_and_coords = add_nearest_park_distance(scores_and_coords, parks)

# %%
# -----------------------------
# Retail / entertainment / tourism
# -----------------------------
entertainment_types = {
    "amenity_bar",
    "amenity_cafe",
    "amenity_restaurant",
    "amenity_pub",
    "amenity_theatre",
    "amenity_cinema",
    "amenity_nightclub",
}

tourism_types = {
    "tourism_attraction",
    "tourism_museum",
    "tourism_gallery",
    "tourism_viewpoint",
}

scores_and_coords = add_count_within_buffer(
    base_df=scores_and_coords,
    source_df=retail,
    out_col="retail_nearby",
    buffer_m=275,
)

scores_and_coords = add_avg_k_nearest_distance(
    base_df=scores_and_coords,
    source_df=retail,
    out_col="avg_dist_3_nearest_retail_m",
    k=3,
)

scores_and_coords = add_count_within_buffer(
    base_df=scores_and_coords,
    source_df=retail,
    out_col="entertainment_nearby",
    buffer_m=275,
    source_filter_col="type",
    source_filter_values=entertainment_types,
)

scores_and_coords = add_avg_k_nearest_distance(
    base_df=scores_and_coords,
    source_df=retail,
    out_col="avg_dist_3_nearest_entertainment_m",
    k=3,
    source_filter_col="type",
    source_filter_values=entertainment_types,
)

scores_and_coords = add_count_within_buffer(
    base_df=scores_and_coords,
    source_df=retail,
    out_col="tourism_nearby",
    buffer_m=275,
    source_filter_col="type",
    source_filter_values=tourism_types,
)

scores_and_coords = add_avg_k_nearest_distance(
    base_df=scores_and_coords,
    source_df=retail,
    out_col="avg_dist_3_nearest_tourism_m",
    k=3,
    source_filter_col="type",
    source_filter_values=tourism_types,
)

# %%
# -----------------------------
# Dining halls + dorms
# -----------------------------
scores_and_coords = add_nearest_distance(
    base_df=scores_and_coords,
    source_df=dining_halls,
    out_col="nearest_dining_hall_dist_m",
)

scores_and_coords = add_nearest_dorm_info(scores_and_coords, dorms)

scores_and_coords = add_sum_within_buffer(
    base_df=scores_and_coords,
    source_df=dorms,
    value_col="population",
    out_col="dorm_pop_within_500m",
    buffer_m=500,
)

# %%
# -----------------------------
# UT hotspots
# -----------------------------
scores_and_coords = add_hotspot_summary_features(
    base_df=scores_and_coords,
    source_df=ut_hotspots,
    prefix="ut_hotspot",
    buffer_m_list=(300, 500),
    k=3,
)

# %%
# -----------------------------
# West Campus hotspots
# -----------------------------
scores_and_coords = add_hotspot_summary_features(
    base_df=scores_and_coords,
    source_df=wampus_hotspots,
    prefix="wampus_hotspot",
    buffer_m_list=(300, 500),
    k=3,
)

# %%
# -----------------------------
# Manual West Campus center point
# Adjust coords if you want a different proxy center
# -----------------------------
scores_and_coords = add_manual_point_distance(
    base_df=scores_and_coords,
    out_col="dist_to_west_campus_center_m",
    point_lat=30.2885,
    point_lon=-97.7475,
)

# %%
# -----------------------------
# UT interaction features
# -----------------------------
scores_and_coords["ut_x_dorm_pop_500m"] = (
    scores_and_coords["is_ut"] * scores_and_coords["dorm_pop_within_500m"]
)

scores_and_coords["ut_x_dining_dist"] = (
    scores_and_coords["is_ut"] * scores_and_coords["nearest_dining_hall_dist_m"]
)

scores_and_coords["ut_x_transit"] = (
    scores_and_coords["is_ut"] * scores_and_coords["transit_nearby"]
)

scores_and_coords["ut_x_housing_275m"] = (
    scores_and_coords["is_ut"] * scores_and_coords["housing_nearby_275m"]
)

scores_and_coords["ut_x_ut_hotspots_300m"] = (
    scores_and_coords["is_ut"] * scores_and_coords["ut_hotspot_within_300m"]
)

scores_and_coords["ut_x_wampus_hotspots_300m"] = (
    scores_and_coords["is_ut"] * scores_and_coords["wampus_hotspot_within_300m"]
)

# %%
# -----------------------------
# Final column selection
# -----------------------------
scores_and_coords = scores_and_coords[
    [
        "id",
        "name",
        "district",
        "total_docks",
        "trips_per_dock",
        "ebs_station",
        "is_ut",
        "lat",
        "lon",
        # transit
        "transit_nearby",
        "nearest_transit_stop_dist_m",
        "avg_dist_3_nearest_transit_stops_m",
        # jobs / housing
        "jobs_nearby_275m",
        "housing_nearby_275m",
        "housing_nearby_1000m",
        "job_housing_ratio_275m",
        "low_income_access_score",
        # amenities / parks / retail
        "amenities_nearby",
        "avg_dist_3_nearest_amenities_m",
        "park_area_nearby",
        "nearest_park_dist_m",
        "bike_infra_score",
        "retail_nearby",
        "avg_dist_3_nearest_retail_m",
        "entertainment_nearby",
        "avg_dist_3_nearest_entertainment_m",
        "tourism_nearby",
        "avg_dist_3_nearest_tourism_m",
        # bikeshare network
        "nearest_station_dist_m",
        "stations_within_500m",
        "stations_within_1000m",
        "avg_stations_dist_3_nearest_m",
        # campus-specific
        "nearest_dining_hall_dist_m",
        "nearest_dorm_dist_m",
        "nearest_dorm_pop",
        "dorm_pop_within_500m",
        "min_dist_to_ut_hotspot_m",
        "avg_dist_3_nearest_ut_hotspot_m",
        "ut_hotspot_within_300m",
        "ut_hotspot_within_500m",
        "min_dist_to_wampus_hotspot_m",
        "avg_dist_3_nearest_wampus_hotspot_m",
        "wampus_hotspot_within_300m",
        "wampus_hotspot_within_500m",
        "dist_to_west_campus_center_m",
        # interactions
        "ut_x_dorm_pop_500m",
        "ut_x_dining_dist",
        "ut_x_transit",
        "ut_x_housing_275m",
        "ut_x_ut_hotspots_300m",
        "ut_x_wampus_hotspots_300m",
    ]
].copy()


# %%
def to_snake_case(text):
    if pd.isna(text):
        return text

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"^_+|_+$", "", text)
    return text


scores_and_coords["name"] = scores_and_coords["name"].apply(to_snake_case)

# %%
# -----------------------------
# Save combined dataset
# -----------------------------
scores_and_coords.to_csv(
    output_prefix + "combined_dataset_v5.csv",
    index=False,
)

# %%
# -----------------------------
# Build ML dataset with scaling
# -----------------------------
df = scores_and_coords.copy()

target = "trips_per_dock"

# keep identifiers / raw text / non-feature cols out of X
drop_cols_for_model = ["id", "name", "district", target]

binary_cols = ["ebs_station", "is_ut"]
ordinal_cols = ["low_income_access_score", "bike_infra_score"]
coord_cols = ["lat", "lon"]

scale_cols = [
    "total_docks",
    "transit_nearby",
    "nearest_transit_stop_dist_m",
    "avg_dist_3_nearest_transit_stops_m",
    "jobs_nearby_275m",
    "housing_nearby_275m",
    "housing_nearby_1000m",
    "job_housing_ratio_275m",
    "amenities_nearby",
    "avg_dist_3_nearest_amenities_m",
    "park_area_nearby",
    "nearest_park_dist_m",
    "retail_nearby",
    "avg_dist_3_nearest_retail_m",
    "entertainment_nearby",
    "avg_dist_3_nearest_entertainment_m",
    "tourism_nearby",
    "avg_dist_3_nearest_tourism_m",
    "nearest_station_dist_m",
    "stations_within_500m",
    "stations_within_1000m",
    "avg_stations_dist_3_nearest_m",
    "nearest_dining_hall_dist_m",
    "nearest_dorm_dist_m",
    "nearest_dorm_pop",
    "dorm_pop_within_500m",
    "min_dist_to_ut_hotspot_m",
    "avg_dist_3_nearest_ut_hotspot_m",
    "ut_hotspot_within_300m",
    "ut_hotspot_within_500m",
    "min_dist_to_wampus_hotspot_m",
    "avg_dist_3_nearest_wampus_hotspot_m",
    "wampus_hotspot_within_300m",
    "wampus_hotspot_within_500m",
    "dist_to_west_campus_center_m",
    "ut_x_dorm_pop_500m",
    "ut_x_dining_dist",
    "ut_x_transit",
    "ut_x_housing_275m",
    "ut_x_ut_hotspots_300m",
    "ut_x_wampus_hotspots_300m",
    "lat",
    "lon",
]

X = df.drop(columns=drop_cols_for_model)
y = df[target]

# make sure scale_cols only includes columns that actually exist
scale_cols = [col for col in scale_cols if col in X.columns]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[scale_cols] = scaler.fit_transform(X[scale_cols])

ml_dataset = X_scaled.copy()
ml_dataset[target] = y

ml_dataset.to_csv(
    output_prefix + "ml_dataset_v5.csv",
    index=False,
)

# %%
print(scores_and_coords.shape)
print(ml_dataset.shape)
print(ml_dataset.head())
