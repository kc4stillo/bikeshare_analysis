# %%
import os
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

# %%
CLEANED_PREFIX = "../../cleaned/"
OUTPUT_PREFIX = "../../cleaned/combined_datasets/v6/"
SOURCE_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:26914"  # UTM 14N, good for Austin-area meter distances
BUFFER_SIZES_M = (275, 500, 1000)
PARK_BUFFER_M = 275

Path(OUTPUT_PREFIX).mkdir(parents=True, exist_ok=True)


# %%
# -----------------------------
# Small utilities
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()
    return out


def to_numeric_inplace(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def safe_load_wkt(value):
    if pd.isna(value):
        return None
    try:
        return wkt.loads(value)
    except Exception:
        return None


def union_geometries(gdf: gpd.GeoDataFrame):
    if gdf.empty:
        return None

    try:
        # shapely/geopandas newer versions
        return gdf.geometry.union_all()
    except Exception:
        # fallback for older versions
        return gdf.geometry.unary_union


# %%
# -----------------------------
# Load datasets
# -----------------------------
amenities = normalize_columns(pd.read_csv(CLEANED_PREFIX + "amenities/amenities.csv"))
coords = normalize_columns(
    pd.read_csv(CLEANED_PREFIX + "coords/bikeshare_stations.csv")
)
housing = normalize_columns(pd.read_csv(CLEANED_PREFIX + "housing/housing.csv"))
jobs = normalize_columns(pd.read_csv(CLEANED_PREFIX + "jobs/jobs.csv"))
retail = normalize_columns(pd.read_csv(CLEANED_PREFIX + "retail/retail.csv"))
scores = normalize_columns(pd.read_csv(CLEANED_PREFIX + "scoring/current_stations.csv"))
transit = normalize_columns(pd.read_csv(CLEANED_PREFIX + "transit/transit.csv"))
parks = normalize_columns(pd.read_csv(CLEANED_PREFIX + "amenities/parks.csv"))
dining_halls = normalize_columns(
    pd.read_csv(CLEANED_PREFIX + "amenities/dining_halls.csv")
)
dorms = normalize_columns(pd.read_csv(CLEANED_PREFIX + "housing/dorms.csv"))
ut_shape = normalize_columns(pd.read_csv(CLEANED_PREFIX + "coords/ut_shape.csv"))
west_campus_shape = normalize_columns(
    pd.read_csv(CLEANED_PREFIX + "coords/west_campus.csv")
)
north_campus_shape = normalize_columns(
    pd.read_csv(CLEANED_PREFIX + "coords/north_campus.csv")
)

# Numeric cleanup for known numeric columns
coords = to_numeric_inplace(coords, ["lat", "lon"])
housing = to_numeric_inplace(housing, ["count", "lat", "lon"])
jobs = to_numeric_inplace(jobs, ["job_count", "lat", "lon"])
amenities = to_numeric_inplace(amenities, ["lat", "lon"])
retail = to_numeric_inplace(retail, ["lat", "lon"])
transit = to_numeric_inplace(transit, ["lat", "lon"])
dining_halls = to_numeric_inplace(dining_halls, ["lat", "lon"])
dorms = to_numeric_inplace(dorms, ["population", "lat", "lon"])
scores = to_numeric_inplace(
    scores,
    [
        "id",
        "trips",
        "total_docks",
        "trips_per_dock",
        "ebs_station",
        "checkouts_rank_per_day",
        "transit_access_score",
        "jobs_access_score",
        "households_access_score",
        "low_income_access_score",
        "public_amenities_access_score",
        "bike_infra_score",
        "retail_entertainment_access_score",
        "existing_bikeshare_access_score",
        "total_score",
        "is_ut",
    ],
)

# %%
# -----------------------------
# Merge scores + coordinates
# -----------------------------
coords_for_merge = coords[["scoring_name", "lat", "lon"]].drop_duplicates(
    subset=["scoring_name"]
)

scores_and_coords = scores.merge(
    coords_for_merge,
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
def make_points_gdf(df, lat_col="lat", lon_col="lon", crs=SOURCE_CRS):
    out = normalize_columns(df)
    out = out.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=[lat_col, lon_col]).copy()

    return gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out[lon_col], out[lat_col]),
        crs=crs,
    )


def make_polygon_gdf(df, shape_col="shape", source_crs=SOURCE_CRS):
    out = normalize_columns(df)
    out = out.copy()

    if shape_col not in out.columns:
        raise KeyError(
            f"Column '{shape_col}' not found. Available columns: {list(out.columns)}"
        )

    out["geometry"] = out[shape_col].apply(safe_load_wkt)
    out = out.dropna(subset=["geometry"]).copy()

    if out.empty:
        return gpd.GeoDataFrame(out, geometry="geometry", crs=source_crs)

    out = gpd.GeoDataFrame(out, geometry="geometry", crs=source_crs)

    # light cleanup for invalid polygons
    out["geometry"] = out.geometry.buffer(0)
    out = out[~out.geometry.is_empty].copy()

    return out


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
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(PROJECTED_CRS)

    src = source_df.copy()
    if source_filter_col is not None and source_filter_values is not None:
        src = src[src[source_filter_col].isin(source_filter_values)].copy()

    if src.empty:
        out = base_df.copy()
        out[out_col] = 0
        return out

    source_gdf = make_points_gdf(src, source_lat_col, source_lon_col).to_crs(
        PROJECTED_CRS
    )

    stations_buffer = stations_gdf[["id", "geometry"]].copy()
    stations_buffer["geometry"] = stations_buffer.geometry.buffer(buffer_m)

    # intersects is slightly safer than within because boundary points still count
    joined = gpd.sjoin(source_gdf, stations_buffer, how="inner", predicate="intersects")
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
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(PROJECTED_CRS)

    src = source_df.copy()
    if source_filter_col is not None and source_filter_values is not None:
        src = src[src[source_filter_col].isin(source_filter_values)].copy()

    if src.empty:
        out = base_df.copy()
        out[out_col] = 0
        return out

    src[value_col] = pd.to_numeric(src[value_col], errors="coerce").fillna(0)
    source_gdf = make_points_gdf(src, source_lat_col, source_lon_col).to_crs(
        PROJECTED_CRS
    )

    stations_buffer = stations_gdf[["id", "geometry"]].copy()
    stations_buffer["geometry"] = stations_buffer.geometry.buffer(buffer_m)

    joined = gpd.sjoin(source_gdf, stations_buffer, how="inner", predicate="intersects")
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
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(PROJECTED_CRS)

    src = source_df.copy()
    if source_filter_col is not None and source_filter_values is not None:
        src = src[src[source_filter_col].isin(source_filter_values)].copy()

    if src.empty:
        out = base_df.copy()
        out[out_col] = np.nan
        return out

    source_gdf = make_points_gdf(src, source_lat_col, source_lon_col).to_crs(
        PROJECTED_CRS
    )

    source_union = union_geometries(source_gdf)
    stations_gdf[out_col] = stations_gdf.geometry.apply(
        lambda station_geom: station_geom.distance(source_union)
        if source_union is not None
        else np.nan
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
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(PROJECTED_CRS)

    src = source_df.copy()
    if source_filter_col is not None and source_filter_values is not None:
        src = src[src[source_filter_col].isin(source_filter_values)].copy()

    if src.empty:
        out = base_df.copy()
        out[out_col] = np.nan
        return out

    source_gdf = make_points_gdf(src, source_lat_col, source_lon_col).to_crs(
        PROJECTED_CRS
    )

    def avg_k_dist(station_geom):
        dists = source_gdf.geometry.distance(station_geom).sort_values().values
        return dists[: min(k, len(dists))].mean() if len(dists) > 0 else np.nan

    stations_gdf[out_col] = stations_gdf.geometry.apply(avg_k_dist)

    out = base_df.merge(stations_gdf[["id", out_col]], on="id", how="left")
    out[out_col] = out[out_col].round(2)
    return out


def add_nearest_dorm_info(base_df, dorms_df):
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(PROJECTED_CRS)
    dorms_gdf = make_points_gdf(dorms_df, "lat", "lon").to_crs(PROJECTED_CRS)

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
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(PROJECTED_CRS)

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


def make_parks_gdf(parks_df, geometry_col="geometry"):
    parks_polys = normalize_columns(parks_df)
    parks_polys = parks_polys.copy()

    if geometry_col not in parks_polys.columns:
        raise KeyError(
            f"Column '{geometry_col}' not found in parks data. Available columns: {list(parks_polys.columns)}"
        )

    parks_polys["geometry"] = parks_polys[geometry_col].apply(safe_load_wkt)
    parks_polys = parks_polys.dropna(subset=["geometry"]).copy()

    if parks_polys.empty:
        return gpd.GeoDataFrame(parks_polys, geometry="geometry", crs=SOURCE_CRS)

    # parks.csv WKT appears to be lon/lat, so initialize as EPSG:4326 first,
    # then project to meters for area + distance operations.
    parks_gdf = gpd.GeoDataFrame(parks_polys, geometry="geometry", crs=SOURCE_CRS)
    parks_gdf["geometry"] = parks_gdf.geometry.buffer(0)
    parks_gdf = parks_gdf[~parks_gdf.geometry.is_empty].copy()
    parks_gdf = parks_gdf.to_crs(PROJECTED_CRS)
    return parks_gdf


def add_park_area_within_buffer(base_df, parks_df, buffer_m=275, out_col=None):
    if out_col is None:
        out_col = f"park_area_within_{buffer_m}m"

    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(PROJECTED_CRS)
    parks_gdf = make_parks_gdf(parks_df)

    out = base_df.copy()
    if parks_gdf.empty:
        out[out_col] = 0.0
        return out

    parks_union = union_geometries(parks_gdf)
    if parks_union is None:
        out[out_col] = 0.0
        return out

    stations_gdf[out_col] = stations_gdf.geometry.buffer(buffer_m).apply(
        lambda geom: geom.intersection(parks_union).area
    )

    out = out.merge(stations_gdf[["id", out_col]], on="id", how="left")
    out[out_col] = out[out_col].fillna(0).round(2)
    return out


def add_nearest_park_distance(base_df, parks_df, out_col="nearest_park_dist_m"):
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(PROJECTED_CRS)
    parks_gdf = make_parks_gdf(parks_df)

    out = base_df.copy()
    if parks_gdf.empty:
        out[out_col] = np.nan
        return out

    parks_union = union_geometries(parks_gdf)
    if parks_union is None:
        out[out_col] = np.nan
        return out

    stations_gdf[out_col] = stations_gdf.geometry.apply(
        lambda station_geom: station_geom.distance(parks_union)
    )

    out = out.merge(stations_gdf[["id", out_col]], on="id", how="left")
    out[out_col] = out[out_col].round(2)
    return out


def add_manual_point_distance(base_df, out_col, point_lat, point_lon):
    stations_gdf = make_points_gdf(base_df, "lat", "lon").to_crs(PROJECTED_CRS)

    point_gdf = gpd.GeoDataFrame(
        {"name": [out_col]},
        geometry=[Point(point_lon, point_lat)],
        crs=SOURCE_CRS,
    ).to_crs(PROJECTED_CRS)

    target_geom = point_gdf.geometry.iloc[0]
    stations_gdf[out_col] = stations_gdf.geometry.distance(target_geom)

    out = base_df.merge(stations_gdf[["id", out_col]], on="id", how="left")
    out[out_col] = out[out_col].round(2)
    return out


def add_polygon_zone_features(
    base_df,
    shape_df,
    prefix,
    buffer_m_list=(275, 500),
    shape_col="shape",
):
    stations_gdf = make_points_gdf(base_df, "lat", "lon")
    zones_gdf = make_polygon_gdf(shape_df, shape_col=shape_col)

    if zones_gdf.empty:
        out = base_df.copy()
        out[f"in_{prefix}"] = 0
        out[f"dist_to_{prefix}_m"] = np.nan
        for buf in buffer_m_list:
            out[f"{prefix}_area_within_{buf}m"] = 0.0
            out[f"{prefix}_share_of_{buf}m_buffer"] = 0.0
            out[f"{prefix}_touches_{buf}m_buffer"] = 0
        return out

    stations_proj = stations_gdf.to_crs(PROJECTED_CRS)
    zones_proj = zones_gdf.to_crs(PROJECTED_CRS)

    # dissolve into one geometry so multiple rows still behave as one zone
    zone_geom = union_geometries(zones_proj)

    stations_proj[f"in_{prefix}"] = stations_proj.geometry.within(zone_geom).astype(int)
    stations_proj[f"dist_to_{prefix}_m"] = stations_proj.geometry.distance(zone_geom)

    for buf in buffer_m_list:
        col_area = f"{prefix}_area_within_{buf}m"
        col_share = f"{prefix}_share_of_{buf}m_buffer"
        col_touch = f"{prefix}_touches_{buf}m_buffer"

        station_buffers = stations_proj[["id", "geometry"]].copy()
        station_buffers["geometry"] = station_buffers.geometry.buffer(buf)
        station_buffers["buffer_area_m2"] = station_buffers.geometry.area

        station_buffers[col_area] = station_buffers.geometry.apply(
            lambda geom: geom.intersection(zone_geom).area
        )
        station_buffers[col_share] = np.where(
            station_buffers["buffer_area_m2"] > 0,
            station_buffers[col_area] / station_buffers["buffer_area_m2"],
            0,
        )
        station_buffers[col_touch] = (station_buffers[col_area] > 0).astype(int)

        stations_proj = stations_proj.merge(
            station_buffers[["id", col_area, col_share, col_touch]],
            on="id",
            how="left",
        )

        stations_proj[col_area] = stations_proj[col_area].fillna(0).round(2)
        stations_proj[col_share] = stations_proj[col_share].fillna(0).round(4)
        stations_proj[col_touch] = stations_proj[col_touch].fillna(0).astype(int)

    keep_cols = ["id", f"in_{prefix}", f"dist_to_{prefix}_m"]
    for buf in buffer_m_list:
        keep_cols.extend(
            [
                f"{prefix}_area_within_{buf}m",
                f"{prefix}_share_of_{buf}m_buffer",
                f"{prefix}_touches_{buf}m_buffer",
            ]
        )

    out = base_df.merge(stations_proj[keep_cols], on="id", how="left")
    out[f"in_{prefix}"] = out[f"in_{prefix}"].fillna(0).astype(int)
    out[f"dist_to_{prefix}_m"] = out[f"dist_to_{prefix}_m"].round(2)

    for buf in buffer_m_list:
        out[f"{prefix}_area_within_{buf}m"] = (
            out[f"{prefix}_area_within_{buf}m"].fillna(0).round(2)
        )
        out[f"{prefix}_share_of_{buf}m_buffer"] = (
            out[f"{prefix}_share_of_{buf}m_buffer"].fillna(0).round(4)
        )
        out[f"{prefix}_touches_{buf}m_buffer"] = (
            out[f"{prefix}_touches_{buf}m_buffer"].fillna(0).astype(int)
        )

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
scores_and_coords = add_park_area_within_buffer(
    scores_and_coords,
    parks,
    buffer_m=PARK_BUFFER_M,
    out_col="park_area_nearby",  # keeps backward-compatible column name
)
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
# Manual West Campus center point
# -----------------------------
scores_and_coords = add_manual_point_distance(
    base_df=scores_and_coords,
    out_col="dist_to_west_campus_center_m",
    point_lat=30.288500,
    point_lon=-97.747500,
)

# %%
# -----------------------------
# Polygon campus-zone features
# -----------------------------
scores_and_coords = add_polygon_zone_features(
    base_df=scores_and_coords,
    shape_df=ut_shape,
    prefix="ut_shape",
    buffer_m_list=BUFFER_SIZES_M,
)

scores_and_coords = add_polygon_zone_features(
    base_df=scores_and_coords,
    shape_df=west_campus_shape,
    prefix="west_campus_shape",
    buffer_m_list=BUFFER_SIZES_M,
)

scores_and_coords = add_polygon_zone_features(
    base_df=scores_and_coords,
    shape_df=north_campus_shape,
    prefix="north_campus_shape",
    buffer_m_list=BUFFER_SIZES_M,
)

# %%
# -----------------------------
# Zone interaction features
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

scores_and_coords["ut_x_in_ut_shape"] = (
    scores_and_coords["is_ut"] * scores_and_coords["in_ut_shape"]
)

scores_and_coords["ut_x_in_west_campus_shape"] = (
    scores_and_coords["is_ut"] * scores_and_coords["in_west_campus_shape"]
)

scores_and_coords["ut_x_in_north_campus_shape"] = (
    scores_and_coords["is_ut"] * scores_and_coords["in_north_campus_shape"]
)

scores_and_coords["ut_x_ut_shape_share_275m"] = (
    scores_and_coords["is_ut"] * scores_and_coords["ut_shape_share_of_275m_buffer"]
)

scores_and_coords["ut_x_west_campus_share_275m"] = (
    scores_and_coords["is_ut"]
    * scores_and_coords["west_campus_shape_share_of_275m_buffer"]
)

scores_and_coords["ut_x_north_campus_share_275m"] = (
    scores_and_coords["is_ut"]
    * scores_and_coords["north_campus_shape_share_of_275m_buffer"]
)

# %%
# -----------------------------
# Final column selection
# -----------------------------
final_columns = [
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
    # campus-specific non-shape
    "nearest_dining_hall_dist_m",
    "nearest_dorm_dist_m",
    "nearest_dorm_pop",
    "dorm_pop_within_500m",
    "dist_to_west_campus_center_m",
    # ut shape features
    "in_ut_shape",
    "dist_to_ut_shape_m",
    "ut_shape_area_within_275m",
    "ut_shape_share_of_275m_buffer",
    "ut_shape_touches_275m_buffer",
    "ut_shape_area_within_500m",
    "ut_shape_share_of_500m_buffer",
    "ut_shape_touches_500m_buffer",
    # west campus shape features
    "in_west_campus_shape",
    "dist_to_west_campus_shape_m",
    "west_campus_shape_area_within_275m",
    "west_campus_shape_share_of_275m_buffer",
    "west_campus_shape_touches_275m_buffer",
    "west_campus_shape_area_within_500m",
    "west_campus_shape_share_of_500m_buffer",
    "west_campus_shape_touches_500m_buffer",
    "west_campus_shape_area_within_1000m",
    # north campus shape features
    "in_north_campus_shape",
    "dist_to_north_campus_shape_m",
    "north_campus_shape_area_within_275m",
    "north_campus_shape_share_of_275m_buffer",
    "north_campus_shape_touches_275m_buffer",
    "north_campus_shape_area_within_500m",
    "north_campus_shape_share_of_500m_buffer",
    "north_campus_shape_touches_500m_buffer",
    # interactions
    "ut_x_dorm_pop_500m",
    "ut_x_dining_dist",
    "ut_x_transit",
    "ut_x_housing_275m",
    "ut_x_in_ut_shape",
    "ut_x_in_west_campus_shape",
    "ut_x_in_north_campus_shape",
    "ut_x_ut_shape_share_275m",
    "ut_x_west_campus_share_275m",
    "ut_x_north_campus_share_275m",
]

existing_final_columns = [
    col for col in final_columns if col in scores_and_coords.columns
]
scores_and_coords = scores_and_coords[existing_final_columns].copy()


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
combined_output_path = os.path.join(OUTPUT_PREFIX, "combined_dataset_v6.csv")
scores_and_coords.to_csv(combined_output_path, index=False)

# %%
# -----------------------------
# Build ML dataset with scaling
# -----------------------------
df = scores_and_coords.copy()

target = "trips_per_dock"

# keep identifiers / raw text / non-feature cols out of X
drop_cols_for_model = ["id", "name", "district", target]

# Only scale continuous numeric columns.
continuous_scale_cols = [
    "total_docks",
    "lat",
    "lon",
    "transit_nearby",
    "nearest_transit_stop_dist_m",
    "avg_dist_3_nearest_transit_stops_m",
    "jobs_nearby_275m",
    "housing_nearby_275m",
    "housing_nearby_1000m",
    "job_housing_ratio_275m",
    "low_income_access_score",
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
    "nearest_station_dist_m",
    "stations_within_500m",
    "stations_within_1000m",
    "avg_stations_dist_3_nearest_m",
    "nearest_dining_hall_dist_m",
    "nearest_dorm_dist_m",
    "nearest_dorm_pop",
    "dorm_pop_within_500m",
    "dist_to_west_campus_center_m",
    "dist_to_ut_shape_m",
    "ut_shape_area_within_275m",
    "ut_shape_share_of_275m_buffer",
    "ut_shape_area_within_500m",
    "ut_shape_share_of_500m_buffer",
    "dist_to_west_campus_shape_m",
    "west_campus_shape_area_within_275m",
    "west_campus_shape_share_of_275m_buffer",
    "west_campus_shape_area_within_500m",
    "west_campus_shape_area_within_1000m",
    "west_campus_shape_share_of_500m_buffer",
    "dist_to_north_campus_shape_m",
    "north_campus_shape_area_within_275m",
    "north_campus_shape_share_of_275m_buffer",
    "north_campus_shape_area_within_500m",
    "north_campus_shape_share_of_500m_buffer",
    "ut_x_dorm_pop_500m",
    "ut_x_dining_dist",
    "ut_x_transit",
    "ut_x_housing_275m",
    "ut_x_ut_shape_share_275m",
    "ut_x_west_campus_share_275m",
    "ut_x_north_campus_share_275m",
]

X = df.drop(columns=[col for col in drop_cols_for_model if col in df.columns])
y = df[target]

continuous_scale_cols = [col for col in continuous_scale_cols if col in X.columns]

scaler = StandardScaler()
X_scaled = X.copy()
if continuous_scale_cols:
    X_scaled[continuous_scale_cols] = scaler.fit_transform(X[continuous_scale_cols])

ml_dataset = X_scaled.copy()
ml_dataset[target] = y

ml_output_path = os.path.join(OUTPUT_PREFIX, "ml_dataset_v6.csv")
ml_dataset.to_csv(ml_output_path, index=False)

# %%
print(f"Combined dataset saved to: {combined_output_path}")
print(f"ML dataset saved to: {ml_output_path}")
print(scores_and_coords.shape)
print(ml_dataset.shape)
print(ml_dataset.head())
