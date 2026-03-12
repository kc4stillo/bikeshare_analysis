# %%
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt

pd.set_option("display.max_rows", 100)

# %%
prefix = "../../data/cleaned/"

amenities = pd.read_csv(prefix + "amenities/amenities.csv")
coords = pd.read_csv(prefix + "coords/coords.csv")
housing = pd.read_csv(prefix + "housing/housing.csv")
jobs = pd.read_csv(prefix + "jobs/jobs.csv")
retail = pd.read_csv(prefix + "retail/retail.csv")
scores = pd.read_csv(prefix + "scoring/current_stations.csv")
transit = pd.read_csv(prefix + "transit/transit.csv")
parks = pd.read_csv(prefix + "amenities/parks.csv")

scores_and_coords = scores.merge(coords, left_on="name", right_on="scoring_name")
scores_and_coords = scores_and_coords[
    ["id", "active_date", "name", "district", "total_docks", "lat", "lon"]
]

# %%
# -----------------------------
# 1. Make copies
# -----------------------------
stations = scores_and_coords.copy()
stops = transit.copy()

# -----------------------------
# 2. Convert to GeoDataFrames
# -----------------------------
stations_gdf = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
    crs="EPSG:4326",
)

stops_gdf = gpd.GeoDataFrame(
    stops, geometry=gpd.points_from_xy(stops["lon"], stops["lat"]), crs="EPSG:4326"
)

# -----------------------------
# 3. Project to meters
#    EPSG:3857 is fine for this
# -----------------------------
stations_gdf = stations_gdf.to_crs(epsg=3857)
stops_gdf = stops_gdf.to_crs(epsg=3857)

# -----------------------------
# 4. Buffer each station by 275m
# -----------------------------
stations_buffer = stations_gdf[["id", "name", "geometry"]].copy()
stations_buffer["geometry"] = stations_buffer.geometry.buffer(275)

# -----------------------------
# 5. Spatial join:
#    which transit stops fall inside each station buffer
# -----------------------------
joined = gpd.sjoin(stops_gdf, stations_buffer, how="inner", predicate="within")

# -----------------------------
# 6. Count stops per station
# -----------------------------
stop_counts = (
    joined.groupby("id").size().rename("transit_stops_within_275m").reset_index()
)

# -----------------------------
# 7. Merge back to original table
# -----------------------------
scores_and_coords = scores_and_coords.merge(stop_counts, on="id", how="left")

# fill stations with no nearby stops as 0
scores_and_coords["transit_nearby"] = (
    scores_and_coords["transit_stops_within_275m"].fillna(0).astype(int)
)

scores_and_coords.drop("transit_stops_within_275m", axis=1, inplace=True)

# %%
# %%
# -----------------------------
# Jobs within 275m
# -----------------------------
stations = scores_and_coords.copy()
jobs_pts = jobs.copy()

stations_gdf = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
    crs="EPSG:4326",
)

jobs_gdf = gpd.GeoDataFrame(
    jobs_pts,
    geometry=gpd.points_from_xy(jobs_pts["lon"], jobs_pts["lat"]),
    crs="EPSG:4326",
)

stations_gdf = stations_gdf.to_crs(epsg=3857)
jobs_gdf = jobs_gdf.to_crs(epsg=3857)

stations_buffer = stations_gdf[["id", "name", "geometry"]].copy()
stations_buffer["geometry"] = stations_buffer.geometry.buffer(275)

joined = gpd.sjoin(jobs_gdf, stations_buffer, how="inner", predicate="within")

job_counts = (
    joined.groupby("id")["job_count"].sum().rename("jobs_within_275m").reset_index()
)

scores_and_coords = scores_and_coords.merge(job_counts, on="id", how="left")

scores_and_coords["jobs_nearby"] = (
    scores_and_coords["jobs_within_275m"].fillna(0).astype(int)
)

scores_and_coords.drop("jobs_within_275m", axis=1, inplace=True)

# %%
# -----------------------------
# Housing within 275m
# -----------------------------
stations = scores_and_coords.copy()
housing_pts = housing.copy()

stations_gdf = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
    crs="EPSG:4326",
)

housing_gdf = gpd.GeoDataFrame(
    housing_pts,
    geometry=gpd.points_from_xy(housing_pts["lon"], housing_pts["lat"]),
    crs="EPSG:4326",
)

stations_gdf = stations_gdf.to_crs(epsg=3857)
housing_gdf = housing_gdf.to_crs(epsg=3857)

stations_buffer = stations_gdf[["id", "name", "geometry"]].copy()
stations_buffer["geometry"] = stations_buffer.geometry.buffer(275)

joined = gpd.sjoin(housing_gdf, stations_buffer, how="inner", predicate="within")

housing_counts = (
    joined.groupby("id")["count"].sum().rename("housing_within_275m").reset_index()
)

scores_and_coords = scores_and_coords.merge(housing_counts, on="id", how="left")

scores_and_coords["housing_nearby"] = (
    scores_and_coords["housing_within_275m"].fillna(0).astype(int)
)

scores_and_coords.drop("housing_within_275m", axis=1, inplace=True)
# %%
# %%
# -----------------------------
# Amenities within 275m
# -----------------------------
stations = scores_and_coords.copy()
amenities_pts = amenities.copy()

stations_gdf = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
    crs="EPSG:4326",
)

amenities_gdf = gpd.GeoDataFrame(
    amenities_pts,
    geometry=gpd.points_from_xy(amenities_pts["lon"], amenities_pts["lat"]),
    crs="EPSG:4326",
)

stations_gdf = stations_gdf.to_crs(epsg=3857)
amenities_gdf = amenities_gdf.to_crs(epsg=3857)

stations_buffer = stations_gdf[["id", "name", "geometry"]].copy()
stations_buffer["geometry"] = stations_buffer.geometry.buffer(275)

joined = gpd.sjoin(amenities_gdf, stations_buffer, how="inner", predicate="within")

amenity_counts = (
    joined.groupby("id").size().rename("amenities_within_275m").reset_index()
)

scores_and_coords = scores_and_coords.merge(amenity_counts, on="id", how="left")

scores_and_coords["amenities_nearby"] = (
    scores_and_coords["amenities_within_275m"].fillna(0).astype(int)
)

scores_and_coords.drop("amenities_within_275m", axis=1, inplace=True)

# %%
# -----------------------------
# Park area within 275m
# -----------------------------
stations = scores_and_coords.copy()
parks_polys = parks.copy()

# station points from lat/lon
stations_gdf = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
    crs="EPSG:4326",
)

# convert park geometry strings into real geometries
parks_polys["geometry"] = parks_polys["geometry"].apply(wkt.loads)

parks_gdf = gpd.GeoDataFrame(
    parks_polys,
    geometry="geometry",
)

# Austin-looking projected coords: try UTM Zone 14N
parks_gdf = parks_gdf.set_crs("EPSG:32614")  # WGS84 / UTM zone 14N

# project stations into same CRS as parks
stations_gdf = stations_gdf.to_crs(parks_gdf.crs)

# create 275m buffers around stations
stations_buffer = stations_gdf[["id", "name", "geometry"]].copy()
stations_buffer["geometry"] = stations_buffer.geometry.buffer(275)

# intersect parks with station buffers
park_intersections = gpd.overlay(parks_gdf, stations_buffer, how="intersection")

# area of intersected park pieces (square meters)
park_intersections["park_area_part"] = park_intersections.geometry.area

# sum park area per station
park_area = (
    park_intersections.groupby("id")["park_area_part"]
    .sum()
    .rename("park_area_within_275m")
    .reset_index()
)

# merge back
scores_and_coords = scores_and_coords.merge(park_area, on="id", how="left")

scores_and_coords["park_area_nearby"] = (
    scores_and_coords["park_area_within_275m"].fillna(0).round().astype(int)
)

scores_and_coords.drop("park_area_within_275m", axis=1, inplace=True)

# %%
# -----------------------------
# Retail within 275m
# -----------------------------
stations = scores_and_coords.copy()
retail_pts = retail.copy()

stations_gdf = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
    crs="EPSG:4326",
)

retail_gdf = gpd.GeoDataFrame(
    retail_pts,
    geometry=gpd.points_from_xy(retail_pts["lon"], retail_pts["lat"]),
    crs="EPSG:4326",
)

stations_gdf = stations_gdf.to_crs(epsg=3857)
retail_gdf = retail_gdf.to_crs(epsg=3857)

stations_buffer = stations_gdf[["id", "name", "geometry"]].copy()
stations_buffer["geometry"] = stations_buffer.geometry.buffer(275)

joined = gpd.sjoin(retail_gdf, stations_buffer, how="inner", predicate="within")

retail_counts = joined.groupby("id").size().rename("retail_within_275m").reset_index()

scores_and_coords = scores_and_coords.merge(retail_counts, on="id", how="left")

scores_and_coords["retail_nearby"] = (
    scores_and_coords["retail_within_275m"].fillna(0).astype(int)
)

scores_and_coords.drop("retail_within_275m", axis=1, inplace=True)

# %%
# -----------------------------
# Station GeoDataFrame
# -----------------------------
stations = scores_and_coords.copy()

stations_gdf = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
    crs="EPSG:4326",
).to_crs(epsg=3857)

stations_gdf = stations_gdf[["id", "name", "geometry"]].copy()

# %%
# -----------------------------
# Pairwise distance matrix (meters)
# -----------------------------
station_ids = stations_gdf["id"].values
n = len(stations_gdf)

dist_array = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        dist_array[i, j] = stations_gdf.geometry.iloc[i].distance(
            stations_gdf.geometry.iloc[j]
        )

dist_matrix = pd.DataFrame(
    dist_array,
    index=station_ids,
    columns=station_ids,
)

# %%
# -----------------------------
# Distance-based connectivity features
# -----------------------------
dist_no_self_array = dist_matrix.to_numpy(copy=True)
np.fill_diagonal(dist_no_self_array, np.nan)

dist_no_self = pd.DataFrame(
    dist_no_self_array,
    index=dist_matrix.index,
    columns=dist_matrix.columns,
)

dist_features = pd.DataFrame(index=dist_no_self.index)

# nearest other station
dist_features["nearest_station_dist_m"] = dist_no_self.min(axis=1)

# number of stations within thresholds
dist_features["stations_within_500m"] = (dist_no_self <= 500).sum(axis=1)
dist_features["stations_within_1000m"] = (dist_no_self <= 1000).sum(axis=1)

# average distance to 3 nearest stations
dist_features["avg_dist_3_nearest_m"] = dist_no_self.apply(
    lambda row: np.sort(row.dropna().values)[:3].mean()
    if row.dropna().shape[0] >= 3
    else np.nan,
    axis=1,
)

# %%
# -----------------------------
# Merge connectivity features back
# -----------------------------
dist_features = dist_features.reset_index().rename(columns={"index": "id"})

scores_and_coords = scores_and_coords.merge(dist_features, on="id", how="left")

scores_and_coords.head()
