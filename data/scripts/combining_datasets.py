# %%
import geopandas as gpd
import pandas as pd

# %%
prefix = "../../data/cleaned/"

amenities = pd.read_csv(prefix + "amenities/amenities.csv")
coords = pd.read_csv(prefix + "coords/coords.csv")
housing = pd.read_csv(prefix + "housing/housing.csv")
jobs = pd.read_csv(prefix + "jobs/jobs.csv")
retail = pd.read_csv(prefix + "retail/retail.csv")
scores = pd.read_csv(prefix + "scoring/current_stations.csv")
transit = pd.read_csv(prefix + "transit/transit.csv")

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
