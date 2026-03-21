# %%
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

# %%
cleaned_prefix = "../../cleaned/"
raw_prefix = "../../raw/"

amenities = pd.read_csv(cleaned_prefix + "amenities/amenities.csv")
coords = pd.read_csv(cleaned_prefix + "coords/coords.csv")
housing = pd.read_csv(cleaned_prefix + "housing/housing.csv")
jobs = pd.read_csv(cleaned_prefix + "jobs/jobs.csv")
retail = pd.read_csv(cleaned_prefix + "retail/retail.csv")
scores = pd.read_csv(cleaned_prefix + "scoring/current_stations.csv")
transit = pd.read_csv(cleaned_prefix + "transit/transit.csv")
parks = pd.read_csv(cleaned_prefix + "amenities/parks.csv")
trips = pd.read_csv(raw_prefix + "scoring/tips_per_station.csv")

scores_and_coords = scores.merge(coords, left_on="name", right_on="scoring_name")

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

scores_and_coords

# %%
scores_and_coords.drop(["total_checkouts", "trips_per_dock"], inplace=True)

# %%
ut_names = [
    "Dean Keeton/Park Place",
    "Dean Keeton/Robert Dedman Dr",
    "Dean Keeton/Speedway",
    "Dean Keeton/Whitis",
    "E 21st/Speedway @ PCL",
    "E 23rd/San Jacinto @ DKR Stadium",
    "Guadalupe/West Mall @ University Co-op",
    "W 21st/Guadalupe",
    "W 21st/University",
    "W 22.5/Rio Grande",
    "W 22nd/Pearl",
    "W 23rd/San Gabriel",
    "W 26th/Nueces",
    "W 28th/Rio Grande",
]

scores_and_coords["is_ut"] = scores_and_coords["name"].isin(ut_names).astype(int)


def to_snake_case(text):
    if pd.isna(text):
        return text

    # lowercase first
    text = str(text).lower()

    # replace any non-alphanumeric character(s) with underscore
    text = re.sub(r"[^a-z0-9]+", "_", text)

    # remove leading/trailing underscores
    text = re.sub(r"^_+|_+$", "", text)

    return text


# apply to your column
scores_and_coords["name"] = scores_and_coords["name"].apply(to_snake_case)

# %%
scores_and_coords = scores_and_coords[
    [
        "id",
        "active_date",
        "name",
        "district",
        "total_checkouts",
        "total_docks",
        "trips_per_dock",
        "ebs_station",
        "transit_nearby",
        "jobs_nearby",
        "housing_nearby",
        "low_income_access_score",
        "amenities_nearby",
        "park_area_nearby",
        "bike_infra_score",
        "retail_nearby",
        "nearest_station_dist_m",
        "stations_within_500m",
        "stations_within_1000m",
        "avg_dist_3_nearest_m",
        "is_ut",
        "lat",
        "lon",
    ]
]


amenities.head()
# name	lat	lon	type
# 0	NaN	30.141048	-97.827575	amenity_post_office
# 1	Walgreens	30.174187	-97.822898	amenity_pharmacy
# 2	Cornerstone Hospital Austin	30.311892	-97.743327	amenity_hospital
# 3	North Austin Optimist Baseball Fields	30.345827	-97.720758	leisure_sports_centre
# 4	Austin Regional Clinic	30.446372	-97.805730	amenity_clinic

parks.head()
# name	geometry
# 0	scofield_farms_neighborhood_park	MULTIPOLYGON (((626537.566663308 3365536.07899...
# 1	bartholomew_district_park	MULTIPOLYGON (((625454.1042703979 3353271.8536...
# 2	marble_creek_greenbelt	MULTIPOLYGON (((621952.7752310712 3337418.2165...
# 3	zilker_metro_park	MULTIPOLYGON (((618728.586566374 3349230.93388...
# 4	lower_bull_creek_greenbelt	MULTIPOLYGON (((618519.9013914722 3361899.4991

coords.head()
# scoring_name	cleaned_name	coordinate_name	lat	lon
# 0	Barton Springs Pool	barton springs pool	Barton Springs Pool	30.264520	-97.771200
# 1	Barton Springs/Azie Morton	azie morton/barton springs	NaN	30.261882	-97.768977
# 2	Barton Springs/Bouldin@ Palmer Auditorium	barton springs/bouldin	NaN	30.259660	-97.753445
# 3	Barton Springs/Kinney	barton springs/kinney	Barton Springs @ Kinney Ave	30.262000	-97.761180
# 4	Cesar Chavez/Congress	cesar chavez/congress	Congress & Cesar Chavez	30.263320	-97.745080


housing.head()
# count	lat	lon
# 0	542	30.323260	-97.747749
# 1	605	30.328906	-97.756538
# 2	431	30.315101	-97.751389
# 3	1010	30.311923	-97.753655
# 4	710	30.334287	-97.769566


jobs.head()
# 	job_count	lat	lon
# 0	40	30.334208	-97.755003
# 1	55	30.336065	-97.755197
# 2	4	30.326063	-97.747348
# 3	99	30.326369	-97.749075
# 4	11	30.321450	-97.748465


retail.head()
# name	lat	lon	type
# 0	South Congress Bat Colony	30.259127	-97.746370	tourism_attraction
# 1	Santa Rita No. 1 Oil Well	30.280025	-97.734710	tourism_attraction
# 2	The Elephant Room	30.265623	-97.743498	amenity_bar
# 3	The Hideout Theater	30.268571	-97.742226	amenity_theatre
# 4	Starbucks	30.268267	-97.742970	amenity_cafe


scores.head()
# id	active_date	name	district	total_checkouts	total_docks	trips_per_dock	trips_per_dock_day	ebs_station	checkouts_rank_per_day	transit_access_score	jobs_access_score	households_access_score	low_income_access_score	public_amenities_access_score	bike_infra_score	retail_entertainment_access_score	existing_bikeshare_access_score	total_score
# 0	36	2024-07-24	Barton Springs Pool	8	2541	11	231.000000	0.641667	0	1.0	2.0	1.0	1.0	1.0	3.0	2.0	3.0	3.0	17.0
# 1	39	2024-11-14	Barton Springs/Azie Morton	9	2389	15	159.266667	0.442407	1	2.0	2.0	3.0	2.0	1.0	2.0	2.0	3.0	3.0	20.0
# 2	37	2024-07-24	Barton Springs/Bouldin@ Palmer Auditorium	9	3425	15	228.333333	0.634259	0	2.0	3.0	2.0	2.0	1.0	3.0	3.0	3.0	3.0	22.0
# 3	38	2024-11-14	Barton Springs/Kinney	9	1982	11	180.181818	0.500505	0	1.0	3.0	3.0	2.0	1.0	2.0	3.0	3.0	3.0	21.0
# 4	41	2024-11-14	Cesar Chavez/Congress	9	2711	11	246.454545	0.684596	0	2.0	3.0	3.0	2.0	1.0	1.0	3.0	2.0	3.0	20.0

transit.head()
# name	lat	lon	type
# 0	riverside_and_burton	30.240341	-97.727308	bus
# 1	2237_riverside_and_willow_creek	30.238275	-97.726015	bus
# 2	2507_riverside_and_pleasant_valley	30.233867	-97.723760	bus
# 3	4411_oltorf_and_huntwick	30.226619	-97.726260	bus
# 4	2401_wickersham_and_oltorf	30.226039	-97.723488	bus


scores_and_coords.head()
# 	id	active_date	name	district	total_checkouts	total_docks	trips_per_dock	ebs_station	transit_nearby	jobs_nearby	housing_nearby	low_income_access_score	amenities_nearby	park_area_nearby	bike_infra_score	retail_nearby	nearest_station_dist_m	stations_within_500m	stations_within_1000m	avg_dist_3_nearest_m	is_ut	lat	lon
# 0	36	2024-07-24	Barton Springs Pool	8	2541	11	231.000000	0	0	34	0	1.0	2	235271	2.0	3	376.406098	3	3	406.615143	0	30.264520	-97.771200
# 1	39	2024-11-14	Barton Springs/Azie Morton	9	2389	15	159.266667	1	0	40	0	1.0	6	116251	2.0	1	379.736553	2	4	441.649726	0	30.261882	-97.768977
# 2	37	2024-07-24	Barton Springs/Bouldin@ Palmer Auditorium	9	3425	15	228.333333	0	4	3647	993	1.0	2	112716	3.0	5	415.522489	1	8	563.327378	0	30.259660	-97.753445
# 3	38	2024-11-14	Barton Springs/Kinney	9	1982	11	180.181818	0	3	819	0	1.0	8	697	3.0	15	595.666914	0	5	792.020875	0	30.262000	-97.761180
# 4	41	2024-11-14	Cesar Chavez/Congress	9	2711	11	246.454545	0	3	6387	0	1.0	7	6908	3.0	34	196.476819	3	14	292.268948	0	30.263320	-97.745080

# %%
scores_and_coords.to_csv(
    "../../cleaned/combined_datasets/v1/combined_dataset_v1.csv", index=False
)

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler

# copy your dataframe
df = scores_and_coords.copy()

# ----------------------------
# 1. choose target
# ----------------------------
target = "trips_per_dock"  # or "total_checkouts"

# ----------------------------
# 3. columns to drop from X
# ----------------------------
drop_cols = ["id", "active_date", "total_checkouts", "active_date", "district"]

# ----------------------------
# 4. columns to leave alone
# ----------------------------
binary_cols = ["ebs_station", "is_ut", "target"]

# ordinal scores: can leave as-is or standardize
ordinal_cols = ["name", "low_income_access_score", "bike_infra_score"]

# lat/lon: optional
coord_cols = ["lat", "lon"]

# ----------------------------
# 5. columns to standardize
# ----------------------------
scale_cols = [
    "transit_nearby",
    "jobs_nearby",
    "housing_nearby",
    "amenities_nearby",
    "park_area_nearby",
    "retail_nearby",
    "nearest_station_dist_m",
    "stations_within_500m",
    "stations_within_1000m",
    "avg_dist_3_nearest_m",
]

# if you want to standardize ordinal scores too, add them:
# scale_cols += ordinal_cols

# if you want to standardize coordinates too, add them:
# scale_cols += coord_cols

# ----------------------------
# 6. build X and y
# ----------------------------
X = df.drop(columns=drop_cols)
y = df[target]

# ----------------------------
# 7. standardize selected columns
# ----------------------------
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[scale_cols] = scaler.fit_transform(X[scale_cols])

X_scaled.head()

X_scaled.to_csv("../../cleaned/combined_datasets/v1/ml_dataset_v1.csv", index=False)
