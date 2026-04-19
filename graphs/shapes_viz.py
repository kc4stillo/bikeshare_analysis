# %%
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt

# %%
# --------------------------------------------------
# Load datasets
# --------------------------------------------------
north_campus_shape = pd.read_csv("../data/d_ut_shapes/clean/north_campus.csv")
wampus_shape = pd.read_csv("../data/d_ut_shapes/clean/west_campus.csv")
ut_shape = pd.read_csv("../data/d_ut_shapes/clean/ut_shape.csv")
bikeshare_stations = pd.read_csv("../data/d_ut_shapes/clean/stations.csv")

# %%
# --------------------------------------------------
# Station whitelist
# --------------------------------------------------
ut = [
    "dean_keeton_park_place",
    "deen_keeton_whitis",
    "dean_keeton_robert_dedman_dr",
    "dean_keeton_whitisdean_keeton_speedway",
    "dean_keeton_whitis",
    "e_21st_speedway_at_pcl",
    "e_23rd_san_jacinto_at_dkr_stadium",
    "guadalupe_west_mall_at_university_co-op",
    "w_21st_guadalupe",
    "w_21st_university",
    "w_225_rio_grande",
    "w_22nd_pearl",
    "w_23rd_san_gabriel",
    "w_26th_nueces",
    "w_28th_rio_grande",
]

# %%
# --------------------------------------------------
# Convert WKT strings to shapely geometries
# --------------------------------------------------
north_campus_shape["geometry"] = north_campus_shape["geometry"].apply(wkt.loads)
wampus_shape["geometry"] = wampus_shape["geometry"].apply(wkt.loads)
ut_shape["geometry"] = ut_shape["geometry"].apply(wkt.loads)

# %%
# --------------------------------------------------
# Convert polygons to GeoDataFrames and project to Web Mercator
# --------------------------------------------------
north_gdf = gpd.GeoDataFrame(north_campus_shape, geometry="geometry", crs="EPSG:4326")
wampus_gdf = gpd.GeoDataFrame(wampus_shape, geometry="geometry", crs="EPSG:4326")
ut_gdf = gpd.GeoDataFrame(ut_shape, geometry="geometry", crs="EPSG:4326")

polygons_gdf = pd.concat([ut_gdf, north_gdf, wampus_gdf], ignore_index=True)
polygons_gdf = gpd.GeoDataFrame(
    polygons_gdf, geometry="geometry", crs="EPSG:4326"
).to_crs(epsg=3857)

# %%
# --------------------------------------------------
# Detect station columns
# --------------------------------------------------
lat_candidates = ["lat", "latitude", "Lat", "Latitude"]
lon_candidates = ["lon", "lng", "long", "longitude", "Lon", "Longitude"]
name_candidates = [
    "station",
    "station_name",
    "name",
    "kiosk_name",
    "slug",
    "id",
    "station_id",
]

lat_col = next((c for c in lat_candidates if c in bikeshare_stations.columns), None)
lon_col = next((c for c in lon_candidates if c in bikeshare_stations.columns), None)
name_col = next((c for c in name_candidates if c in bikeshare_stations.columns), None)

if lat_col is None or lon_col is None:
    raise ValueError(
        f"Could not find station latitude/longitude columns. "
        f"Available columns: {list(bikeshare_stations.columns)}"
    )

if name_col is None:
    raise ValueError(
        f"Could not find a station name/id column to filter on. "
        f"Available columns: {list(bikeshare_stations.columns)}"
    )

# %%
# --------------------------------------------------
# Filter to selected stations only
# --------------------------------------------------
stations_df = bikeshare_stations.copy()
stations_df[name_col] = stations_df[name_col].astype(str).str.strip()

stations_df = (
    stations_df[stations_df[name_col].isin(ut)].dropna(subset=[lat_col, lon_col]).copy()
)

stations_gdf = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(stations_df[lon_col], stations_df[lat_col]),
    crs="EPSG:4326",
).to_crs(epsg=3857)

# %%
# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.close("all")
fig, ax = plt.subplots(figsize=(30, 20), dpi=150)

color = "#001589"

# Plot polygons first
polygons_gdf.plot(
    ax=ax,
    color=color,
    alpha=0.18,
    edgecolor=color,
    linewidth=1.5,
    zorder=2,
)

# Plot stations as dots
stations_gdf.plot(
    ax=ax,
    markersize=70,
    color=color,
    alpha=0.9,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)

# %%

# Add basemap and preserve the current extent
cx.add_basemap(
    ax,
    source=cx.providers.CartoDB.Positron,
    crs=polygons_gdf.crs,
    reset_extent=False,
    zoom="auto",
)

ax.set_axis_off()
plt.tight_layout()
plt.savefig("campus_polygons_with_ut_stations.png", dpi=300, bbox_inches="tight")
plt.show()
