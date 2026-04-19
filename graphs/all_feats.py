import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt
from shapely.geometry import Point

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
stations = pd.read_csv("../data/a_stations/stations.csv")
amenities = pd.read_csv("../data/b_amenities/clean/amenities.csv")
demo = pd.read_csv("../data/c_demographics/clean/demographics.csv")
jobs = pd.read_csv("../data/e_jobs/clean/jobs.csv")
retail = pd.read_csv("../data/f_retail/clean/retail.csv")
transit = pd.read_csv("../data/g_transit/clean/transit.csv")
parks = pd.read_csv("../data/b_amenities/clean/parks.csv")

# --------------------------------------------------
# Demo polygons
# --------------------------------------------------
demo = demo.copy()
demo["geometry"] = demo["geometry"].apply(
    lambda x: wkt.loads(x) if isinstance(x, str) else x
)

demo_gdf = gpd.GeoDataFrame(demo, geometry="geometry", crs="EPSG:4326").to_crs(3857)

travis_union = demo_gdf.unary_union


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def make_points_gdf(df, lat_col="lat", lon_col="lon"):
    df = df.dropna(subset=[lat_col, lon_col]).copy()

    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).to_crs(3857)

    pts = gpd.sjoin(
        pts,
        demo_gdf[["geometry"]],
        how="inner",
        predicate="intersects",
    ).drop(columns="index_right")

    pts = pts.loc[~pts.index.duplicated()]
    return pts


def make_polygon_gdf(df, geom_col="geometry"):
    df = df.copy()
    df[geom_col] = df[geom_col].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) else x
    )

    gdf = gpd.GeoDataFrame(df, geometry=geom_col, crs="EPSG:4326").to_crs(3857)

    # keep only polygons that touch/intersect the demo area
    gdf = gdf[gdf.geometry.intersects(travis_union)].copy()
    return gdf


# --------------------------------------------------
# Convert layers
# --------------------------------------------------
stations_gdf = make_points_gdf(stations)
amenities_gdf = make_points_gdf(amenities)
jobs_gdf = make_points_gdf(jobs)
retail_gdf = make_points_gdf(retail)
transit_gdf = make_points_gdf(transit)
parks_gdf = make_polygon_gdf(parks)

# --------------------------------------------------
# Marker sizes
# --------------------------------------------------
if "trips_per_dock" in stations_gdf.columns:
    station_sizes = stations_gdf["trips_per_dock"].fillna(0) * 5
else:
    station_sizes = 60

if "job_count" in jobs_gdf.columns:
    job_sizes = jobs_gdf["job_count"].fillna(0).clip(lower=1) * 1.5
else:
    job_sizes = 20

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 14), dpi=150)

# draw polygons first
demo_gdf.boundary.plot(
    ax=ax,
    edgecolor="#4d4d4d",
    linewidth=1.5,
    alpha=0.8,
    zorder=2,
)

parks_gdf.plot(
    ax=ax,
    facecolor="#7fc97f",
    edgecolor="#2e7d32",
    linewidth=0.4,
    alpha=0.35,
    zorder=3,
    label="Parks",
)

# points on top
amenities_gdf.plot(
    ax=ax,
    markersize=20,
    color="#FF5DF4",
    alpha=1,
    edgecolor="none",
    zorder=4,
    label="Amenities",
)

retail_gdf.plot(
    ax=ax,
    markersize=20,
    color="#FFA43D",
    alpha=1,
    edgecolor="none",
    zorder=4,
    label="Retail",
)

transit_gdf.plot(
    ax=ax,
    markersize=20,
    color="#001589",
    alpha=1,
    edgecolor="none",
    zorder=5,
    label="Transit",
)

# jobs_gdf.plot(
#     ax=ax,
#     markersize=80,
#     color="#ED3B97",
#     alpha=1,
#     edgecolor="none",
#     zorder=6,
#     label="Jobs",
# )

stations_gdf.plot(
    ax=ax,
    markersize=2000,
    color="#3BBFEF",
    alpha=0.3,
    edgecolor="none",
    zorder=7,
    label="Bikehare Station Radius ~275m",
)

stations_gdf.plot(
    ax=ax,
    markersize=80,
    color="#3BBFEF",
    alpha=1,
    edgecolor="none",
    zorder=7,
    label="Bikeshare Stations",
)

# --------------------------------------------------
# Zoom out more
# --------------------------------------------------
downtown_pt = gpd.GeoSeries([Point(-97.7431, 30.2672)], crs="EPSG:4326").to_crs(3857)

cx0 = downtown_pt.iloc[0].x
cy0 = downtown_pt.iloc[0].y

# was 1000; this is much more zoomed out
# smaller = more zoomed in
half_width = 3800
half_height = 3750

ax.set_xlim(cx0 - half_width, cx0 + half_width)
ax.set_ylim(cy0 - half_height, cy0 + half_height)

# basemap AFTER limits/data
cx.add_basemap(
    ax,
    source=cx.providers.CartoDB.Positron,
    crs=demo_gdf.crs,
)

ax.set_axis_off()
ax.legend(loc="lower left", frameon=True, fontsize=18)
plt.tight_layout()
plt.title("Current Bikeshare Stations With Influencing Factors", fontsize=24)
plt.savefig("all_attributes_with_parks_zoomed_out.png", dpi=300, bbox_inches="tight")
plt.show()


# capmetro blue #3bbfef
# electric blue  #001589
# white
# grey
