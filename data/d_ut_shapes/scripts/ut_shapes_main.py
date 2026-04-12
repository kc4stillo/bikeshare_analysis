# %%
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt

sys.path.insert(0, str(Path.cwd().parents[2]))

from utilities.df import (
    area_covered_within_radius,
    nearest_distance_to_polygons,
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# %%
stations = pd.read_csv("../../c_demographics/clean/stations.csv")

north_campus = pd.read_csv("../clean/north_campus.csv")
west_campus = pd.read_csv("../clean/west_campus.csv")
ut = pd.read_csv("../clean/ut_shape.csv")


# turn df in geodatafram
north_campus = north_campus.copy()
north_campus["geometry"] = north_campus["geometry"].apply(wkt.loads)
north_campus = gpd.GeoDataFrame(north_campus, geometry="geometry", crs="EPSG:4326")

west_campus = west_campus.copy()
west_campus["geometry"] = west_campus["geometry"].apply(wkt.loads)
west_campus = gpd.GeoDataFrame(west_campus, geometry="geometry", crs="EPSG:4326")

ut = ut.copy()
ut["geometry"] = ut["geometry"].apply(wkt.loads)
ut = gpd.GeoDataFrame(ut, geometry="geometry", crs="EPSG:4326")

# %%
stations = area_covered_within_radius(
    stations, north_campus, "north_campus_area_within_275m"
)
stations = area_covered_within_radius(
    stations, north_campus, radius_m=550, new_col="north_campus_area_within_550m"
)


stations = area_covered_within_radius(
    stations, west_campus, "west_campus_area_within_275m"
)
stations = area_covered_within_radius(
    stations, west_campus, radius_m=550, new_col="west_campus_area_within_550m"
)
stations = area_covered_within_radius(
    stations, west_campus, radius_m=825, new_col="west_campus_area_within_825m"
)
stations = nearest_distance_to_polygons(
    stations, west_campus, "distance_to_west_campus_m"
)


stations = nearest_distance_to_polygons(stations, ut, "distance_to_ut_m")
stations = area_covered_within_radius(stations, ut, "ut_area_within_275m")
stations = area_covered_within_radius(
    stations, ut, radius_m=550, new_col="ut_area_within_550m"
)


# %%
stations.to_csv("../clean/stations.csv", index=False)
