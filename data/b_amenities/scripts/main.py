# %%
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt

sys.path.insert(0, str(Path.cwd().parents[2]))

from utilities import (
    area_covered_within_radius,
    avg_nearest_3_distance,
    nearest_distance,
    nearest_distance_to_polygons,
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# %%
stations = pd.read_csv("../../a_stations/stations.csv")
amenities = pd.read_csv("../clean/amenities.csv")
dining_halls = pd.read_csv("../clean/dining_halls.csv")
parks = pd.read_csv("../clean/parks.csv")

# turn parks in geodatafram
parks = parks.copy()
parks["geometry"] = parks["geometry"].apply(wkt.loads)
parks = gpd.GeoDataFrame(parks, geometry="geometry", crs="EPSG:4326")

# %%
stations = nearest_distance(stations, dining_halls, "nearest_dining_hall_m")
stations = nearest_distance(stations, amenities, "nearest_amenity_m")
stations = nearest_distance_to_polygons(stations, parks, "nearest_park_m")

stations = avg_nearest_3_distance(stations, amenities, "avg_dist_3_amenities_m")

stations = area_covered_within_radius(stations, parks, "park_area_within_275m")
stations = area_covered_within_radius(
    stations, parks, radius_m=550, new_col="park_area_within_550m"
)

stations.head()

# %%
stations.to_csv("../clean/stations.csv", index=None)
