# %%
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt

sys.path.insert(0, str(Path.cwd().parents[2]))

from utilities import attach_polygon_stats

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# %%
demographics = pd.read_csv("../clean/demographics.csv")
stations = pd.read_csv("../../b_amenities/clean/stations.csv")

demographics = demographics.copy()
demographics["geometry"] = demographics["geometry"].apply(wkt.loads)
demographics = gpd.GeoDataFrame(demographics, geometry="geometry", crs="EPSG:4326")

# %%
stations = attach_polygon_stats(
    stations=stations,
    polygon_gdf=demographics,
    polygon_cols=("age", "income", "population", "undergrad", "grad"),
)

stations["undergrad_percentage"] = stations["undergrad"] / stations["population"]
stations["grad_percentage"] = stations["grad"] / stations["population"]

stations.to_csv("../clean/stations.csv", index=False)
