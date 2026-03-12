import geopandas as gpd
import pandas as pd
from shapely import wkt

parks = pd.read_csv("../raw/amenities/park_borders.csv")


def clean_names(x):
    """
    Clean names:
      - remove quotes (straight + curly)
      - lowercase everything
      - replace spaces with _
      - replace / with _and_
      - replace @ with _at_
      - strip leading/trailing whitespace
      - collapse repeated underscores
    Works for a string or a pandas Series.
    """
    if isinstance(x, pd.Series):
        s = x.astype("string")
    else:
        s = pd.Series([x], dtype="string")

    s = (
        s.str.strip()
        # remove quotations
        .str.replace('"', "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace("“", "", regex=False)
        .str.replace("”", "", regex=False)
        .str.replace("‘", "", regex=False)
        .str.replace("’", "", regex=False)
        .str.replace(".", "", regex=False)
        # your existing rules
        .str.lower()
        .str.replace("&", "_and_", regex=False)
        .str.replace("/", "_and_", regex=False)
        .str.replace("@", "_at_", regex=False)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    return s if isinstance(x, pd.Series) else s.iloc[0]


parks.head()

parks = parks[["LOCATION_NAME", "the_geom"]].dropna()

parks_gdf = gpd.GeoDataFrame(
    parks, geometry=parks["the_geom"].apply(wkt.loads), crs="EPSG:4326"
)

# Project to meters (Austin ≈ UTM 14N)
parks_m = parks_gdf.to_crs("EPSG:26914")

# parks_m already projected to EPSG:26914
parks_m["area_m2"] = parks_m.geometry.area
parks_m["area_acres"] = parks_m["area_m2"] / 4046.86

# Keep only parks larger than 10 acres
parks = parks_m[parks_m["area_acres"] > 15].copy()[["LOCATION_NAME", "geometry"]]
parks.columns = ["name", "geometry"]

# standardize names
parks["name"] = clean_names(parks["name"])

parks

parks.to_csv("../cleaned/amenities/parks.csv", index=False)
