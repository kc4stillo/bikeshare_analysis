from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

pools = pd.read_csv("../raw/amenities/austin_pools.csv")
libraries = pd.read_csv("../raw/amenities/library_locations.csv")
recs = pd.read_csv("../raw/amenities/rec_center_locations.csv")
parks = pd.read_csv("../raw/amenities/park_borders.csv")

# %%
# extract lat/long from pools
coords = (
    pools["Location 1"]
    .astype(str)
    .str.extract(r"\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)")
)

pools["lat"] = coords[0].astype(float)
pools["lon"] = coords[1].astype(float)

# select + rename
pools = pools[["Pool Name", "lat", "lon"]].copy()

# rename column to lowercase "name"
pools = pools.rename(columns={"Pool Name": "name"})

# make all column names lowercase (safe guard)
pools.columns = pools.columns.str.lower()

# add amenity column
pools["amenity"] = "pool"

pools.head()

# %%
coords = libraries["Latitude / Longitude"].str.extract(
    r"\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)"
)

libraries["lat"] = coords[0].astype(float)
libraries["lon"] = coords[1].astype(float)

public_libraries = libraries[["Name", "lat", "lon"]].copy()
public_libraries.columns = public_libraries.columns.str.lower()  # make lowercase


# --- Extra libraries (fixed longitude sign + column names consistent) ---
extra_libraries = pd.DataFrame(
    {
        "name": [
            "Perry-Castañeda Library",
            "Harry Ransom Center",
            "Classics Library",
            "Fine Arts Library",
            "Architecture and Planning Library",
            "Life Science Library",
            "Mallet Chemistry Library",
            "Kuehne Physics Mathematics Astronomy Library",
            "Masterson Library",
            "Wright Learning and Information Center (Stitt Library)",
            "LBJ Presidential Library",
            "Travis County Law Library & Self-Help Center",
            "Legislative Library",
            "Texas State Library and Archives Commission",
        ],
        "lat": [
            30.28309620135224,
            30.284909196600626,
            30.285694518893653,
            30.286082575971204,
            30.28549697243695,
            30.286222065016737,
            30.286865523109753,
            30.28938992969517,
            30.291928752326363,
            30.292277580745534,
            30.286376941567713,
            30.280399684477764,
            30.276041773045325,
            30.27458498763517,
        ],
        "lon": [  # renamed from "long"
            -97.7383782827987,
            -97.74107219687771,
            -97.73781814401929,
            -97.7317382624085,
            -97.740770264628,
            -97.73727582125527,
            -97.73788442524756,
            -97.73649160949135,
            -97.74003773918234,
            -97.73743394220428,
            -97.72924465621654,
            -97.74254569872491,
            -97.74040857314445,
            -97.7385147865229,  # fixed sign
        ],
    }
)


# --- Concatenate correctly ---
libraries = pd.concat([public_libraries, extra_libraries], ignore_index=True)

libraries["amenity"] = "library"

# %%
# extract lat/long from recs
coords = (
    recs["Location 1"].astype(str).str.extract(r"\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)")
)

recs["lat"] = coords[0].astype(float)
recs["lon"] = coords[1].astype(float)

# select + clean
recs = recs[["Recreation Centers", "lat", "lon"]].copy()

# rename to match schema
recs = recs.rename(columns={"Recreation Centers Name": "name"})

# lowercase column names
recs.columns = ["name", "lat", "lon"]

# add amenity column
recs["amenity"] = "rec_center"

recs
# %%
amenities = pd.concat([pools, recs, libraries])
amenities.head()

# %%
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

parks

# %%
out_dir = Path("../../cleaned/amenities")
out_dir.mkdir(parents=True, exist_ok=True)

amenities.to_csv(out_dir / "amenities.csv", index=False)
parks.to_csv(out_dir / "parks.csv", index=False)
