# %%
import folium
import geopandas as gpd
import numpy as np
import pandas as pd

# %%
# --------------------------------------------------
# Load raw housing geometry + housing attribute data
# --------------------------------------------------
bg = gpd.read_file(r"../raw/housing/block_shapes.shp")
df = pd.read_csv("../raw/housing/house_data.csv")

# %%
# --------------------------------------------------
# Merge geometry with housing counts
# AUUDE002 = occupied housing units
# --------------------------------------------------
housing = bg.merge(df[["GISJOIN", "AUUDE002"]], on="GISJOIN", how="left")

# keep Travis County only
housing = housing[housing["COUNTYFP"] == "453"].copy()

# %%
# --------------------------------------------------
# Clean numeric housing count
# --------------------------------------------------
housing["AUUDE002"] = pd.to_numeric(housing["AUUDE002"], errors="coerce").fillna(0)

# %%
# --------------------------------------------------
# Convert NHGIS internal point strings to numeric lat/lon
# Example: "+30.2178959", "-097.7431000"
# --------------------------------------------------
housing["lat"] = (
    housing["INTPTLAT"].astype(str).str.replace("+", "", regex=False).astype(float)
)

housing["lon"] = (
    housing["INTPTLON"].astype(str).str.replace("+", "", regex=False).astype(float)
)

# %%
# --------------------------------------------------
# Optional quick map check
# --------------------------------------------------
center_lat = housing["lat"].mean()
center_lon = housing["lon"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="CartoDB positron",
)

vals = housing["AUUDE002"].to_numpy()
vmax = vals.max() if vals.max() > 0 else 1

min_radius = 50
max_radius = 600


def scale_radius(x):
    return min_radius + (np.sqrt(x) / np.sqrt(vmax)) * (max_radius - min_radius)


for _, r in housing.iterrows():
    hh = float(r["AUUDE002"])
    folium.Circle(
        location=[r["lat"], r["lon"]],
        radius=scale_radius(hh),
        fill=True,
        fill_opacity=0.35,
        opacity=0.6,
        popup=f"{r.get('NAMELSAD', 'Block Group')}<br>Occupied units: {int(hh):,}",
    ).add_to(m)

m

# %%
# --------------------------------------------------
# Keep cleaned columns for downstream station scoring
# Treat each block group as a point with a housing count
# --------------------------------------------------
housing_clean = housing[["AUUDE002", "lat", "lon"]].copy()
housing_clean.columns = ["count", "lat", "lon"]

# %%
# Optional checks
print(housing_clean.head())
print(housing_clean.dtypes)
print(housing_clean.shape)

# %%
# Save cleaned CSV
housing_clean.to_csv("../cleaned/housing/housing.csv", index=False)
