# %%
import folium
import pandas as pd
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon

# %%
# --------------------------------------------------
# Load datasets
# --------------------------------------------------
north_campus_shape = pd.read_csv("../data/cleaned/coords/north_campus.csv")
wampus_shape = pd.read_csv("../data/cleaned/coords/west_campus.csv")
ut_shape = pd.read_csv("../data/cleaned/coords/ut_shape.csv")
bikeshare_stations = pd.read_csv("../data/cleaned/coords/bikeshare_stations.csv")

# %%
# --------------------------------------------------
# Convert shape strings to shapely geometries
# --------------------------------------------------
north_campus_shape["geometry"] = north_campus_shape["shape"].apply(wkt.loads)
wampus_shape["geometry"] = wampus_shape["shape"].apply(wkt.loads)
ut_shape["geometry"] = ut_shape["shape"].apply(wkt.loads)


# %%
# --------------------------------------------------
# Helper function to add shapely polygons to folium
# Shapely uses (lon, lat), but folium wants (lat, lon)
# --------------------------------------------------
def add_shape_to_map(fg, geometry, popup_text=None, color="blue", fill_opacity=0.25):
    if isinstance(geometry, Polygon):
        coords = [(lat, lon) for lon, lat in geometry.exterior.coords]
        folium.Polygon(
            locations=coords,
            color=color,
            weight=2,
            fill=True,
            fill_opacity=fill_opacity,
            popup=popup_text,
        ).add_to(fg)

    elif isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            coords = [(lat, lon) for lon, lat in poly.exterior.coords]
            folium.Polygon(
                locations=coords,
                color=color,
                weight=2,
                fill=True,
                fill_opacity=fill_opacity,
                popup=popup_text,
            ).add_to(fg)


# %%
# --------------------------------------------------
# Create base map
# Centered roughly around UT / central Austin
# --------------------------------------------------
m = folium.Map(location=[30.285, -97.745], zoom_start=14, tiles="CartoDB positron")

# %%
# --------------------------------------------------
# Feature groups
# --------------------------------------------------
fg_north = folium.FeatureGroup(name="North Campus", show=True)
fg_wampus = folium.FeatureGroup(name="West Campus", show=True)
fg_ut = folium.FeatureGroup(name="UT Shapes", show=True)
fg_stations = folium.FeatureGroup(name="Bikeshare Stations", show=True)

# %%
# --------------------------------------------------
# Add North Campus polygon(s)
# --------------------------------------------------
for _, row in north_campus_shape.iterrows():
    add_shape_to_map(
        fg_north,
        row["geometry"],
        popup_text=row["name"],
        color="green",
        fill_opacity=0.20,
    )

# %%
# --------------------------------------------------
# Add West Campus polygon(s)
# --------------------------------------------------
for _, row in wampus_shape.iterrows():
    add_shape_to_map(
        fg_wampus,
        row["geometry"],
        popup_text=row["name"],
        color="orange",
        fill_opacity=0.20,
    )

# %%
# --------------------------------------------------
# Add UT polygon(s)
# --------------------------------------------------
for _, row in ut_shape.iterrows():
    add_shape_to_map(
        fg_ut,
        row["geometry"],
        popup_text=row["name"],
        color="red",
        fill_opacity=0.15,
    )

# %%
# --------------------------------------------------
# Add bikeshare stations
# --------------------------------------------------
for _, row in bikeshare_stations.iterrows():
    if pd.notnull(row["lat"]) and pd.notnull(row["lon"]):
        popup_text = f"""
        <b>Scoring Name:</b> {row["scoring_name"]}<br>
        <b>Cleaned Name:</b> {row["cleaned_name"]}<br>
        <b>Coordinate Name:</b> {row["coordinate_name"]}
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            popup=folium.Popup(popup_text, max_width=300),
            color="blue",
            fill=True,
            fill_opacity=0.9,
        ).add_to(fg_stations)

# %%
# --------------------------------------------------
# Add layers to map
# --------------------------------------------------
fg_north.add_to(m)
fg_wampus.add_to(m)
fg_ut.add_to(m)
fg_stations.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

m
