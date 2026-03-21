# %%
import folium
import numpy as np
import pandas as pd

# ============================================================
# Load data
# ============================================================
prefix = "../../data/cleaned/"

amenities = pd.read_csv(prefix + "amenities/amenities.csv")
coords = pd.read_csv(prefix + "coords/coords.csv")
jobs = pd.read_csv(prefix + "jobs/jobs.csv")
retail = pd.read_csv(prefix + "retail/retail.csv")
transit = pd.read_csv(prefix + "transit/transit.csv")
housing = pd.read_csv(prefix + "housing/housing.csv")

# ============================================================
# Styling
# ============================================================
MARKER_OPACITY = 0.45
FILL_OPACITY = 0.25

COLORS = {
    "stations": "#377eb8",
    "amenities": "#4daf4a",
    "retail": "#984ea3",
    "jobs": "#ff7f00",
    "scored": "#e41a1c",
    "transit": "#a65628",
    "housing": "#f781bf",
    "radius": "#e41a1c",
}


# ============================================================
# Helpers
# ============================================================
def clean_latlon(df):
    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon"])


def retail_type_str(row):
    if "type" in retail.columns:
        return str(row.get("type", ""))
    if "type1" in retail.columns and "type2" in retail.columns:
        return f"{row.get('type1', '')}_{row.get('type2', '')}".strip("_")
    return ""


def scale_sqrt(x, vmax, min_px=2, max_px=12):
    if vmax <= 0:
        return min_px
    return min_px + (np.sqrt(x) / np.sqrt(vmax)) * (max_px - min_px)


def add_point_layer(
    m,
    df,
    name,
    color,
    popup_fn,
    radius=3,
    show=False,
):
    fg = folium.FeatureGroup(name=name, show=show)
    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            opacity=MARKER_OPACITY,
            fill_opacity=FILL_OPACITY,
            popup=folium.Popup(popup_fn(r), max_width=350),
        ).add_to(fg)
    fg.add_to(m)
    return fg


def add_scaled_layer(
    m,
    df,
    value_col,
    name,
    color,
    popup_fn,
    min_px=2,
    max_px=12,
    show=False,
):
    fg = folium.FeatureGroup(name=name, show=show)
    vals = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
    vmax = vals.max() if len(vals) else 1
    vmax = vmax if vmax > 0 else 1

    for _, r in df.iterrows():
        val = float(r[value_col])
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=scale_sqrt(val, vmax, min_px=min_px, max_px=max_px),
            color=color,
            fill=True,
            fill_color=color,
            opacity=MARKER_OPACITY,
            fill_opacity=FILL_OPACITY,
            popup=folium.Popup(popup_fn(r), max_width=250),
        ).add_to(fg)
    fg.add_to(m)
    return fg


def add_station_layer(m, coords_df):
    fg = folium.FeatureGroup(name="MetroBike Stations", show=True)
    for _, r in coords_df.iterrows():
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=5,
            color=COLORS["stations"],
            fill=True,
            fill_color=COLORS["stations"],
            opacity=MARKER_OPACITY,
            fill_opacity=FILL_OPACITY,
            popup=folium.Popup(
                f"<b>{r.get('scoring_name', '')}</b><br>"
                f"cleaned_name: {r.get('cleaned_name', '')}<br>"
                f"coordinate_name: {r.get('coordinate_name', '')}",
                max_width=350,
            ),
        ).add_to(fg)
    fg.add_to(m)


def add_scored_layer(m, scoring_df):
    if not {"lat", "lon"}.issubset(scoring_df.columns):
        return

    fg = folium.FeatureGroup(name="Scored Stations", show=False)
    popup_cols = [
        c
        for c in ["name", "scoring_name", "Total Score", "total_score", "score"]
        if c in scoring_df.columns
    ]

    for _, r in scoring_df.iterrows():
        popup = (
            "<br>".join([f"{c}: {r.get(c)}" for c in popup_cols]) if popup_cols else ""
        )
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6,
            color=COLORS["scored"],
            fill=True,
            fill_color=COLORS["scored"],
            opacity=MARKER_OPACITY,
            fill_opacity=FILL_OPACITY,
            popup=folium.Popup(popup, max_width=350) if popup else None,
        ).add_to(fg)
    fg.add_to(m)


def add_common_layers(m):
    add_point_layer(
        m,
        clean_latlon(amenities),
        "Public Amenities",
        COLORS["amenities"],
        lambda r: f"{r.get('name', '')}<br>type: {r.get('type', '')}",
    )

    add_point_layer(
        m,
        clean_latlon(retail),
        "Retail / Entertainment",
        COLORS["retail"],
        lambda r: f"{r.get('name', '')}<br>{retail_type_str(r)}",
    )

    add_scaled_layer(
        m,
        clean_latlon(
            jobs.assign(
                job_count=pd.to_numeric(jobs["job_count"], errors="coerce").fillna(0)
            )
        ),
        "job_count",
        "Jobs",
        COLORS["jobs"],
        lambda r: f"Jobs: {int(r['job_count']):,}",
    )

    add_scaled_layer(
        m,
        clean_latlon(
            housing.assign(
                count=pd.to_numeric(housing["count"], errors="coerce").fillna(0)
            )
        ),
        "count",
        "Housing",
        COLORS["housing"],
        lambda r: f"Housing units: {int(r['count']):,}",
    )

    add_point_layer(
        m,
        clean_latlon(transit),
        "Transit Stops",
        COLORS["transit"],
        lambda r: f"{r.get('name', '')}<br>type: {r.get('type', '')}",
    )


# ============================================================
# Full map
# ============================================================
coords = clean_latlon(coords)
# scoring = clean_latlon(scoring)

center_lat = coords["lat"].mean()
center_lon = coords["lon"].mean()

m = folium.Map(
    location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron"
)

add_station_layer(m, coords)
add_common_layers(m)
# add_scored_layer(m, scoring)

folium.LayerControl(collapsed=False).add_to(m)
m

# Optional save
# m.save("capmetro_layers_map_points_only.html")


# ============================================================
# Station radius function
# ============================================================
def plot_station_radius_layers(station_name: str, radius_m: int = 275):
    df = clean_latlon(coords)

    possible_name_cols = [
        c
        for c in ["scoring_name", "cleaned_name", "coordinate_name", "name"]
        if c in df.columns
    ]
    if not possible_name_cols:
        raise ValueError("No station name columns found in coords.")

    station_name_clean = station_name.strip().lower()
    mask = pd.Series(False, index=df.index)

    for col in possible_name_cols:
        mask |= df[col].astype(str).str.strip().str.lower().eq(station_name_clean)

    match = df.loc[mask]
    if match.empty:
        examples = (
            df[possible_name_cols[0]]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"No station found for '{station_name}'. Example station names: {examples}"
        )

    r = match.iloc[0]
    lat, lon = float(r["lat"]), float(r["lon"])

    m = folium.Map(location=[lat, lon], zoom_start=16, tiles="CartoDB positron")

    fg_station = folium.FeatureGroup(name="Selected MetroBike Station", show=True)

    popup_html = "<br>".join(
        [
            f"<b>{r.get('scoring_name', '')}</b>",
            f"cleaned_name: {r.get('cleaned_name', '')}",
            f"coordinate_name: {r.get('coordinate_name', '')}",
            f"lat: {lat}",
            f"lon: {lon}",
        ]
    )

    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=COLORS["stations"],
        fill=True,
        fill_color=COLORS["stations"],
        opacity=0.9,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=350),
    ).add_to(fg_station)

    folium.Circle(
        location=[lat, lon],
        radius=radius_m,
        color=COLORS["radius"],
        weight=2,
        fill=True,
        fill_color=COLORS["radius"],
        fill_opacity=0.15,
        popup=f"{radius_m} meter radius",
    ).add_to(fg_station)

    fg_station.add_to(m)
    add_common_layers(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# %%
plot_station_radius_layers("W 6th/Lavaca")
