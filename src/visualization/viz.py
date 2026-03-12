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
scoring = pd.read_csv(prefix + "scoring/current_stations.csv")
transit = pd.read_csv(prefix + "transit/transit.csv")

# %%
# ============================================================
# Tunable styling
# ============================================================
MARKER_OPACITY = 0.45
FILL_OPACITY = 0.25

COLOR_STATIONS = "#377eb8"  # blue
COLOR_AMENITIES = "#4daf4a"  # green
COLOR_RETAIL = "#984ea3"  # purple
COLOR_JOBS = "#ff7f00"  # orange
COLOR_SCORED = "#e41a1c"  # red
COLOR_TRANSIT = "#a65628"  # brown


# %%
# ============================================================
# Helpers
# ============================================================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def retail_type_str(row):
    if "type" in retail.columns:
        return str(row.get("type", ""))
    if "type1" in retail.columns and "type2" in retail.columns:
        t1 = row.get("type1", "")
        t2 = row.get("type2", "")
        return f"{t1}_{t2}".strip("_")
    return ""


# ============================================================
# Base map
# ============================================================
coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
coords = coords.dropna(subset=["lat", "lon"])

center_lat = coords["lat"].mean()
center_lon = coords["lon"].mean()

m = folium.Map(
    location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron"
)

# %%
# ============================================================
# MetroBike Stations
# ============================================================
fg_stations = folium.FeatureGroup(name="MetroBike Stations", show=True)

for _, r in coords.iterrows():
    folium.CircleMarker(
        location=[float(r["lat"]), float(r["lon"])],
        radius=5,
        color=COLOR_STATIONS,
        fill=True,
        fill_color=COLOR_STATIONS,
        opacity=MARKER_OPACITY,
        fill_opacity=FILL_OPACITY,
        popup=folium.Popup(
            f"<b>{r.get('scoring_name', '')}</b><br>"
            f"cleaned_name: {r.get('cleaned_name', '')}<br>"
            f"coordinate_name: {r.get('coordinate_name', '')}",
            max_width=350,
        ),
    ).add_to(fg_stations)

fg_stations.add_to(m)

# ============================================================
# Public Amenities
# ============================================================
fg_amen = folium.FeatureGroup(name="Public Amenities (points)", show=False)

amenities_local = amenities.copy()
amenities_local["lat"] = pd.to_numeric(amenities_local["lat"], errors="coerce")
amenities_local["lon"] = pd.to_numeric(amenities_local["lon"], errors="coerce")
amenities_local = amenities_local.dropna(subset=["lat", "lon"])

for _, r in amenities_local.iterrows():
    folium.CircleMarker(
        location=[float(r["lat"]), float(r["lon"])],
        radius=3,
        color=COLOR_AMENITIES,
        fill=True,
        fill_color=COLOR_AMENITIES,
        opacity=MARKER_OPACITY,
        fill_opacity=FILL_OPACITY,
        popup=folium.Popup(
            f"{r.get('name', '')}<br>type: {r.get('type', '')}", max_width=350
        ),
    ).add_to(fg_amen)

fg_amen.add_to(m)

# ============================================================
# Retail / Entertainment
# ============================================================
fg_retail = folium.FeatureGroup(name="Retail / Entertainment (points)", show=False)

retail_local = retail.copy()
retail_local["lat"] = pd.to_numeric(retail_local["lat"], errors="coerce")
retail_local["lon"] = pd.to_numeric(retail_local["lon"], errors="coerce")
retail_local = retail_local.dropna(subset=["lat", "lon"])

for _, r in retail_local.iterrows():
    folium.CircleMarker(
        location=[float(r["lat"]), float(r["lon"])],
        radius=3,
        color=COLOR_RETAIL,
        fill=True,
        fill_color=COLOR_RETAIL,
        opacity=MARKER_OPACITY,
        fill_opacity=FILL_OPACITY,
        popup=folium.Popup(
            f"{r.get('name', '')}<br>{retail_type_str(r)}", max_width=350
        ),
    ).add_to(fg_retail)

fg_retail.add_to(m)

# ============================================================
# Jobs (scaled by job_count)
# ============================================================
fg_jobs = folium.FeatureGroup(name="Jobs (sized by job_count)", show=False)

jobs = jobs.copy()
jobs["job_count"] = pd.to_numeric(jobs["job_count"], errors="coerce").fillna(0)
jobs["lat"] = pd.to_numeric(jobs["lat"], errors="coerce")
jobs["lon"] = pd.to_numeric(jobs["lon"], errors="coerce")
jobs = jobs.dropna(subset=["lat", "lon"])

vmax = jobs["job_count"].max() if len(jobs) else 1
vmax = vmax if vmax > 0 else 1


def scale_px(x, min_px=2, max_px=12):
    return min_px + (np.sqrt(x) / np.sqrt(vmax)) * (max_px - min_px)


for _, r in jobs.iterrows():
    jc = float(r["job_count"])

    folium.CircleMarker(
        location=[float(r["lat"]), float(r["lon"])],
        radius=scale_px(jc),
        color=COLOR_JOBS,
        fill=True,
        fill_color=COLOR_JOBS,
        opacity=MARKER_OPACITY,
        fill_opacity=FILL_OPACITY,
        popup=folium.Popup(f"Jobs: {int(jc):,}", max_width=250),
    ).add_to(fg_jobs)

fg_jobs.add_to(m)

# ============================================================
# Transit
# ============================================================
fg_transit = folium.FeatureGroup(name="Transit Stops", show=False)

transit_local = transit.copy()
transit_local["lat"] = pd.to_numeric(transit_local["lat"], errors="coerce")
transit_local["lon"] = pd.to_numeric(transit_local["lon"], errors="coerce")
transit_local = transit_local.dropna(subset=["lat", "lon"])

for _, r in transit_local.iterrows():
    popup_html = f"{r.get('name', '')}<br>type: {r.get('type', '')}"

    folium.CircleMarker(
        location=[float(r["lat"]), float(r["lon"])],
        radius=3,
        color=COLOR_TRANSIT,
        fill=True,
        fill_color=COLOR_TRANSIT,
        opacity=MARKER_OPACITY,
        fill_opacity=FILL_OPACITY,
        popup=folium.Popup(popup_html, max_width=350),
    ).add_to(fg_transit)

fg_transit.add_to(m)

# ============================================================
# Scored Stations
# ============================================================
if "lat" in scoring.columns and "lon" in scoring.columns:
    fg_score = folium.FeatureGroup(name="Scored Stations", show=False)

    scoring = scoring.copy()
    scoring["lat"] = pd.to_numeric(scoring["lat"], errors="coerce")
    scoring["lon"] = pd.to_numeric(scoring["lon"], errors="coerce")
    scoring = scoring.dropna(subset=["lat", "lon"])

    popup_cols = [
        c
        for c in ["name", "scoring_name", "Total Score", "total_score", "score"]
        if c in scoring.columns
    ]

    for _, r in scoring.iterrows():
        popup = (
            "<br>".join([f"{c}: {r.get(c)}" for c in popup_cols])
            if popup_cols
            else None
        )

        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6,
            color=COLOR_SCORED,
            fill=True,
            fill_color=COLOR_SCORED,
            opacity=MARKER_OPACITY,
            fill_opacity=FILL_OPACITY,
            popup=folium.Popup(popup, max_width=350) if popup else None,
        ).add_to(fg_score)

    fg_score.add_to(m)

# %%
# ============================================================
# Layer control + output
# ============================================================
folium.LayerControl(collapsed=False).add_to(m)

m

# Optional save
# m.save("capmetro_layers_map_points_only.html")

# %%
import folium
import pandas as pd


# ============================================================
# Helpers
# ============================================================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def retail_type_str(row):
    if "type" in retail.columns:
        return str(row.get("type", ""))
    if "type1" in retail.columns and "type2" in retail.columns:
        t1 = row.get("type1", "")
        t2 = row.get("type2", "")
        return f"{t1}_{t2}".strip("_")
    return ""


# ============================================================
# Function
# ============================================================
def plot_station_radius_layers(station_name: str, radius_m: int = 275):
    """
    Return a folium map showing:
    - one MetroBike station
    - a radius circle around it
    - checkbox layers for all amenities, retail, jobs, and transit points
    """

    MARKER_OPACITY = 0.45
    FILL_OPACITY = 0.25

    COLOR_STATIONS = "#377eb8"  # blue
    COLOR_AMENITIES = "#4daf4a"  # green
    COLOR_RETAIL = "#984ea3"  # purple
    COLOR_JOBS = "#ff7f00"  # orange
    COLOR_RADIUS = "#e41a1c"  # red
    COLOR_TRANSIT = "#a65628"  # brown

    # -----------------------------
    # Clean station coords
    # -----------------------------
    df = coords.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

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
        mask = mask | (
            df[col].astype(str).str.strip().str.lower() == station_name_clean
        )

    match = df.loc[mask].copy()

    if match.empty:
        available_examples = (
            df[possible_name_cols[0]]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"No station found for '{station_name}'. "
            f"Example station names: {available_examples}"
        )

    # use first match
    r = match.iloc[0]
    station_lat = float(r["lat"])
    station_lon = float(r["lon"])

    # -----------------------------
    # Base map
    # -----------------------------
    m = folium.Map(
        location=[station_lat, station_lon], zoom_start=16, tiles="CartoDB positron"
    )

    # -----------------------------
    # Selected station + radius
    # -----------------------------
    fg_station = folium.FeatureGroup(name="Selected MetroBike Station", show=True)

    popup_html = "<br>".join(
        [
            f"<b>{r.get('scoring_name', '')}</b>",
            f"cleaned_name: {r.get('cleaned_name', '')}",
            f"coordinate_name: {r.get('coordinate_name', '')}",
            f"lat: {station_lat}",
            f"lon: {station_lon}",
        ]
    )

    folium.CircleMarker(
        location=[station_lat, station_lon],
        radius=6,
        color=COLOR_STATIONS,
        fill=True,
        fill_color=COLOR_STATIONS,
        opacity=0.9,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=350),
    ).add_to(fg_station)

    folium.Circle(
        location=[station_lat, station_lon],
        radius=radius_m,
        color=COLOR_RADIUS,
        weight=2,
        fill=True,
        fill_color=COLOR_RADIUS,
        fill_opacity=0.15,
        popup=f"{radius_m} meter radius",
    ).add_to(fg_station)

    fg_station.add_to(m)

    # -----------------------------
    # Public Amenities layer
    # -----------------------------
    fg_amen = folium.FeatureGroup(name="Public Amenities", show=False)

    amenities_local = amenities.copy()
    amenities_local["lat"] = pd.to_numeric(amenities_local["lat"], errors="coerce")
    amenities_local["lon"] = pd.to_numeric(amenities_local["lon"], errors="coerce")
    amenities_local = amenities_local.dropna(subset=["lat", "lon"])

    for _, row in amenities_local.iterrows():
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=3,
            color=COLOR_AMENITIES,
            fill=True,
            fill_color=COLOR_AMENITIES,
            opacity=MARKER_OPACITY,
            fill_opacity=FILL_OPACITY,
            popup=folium.Popup(
                f"{row.get('name', '')}<br>type: {row.get('type', '')}", max_width=350
            ),
        ).add_to(fg_amen)

    fg_amen.add_to(m)

    # -----------------------------
    # Retail / Entertainment layer
    # -----------------------------
    fg_retail = folium.FeatureGroup(name="Retail / Entertainment", show=False)

    retail_local = retail.copy()
    retail_local["lat"] = pd.to_numeric(retail_local["lat"], errors="coerce")
    retail_local["lon"] = pd.to_numeric(retail_local["lon"], errors="coerce")
    retail_local = retail_local.dropna(subset=["lat", "lon"])

    for _, row in retail_local.iterrows():
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=3,
            color=COLOR_RETAIL,
            fill=True,
            fill_color=COLOR_RETAIL,
            opacity=MARKER_OPACITY,
            fill_opacity=FILL_OPACITY,
            popup=folium.Popup(
                f"{row.get('name', '')}<br>{retail_type_str(row)}", max_width=350
            ),
        ).add_to(fg_retail)

    fg_retail.add_to(m)

    # -----------------------------
    # Jobs layer
    # -----------------------------
    fg_jobs = folium.FeatureGroup(name="Jobs", show=False)

    jobs_local = jobs.copy()
    jobs_local["job_count"] = pd.to_numeric(
        jobs_local["job_count"], errors="coerce"
    ).fillna(0)
    jobs_local["lat"] = pd.to_numeric(jobs_local["lat"], errors="coerce")
    jobs_local["lon"] = pd.to_numeric(jobs_local["lon"], errors="coerce")
    jobs_local = jobs_local.dropna(subset=["lat", "lon"])

    vmax = jobs_local["job_count"].max() if len(jobs_local) else 1
    vmax = vmax if vmax > 0 else 1

    def scale_px(x, min_px=2, max_px=12):
        return min_px + (np.sqrt(x) / np.sqrt(vmax)) * (max_px - min_px)

    for _, row in jobs_local.iterrows():
        jc = float(row["job_count"])

        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=scale_px(jc),
            color=COLOR_JOBS,
            fill=True,
            fill_color=COLOR_JOBS,
            opacity=MARKER_OPACITY,
            fill_opacity=FILL_OPACITY,
            popup=folium.Popup(f"Jobs: {int(jc):,}", max_width=250),
        ).add_to(fg_jobs)

    fg_jobs.add_to(m)

    # -----------------------------
    # Transit layer
    # -----------------------------
    fg_transit = folium.FeatureGroup(name="Transit Stops", show=False)

    transit_local = transit.copy()
    transit_local["lat"] = pd.to_numeric(transit_local["lat"], errors="coerce")
    transit_local["lon"] = pd.to_numeric(transit_local["lon"], errors="coerce")
    transit_local = transit_local.dropna(subset=["lat", "lon"])

    for _, row in transit_local.iterrows():
        popup_html = f"{row.get('name', '')}<br>type: {row.get('type', '')}"

        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=3,
            color=COLOR_TRANSIT,
            fill=True,
            fill_color=COLOR_TRANSIT,
            opacity=MARKER_OPACITY,
            fill_opacity=FILL_OPACITY,
            popup=folium.Popup(popup_html, max_width=350),
        ).add_to(fg_transit)

    fg_transit.add_to(m)

    # -----------------------------
    # Layer control
    # -----------------------------
    folium.LayerControl(collapsed=False).add_to(m)

    return m


# %%
plot_station_radius_layers("Dean Keeton/Park Place")
