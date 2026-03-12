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
scoring = pd.read_csv(prefix + "scoring/curr_stations_scored.csv")

# ============================================================
# Tunable styling
# ============================================================
MARKER_OPACITY = 0.45  # outline opacity
FILL_OPACITY = 0.25  # fill opacity

# Effective categorical palette
COLOR_STATIONS = "#377eb8"  # blue
COLOR_AMENITIES = "#4daf4a"  # green
COLOR_RETAIL = "#984ea3"  # purple
COLOR_JOBS = "#ff7f00"  # orange
COLOR_SCORED = "#e41a1c"  # red


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

for _, r in amenities.iterrows():
    lat, lon = safe_float(r.get("lat")), safe_float(r.get("lon"))
    if lat is None or lon is None:
        continue

    folium.CircleMarker(
        location=[lat, lon],
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

for _, r in retail.iterrows():
    lat, lon = safe_float(r.get("lat")), safe_float(r.get("lon"))
    if lat is None or lon is None:
        continue

    folium.CircleMarker(
        location=[lat, lon],
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

# ============================================================
# Layer control + output
# ============================================================
folium.LayerControl(collapsed=False).add_to(m)

m

# Optional save
# m.save("capmetro_layers_map_points_only.html")
