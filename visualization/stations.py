import folium
import pandas as pd

df = pd.read_csv("../data/scoring/df.csv")

# center map on your station coordinates
m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=14)

for _, row in df.iterrows():
    popup_text = f"""
    <b>{row["name"]}</b><br>
    Total Docks: {row["total Docks"]}<br>
    Trips: {row["trips"]}<br>
    Trips per Dock: {row["trips_per_dock"]:.2f}<br>
    Bikeable Infrastructure: {row["bikeable_infrastructure"]}
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=max(5, row["trips_per_dock"] / 100),
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=row["name"],
        fill=True,
        fill_opacity=0.7,
    ).add_to(m)

m
