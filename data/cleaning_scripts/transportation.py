import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

metrobike_stations = pd.read_csv("../raw/scoring/raw_scores_with_coords.csv")
rail_and_bus = pd.read_csv("../raw/transit/stops.txt")

# %%
rail_stations = {
    "Downtown Station",
    "Plaza Saltillo Station",
    "MLK Jr Station",
    "Highland Station",
    "Crestview Station",
    "McKalla Station",
    "Kramer Station",
    "Howard Station",
    "Lakeline Station",
    "Leander Station",
}

# 1) make the flag
rail_and_bus["type"] = (
    rail_and_bus["stop_name"].isin(rail_stations).map({True: "rail", False: "bus"})
)

rail_and_bus = rail_and_bus[["stop_name", "stop_lat", "stop_lon", "type"]]

rail_and_bus.columns = ["name", "lat", "lon", "type"]

# %%
metrobike_stations = metrobike_stations[["name", "lat", "lon"]]
metrobike_stations["type"] = "bike"
metrobike_stations.columns = ["name", "lat", "lon", "type"]

# %%
transportation = pd.concat(
    [metrobike_stations, rail_and_bus[["name", "lat", "lon", "type"]]],
    axis=0,
    ignore_index=True,
)

# %%
# TODO: CREATE CLEANING SCRIPT FOR TRANSPORTATION NAME COL
