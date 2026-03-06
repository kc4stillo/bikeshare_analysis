from pathlib import Path

import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

metrobike_stations = pd.read_csv("../raw/scoring/raw_scores_with_coords.csv")
rail_and_bus = pd.read_csv("../raw/transit/stops.txt")


# %%
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

transportation["name"] = clean_names(transportation["name"])

out_dir = Path("../cleaned/transportation")
out_dir.mkdir(parents=True, exist_ok=True)

transportation.to_csv(out_dir / "transit.csv", index=False)
