from pathlib import Path

import pandas as pd

amenities_file_path = "../cleaned/amenities/amenities.csv"
parks_file_path = "../cleaned/amenities/parks.csv"

stations_file_path = "../cleaned/scoring/current_stations_cleaned.csv"

stations_df = pd.read_csv(stations_file_path)


# %%
out_dir = Path("../cleaned/scoring")
out_dir.mkdir(parents=True, exist_ok=True)
