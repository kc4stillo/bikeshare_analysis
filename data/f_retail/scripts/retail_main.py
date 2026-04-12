import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd().parents[2]))

from utilities.df import avg_nearest_3_distance, count_within_radius, nearest_distance

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# %%
stations = pd.read_csv("../../e_jobs/clean/stations.csv")
retail = pd.read_csv("../clean/retail.csv")

# %%
stations = nearest_distance(stations, retail, "nearest_retail_m")
stations = count_within_radius(
    stations, retail, radius_m=275, output_col="count_retail_275m"
)
stations = count_within_radius(
    stations, retail, radius_m=550, output_col="count_retail_550m"
)
stations = avg_nearest_3_distance(stations, retail, "avg_dist_3_retail_m")

stations.head()

# %%
stations.to_csv("../clean/stations.csv", index=None)
