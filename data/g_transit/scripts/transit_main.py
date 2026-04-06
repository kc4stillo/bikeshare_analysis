import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd().parents[2]))

from utilities import (
    avg_distance_k_nearest_stations,
    count_stations_within_radius,
    count_within_radius,
    nearest_distance,
    nearest_station_distance,
)

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# %%
stations = pd.read_csv("../../f_retail/clean/stations.csv")

stations_right = pd.read_csv("../../a_stations/stations.csv")
transit_stops = pd.read_csv("../clean/transit.csv")

bus_stops = transit_stops[transit_stops["type"] == "bus"]
rail_stops = transit_stops[transit_stops["type"] == "rail"]

# %%
stations = nearest_station_distance(stations, "nearest_bikeshare_station_m")
stations = avg_distance_k_nearest_stations(
    stations, k=3, output_col="avg_dist_3_stations"
)
stations = count_stations_within_radius(
    stations, radius_m=275, output_col="bikeshare_station_count_within_275m"
)
stations = count_stations_within_radius(
    stations, radius_m=550, output_col="bikeshare_station_count_within_550m"
)

stations = nearest_distance(stations, bus_stops, new_col="nearest_bus_stop_distance_m")

stations = count_within_radius(
    stations, bus_stops, radius_m=275, output_col="count_bus_stop_275m"
)
stations = count_within_radius(
    stations, bus_stops, radius_m=550, output_col="count_bus_stop_550m"
)

stations = nearest_distance(
    stations, rail_stops, new_col="nearest_rail_stop_distance_m"
)
stations = count_within_radius(
    stations, rail_stops, radius_m=275, output_col="count_rail_stop_275m"
)
stations = count_within_radius(
    stations, rail_stops, radius_m=550, output_col="count_rail_stop_550m"
)


stations.head()

# %%
stations.to_csv("../clean/stations.csv", index=None)
