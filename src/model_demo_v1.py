import numpy as np
import pandas as pd

stations = pd.read_csv("../data/a_stations/stations.csv")
amenities = pd.read_csv("../data/b_amenities/clean/amenities.csv")
dining_halls = pd.read_csv("../data/b_amenities/clean/dining_halls.csv")
parks = pd.read_csv("../data/b_amenities/clean/dining_halls.csv")
demographics = pd.read_csv("../data/c_demographics/clean/demographics.csv")
ut_shape = pd.read_csv("../data/d_ut_shapes/clean/ut_shape.csv")
west_campus = pd.read_csv("../data/d_ut_shapes/clean/west_campus.csv")
jobs = pd.read_csv("../data/e_jobs/clean/jobs.csv")
retail = pd.read_csv("../data/f_retail/clean/retail.csv")
tranit = pd.read_csv("../data/g_transit/clean/transit.csv")

# barton_springs_pool lat/lon
LAT = 30.264500
LON = -97.771359

DOCKS = 3


def find_nearest_point(lat, lon, df, lat_col="lat", lon_col="lon", return_row=False):
    """
    Given a lat/lon, find the distance in meters to the nearest row in df.

    Parameters
    ----------
    lat : float
        Latitude of the input point.
    lon : float
        Longitude of the input point.
    df : pandas.DataFrame
        DataFrame that must contain latitude and longitude columns.
    lat_col : str, default="lat"
        Name of the latitude column in df.
    lon_col : str, default="lon"
        Name of the longitude column in df.
    return_row : bool, default=False
        If True, also returns the nearest row.

    Returns
    -------
    float
        Distance in meters to the nearest point in df.
    OR
    tuple
        (nearest_distance_m, nearest_row) if return_row=True
    """

    if df.empty:
        raise ValueError("df is empty.")

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"df must contain '{lat_col}' and '{lon_col}' columns.")

    # drop rows with missing coords
    valid_df = df.dropna(subset=[lat_col, lon_col]).copy()

    if valid_df.empty:
        raise ValueError("df has no valid lat/lon rows.")

    # convert degrees to radians
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(valid_df[lat_col].values)
    lon2 = np.radians(valid_df[lon_col].values)

    # haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_m = 6_371_000
    distances_m = earth_radius_m * c

    nearest_idx = np.argmin(distances_m)
    nearest_distance_m = distances_m[nearest_idx]

    if return_row:
        nearest_row = valid_df.iloc[nearest_idx]
        return nearest_distance_m, nearest_row

    return nearest_distance_m


find_nearest_point(LAT, LON, dining_halls)
# ACTUAL POINT: nearest_dining_hall_m: 3903.249201

find_nearest_point(LAT, LON, amenities)
# ACTUAL POINT: nearest_amenity_m: 48.268192
