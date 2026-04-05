import numpy as np
from sklearn.neighbors import BallTree


def nearest_distance(stations, new_df, new_col):
    """
    Add a column to `stations` with the distance in meters
    to the nearest object.

    Parameters
    ----------
    stations : pd.DataFrame
        Must contain 'lat' and 'lon' columns.
    new_df : pd.DataFrame
        Must contain 'lat' and 'lon' columns.
    new_col : str
        Name of new column

    Returns
    -------
    pd.DataFrame
        Copy of stations with a new column:
        - 'new_col'
    """
    stations_out = stations.copy()

    if new_df.empty:
        stations_out[new_col] = np.nan
        return stations_out

    earth_radius_m = 6_371_000

    station_coords = np.radians(stations_out[["lat", "lon"]].to_numpy())
    new_coords = np.radians(new_df[["lat", "lon"]].to_numpy())

    tree = BallTree(new_coords, metric="haversine")

    # Query the single nearest amenity
    distances, indices = tree.query(station_coords, k=1)

    # Convert radians to meters
    stations_out[new_col] = distances[:, 0] * earth_radius_m

    return stations_out


def avg_nearest_3_distance(stations, new_df, new_col):
    """
    Add a column to `stations` with the average distance in meters
    to the nearest 3 objects.

    Parameters
    ----------
    stations : pd.DataFrame
        Must contain 'lat' and 'lon' columns.
    new_df : pd.DataFrame
        Must contain 'lat' and 'lon' columns.
    new_col : str
        Name of new column.

    Returns
    -------
    pd.DataFrame
        Copy of stations with a new column:
        - `new_col`: average distance to nearest 3 objects in meters
    """
    stations_out = stations.copy()

    if new_df.empty:
        stations_out[new_col] = np.nan
        return stations_out

    earth_radius_m = 6_371_000

    station_coords = np.radians(stations_out[["lat", "lon"]].to_numpy())
    new_coords = np.radians(new_df[["lat", "lon"]].to_numpy())

    tree = BallTree(new_coords, metric="haversine")

    # Use up to 3 neighbors in case new_df has fewer than 3 rows
    k = min(3, len(new_df))

    distances, indices = tree.query(station_coords, k=k)

    # Convert from radians to meters, then average across nearest neighbors
    stations_out[new_col] = distances.mean(axis=1) * earth_radius_m

    return stations_out


def counts_within_radius(stations, new_df, new_col, radius_m=275):
    """
    Add a column to `stations` with the number of objects within `radius_m` meters.

    Parameters
    ----------
    stations : pd.DataFrame
        Must contain 'lat' and 'lon' columns.
    amenities : pd.DataFrame
        Must contain 'lat' and 'lon' columns.
    radius_m : float, default=275
        Search radius in meters.
    new_col : str
        Name of new column

    Returns
    -------
    pd.DataFrame
        Copy of stations with a new column:
        - 'new_col'
    """
    stations_out = stations.copy()

    # Earth radius in meters
    earth_radius_m = 6_371_000

    # Convert lat/lon to radians for haversine distance
    station_coords = np.radians(stations_out[["lat", "lon"]].to_numpy())
    new_coords = np.radians(new_df[["lat", "lon"]].to_numpy())

    # Build BallTree on amenity coordinates
    tree = BallTree(new_coords, metric="haversine")

    # Radius in radians
    radius_rad = radius_m / earth_radius_m

    # Find amenities within radius for each station
    indices = tree.query_radius(station_coords, r=radius_rad)

    # Count amenities for each station
    col_name = new_col
    stations_out[col_name] = [len(i) for i in indices]

    return stations_out
