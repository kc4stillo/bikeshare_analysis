import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Point
from shapely.ops import unary_union


def lat_lon_get_polygon_attributes_with_nearest_fill(
    lat,
    lon,
    gdf,
    geometry_col="geometry",
    target_crs="EPSG:26914",
    columns=None,
    return_source_info=False,
):
    """
    Given a lat/lon and a GeoDataFrame of polygons, find the polygon that
    contains the point. Return its attribute values, but if any attribute
    is NaN, fill that specific attribute from the nearest polygon with a
    non-NaN value in that column.

    If the point is not inside any polygon, use the nearest polygon overall
    as the base row.

    Parameters
    ----------
    lat : float
        Latitude of the input point.
    lon : float
        Longitude of the input point.
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries and attributes.
    geometry_col : str, default="geometry"
        Name of the geometry column.
    target_crs : str, default="EPSG:26914"
        Projected CRS used for nearest-distance calculations in meters.
    columns : list or None, default=None
        List of attribute columns to return.
        If None, returns all non-geometry columns.
    return_source_info : bool, default=False
        If True, also returns a dict showing where each value came from.

    Returns
    -------
    pandas.Series
        Attribute values for the point.
    OR
    tuple
        (result_series, source_info_dict) if return_source_info=True
    """

    if gdf.empty:
        raise ValueError("gdf is empty.")

    if geometry_col not in gdf.columns:
        raise ValueError(f"gdf must contain '{geometry_col}' column.")

    if gdf.crs is None:
        raise ValueError("gdf must have a CRS set.")

    valid_gdf = gdf.dropna(subset=[geometry_col]).copy()

    if valid_gdf.empty:
        raise ValueError("gdf has no valid geometries.")

    valid_gdf = valid_gdf.set_geometry(geometry_col)

    # decide which columns to return
    if columns is None:
        columns = [col for col in valid_gdf.columns if col != geometry_col]

    # create point in WGS84, then project to gdf CRS
    point_gdf = gpd.GeoDataFrame({"geometry": [Point(lon, lat)]}, crs="EPSG:4326")
    point_in_gdf_crs = point_gdf.to_crs(valid_gdf.crs)
    point_geom = point_in_gdf_crs.geometry.iloc[0]

    # find containing polygon
    containing = valid_gdf[valid_gdf.geometry.covers(point_geom)]

    if not containing.empty:
        base_idx = containing.index[0]
        base_row = valid_gdf.loc[base_idx]
        base_source = "containing_polygon"
    else:
        # fallback: nearest polygon overall
        valid_gdf_proj = valid_gdf.to_crs(target_crs)
        point_proj = point_gdf.to_crs(target_crs)
        point_proj_geom = point_proj.geometry.iloc[0]

        distances = valid_gdf_proj.geometry.distance(point_proj_geom)
        base_idx = distances.idxmin()
        base_row = valid_gdf.loc[base_idx]
        base_source = "nearest_polygon_no_containment"

    result = {}
    source_info = {}

    # projected version for nearest-neighbor fallback by column
    valid_gdf_proj = valid_gdf.to_crs(target_crs)
    point_proj = point_gdf.to_crs(target_crs)
    point_proj_geom = point_proj.geometry.iloc[0]

    for col in columns:
        base_val = base_row[col]

        if pd.notna(base_val):
            result[col] = base_val
            source_info[col] = {"source_index": base_idx, "source_type": base_source}
        else:
            # nearest polygon with non-null value in this specific column
            candidate_idx = valid_gdf[valid_gdf[col].notna()].index

            if len(candidate_idx) == 0:
                result[col] = pd.NA
                source_info[col] = {
                    "source_index": None,
                    "source_type": "no_non_null_polygon_found",
                }
            else:
                candidate_proj = valid_gdf_proj.loc[candidate_idx]
                distances = candidate_proj.geometry.distance(point_proj_geom)
                nearest_idx = distances.idxmin()

                result[col] = valid_gdf.loc[nearest_idx, col]
                source_info[col] = {
                    "source_index": nearest_idx,
                    "source_type": f"nearest_non_null_for_{col}",
                }

    result_series = pd.Series(result)

    if return_source_info:
        return result_series, source_info

    return result_series


def lat_lon_area_occupied_within_radius(
    lat,
    lon,
    gdf,
    radius_m,
    geometry_col="geometry",
    target_crs="EPSG:26914",
    return_fraction=False,
):
    """
    Given a lat/lon and a GeoDataFrame of polygons/multipolygons,
    calculate how much area within a radius around the point is occupied
    by any geometry in the GeoDataFrame.

    Parameters
    ----------
    lat : float
        Latitude of the input point.
    lon : float
        Longitude of the input point.
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon or multipolygon geometries.
    radius_m : float
        Radius around the point, in meters.
    geometry_col : str, default="geometry"
        Name of the geometry column in gdf.
    target_crs : str, default="EPSG:26914"
        Projected CRS to use for meter-based area/distance calculations.
        EPSG:26914 works well for Austin-area work.
    return_fraction : bool, default=False
        If True, also return the fraction of the circular buffer occupied.

    Returns
    -------
    float
        Area in square meters within the radius that is occupied by
        any geometry in gdf.
    OR
    tuple
        (occupied_area_m2, occupied_fraction) if return_fraction=True
    """

    if gdf.empty:
        raise ValueError("gdf is empty.")

    if geometry_col not in gdf.columns:
        raise ValueError(f"gdf must contain a '{geometry_col}' column.")

    if gdf.crs is None:
        raise ValueError("gdf must have a CRS set before area calculations.")

    valid_gdf = gdf.dropna(subset=[geometry_col]).copy()

    if valid_gdf.empty:
        raise ValueError("gdf has no valid geometries.")

    valid_gdf = valid_gdf.set_geometry(geometry_col)

    # input point in WGS84
    point_gdf = gpd.GeoDataFrame({"geometry": [Point(lon, lat)]}, crs="EPSG:4326")

    # project to meter-based CRS
    valid_gdf_proj = valid_gdf.to_crs(target_crs)
    point_proj = point_gdf.to_crs(target_crs)

    point_geom = point_proj.geometry.iloc[0]

    # circular buffer around the point
    buffer_geom = point_geom.buffer(radius_m)

    # combine all polygons so overlaps are not double-counted
    all_geom = unary_union(valid_gdf_proj.geometry)

    # clip unioned geometry to the buffer
    occupied_geom = all_geom.intersection(buffer_geom)

    occupied_area_m2 = occupied_geom.area
    buffer_area_m2 = buffer_geom.area

    if return_fraction:
        occupied_fraction = (
            occupied_area_m2 / buffer_area_m2 if buffer_area_m2 > 0 else 0.0
        )
        return float(occupied_area_m2), float(occupied_fraction)

    return float(occupied_area_m2)


def lat_lon_average_distance_to_3_nearest(lat, lon, df, lat_col="lat", lon_col="lon"):
    """
    Given a lat/lon, calculate the average distance in meters
    to the 3 nearest rows in df.

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

    Returns
    -------
    float
        Average distance in meters to the 3 nearest rows.
    """

    if df.empty:
        raise ValueError("df is empty.")

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"df must contain '{lat_col}' and '{lon_col}' columns.")

    valid_df = df.dropna(subset=[lat_col, lon_col]).copy()

    if len(valid_df) < 3:
        raise ValueError("df must have at least 3 valid lat/lon rows.")

    # convert to radians
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

    # get 3 smallest distances
    three_nearest = np.sort(distances_m)[:3]

    return float(np.mean(three_nearest))


def lat_lon_count_points_within_radius(
    lat, lon, df, radius_m, lat_col="lat", lon_col="lon"
):
    """
    Given a lat/lon, count how many rows in df fall within radius_m meters.

    Parameters
    ----------
    lat : float
        Latitude of the input point.
    lon : float
        Longitude of the input point.
    df : pandas.DataFrame
        DataFrame that must contain latitude and longitude columns.
    radius_m : float
        Radius in meters.
    lat_col : str, default="lat"
        Name of the latitude column in df.
    lon_col : str, default="lon"
        Name of the longitude column in df.

    Returns
    -------
    int
        Number of rows within radius_m meters of the input point.
    """

    if df.empty:
        raise ValueError("df is empty.")

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"df must contain '{lat_col}' and '{lon_col}' columns.")

    valid_df = df.dropna(subset=[lat_col, lon_col]).copy()

    if valid_df.empty:
        raise ValueError("df has no valid lat/lon rows.")

    # convert to radians
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(valid_df[lat_col].values)
    lon2 = np.radians(valid_df[lon_col].values)

    # haversine distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_m = 6_371_000
    distances_m = earth_radius_m * c

    return int(np.sum(distances_m <= radius_m))


def lat_lon_find_nearest_geometry_distance(
    lat,
    lon,
    gdf,
    geometry_col="geometry",
    target_crs="EPSG:26914",
    return_row=False,
):
    """
    Given a lat/lon, find the distance in meters to the nearest geometry
    in a GeoDataFrame.

    Parameters
    ----------
    lat : float
        Latitude of the input point.
    lon : float
        Longitude of the input point.
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon/multipolygon geometries.
    geometry_col : str, default="geometry"
        Name of the geometry column.
    target_crs : str, default="EPSG:26914"
        Projected CRS to use for distance calculation in meters.
        EPSG:26914 works well for Austin / Texas area work.
    return_row : bool, default=False
        If True, also returns the nearest row.

    Returns
    -------
    float
        Distance in meters to the nearest geometry.
    OR
    tuple
        (nearest_distance_m, nearest_row) if return_row=True
    """

    if gdf.empty:
        raise ValueError("gdf is empty.")

    if geometry_col not in gdf.columns:
        raise ValueError(f"gdf must contain a '{geometry_col}' column.")

    if gdf.crs is None:
        raise ValueError("gdf must have a CRS set before distance calculations.")

    valid_gdf = gdf.dropna(subset=[geometry_col]).copy()

    if valid_gdf.empty:
        raise ValueError("gdf has no valid geometries.")

    # make sure geometry column is active
    valid_gdf = valid_gdf.set_geometry(geometry_col)

    # create input point in WGS84
    point_gdf = gpd.GeoDataFrame({"geometry": [Point(lon, lat)]}, crs="EPSG:4326")

    # project both to a CRS in meters
    valid_gdf_proj = valid_gdf.to_crs(target_crs)
    point_proj = point_gdf.to_crs(target_crs)

    point_geom = point_proj.geometry.iloc[0]

    distances_m = valid_gdf_proj.geometry.distance(point_geom)

    nearest_idx = distances_m.idxmin()
    nearest_distance_m = distances_m.loc[nearest_idx]

    if return_row:
        nearest_row = valid_gdf.loc[nearest_idx]
        return nearest_distance_m, nearest_row

    return nearest_distance_m


def lat_lon_find_nearest_point(
    lat, lon, df, lat_col="lat", lon_col="lon", return_row=False
):
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
