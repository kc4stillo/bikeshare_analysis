import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from sklearn.neighbors import BallTree


def count_stations_within_radius(
    stations,
    radius_m=550,
    output_col=None,
    station_crs="EPSG:4326",
    projected_crs=None,
):
    """
    Add a column to stations with the number of OTHER stations within radius_m,
    excluding the station itself.

    Parameters
    ----------
    stations : pd.DataFrame
        Must contain: ['lat', 'lon']
    radius_m : float, default 550
        Radius around each station in meters
    output_col : str or None, default None
        Name of output column.
        If None, uses f"stations_within_{radius_m}m"
    station_crs : str, default "EPSG:4326"
        CRS of station lat/lon
    projected_crs : str or None, default None
        Projected CRS to use for buffering in meters.
        If None, estimated from station locations.

    Returns
    -------
    pd.DataFrame
        Original stations DataFrame with one added column
    """

    if output_col is None:
        output_col = f"stations_within_{int(radius_m)}m"

    stations_out = stations.copy()

    # -----------------------------
    # Validate required columns
    # -----------------------------
    required_station_cols = {"lat", "lon"}
    missing_station = required_station_cols - set(stations_out.columns)

    if missing_station:
        raise ValueError(f"stations is missing required columns: {missing_station}")

    if radius_m < 0:
        raise ValueError("radius_m must be non-negative.")

    # -----------------------------
    # Clean numeric columns
    # -----------------------------
    for col in ["lat", "lon"]:
        stations_out[col] = pd.to_numeric(stations_out[col], errors="coerce")

    if stations_out[["lat", "lon"]].isna().any().any():
        raise ValueError("stations has invalid lat/lon values.")

    # -----------------------------
    # Build GeoDataFrame
    # -----------------------------
    station_gdf = gpd.GeoDataFrame(
        stations_out.copy(),
        geometry=gpd.points_from_xy(stations_out["lon"], stations_out["lat"]),
        crs=station_crs,
    ).reset_index(drop=True)

    station_gdf["_station_id"] = station_gdf.index

    # -----------------------------
    # Pick projected CRS for meter-based buffering
    # -----------------------------
    if projected_crs is None:
        try:
            projected_crs = station_gdf.estimate_utm_crs()
        except Exception:
            projected_crs = "EPSG:32614"

    station_proj = station_gdf.to_crs(projected_crs)

    # -----------------------------
    # Create buffers around each station
    # -----------------------------
    station_buffers = station_proj[["_station_id", "geometry"]].copy()
    station_buffers["geometry"] = station_buffers.geometry.buffer(radius_m)

    # Rename point-side station id so we can distinguish
    station_points = station_proj[["_station_id", "geometry"]].copy()
    station_points = station_points.rename(columns={"_station_id": "_other_station_id"})

    # -----------------------------
    # Spatial join: which station points fall inside each station buffer
    # -----------------------------
    joined = gpd.sjoin(
        station_points,
        station_buffers,
        how="left",
        predicate="intersects",
    ).drop(columns=["index_right"], errors="ignore")

    # -----------------------------
    # Exclude self-matches
    # -----------------------------
    joined = joined[joined["_station_id"].notna()].copy()
    joined = joined[joined["_other_station_id"] != joined["_station_id"]].copy()

    # -----------------------------
    # Count other stations per station
    # -----------------------------
    counts = (
        joined.groupby("_station_id")
        .size()
        .reindex(station_gdf["_station_id"], fill_value=0)
        .rename(output_col)
    )

    # -----------------------------
    # Merge back to original stations
    # -----------------------------
    result = station_gdf.drop(columns=["geometry"]).merge(
        counts.reset_index(),
        on="_station_id",
        how="left",
    )

    result[output_col] = result[output_col].fillna(0).astype(int)
    result = result.drop(columns=["_station_id"], errors="ignore")

    return result


def avg_distance_k_nearest_stations(
    stations,
    k=3,
    output_col=None,
    station_crs="EPSG:4326",
    projected_crs=None,
):
    """
    Add a column to stations with the average distance (in meters) to the
    k nearest other stations, excluding itself.

    Parameters
    ----------
    stations : pd.DataFrame
        Must contain: ['lat', 'lon']
    k : int, default 3
        Number of nearest other stations to average
    output_col : str or None, default None
        Name of output column.
        If None, uses f"avg_dist_{k}_nearest_stations_m"
    station_crs : str, default "EPSG:4326"
        CRS of station lat/lon
    projected_crs : str or None, default None
        Projected CRS to use for distance calculations in meters.
        If None, estimated from station locations.

    Returns
    -------
    pd.DataFrame
        Original stations DataFrame with one added column
    """

    if output_col is None:
        output_col = f"avg_dist_{k}_nearest_stations_m"

    stations_out = stations.copy()

    # -----------------------------
    # Validate required columns
    # -----------------------------
    required_station_cols = {"lat", "lon"}
    missing_station = required_station_cols - set(stations_out.columns)

    if missing_station:
        raise ValueError(f"stations is missing required columns: {missing_station}")

    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")

    if len(stations_out) <= k:
        raise ValueError(
            f"stations must contain at least {k + 1} rows to compute the "
            f"average distance to {k} nearest other stations."
        )

    # -----------------------------
    # Clean numeric columns
    # -----------------------------
    for col in ["lat", "lon"]:
        stations_out[col] = pd.to_numeric(stations_out[col], errors="coerce")

    if stations_out[["lat", "lon"]].isna().any().any():
        raise ValueError("stations has invalid lat/lon values.")

    # -----------------------------
    # Build GeoDataFrame
    # -----------------------------
    station_gdf = gpd.GeoDataFrame(
        stations_out.copy(),
        geometry=gpd.points_from_xy(stations_out["lon"], stations_out["lat"]),
        crs=station_crs,
    ).reset_index(drop=True)

    # -----------------------------
    # Pick projected CRS for meter-based distances
    # -----------------------------
    if projected_crs is None:
        try:
            projected_crs = station_gdf.estimate_utm_crs()
        except Exception:
            projected_crs = "EPSG:32614"

    station_proj = station_gdf.to_crs(projected_crs)

    # -----------------------------
    # Extract x/y coordinates
    # -----------------------------
    coords = np.column_stack((station_proj.geometry.x, station_proj.geometry.y))

    # -----------------------------
    # Pairwise distance matrix
    # -----------------------------
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist_matrix = np.sqrt(dx**2 + dy**2)

    # Ignore self-distance
    np.fill_diagonal(dist_matrix, np.inf)

    # -----------------------------
    # Average of k nearest other stations
    # -----------------------------
    k_nearest = np.partition(dist_matrix, kth=k - 1, axis=1)[:, :k]
    avg_k_nearest = k_nearest.mean(axis=1)

    # -----------------------------
    # Attach result
    # -----------------------------
    result = station_gdf.drop(columns=["geometry"]).copy()
    result[output_col] = avg_k_nearest

    return result


def nearest_station_distance(
    stations,
    output_col="nearest_station_m",
    station_crs="EPSG:4326",
    projected_crs=None,
):
    """
    Add a column to stations with the distance (in meters) to the nearest
    other station, excluding itself.

    Parameters
    ----------
    stations : pd.DataFrame
        Must contain: ['lat', 'lon']
    output_col : str, default "nearest_station_m"
        Name of output column
    station_crs : str, default "EPSG:4326"
        CRS of station lat/lon
    projected_crs : str or None, default None
        Projected CRS to use for distance calculations in meters.
        If None, estimated from station locations.

    Returns
    -------
    pd.DataFrame
        Original stations DataFrame with one added column
    """

    stations_out = stations.copy()

    # -----------------------------
    # Validate required columns
    # -----------------------------
    required_station_cols = {"lat", "lon"}
    missing_station = required_station_cols - set(stations_out.columns)

    if missing_station:
        raise ValueError(f"stations is missing required columns: {missing_station}")

    if len(stations_out) < 2:
        raise ValueError(
            "stations must contain at least 2 rows to compute nearest-station distance."
        )

    # -----------------------------
    # Clean numeric columns
    # -----------------------------
    for col in ["lat", "lon"]:
        stations_out[col] = pd.to_numeric(stations_out[col], errors="coerce")

    if stations_out[["lat", "lon"]].isna().any().any():
        raise ValueError("stations has invalid lat/lon values.")

    # -----------------------------
    # Build GeoDataFrame
    # -----------------------------
    station_gdf = gpd.GeoDataFrame(
        stations_out.copy(),
        geometry=gpd.points_from_xy(stations_out["lon"], stations_out["lat"]),
        crs=station_crs,
    ).reset_index(drop=True)

    # -----------------------------
    # Pick projected CRS for meter-based distances
    # -----------------------------
    if projected_crs is None:
        try:
            projected_crs = station_gdf.estimate_utm_crs()
        except Exception:
            projected_crs = "EPSG:32614"

    station_proj = station_gdf.to_crs(projected_crs)

    # -----------------------------
    # Extract x/y coordinates
    # -----------------------------
    coords = np.column_stack((station_proj.geometry.x, station_proj.geometry.y))

    # -----------------------------
    # Pairwise distance matrix
    # -----------------------------
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist_matrix = np.sqrt(dx**2 + dy**2)

    # Ignore self-distance by setting diagonal to infinity
    np.fill_diagonal(dist_matrix, np.inf)

    # Nearest other station distance
    nearest_dist = dist_matrix.min(axis=1)

    # -----------------------------
    # Attach result
    # -----------------------------
    result = station_gdf.drop(columns=["geometry"]).copy()
    result[output_col] = nearest_dist

    return result


def count_within_radius(
    stations,
    points,
    radius_m=275,
    station_crs="EPSG:4326",
    points_crs="EPSG:4326",
    projected_crs=None,
    count_col=None,
    output_col=None,
):
    """
    Add a column to stations with either:
    - the sum of `count_col` within radius_m of each station, or
    - the number of points within radius_m if count_col is None

    Parameters
    ----------
    stations : pd.DataFrame
        Must contain: ['lat', 'lon']
    points : pd.DataFrame
        Must contain: ['lat', 'lon']
        If count_col is provided, must also contain that column.
    radius_m : float, default 275
        Radius around each station in meters
    station_crs : str, default "EPSG:4326"
        CRS of station lat/lon
    points_crs : str, default "EPSG:4326"
        CRS of point lat/lon
    projected_crs : str or None, default None
        Projected CRS to use for buffering in meters.
        If None, estimated from station locations.
    count_col : str or None, default None
        If provided, sums this column within the radius.
        If None, counts the number of points within the radius.
    output_col : str or None, default None
        Name of output column.
        If None:
          - uses f"{count_col}_within_{radius_m}m" when count_col is provided
          - uses f"count_within_{radius_m}m" when count_col is None

    Returns
    -------
    pd.DataFrame
        Original stations DataFrame with one added column
    """

    if output_col is None:
        if count_col is None:
            output_col = f"count_within_{int(radius_m)}m"
        else:
            output_col = f"{count_col}_within_{int(radius_m)}m"

    stations_out = stations.copy()
    points_out = points.copy()

    # -----------------------------
    # Validate required columns
    # -----------------------------
    required_station_cols = {"lat", "lon"}
    required_point_cols = {"lat", "lon"}

    if count_col is not None:
        required_point_cols.add(count_col)

    missing_station = required_station_cols - set(stations_out.columns)
    missing_points = required_point_cols - set(points_out.columns)

    if missing_station:
        raise ValueError(f"stations is missing required columns: {missing_station}")
    if missing_points:
        raise ValueError(f"points is missing required columns: {missing_points}")

    # -----------------------------
    # Clean numeric columns
    # -----------------------------
    for col in ["lat", "lon"]:
        stations_out[col] = pd.to_numeric(stations_out[col], errors="coerce")
        points_out[col] = pd.to_numeric(points_out[col], errors="coerce")

    if count_col is not None:
        points_out[count_col] = pd.to_numeric(points_out[count_col], errors="coerce")

    if stations_out[["lat", "lon"]].isna().any().any():
        raise ValueError("stations has invalid lat/lon values.")
    if points_out[["lat", "lon"]].isna().any().any():
        raise ValueError("points has invalid lat/lon values.")
    if count_col is not None and points_out[count_col].isna().any():
        raise ValueError(f"points column '{count_col}' has invalid or missing values.")

    # -----------------------------
    # Build GeoDataFrames
    # -----------------------------
    station_gdf = gpd.GeoDataFrame(
        stations_out.copy(),
        geometry=gpd.points_from_xy(stations_out["lon"], stations_out["lat"]),
        crs=station_crs,
    ).reset_index(drop=True)

    point_gdf = gpd.GeoDataFrame(
        points_out.copy(),
        geometry=gpd.points_from_xy(points_out["lon"], points_out["lat"]),
        crs=points_crs,
    ).reset_index(drop=True)

    station_gdf["_station_id"] = station_gdf.index

    # -----------------------------
    # Pick projected CRS for meter-based buffering
    # -----------------------------
    if projected_crs is None:
        try:
            projected_crs = station_gdf.estimate_utm_crs()
        except Exception:
            projected_crs = "EPSG:32614"

    station_proj = station_gdf.to_crs(projected_crs)
    point_proj = point_gdf.to_crs(projected_crs)

    # -----------------------------
    # Buffer stations by radius_m
    # -----------------------------
    station_buffers = station_proj[["_station_id", "geometry"]].copy()
    station_buffers["geometry"] = station_buffers.geometry.buffer(radius_m)

    # -----------------------------
    # Spatial join: points inside each station buffer
    # -----------------------------
    join_cols = ["geometry"] if count_col is None else [count_col, "geometry"]

    joined = gpd.sjoin(
        point_proj[join_cols],
        station_buffers,
        how="left",
        predicate="intersects",
    ).drop(columns=["index_right"], errors="ignore")

    joined = joined[joined["_station_id"].notna()].copy()

    # -----------------------------
    # Aggregate by station
    # -----------------------------
    if joined.empty:
        agg = pd.Series(0, index=station_gdf["_station_id"], name=output_col)
    else:
        if count_col is None:
            agg = (
                joined.groupby("_station_id")
                .size()
                .reindex(station_gdf["_station_id"], fill_value=0)
                .rename(output_col)
            )
        else:
            agg = (
                joined.groupby("_station_id")[count_col]
                .sum()
                .reindex(station_gdf["_station_id"], fill_value=0)
                .rename(output_col)
            )

    # -----------------------------
    # Merge back to original stations
    # -----------------------------
    result = station_gdf.drop(columns=["geometry"]).merge(
        agg.reset_index(),
        on="_station_id",
        how="left",
    )

    result[output_col] = result[output_col].fillna(0)
    result = result.drop(columns=["_station_id"], errors="ignore")

    return result


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


def area_covered_within_radius(
    stations,
    polygons_gdf,
    new_col,
    radius_m=275,
    station_crs="EPSG:4326",
    projected_crs=None,
):
    stations_out = stations.copy()

    if "geometry" not in polygons_gdf.columns:
        raise ValueError("polygons_gdf must contain a 'geometry' column.")

    # Convert WKT strings to shapely geometries if needed
    if polygons_gdf["geometry"].dtype == "object":
        sample = (
            polygons_gdf["geometry"].dropna().iloc[0]
            if not polygons_gdf["geometry"].dropna().empty
            else None
        )
        if isinstance(sample, str):
            polygons_gdf = polygons_gdf.copy()
            polygons_gdf["geometry"] = polygons_gdf["geometry"].apply(wkt.loads)
            polygons_gdf = gpd.GeoDataFrame(
                polygons_gdf, geometry="geometry", crs=station_crs
            )

    if not isinstance(polygons_gdf, gpd.GeoDataFrame):
        polygons_gdf = gpd.GeoDataFrame(
            polygons_gdf, geometry="geometry", crs=station_crs
        )

    required_cols = {"lat", "lon"}
    missing = required_cols - set(stations_out.columns)
    if missing:
        raise ValueError(f"stations is missing required columns: {missing}")

    if polygons_gdf.crs is None:
        polygons_gdf = polygons_gdf.set_crs(station_crs)

    if polygons_gdf.empty:
        stations_out[new_col] = 0.0
        return stations_out

    station_gdf = gpd.GeoDataFrame(
        stations_out.copy(),
        geometry=gpd.points_from_xy(stations_out["lon"], stations_out["lat"]),
        crs=station_crs,
    )

    if projected_crs is None:
        try:
            projected_crs = station_gdf.estimate_utm_crs()
        except Exception:
            projected_crs = "EPSG:32614"

    station_proj = station_gdf.to_crs(projected_crs)
    polygons_proj = polygons_gdf.to_crs(projected_crs).copy()
    polygons_proj = polygons_proj[
        polygons_proj.geometry.notna() & ~polygons_proj.geometry.is_empty
    ]

    if polygons_proj.empty:
        stations_out[new_col] = 0.0
        return stations_out

    sindex = polygons_proj.sindex
    covered_areas = []

    for point in station_proj.geometry:
        circle = point.buffer(radius_m)
        candidate_idx = list(sindex.query(circle, predicate="intersects"))

        if not candidate_idx:
            covered_areas.append(0.0)
            continue

        candidate_geoms = polygons_proj.iloc[candidate_idx].geometry

        try:
            merged = candidate_geoms.union_all()
        except AttributeError:
            merged = candidate_geoms.unary_union

        covered_areas.append(float(merged.intersection(circle).area))

    stations_out[new_col] = covered_areas
    return stations_out


def nearest_distance_to_polygons(
    stations,
    polygons_gdf,
    new_col,
    station_crs="EPSG:4326",
    projected_crs=None,
):
    stations_out = stations.copy()

    if "geometry" not in polygons_gdf.columns:
        raise ValueError("polygons_gdf must contain a 'geometry' column.")

    # Convert WKT strings to shapely geometries if needed
    if polygons_gdf["geometry"].dtype == "object":
        sample = (
            polygons_gdf["geometry"].dropna().iloc[0]
            if not polygons_gdf["geometry"].dropna().empty
            else None
        )
        if isinstance(sample, str):
            polygons_gdf = polygons_gdf.copy()
            polygons_gdf["geometry"] = polygons_gdf["geometry"].apply(wkt.loads)
            polygons_gdf = gpd.GeoDataFrame(
                polygons_gdf, geometry="geometry", crs=station_crs
            )

    if not isinstance(polygons_gdf, gpd.GeoDataFrame):
        polygons_gdf = gpd.GeoDataFrame(
            polygons_gdf, geometry="geometry", crs=station_crs
        )

    required_cols = {"lat", "lon"}
    missing = required_cols - set(stations_out.columns)
    if missing:
        raise ValueError(f"stations is missing required columns: {missing}")

    if polygons_gdf.crs is None:
        polygons_gdf = polygons_gdf.set_crs(station_crs)

    if polygons_gdf.empty:
        stations_out[new_col] = np.nan
        return stations_out

    station_gdf = gpd.GeoDataFrame(
        stations_out.copy(),
        geometry=gpd.points_from_xy(stations_out["lon"], stations_out["lat"]),
        crs=station_crs,
    )

    if projected_crs is None:
        try:
            projected_crs = station_gdf.estimate_utm_crs()
        except Exception:
            projected_crs = "EPSG:32614"

    station_proj = station_gdf.to_crs(projected_crs)
    polygons_proj = polygons_gdf.to_crs(projected_crs).copy()
    polygons_proj = polygons_proj[
        polygons_proj.geometry.notna() & ~polygons_proj.geometry.is_empty
    ]

    if polygons_proj.empty:
        stations_out[new_col] = np.nan
        return stations_out

    # Merge all polygons into one geometry
    try:
        merged_polygons = polygons_proj.geometry.union_all()
    except AttributeError:
        merged_polygons = polygons_proj.geometry.unary_union

    # Distance is in meters because we're in a projected CRS
    stations_out[new_col] = station_proj.geometry.distance(merged_polygons).astype(
        float
    )

    return stations_out


def attach_polygon_stats(
    stations,
    polygon_gdf,
    station_crs="EPSG:4326",
    polygon_cols=("age", "income", "population", "undergrad", "grad"),
    projected_crs=None,
    join_predicate="intersects",
):
    stations_out = stations.copy()

    # -----------------------------
    # Validate station columns
    # -----------------------------
    required_station_cols = {"lat", "lon"}
    missing_station = required_station_cols - set(stations_out.columns)
    if missing_station:
        raise ValueError(f"stations is missing required columns: {missing_station}")

    # -----------------------------
    # Validate polygon columns
    # -----------------------------
    if "geometry" not in polygon_gdf.columns:
        raise ValueError("polygon_gdf must contain a 'geometry' column.")

    missing_polygon = set(polygon_cols) - set(polygon_gdf.columns)
    if missing_polygon:
        raise ValueError(f"polygon_gdf is missing required columns: {missing_polygon}")

    # -----------------------------
    # Clean / build polygon GeoDataFrame
    # -----------------------------
    poly = polygon_gdf.copy()

    if poly["geometry"].dtype == "object":
        sample = (
            poly["geometry"].dropna().iloc[0]
            if not poly["geometry"].dropna().empty
            else None
        )
        if isinstance(sample, str):
            poly["geometry"] = poly["geometry"].apply(wkt.loads)

    if not isinstance(poly, gpd.GeoDataFrame):
        poly = gpd.GeoDataFrame(poly, geometry="geometry", crs=station_crs)

    if poly.crs is None:
        poly = poly.set_crs(station_crs)

    poly = poly[poly.geometry.notna() & ~poly.geometry.is_empty].copy()

    if poly.empty:
        raise ValueError("polygon_gdf has no valid geometries after cleaning.")

    # -----------------------------
    # Build station GeoDataFrame
    # -----------------------------
    station_gdf = gpd.GeoDataFrame(
        stations_out.copy(),
        geometry=gpd.points_from_xy(stations_out["lon"], stations_out["lat"]),
        crs=station_crs,
    ).reset_index(drop=True)

    station_gdf["_station_id"] = station_gdf.index

    # Match CRS
    if station_gdf.crs != poly.crs:
        poly = poly.to_crs(station_gdf.crs)

    poly_subset = poly[list(polygon_cols) + ["geometry"]].copy()

    # -----------------------------
    # Spatial join only (NO fallback)
    # -----------------------------
    inside_join = gpd.sjoin(
        station_gdf[["_station_id", "geometry"]],
        poly_subset,
        how="left",
        predicate=join_predicate,
    ).drop(columns=["index_right"], errors="ignore")

    # If multiple matches, keep the polygon with most non-null attributes
    inside_join["_nonnull_count"] = inside_join[list(polygon_cols)].notna().sum(axis=1)

    inside_join = (
        inside_join.sort_values(
            ["_station_id", "_nonnull_count"], ascending=[True, False]
        )
        .drop_duplicates(subset="_station_id", keep="first")
        .drop(columns=["_nonnull_count"], errors="ignore")
    )

    result = station_gdf.drop(columns=["geometry"]).merge(
        inside_join[["_station_id", *polygon_cols]],
        on="_station_id",
        how="left",
    )

    # -----------------------------
    # Final cleanup
    # -----------------------------
    result = result.drop(columns=["_station_id"], errors="ignore")

    return result
