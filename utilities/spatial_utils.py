import geopandas as gpd
import numpy as np
from shapely import wkt
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
    new_df : pd.DataFrame
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

    # Find new_df within radius for each station
    indices = tree.query_radius(station_coords, r=radius_rad)

    # Count new_df for each station
    col_name = new_col
    stations_out[col_name] = [len(i) for i in indices]

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
    # Step 1: spatial join to get containing/intersecting polygon
    # -----------------------------
    inside_join = gpd.sjoin(
        station_gdf[["_station_id", "geometry"]],
        poly_subset,
        how="left",
        predicate=join_predicate,
    ).drop(columns=["index_right"], errors="ignore")

    # If multiple matches, keep the polygon with the most non-null attributes
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
    # Step 2: project for nearest fallback
    # -----------------------------
    if projected_crs is None:
        try:
            projected_crs = station_gdf.estimate_utm_crs()
        except Exception:
            projected_crs = "EPSG:32614"

    station_proj = station_gdf[["_station_id", "geometry"]].copy().to_crs(projected_crs)
    poly_proj = poly_subset.to_crs(projected_crs)

    # -----------------------------
    # Step 3: fill missing values column-by-column
    # -----------------------------
    for col in polygon_cols:
        missing_ids = result.loc[result[col].isna(), "_station_id"]

        if missing_ids.empty:
            continue

        # only polygons with a real value for this column
        valid_poly = poly_proj.loc[poly_proj[col].notna(), [col, "geometry"]].copy()

        # if no polygon anywhere has a usable value, fail loudly
        if valid_poly.empty:
            raise ValueError(
                f"Column '{col}' has no non-null values in polygon_gdf, "
                "so nearest-shape fallback cannot fill missing values."
            )

        missing_points = station_proj[
            station_proj["_station_id"].isin(missing_ids)
        ].copy()

        nearest_join = gpd.sjoin_nearest(
            missing_points, valid_poly, how="left", distance_col="_nearest_dist"
        ).drop(columns=["index_right"], errors="ignore")

        # if ties happen, keep the first nearest result per station
        nearest_join = nearest_join.sort_values(
            ["_station_id", "_nearest_dist"]
        ).drop_duplicates(subset="_station_id", keep="first")

        fill_map = nearest_join.set_index("_station_id")[col]

        result.loc[result[col].isna(), col] = result.loc[
            result[col].isna(), "_station_id"
        ].map(fill_map)

        # strict guarantee: no NaNs allowed after fallback
        if result[col].isna().any():
            unresolved = result.loc[result[col].isna(), "_station_id"].tolist()
            raise ValueError(
                f"Column '{col}' still has missing values after nearest fallback. "
                f"Unresolved station ids: {unresolved}"
            )

    # -----------------------------
    # Final cleanup
    # -----------------------------
    result = result.drop(columns=["_station_id"], errors="ignore")

    return result
