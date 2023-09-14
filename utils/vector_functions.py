from typing import List, Dict

import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from shapely import geometry, Point, box


def add_rows_to_gdf(
    gdf: GeoDataFrame, new_geometries: List[geometry], features: Dict[str, List]
) -> GeoDataFrame:
    """Add a row to a geodataframe

    Args:
        gdf (GeoDataFrame): geodataframe to add row to
        new_geometries (geometry): geometry of new row
        features (dict): features of new row

    Returns:
        GeoDataFrame: geodataframe with new row added
    """

    columns_gdf = set(gdf.columns)
    columns_gdf.remove("geometry")
    if not set(features.keys()) == columns_gdf:
        raise ValueError("features keys are not the same as gdf columns")

    gdf_to_add = GeoDataFrame(features, geometry=new_geometries, crs=gdf.crs)
    gdf_out = pd.concat([gdf, gdf_to_add])  # returns a GeoDataFrame

    return gdf_out


def remove_points_based_on_lat_lon(
    gdf: GeoDataFrame, coordinate_to_remove: Point
) -> GeoDataFrame:
    """Remove points from a geodataframe based on latitude and longitude

    Args:
        gdf (GeoDataFrame): geodataframe to remove points from
        coordinate_to_remove (Point): point to remove

    Returns:
        GeoDataFrame: geodataframe with points removed
    """
    # check numerically based on the coordinate values
    lons_is_equal = [
        np.isclose(point.x, coordinate_to_remove.x) for point in gdf["geometry"]
    ]
    lat_is_equal = [
        np.isclose(point.y, coordinate_to_remove.y) for point in gdf["geometry"]
    ]
    gdf = gdf[~(np.array(lons_is_equal) & np.array(lat_is_equal))]
    return gdf


def filter_gdf_points_by_extent(gdf: GeoDataFrame, extent: box) -> GeoDataFrame:
    """Filter a geodataframe based on an extent

    Args:
        gdf (GeoDataFrame): geodataframe to filter
        extent (box): extent to filter by

    Returns:
        GeoDataFrame: filtered geodataframe
    """
    # check numerically based on the coordinate values
    gdf["geometry"].apply(lambda x: x.within(extent))
    gdf = gdf[gdf["geometry"].apply(lambda x: x.within(extent))]
    return gdf
