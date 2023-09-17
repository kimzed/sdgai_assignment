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
        gdf (GeoDataFrame): geodataframe to add rows to
        new_geometries (geometry): list of new geometries to add to gdf
        features (dict): features of new row

    Returns:
        GeoDataFrame: geodataframe with new row added
    """

    columns_gdf = set(gdf.columns)
    columns_gdf.remove("geometry")
    if set(features.keys()) != columns_gdf:
        raise ValueError("features keys are not the same as gdf columns")

    gdf_to_add = GeoDataFrame(features, geometry=new_geometries, crs=gdf.crs)
    gdf_out = pd.concat([gdf, gdf_to_add])  # returns a GeoDataFrame

    return gdf_out


def remove_point_by_coordinate(
    gdf: GeoDataFrame, coordinate_to_remove: Point
) -> GeoDataFrame:
    """Remove points from a geodataframe based on latitude and longitude

    Args:
        gdf (GeoDataFrame): geodataframe to remove points from
        coordinate_to_remove (Point): point to remove

    Returns:
        GeoDataFrame: geodataframe with points removed
    """

    # we create a list of booleans, where True means that the point
    # is not the one we want to remove
    mask = ~gdf["geometry"].apply(
        lambda point: np.isclose(point.x, coordinate_to_remove.x)
        and np.isclose(point.y, coordinate_to_remove.y)
    )
    gdf = gdf[mask]
    return gdf
