import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely import Point

from utils.visualization_functions import (
    scatter_plot,
    plot_with_basemap,
    visualize_image_array,
)


def test_plot_with_basemap_file_created_is_not_empty():
    # arrange
    geometry = [Point(0, 0), Point(10, 10), Point(0, 0)]
    df = {"geometry": geometry}
    gdf = GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # act
    with tempfile.TemporaryDirectory() as temp_dir:
        file_out = Path(Path(temp_dir + "test.png"))
        plot_with_basemap(gdf, output_file=file_out, show=False)

        # assert
        image = Image.open(file_out)
        image_array = np.array(image)
        # assert values in the image are varied
        assert np.var(image_array) > 100


def test_visualize_image_array_file_created_is_not_empty():
    # arrange
    image = np.random.rand(100, 100, 3)

    # act
    with tempfile.TemporaryDirectory() as temp_dir:
        file_out = Path(Path(temp_dir + "test.png"))
        visualize_image_array(image, file_save=file_out, show=False)

        # assert
        image = Image.open(file_out)
        image_array = np.array(image)
        # assert values in the image are varied
        assert np.var(image_array) > 100


def test_scatter_plot_file_created_is_not_empty():
    # arrange
    x = np.random.rand(100)
    y = np.random.rand(100)
    df = DataFrame({"x": x, "y": y})

    # act
    with tempfile.TemporaryDirectory() as temp_dir:
        file_out = Path(Path(temp_dir + "test.png"))
        scatter_plot(
            data=df,
            x_feature="x",
            y_feature="y",
            title="test",
            output_file=file_out,
            show=False,
        )

        # assert
        image = Image.open(file_out)
        image_array = np.array(image)
        # assert values in the image are varied
        assert np.var(image_array) > 100
