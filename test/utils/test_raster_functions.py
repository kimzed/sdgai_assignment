import tempfile
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
import pytest
from shapely import Polygon

from utils.raster_functions import (
    stack_raster_bands_into_single_tif_raster,
    save_2d_array_as_tif,
    minmax_scale_on_multi_band_raster,
    crop_raster_with_polygon,
    resample_raster,
)


@pytest.fixture(scope="function")
def mock_raster_file():
    # Create a mock raster
    mock_raster_data = np.random.randint(0, 255, (1, 100, 100)).astype(np.uint8)

    # Define a transform (Affine transform)
    # This indicates a top-left corner at (0, 0) and a pixel size of 1x1
    transform = rasterio.transform.from_origin(0, 100, 1, 1)

    with tempfile.TemporaryDirectory() as temp_dir:
        raster_path = temp_dir + "/mock_raster.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=1,
            dtype=mock_raster_data.dtype,
            transform=transform,
        ) as mock_raster:
            mock_raster.write(mock_raster_data)

        yield raster_path  # This allows the raster path to be accessible to the test functions


def test_stack_raster_bands_into_single_raster_correct_values_are_saved():
    """
    Test that the values in the stacked raster are correct
    :return:
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        # create different rasters bands and save them to disk
        raster_band_1 = np.arange(100).reshape(10, 10) + 10
        raster_band_2 = np.arange(100).reshape(10, 10) + 20
        raster_band_3 = np.arange(100).reshape(10, 10) + 30

        raster_band_1_file = Path(temp_dir) / "raster_band_1.tif"
        raster_band_2_file = Path(temp_dir) / "raster_band_2.tif"
        raster_band_3_file = Path(temp_dir) / "raster_band_3.tif"

        metadata_rasters = {
            "driver": "GTiff",
            "dtype": "uint16",
            "nodata": None,
            "width": raster_band_1.shape[0],
            "height": raster_band_1.shape[1],
            "count": 1,
            "crs": CRS.from_epsg(32628),
        }

        save_2d_array_as_tif(raster_band_1, metadata_rasters, raster_band_1_file)
        save_2d_array_as_tif(raster_band_2, metadata_rasters, raster_band_2_file)
        save_2d_array_as_tif(raster_band_3, metadata_rasters, raster_band_3_file)

        # stack the rasters
        raster_files = [raster_band_1_file, raster_band_2_file, raster_band_3_file]
        out_raster_file = Path(temp_dir) / "stacked_raster.tif"
        stack_raster_bands_into_single_tif_raster(raster_files, out_raster_file)

        # read the stacked raster
        with rasterio.open(out_raster_file) as raster:
            raster_array = raster.read()

        # assert that the values are correct
        assert np.all(raster_array[0] == raster_band_1)
        assert np.all(raster_array[1] == raster_band_2)
        assert np.all(raster_array[2] == raster_band_3)


def test_resample_raster_resolution_doubled_values_are_clustered(mock_raster_file):
    """
    Test that the resampled raster has the correct shape
    :return:
    """
    values_before = rasterio.open(mock_raster_file).read()
    with tempfile.TemporaryDirectory() as temp_dir:
        out_raster_file = Path(temp_dir) / "resampled_raster.tif"
        # original spatial resolution is of 1 meter
        resample_raster(
            raster=mock_raster_file,
            file_save=out_raster_file,
            new_spatial_resolution_meter=2,
        )

        with rasterio.open(out_raster_file) as raster:
            values_after = raster.read()

        # number of pixels is the same since we keep the raster original extent
        assert values_after.shape == (1, 100, 100)

        # Check right neighbor similarity
        similarity_before = np.sum(values_before[:, :, :-1] == values_before[:, :, 1:])
        similarity_after = np.sum(values_after[:, :, :-1] == values_after[:, :, 1:])

        # This way shows that the new raster has groups of pixels with the same value
        # because of the resampling
        assert similarity_after > 1000
        assert similarity_before < 100



def test_crop_raster_with_polygon_shape_after_cropping_is_correct(mock_raster_file):

    polygon = Polygon([(0, 0), (0, 50), (50, 50), (50, 0)])
    cropped_raster = crop_raster_with_polygon(mock_raster_file, [polygon], None)

    assert cropped_raster.shape == (1, 50, 50)


def test_minmax_scale_on_multi_band_raster_correct_values_are_computed():

    raster_band_1 = np.array([[0, 0, 3000], [0, 0, 3000], [0, 0, 3000]])
    raster_band_2 = np.array([[3000, 0, 0], [0, 0, 0], [0, 0, 3000]])

    raster = np.stack([raster_band_1, raster_band_2], axis=0)

    min_value = 0
    max_value = 255

    rescaled_raster = minmax_scale_on_multi_band_raster(raster, min_value, max_value)

    # assert that the values are correct
    assert np.all(
        np.isclose(
            rescaled_raster[0], np.array([[0, 0, 255], [0, 0, 255], [0, 0, 255]])
        )
    )
    assert np.all(
        np.isclose(rescaled_raster[1], np.array([[255, 0, 0], [0, 0, 0], [0, 0, 255]]))
    )
