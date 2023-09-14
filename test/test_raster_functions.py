import tempfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS

from utils.raster_functions import stack_raster_bands_into_single_tif_raster, save_2d_array_as_tif


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
