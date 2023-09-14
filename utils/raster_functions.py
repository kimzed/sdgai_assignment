from typing import List

import numpy as np
import rasterio

from pathlib import Path

from affine import Affine


def stack_raster_bands_into_single_tif_raster(files: List[Path], out_file: Path) -> None:
    """
    Stacks raster bands into a single raster file

    :param files:
    :param out_file:
    :return:
    """

    # pathlib is not handled by rasterio, so we need to convert to string
    files_raw = [str(file) for file in files]

    # Read metadata of first file
    with rasterio.open(files_raw[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of bands
    meta.update(count=len(files_raw))
    meta.update(driver="GTiff")

    # Read each band and write it to stack
    with rasterio.open(out_file, "w", **meta) as destination_raster:
        for id_layer, layer in enumerate(files_raw, start=1):
            with rasterio.open(layer) as band:
                destination_raster.write_band(id_layer, band.read(1))


def save_2d_array_as_tif(
    array: np.ndarray, metadata: dict, name_file: Path
) -> None:

    if array.ndim != 2:
        raise ValueError(f"Array must be 2D. Got {array.ndim}D array")
    height, width = array.shape
    if metadata["width"] != width or metadata["height"] != height:
        raise ValueError(
            f"Array shape ({array.shape}) does not match metadata shape ({metadata['width']}, {metadata['height']})"
        )

    with rasterio.open(str(name_file), "w", **metadata) as dest:
        dest.write(array, 1) # write array to first band
