from typing import List

import cv2
import numpy as np
import rasterio

from pathlib import Path

from sklearn.preprocessing import minmax_scale


def stack_raster_bands_into_single_tif_raster(
    files: List[Path], out_file: Path
) -> None:
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


def minmax_scale_on_multi_band_raster(
    raster: np.ndarray, min_value: float, max_value: float
) -> np.ndarray:
    """
    Performs min-max scaling on a multi band raster

    :param raster: raster to be scaled
    :param min_value: minimum value of the raster
    :param max_value: maximum value of the raster
    :return: scaled raster
    """

    if min_value >= max_value:
        raise ValueError(
            f"min_value ({min_value}) must be smaller than max_value ({max_value})"
        )
    # formula: (band - band.min()) / (band.max() - band.min()) * max_value
    # we need to lower the data type to avoid memory overflow
    rescaled_bands = [
        cv2.normalize(
            band,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,  # CV_16F
        )
        for band in raster
    ]

    # stack the bands back together
    return np.stack(rescaled_bands, axis=0)


def save_2d_array_as_tif(array: np.ndarray, metadata: dict, name_file: Path) -> None:

    if array.ndim != 2:
        raise ValueError(f"Array must be 2D. Got {array.ndim}D array")
    height, width = array.shape
    if metadata["width"] != width or metadata["height"] != height:
        raise ValueError(
            f"Array shape ({array.shape}) does not match metadata shape ({metadata['width']}, {metadata['height']})"
        )

    with rasterio.open(str(name_file), "w", **metadata) as dest:
        dest.write(array, 1)  # write array to first band
