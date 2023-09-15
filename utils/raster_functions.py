from enum import Enum
from typing import List, Optional, Iterable
from pathlib import Path

from rasterio import DatasetReader
from rasterio.enums import Resampling
from rasterio.mask import mask
from shapely.geometry.polygon import Polygon
import cv2
import numpy as np
import rasterio


class SatelliteRgbBandsIndexes(Enum):
    LANDSAT8 = [3, 2, 1]
    SENTINEL2_R10 = [3, 2, 1]


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

    # we make a copy to avoid modifying the original raster
    raster_out = raster.copy()

    if min_value >= max_value:
        raise ValueError(
            f"min_value ({min_value}) must be smaller than max_value ({max_value})"
        )
    # formula: (band - band.min()) / (band.max() - band.min()) * max_value
    # we convert in the raster to avoid memory overflow
    for i_band, band in enumerate(raster_out):

        raster_out[i_band] = cv2.normalize(
            band,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )

    return raster_out


def resample_raster(
    raster: Path, new_spatial_resolution_meter: int, file_save: Optional[Path] = None
) -> np.ndarray:

    with rasterio.open(raster) as raster_reader:
        # we use the raster resolution to find the scale factor
        scale_factor_x = raster_reader.res[0] / new_spatial_resolution_meter
        scale_factor_y = raster_reader.res[1] / new_spatial_resolution_meter

        profile = raster_reader.profile.copy()
        # resample data to target shape
        raster_resampled = raster_reader.read(
            out_shape=(
                raster_reader.count,
                int(raster_reader.height * scale_factor_y),
                int(raster_reader.width * scale_factor_x),
            ),
            resampling=Resampling.bilinear,
        )

        profile.update(
            {
                "height": raster_reader.shape[-2],
                "width": raster_reader.shape[-1],
            }
        )

    if file_save:
        with rasterio.open(file_save, "w", **profile) as dataset:
            dataset.write(raster_resampled)

    return raster_resampled


def get_rgb_bands(
    raster_reader: DatasetReader, satellite_rgb_bands_ids: SatelliteRgbBandsIndexes
):

    rgb_indices = satellite_rgb_bands_ids.value
    return np.stack([raster_reader.read(i) for i in rgb_indices])


def crop_raster_with_polygon(
    raster_file: Path, polygon: Iterable[Polygon], file_save: Optional[Path]
) -> np.ndarray:

    with rasterio.open(raster_file) as stacked_raster_reader:
        raster_profile = stacked_raster_reader.profile
        cropped_raster, out_transform = mask(
            dataset=stacked_raster_reader,
            shapes=polygon,
            crop=True,
        )

    if file_save:
        # we update the metadata of the raster
        raster_profile_out = raster_profile.copy()
        raster_profile_out.update(
            {
                "driver": "GTiff",
                "height": cropped_raster.shape[1],
                "width": cropped_raster.shape[2],
                "transform": out_transform,
            }
        )
        with rasterio.open(file_save, "w", **raster_profile_out) as dst:
            dst.write(cropped_raster)

    return cropped_raster


def save_2d_array_as_tif(array: np.ndarray, metadata: dict, name_file: Path) -> None:

    if array.ndim != 2:
        raise ValueError(f"Array must be 2D. Got {array.ndim}D array")
    height, width = array.shape
    if metadata["width"] != width or metadata["height"] != height:
        raise ValueError(
            f"Array shape ({array.shape}) does not match metadata "
            f"shape ({metadata['width']}, {metadata['height']})"
        )

    with rasterio.open(str(name_file), "w", **metadata) as dest:
        dest.write(array, 1)  # write array to first band
