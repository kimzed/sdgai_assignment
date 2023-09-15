from pathlib import Path


import rasterio
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.mask import mask
from shapely.ops import unary_union


from utils.raster_functions import (
    stack_raster_bands_into_single_tif_raster,
    minmax_scale_on_multi_band_raster,
    crop_raster_with_polygon,
    resample_raster, get_rgb_bands, SatelliteRgbBandsIndexes,
)

DATA_DIR = Path(__file__).parent.joinpath(
    "TASK 2 Data-20230913T191955Z-001/TASK 2 Data/"
)
DATA_DIR = Path(
    "/home/cedric/repos/sdgai_assignment/TASK 2 Data-20230913T191955Z-001/TASK 2 Data"
)
RASTER_DIR = (
    DATA_DIR
    / "Sentinel_Footprints/S2A_MSIL2A_20210910T120321_N0301_R023_T28RBS_20210910T144004.SAFE/GRANULE/L2A_T28RBS_A032481_20210910T120324/IMG_DATA"
)
FILE_LA_PALMA_BOUNDS = DATA_DIR / "La_palma_bounds.geojson"


R10m_RASTER_DIR = RASTER_DIR / "R10m"
R20m_RASTER_DIR = RASTER_DIR / "R20m"

# B02.jp2 - the
# blue
# band
# B03.jp2 - the
# green
# band
# B04.jp2 - the
# red
# band


def main():
    # Step 1
    band_files = list(R10m_RASTER_DIR.glob("*.jp2"))

    # adding band 11
    r20m_band_files = list(R20m_RASTER_DIR.glob("*.jp2"))
    # get the 11th band
    b11_file = next(
        (file for file in r20m_band_files if "B11_20m.jp2" in file.name), None
    )

    band_files.append(b11_file)
    band_files.sort()  # sort the bands in order

    # we will store the stacked rasters in a new directory
    out_raster_dir = DATA_DIR / "stacked_rasters/"
    if out_raster_dir.parent.exists() and not out_raster_dir.exists():
        out_raster_dir.mkdir()

    stacked_raster_file = out_raster_dir / "r10m_stacked.tif"
    stack_raster_bands_into_single_tif_raster(band_files, stacked_raster_file)

    # Step 2
    # load FILE_LA_PALMA_BOUNDS into gpd
    gdf_bounds_la_palma = gpd.read_file(FILE_LA_PALMA_BOUNDS)
    # since the gdf is a multi polygon, we need to merge it into a single polygon
    boundary_la_palma = gpd.GeoSeries(
        unary_union(list(gdf_bounds_la_palma["geometry"]))
    )
    boundary_la_palma.crs = gdf_bounds_la_palma.crs
    # save into a new file
    gdf_lapalma = gpd.GeoDataFrame(geometry=boundary_la_palma)
    gdf_lapalma.to_file(str(DATA_DIR / "boundary_la_palma.gpkg"), driver="GPKG")

    cropped_raster_file = DATA_DIR / "cropped_raster.tif"
    cropped_raster = crop_raster_with_polygon(
        raster_file=stacked_raster_file,
        polygon=gdf_lapalma.geometry,
        file_save=cropped_raster_file,
    )
    # Step 3
    rescaled_values = minmax_scale_on_multi_band_raster(
        cropped_raster, min_value=0, max_value=255
    )

    rescaled_raster = DATA_DIR / "scaled_raster.tif"
    with rasterio.open(cropped_raster_file) as cropped_raster_reader:
        cropped_raster_profile = cropped_raster_reader.profile

    with rasterio.open(rescaled_raster, "w", **cropped_raster_profile) as dst:
        dst.write(rescaled_values)

    # Step 4 and 5
    resampled_raster_file = DATA_DIR / "resampled_raster.tif"
    resampled_raster = resample_raster(
        raster=rescaled_raster,
        new_spatial_resolution_meter=20,
        file_save=resampled_raster_file,
    )


    # Step 6
    from rasterio.plot import show
    with rasterio.open(resampled_raster_file) as raster:
        rgb_bands = get_rgb_bands(raster_reader=raster,
                                  satellite_rgb_bands_ids=SatelliteRgbBandsIndexes.Sentinel2_R10)
        show(rgb_bands, adjust=True)


if __name__ == "__main__":
    main()
