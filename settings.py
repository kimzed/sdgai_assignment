from pathlib import Path

import torch

DELIVERABLE_DIR = Path(__file__).cwd().joinpath("deliverables")

# Task 1
DATA_DIR_TASK1 = Path(__file__).cwd().joinpath(
    "TASK 1 Data-20230913T191950Z-001/TASK 1 Data"
)
POINTS2_SHP = DATA_DIR_TASK1.joinpath("points2.shp")
ADDITIONAL_DATA_CSV = DATA_DIR_TASK1.joinpath("additional_data.csv")


# Task 2
DATA_DIR_TASK2 = Path(__file__).cwd().joinpath(
    "TASK 2 Data-20230913T191955Z-001/TASK 2 Data/"
)
RASTER_DIR = (
    DATA_DIR_TASK2 / "Sentinel_Footprints/S2A_MSIL2A_20210910T120321_N0301_R0"
    "23_T28RBS_20210910T144004.SAFE/GRANULE/L2A_T28RBS_A03"
    "2481_20210910T120324/IMG_DATA"
)
FILE_LA_PALMA_BOUNDS = DATA_DIR_TASK2 / "La_palma_bounds.geojson"
R10m_RASTER_DIR = RASTER_DIR / "R10m"
R20m_RASTER_DIR = RASTER_DIR / "R20m"


# Task 3

TASK_3_DIR = Path.cwd().joinpath("TASK 3 Data-20230913T191958Z-001/TASK 3 Data")
WEIGHTS_PATH = TASK_3_DIR / "model.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
