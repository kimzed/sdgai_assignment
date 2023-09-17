from settings import DELIVERABLE_DIR_TASK1, DELIVERABLE_DIR_TASK2, DELIVERABLE_DIR_TASK3
from task_1 import main as task_1_main
from task_2 import main as task_2_main
from task_3 import main as task_3_main
import pytest
import matplotlib

# we disable display of plots
matplotlib.use("Agg")


@pytest.fixture
def ensure_empty_deliverable_dir():
    # If the directory exists, remove all files in it
    if DELIVERABLE_DIR_TASK1.exists():
        for item in DELIVERABLE_DIR_TASK1.iterdir():
            if item.is_file():
                item.unlink()

    if DELIVERABLE_DIR_TASK2.exists():
        for item in DELIVERABLE_DIR_TASK2.iterdir():
            if item.is_file():
                item.unlink()

    if DELIVERABLE_DIR_TASK3.exists():
        for item in DELIVERABLE_DIR_TASK3.iterdir():
            if item.is_file():
                item.unlink()


def test_task1_main_deliverables_are_computed(ensure_empty_deliverable_dir):

    task_1_main()

    # check that deliverables are created
    assert DELIVERABLE_DIR_TASK1.exists()
    assert DELIVERABLE_DIR_TASK1.joinpath("lat_lon.csv").exists()
    assert DELIVERABLE_DIR_TASK1.joinpath("points_within_extent.png").exists()
    assert DELIVERABLE_DIR_TASK1.joinpath("scatter_plot_clustered.png").exists()
    assert DELIVERABLE_DIR_TASK1.joinpath("scatter_plot_unclustered.png").exists()
    assert DELIVERABLE_DIR_TASK1.joinpath("histograms.png").exists()


def test_task2_main_deliverables_are_computed(ensure_empty_deliverable_dir):

    task_2_main()

    assert DELIVERABLE_DIR_TASK2.joinpath("resampled_raster.tif").exists()


def test_task3_main_deliverables_are_computed(ensure_empty_deliverable_dir):

    task_3_main()

    assert DELIVERABLE_DIR_TASK3.joinpath("confusion_matrix.png").exists()
    assert DELIVERABLE_DIR_TASK3.joinpath("classification_report.csv").exists()
