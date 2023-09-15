import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore
from shapely import Point, box

from settings import POINTS2_SHP, DELIVERABLE_DIR, DATA_DIR_TASK1, ADDITIONAL_DATA_CSV
from utils.visualization_functions import plot_with_basemap, scatter_plot
from utils.vector_functions import (
    add_rows_to_gdf,
    remove_points_based_on_lat_lon,
    filter_gdf_points_by_extent,
)


def main():

    # Step 1
    gdf = gpd.read_file(POINTS2_SHP)

    # Step 2
    new_point = Point(-2.00144, 11.76553)
    # since we do not have specification for other features, we set them to NAN
    features = {
        "URBAN_RURA": ["U"],
        "ALT_DEM": [42],
        "DHSCLUST": [np.NAN],
        "LATNUM": [new_point.y],
        "LONGNUM": [new_point.x],
    }

    gdf_with_point = add_rows_to_gdf(
        gdf=gdf, new_geometries=[new_point], features=features
    )

    # Step 3
    # through visual inspection, we identify that the 0, 0 points are outliers. we remove them
    coordinate_to_remove = Point(0, 0)
    gdf_clean = remove_points_based_on_lat_lon(
        gdf=gdf_with_point, coordinate_to_remove=coordinate_to_remove
    )

    # Step 4
    # only column 'LATNUM', 'LONGNUM' into a csv file
    lat_lon_to_save = gdf_clean[["LATNUM", "LONGNUM"]]
    lat_lon_to_save.to_csv(DELIVERABLE_DIR.joinpath("lat_lon.csv"))

    # Step 5
    # we add the value given in the assignment
    extent = box(
        xmin=-2.001880227, ymin=12.118472459, xmax=-0.967547730, ymax=12.593677957
    )
    gdf_filtered = filter_gdf_points_by_extent(gdf_clean, extent)
    print("Number of urban and rural points within the bounding box:\n")
    print(gdf_filtered["URBAN_RURA"].value_counts())

    # Step 6
    plot_with_basemap(
        gdf=gdf_filtered,
        output_file=DATA_DIR_TASK1.joinpath("points_within_extent.png"),
    )

    # Step 7
    # since there are several values for the same cluster number,
    # we group by mean for other features
    additional_data = pd.read_csv(ADDITIONAL_DATA_CSV)
    additional_data_grouped = additional_data.groupby("Cluster number").mean()
    gdf_merged = pd.merge(
        gdf,  # we use the .shp file as mentioned in instructions
        additional_data_grouped,
        left_on="DHSCLUST",
        right_on="Cluster number",
        how="left",
    )
    assert (
        gdf_merged["DHSCLUST"].value_counts().max() == 1
    ), "Not all cluster id are unique"

    # Step 8
    # to make a comparison, we plot two scatter plots,
    # # one on the clustered data, one on an unclustered data
    scatter_plot(
        data=additional_data,
        x_feature="wealth index",
        y_feature="number of household members",
        title="Scatterplot between Wealth Index and Number of Household Members (unclustered data)",
        output_file=DELIVERABLE_DIR.joinpath("scatter_plot_unclustered.png"),
    )
    scatter_plot(
        data=gdf_merged,
        x_feature="wealth index",
        y_feature="number of household members",
        title="Scatterplot between Wealth Index and Number of Household Members (clustered data)",
        output_file=DELIVERABLE_DIR.joinpath("scatter_plot_clustered.png"),
    )

    # Step 9
    # TODO extract the plotting into a function
    gdf_merged["number_of_household_members_zscore"] = zscore(
        gdf_merged["number of household members"]
    )

    # Setup figure and axes
    _, ax_plot = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    # Plot original data
    ax_plot[0].hist(
        gdf_merged["number of household members"], bins=30, color="blue", alpha=0.7
    )
    ax_plot[0].set_title("Histogram of Number of Household Members")
    ax_plot[0].set_xlabel("Number of Household Members")
    ax_plot[0].set_ylabel("Frequency")
    ax_plot[0].grid(True)

    # Plot standardized data
    ax_plot[1].hist(
        gdf_merged["number_of_household_members_zscore"],
        bins=30,
        color="green",
        alpha=0.7,
    )
    ax_plot[1].set_title("Histogram of Standardized Number of Household Members")
    ax_plot[1].set_xlabel("Z-Score")
    ax_plot[1].set_ylabel("Frequency")
    ax_plot[1].grid(True)

    plt.tight_layout()
    plt.savefig(DELIVERABLE_DIR / "histograms.png")
    plt.show()


if __name__ == "__main__":
    main()
