from pathlib import Path
from typing import Optional

import pandas as pd
from geopandas import GeoDataFrame
import seaborn as sns
import contextily as ctx
from matplotlib import pyplot as plt
from scipy.stats import linregress
from matplotlib_scalebar.scalebar import ScaleBar


def plot_with_basemap(
    gdf: GeoDataFrame,
    output_file: Optional[Path] = None,
    show: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Plots a GeoDataFrame with a basemap with cartographic standards.

    Parameters:
    - gdf: GeoDataFrame containing the geospatial data.
    - output_file: Path where the visualization should be saved.
    - show: Boolean indicating whether to display the plot.

    Returns:
    - A cartographically appealing map.
    """

    # Convert the data to Web Mercator
    gdf_web_mercator = gdf.to_crs(epsg=3857)

    _, ax_plot = plt.subplots(figsize=(12, 12))

    gdf_web_mercator.plot(
        ax=ax_plot, color="blue", edgecolor="white", markersize=50, alpha=0.7
    )
    ctx.add_basemap(ax_plot)

    # Add scale bar
    ax_plot.add_artist(ScaleBar(1))

    # Add north arrow
    x_letter, y_letter, arrow_length = 0.07, 0.2, 0.1
    ax_plot.annotate(
        "N",
        xy=(x_letter, y_letter),
        xytext=(x_letter, y_letter - arrow_length),
        arrowprops={"facecolor": 'black', "width": 5, "headwidth": 15},

        ha="center",
        va="center",
        fontsize=20,
        xycoords=ax_plot.transAxes,
    )

    if title:
        ax_plot.set_title(title, fontweight="bold", fontsize=16)
        ax_plot.set_axis_off()

    # Adding border to the map
    for spine in ax_plot.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")

    # Remove values on the axis
    ax_plot.set_xticks([])
    ax_plot.set_yticks([])

    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()

# pylint: disable=[too-many-arguments]
def scatter_plot(
    data: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    title: str,
    output_file: Optional[Path] = None,
    show: Optional[bool] = True,
) -> None:
    # pylint: disable=W0212
    """
    Plots a scatterplot for given x and y features from a dataframe with a given title,
    and also fits a regression line with the R^2 score.

    Parameters:
    - df: DataFrame containing the data.
    - x_feature: Column name for the x-axis.
    - y_feature: Column name for the y-axis.
    - title: Title of the scatterplot.

    Returns:
    - A visually appealing scatter plot with a regression line and R^2 score.
    """

    # Set style and context for the plot
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})

    plt.figure(figsize=(12, 7))

    # Plotting the data
    color = sns.color_palette("viridis", as_cmap=True)(0.5)
    plt.scatter(
        data[x_feature], data[y_feature], alpha=0.7, c=color, edgecolors="w", linewidth=0.5
    )

    # Plotting the regression line using seaborn
    sns.regplot(
        x=data[x_feature],
        y=data[y_feature],
        scatter=False,
        color="red",
        line_kws={"linewidth": 2},
    )

    # Calculating the R^2 score
    _, _, r_value, _, _ = linregress(data[x_feature], data[y_feature])
    r_squared = r_value ** 2
    plt.text(
        0.05,
        0.95,
        f"$R^2 = {r_squared:.2f}$",
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
    )

    plt.title(title, fontsize=18)
    plt.xlabel(x_feature, fontsize=16)
    plt.ylabel(y_feature, fontsize=16)

    # Remove top and right spines for a cleaner look
    sns.despine()

    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()
