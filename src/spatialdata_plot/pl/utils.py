import random
from collections.abc import Iterable
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import xarray as xr


def _get_subplots(num_images: int, ncols: int = 4, width: int = 4, height: int = 3) -> Union[plt.Figure, plt.Axes]:
    """Helper function to set up axes for plotting.

    Parameters
    ----------
    num_images : int
        Number of images to plot. Must be greater than 1.
    ncols : int, optional
        Number of columns in the subplot grid, by default 4
    width : int, optional
        Width of each subplot, by default 4

    Returns
    -------
    Union[plt.Figure, plt.Axes]
        Matplotlib figure and axes object.
    """
    # if num_images <= 1:
    # raise ValueError("Number of images must be greater than 1.")

    if num_images < ncols:
        nrows = 1
        ncols = num_images
    else:
        nrows, reminder = divmod(num_images, ncols)

        if nrows == 0:
            nrows = 1
        if reminder > 0:
            nrows += 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))

    if not isinstance(axes, Iterable):
        axes = np.array([axes])

    # get rid of the empty axes
    _ = [ax.axis("off") for ax in axes.flatten()[num_images:]]
    return fig, axes


def _get_random_hex_colors(num_colors: int) -> set[str]:
    """Helper function to get random colors.

    Parameters
    ----------
    num_colors : int
        Number of colors to generate.

    Returns
    -------
    list
        List of random colors.
    """
    colors: set[str] = set()
    while len(colors) < num_colors:
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        color = f"#{r:02x}{g:02x}{b:02x}"
        colors.add(color)

    return colors


def _get_hex_colors_for_continous_values(values: pd.Series, cmap_name: str = "viridis") -> list[str]:
    """Converts a series of continuous numerical values to hex color values using a colormap.

    Parameters
    ----------
    values : pd.Series
        The values to be converted to colors.
    cmap_name : str, optional
        The name of the colormap to be used, by default 'viridis'.

    Returns
    -------
    pd.Series
        The converted color values as hex strings.
    """
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    colors = cmap(norm(values))
    hex_colors = [mpl.colors.to_hex(color) for color in colors]

    return hex_colors


def _normalize(
    img: xr.DataArray,
    pmin: float = 3.0,
    pmax: float = 99.8,
    eps: float = 1e-20,
    clip: bool = False,
    name: str = "normed",
) -> xr.DataArray:
    """Performs a min max normalisation.

    This function was adapted from the csbdeep package.

    Parameters
    ----------
    dataarray: xr.DataArray
        A xarray DataArray with an image field.
    pmin: float
        Lower quantile (min value) used to perform qunatile normalization.
    pmax: float
        Upper quantile (max value) used to perform qunatile normalization.
    eps: float
        Epsilon float added to prevent 0 division.
    clip: bool
        Ensures that normed image array contains no values greater than 1.

    Returns
    -------
    xr.DataArray
        A min-max normalized image.
    """
    perc = np.percentile(img, [pmin, pmax], axis=(1, 2)).T

    norm = (img - np.expand_dims(perc[:, 0], (1, 2))) / (np.expand_dims(perc[:, 1] - perc[:, 0], (1, 2)) + eps)

    if clip:
        norm = np.clip(norm, 0, 1)

    return norm


def _get_color_key_values(
    sdata: sd.SpatialData,
    color_key: str,
) -> pd.Series:
    """Helper function to extract the values corresponding to the color key.

    The 'color_key' indicates a column, either found in sdata.table.obs or
    sdata.table.var, that will be used to color the images.

    Parameters
    ----------
    sdata: sd.SpatialData
        A spatial data object.
    color_key: str
        The column name of the color key.

    Returns
    -------
    pd.Series
        A series containing the values of the color key.
    """
    return sc.get.obs_df(sdata.table, color_key)
