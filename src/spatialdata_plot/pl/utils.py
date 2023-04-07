from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, NamedTuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from cycler import Cycler, cycler
from matplotlib import colors
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from matpltolib.cm import Colormap
from numpy.random import default_rng
from pandas.api.types import CategoricalDtype, is_categorical_dtype
from skimage.color import label2rgb
from skimage.morphology import erosion, square
from skimage.segmentation import find_boundaries
from skimage.util import map_array
from spatialdata._types import ArrayLike

Palette_t = Optional[Union[str, ListedColormap]]
_Normalize = Union[Normalize, Sequence[Normalize]]
_SeqStr = Union[str, Sequence[str]]

to_hex = partial(colors.to_hex, keep_alpha=True)


class CmapParams(NamedTuple):
    """Cmap params."""

    cmap: Colormap
    img_cmap: Colormap
    norm: Normalize


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


def _get_random_hex_colors(num_colors: int, seed: int | None = None) -> set[str]:
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
    rng = default_rng(seed)
    colors: set[str] = set()
    while len(colors) < num_colors:
        r, g, b = rng.integers(0, 255), rng.integers(0, 255), rng.integers(0, 255)
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
    hex_colors = [colors.to_hex(color) for color in colors]

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


def _set_color_source_vec(
    adata: AnnData,
    value_to_plot: str | None,
    use_raw: bool | None = None,
    alt_var: str | None = None,
    layer: str | None = None,
    groups: _SeqStr | None = None,
    palette: Palette_t = None,
    na_color: str | tuple[float, ...] | None = None,
    alpha: float = 1.0,
) -> tuple[ArrayLike | pd.Series | None, ArrayLike, bool]:
    if value_to_plot is None:
        color = np.full(adata.n_obs, to_hex(na_color))
        return color, color, False

    if alt_var is not None and value_to_plot not in adata.obs and value_to_plot not in adata.var_names:
        value_to_plot = adata.var_names[adata.var[alt_var] == value_to_plot][0]
    if use_raw and value_to_plot not in adata.obs:
        color_source_vector = adata.raw.obs_vector(value_to_plot)
    else:
        color_source_vector = adata.obs_vector(value_to_plot, layer=layer)

    if not is_categorical_dtype(color_source_vector):
        return None, color_source_vector, False

    color_source_vector = pd.Categorical(color_source_vector)  # convert, e.g., `pd.Series`
    categories = color_source_vector.categories
    if groups is not None:
        color_source_vector = color_source_vector.remove_categories(categories.difference(groups))

    color_map = _get_palette(adata, cluster_key=value_to_plot, categories=categories, palette=palette, alpha=alpha)
    if color_map is None:
        raise ValueError("Unable to create color palette.")
    # do not rename categories, as colors need not be unique
    color_vector = color_source_vector.map(color_map)
    if color_vector.isna().any():
        color_vector = color_vector.add_categories([to_hex(na_color)])
        color_vector = color_vector.fillna(to_hex(na_color))

    return color_source_vector, color_vector, True


def _map_color_seg(
    seg: ArrayLike,
    cell_id: ArrayLike,
    color_vector: ArrayLike | pd.Series[CategoricalDtype],
    color_source_vector: pd.Series[CategoricalDtype],
    cmap_params: CmapParams,
    seg_erosionpx: int | None = None,
    seg_boundaries: bool = False,
    na_color: str | tuple[float, ...] = (0, 0, 0, 0),
) -> ArrayLike:
    cell_id = np.array(cell_id)

    if is_categorical_dtype(color_vector):
        if isinstance(na_color, tuple) and len(na_color) == 4 and np.any(color_source_vector.isna()):
            cell_id[color_source_vector.isna()] = 0
        val_im: ArrayLike = map_array(seg, cell_id, color_vector.codes + 1)  # type: ignore
        cols = colors.to_rgba_array(color_vector.categories)  # type: ignore
    else:
        val_im = map_array(seg, cell_id, cell_id)  # replace with same seg id to remove missing segs
        try:
            cols = cmap_params.cmap(cmap_params.norm(color_vector))
        except TypeError:
            assert all(colors.is_color_like(c) for c in color_vector), "Not all values are color-like."
            cols = colors.to_rgba_array(color_vector)

    if seg_erosionpx is not None:
        val_im[val_im == erosion(val_im, square(seg_erosionpx))] = 0

    seg_im: ArrayLike = label2rgb(
        label=val_im,
        colors=cols,
        bg_label=0,
        bg_color=(1, 1, 1),  # transparency doesn't really work
    )

    if seg_boundaries:
        seg_bound: ArrayLike = np.clip(seg_im - find_boundaries(seg)[:, :, None], 0, 1)
        seg_bound = np.dstack((seg_bound, np.where(val_im > 0, 1, 0)))  # add transparency here
        return seg_bound
    seg_im = np.dstack((seg_im, np.where(val_im > 0, 1, 0)))  # add transparency here
    return seg_im


def _get_palette(
    adata: AnnData,
    cluster_key: Optional[str],
    categories: Sequence[Any],
    palette: Palette_t = None,
    alpha: float = 1.0,
) -> Mapping[str, str] | None:
    if palette is None:
        try:
            palette = adata.uns[f"{cluster_key}_colors"]  # type: ignore[arg-type]
            if len(palette) != len(categories):
                raise ValueError(
                    f"Expected palette to be of length `{len(categories)}`, found `{len(palette)}`. "
                    + f"Removing the colors in `adata.uns` with `adata.uns.pop('{cluster_key}_colors')` may help."
                )
            return {cat: to_hex(to_rgba(col)[:3] + (alpha,), keep_alpha=True) for cat, col in zip(categories, palette)}
        except KeyError as e:
            print(e)
            return None

    len_cat = len(categories)
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        palette = [to_hex(x, keep_alpha=True) for x in cmap(np.linspace(0, 1, len_cat), alpha=alpha)]
    elif isinstance(palette, ListedColormap):
        palette = [to_hex(x, keep_alpha=True) for x in palette(np.linspace(0, 1, len_cat), alpha=alpha)]
    else:
        raise TypeError(f"Palette is {type(palette)} but should be string or `ListedColormap`.")

    return dict(zip(categories, palette))


def _maybe_set_colors(
    source: AnnData, target: AnnData, key: str, palette: str | ListedColormap | Cycler | Sequence[Any] | None = None
) -> None:
    color_key = f"{key}_colors"
    from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation

    # this is insane, basically the version copied here was from napari
    # in napari is modified because we have to do some tricks to plot the categorical values
    # hence it requires the argument vec. But here we don't, so am re importing the original one
    # from scanpy here.
    # this is a testament to how broken the categorical color handling is in the scanpy ecosystem and
    # to the fact that, because I've never fixed it, an embarassing amount of intellectual debt has
    # been accumulated.

    try:
        if palette is not None:
            raise KeyError("Unable to copy the palette when there was other explicitly specified.")
        target.uns[color_key] = source.uns[color_key]
    except KeyError:
        if isinstance(palette, ListedColormap):  # `scanpy` requires it
            palette = cycler(color=palette.colors)
        add_colors_for_categorical_sample_annotation(target, key=key, force_update_colors=True, palette=palette)
