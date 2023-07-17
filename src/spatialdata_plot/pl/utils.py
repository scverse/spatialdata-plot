from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import multiscale_spatial_image as msi
import numpy as np
import pandas as pd
import spatial_image
import spatialdata as sd
import xarray as xr
from anndata import AnnData
from cycler import Cycler, cycler
from matplotlib import colors, patheffects, rcParams
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm, to_rgba
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib_scalebar.scalebar import ScaleBar
from numpy.random import default_rng
from pandas.api.types import CategoricalDtype, is_categorical_dtype
from scanpy import settings
from scanpy.plotting._tools.scatterplots import _add_categorical_legend
from scanpy.plotting.palettes import default_20, default_28, default_102
from shapely.geometry import LineString, Point, Polygon
from skimage.color import label2rgb
from skimage.morphology import erosion, square
from skimage.segmentation import find_boundaries
from skimage.util import map_array
from spatialdata import transform
from spatialdata._logging import logger as logging
from spatialdata._types import ArrayLike
from spatialdata.models import (
    Image2DModel,
)
from spatialdata.transformations import get_transformation

from spatialdata_plot.pp.utils import _get_coordinate_system_mapping

_FontWeight = Literal["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
_FontSize = Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]

to_hex = partial(colors.to_hex, keep_alpha=True)


@dataclass
class FigParams:
    """Figure params."""

    fig: Figure
    ax: Axes
    num_panels: int
    axs: Sequence[Axes] | None = None
    title: str | Sequence[str] | None = None
    ax_labels: Sequence[str] | None = None
    frameon: bool | None = None


@dataclass
class ScalebarParams:
    """Scalebar params."""

    scalebar_dx: Sequence[float] | None = None
    scalebar_units: Sequence[str] | None = None


def _prepare_params_plot(
    # this param is inferred when `pl.show`` is called
    num_panels: int,
    # this args are passed at `pl.show``
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    fig: Figure | None = None,
    ax: Axes | Sequence[Axes] | None = None,
    wspace: float | None = None,
    hspace: float = 0.25,
    ncols: int = 4,
    frameon: bool | None = None,
    # this is passed at `render_*`
    cmap: Colormap | str | None = None,
    norm: Normalize | Sequence[Normalize] | None = None,
    na_color: str | tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0),
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    # this args will be inferred from coordinate system
    scalebar_dx: float | Sequence[float] | None = None,
    scalebar_units: str | Sequence[str] | None = None,
) -> tuple[FigParams, ScalebarParams]:
    # len(list(itertools.product(*iter_panels)))

    # handle axes and size
    wspace = 0.75 / rcParams["figure.figsize"][0] + 0.02 if wspace is None else wspace
    figsize = rcParams["figure.figsize"] if figsize is None else figsize
    dpi = rcParams["figure.dpi"] if dpi is None else dpi
    if num_panels > 1 and ax is None:
        fig, grid = _panel_grid(
            num_panels=num_panels, hspace=hspace, wspace=wspace, ncols=ncols, dpi=dpi, figsize=figsize
        )
        axs: None | Sequence[Axes] = [plt.subplot(grid[c]) for c in range(num_panels)]
    elif num_panels > 1 and ax is not None:
        if len(ax) != num_panels:
            raise ValueError(f"Len of `ax`: {len(ax)} is not equal to number of panels: {num_panels}.")
        if fig is None:
            raise ValueError(
                f"Invalid value of `fig`: {fig}. If a list of `Axes` is passed, a `Figure` must also be specified."
            )
        assert isinstance(ax, Sequence), f"Invalid type of `ax`: {type(ax)}, expected `Sequence`."
        axs = ax
    else:
        axs = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    # set scalebar
    if scalebar_dx is not None:
        scalebar_dx, scalebar_units = _get_scalebar(scalebar_dx, scalebar_units, num_panels)

    fig_params = FigParams(
        fig=fig,
        ax=ax,
        axs=axs,
        num_panels=num_panels,
        frameon=frameon,
    )
    scalebar_params = ScalebarParams(scalebar_dx=scalebar_dx, scalebar_units=scalebar_units)

    return fig_params, scalebar_params


def _get_cs_contents(sdata: sd.SpatialData) -> pd.DataFrame:
    """Check which coordinate systems contain which elements and return that info."""
    cs_mapping = _get_coordinate_system_mapping(sdata)
    content_flags = ["has_images", "has_labels", "has_points", "has_shapes"]
    cs_contents = pd.DataFrame(columns=["cs"] + content_flags)

    for cs_name, element_ids in cs_mapping.items():
        # determine if coordinate system has the respective elements
        cs_has_images = bool(any((e in sdata.images) for e in element_ids))
        cs_has_labels = bool(any((e in sdata.labels) for e in element_ids))
        cs_has_points = bool(any((e in sdata.points) for e in element_ids))
        cs_has_shapes = bool(any((e in sdata.shapes) for e in element_ids))

        cs_contents = pd.concat(
            [
                cs_contents,
                pd.DataFrame(
                    {
                        "cs": cs_name,
                        "has_images": [cs_has_images],
                        "has_labels": [cs_has_labels],
                        "has_points": [cs_has_points],
                        "has_shapes": [cs_has_shapes],
                    }
                ),
            ]
        )

        cs_contents["has_images"] = cs_contents["has_images"].astype("bool")
        cs_contents["has_labels"] = cs_contents["has_labels"].astype("bool")
        cs_contents["has_points"] = cs_contents["has_points"].astype("bool")
        cs_contents["has_shapes"] = cs_contents["has_shapes"].astype("bool")

    return cs_contents


def _get_extent(
    sdata: sd.SpatialData,
    coordinate_systems: Sequence[str] | str | None = None,
    has_images: bool = True,
    has_labels: bool = True,
    has_points: bool = True,
    has_shapes: bool = True,
    elements: Iterable[Any] | None = None,
    share_extent: bool = False,
) -> dict[str, tuple[int, int, int, int]]:
    """Return the extent of all elements in their respective coordinate systems.

    Parameters
    ----------
    sdata
        The sd.SpatialData object to retrieve the extent from
    has_images
        Flag indicating whether to consider images when calculating the extent
    has_labels
        Flag indicating whether to consider labels when calculating the extent
    has_points
        Flag indicating whether to consider points when calculating the extent
    has_shapes
        Flag indicating whether to consider shapes when calculating the extent
    elements
        Optional list of element names to be considered. When None, all are used.
    share_extent
        Flag indicating whether to use the same extent for all coordinate systems

    Returns
    -------
    A dict of tuples with the shape (xmin, xmax, ymin, ymax). The keys of the
        dict are the coordinate_system keys.

    """
    extent: dict[str, dict[str, Sequence[int]]] = {}
    cs_mapping = _get_coordinate_system_mapping(sdata)
    cs_contents = _get_cs_contents(sdata)

    if elements is None:  # to shut up ruff
        elements = []

    if not isinstance(elements, list):
        raise ValueError(f"Invalid type of `elements`: {type(elements)}, expected `list`.")

    if coordinate_systems is not None:
        if isinstance(coordinate_systems, str):
            coordinate_systems = [coordinate_systems]
        cs_contents = cs_contents[cs_contents["cs"].isin(coordinate_systems)]
        cs_mapping = {k: v for k, v in cs_mapping.items() if k in coordinate_systems}

    for cs_name, element_ids in cs_mapping.items():
        extent[cs_name] = {}
        if len(elements) > 0:
            element_ids = [e for e in element_ids if e in elements]

        def _get_extent_after_transformations(element: Any, cs_name: str) -> Sequence[int]:
            tmp = element.copy()
            if len(tmp.shape) == 3:
                x_idx = 2
                y_idx = 1
            elif len(tmp.shape) == 2:
                x_idx = 1
                y_idx = 0

            transformations = get_transformation(tmp, to_coordinate_system=cs_name)
            transformations = _flatten_transformation_sequence(transformations)

            if len(transformations) == 1 and isinstance(
                transformations[0], sd.transformations.transformations.Identity
            ):
                result = (0, tmp.shape[x_idx], 0, tmp.shape[y_idx])

            else:
                origin = {
                    "x": 0,
                    "y": 0,
                }
                for t in transformations:
                    if isinstance(t, sd.transformations.transformations.Translation):
                        tmp = _translate_image(image=tmp, translation=t)

                        for idx, ax in enumerate(t.axes):
                            origin["x"] += t.translation[idx] if ax == "x" else 0
                            origin["y"] += t.translation[idx] if ax == "y" else 0

                    else:
                        tmp = transform(tmp, t)

                        if isinstance(t, sd.transformations.transformations.Scale):
                            for idx, ax in enumerate(t.axes):
                                origin["x"] *= t.scale[idx] if ax == "x" else 1
                                origin["y"] *= t.scale[idx] if ax == "y" else 1

                        elif isinstance(t, sd.transformations.transformations.Affine):
                            pass

                result = (origin["x"], tmp.shape[x_idx], origin["y"], tmp.shape[y_idx])

            del tmp
            return result

        if has_images and cs_contents.query(f"cs == '{cs_name}'")["has_images"][0]:
            for images_key in sdata.images:
                for e_id in element_ids:
                    if images_key == e_id:
                        if not isinstance(sdata.images[e_id], msi.multiscale_spatial_image.MultiscaleSpatialImage):
                            extent[cs_name][e_id] = _get_extent_after_transformations(sdata.images[e_id], cs_name)
                        else:
                            pass

        if has_labels and cs_contents.query(f"cs == '{cs_name}'")["has_labels"][0]:
            for labels_key in sdata.labels:
                for e_id in element_ids:
                    if labels_key == e_id:
                        if not isinstance(sdata.labels[e_id], msi.multiscale_spatial_image.MultiscaleSpatialImage):
                            extent[cs_name][e_id] = _get_extent_after_transformations(sdata.labels[e_id], cs_name)
                        else:
                            pass

        if has_shapes and cs_contents.query(f"cs == '{cs_name}'")["has_shapes"][0]:
            for shapes_key in sdata.shapes:
                for e_id in element_ids:
                    if shapes_key == e_id:

                        def get_point_bb(
                            point: Point, radius: int, method: Literal["topleft", "bottomright"], buffer: int = 1
                        ) -> Point:
                            x, y = point.coords[0]
                            if method == "topleft":
                                point_bb = Point(x - radius - buffer, y - radius - buffer)
                            else:
                                point_bb = Point(x + radius + buffer, y + radius + buffer)

                            return point_bb

                        y_dims = []
                        x_dims = []

                        # Split by Point and Polygon:
                        tmp_points = sdata.shapes[e_id][
                            sdata.shapes[e_id]["geometry"].apply(lambda geom: geom.geom_type == "Point")
                        ]
                        tmp_polygons = sdata.shapes[e_id][
                            sdata.shapes[e_id]["geometry"].apply(
                                lambda geom: geom.geom_type in ["Polygon", "MultiPolygon"]
                            )
                        ]

                        if not tmp_points.empty:
                            tmp_points["point_topleft"] = tmp_points.apply(
                                lambda row: get_point_bb(row["geometry"], row["radius"], "topleft"),
                                axis=1,
                            )
                            tmp_points["point_bottomright"] = tmp_points.apply(
                                lambda row: get_point_bb(row["geometry"], row["radius"], "bottomright"),
                                axis=1,
                            )
                            xmin_tl, ymin_tl, xmax_tl, ymax_tl = tmp_points["point_topleft"].total_bounds
                            xmin_br, ymin_br, xmax_br, ymax_br = tmp_points["point_bottomright"].total_bounds
                            y_dims += [min(ymin_tl, ymin_br), max(ymax_tl, ymax_br)]
                            x_dims += [min(xmin_tl, xmin_br), max(xmax_tl, xmax_br)]

                        if not tmp_polygons.empty:
                            xmin, ymin, xmax, ymax = tmp_polygons.total_bounds
                            y_dims += [ymin, ymax]
                            x_dims += [xmin, xmax]

                        del tmp_points
                        del tmp_polygons

                        extent[cs_name][e_id] = x_dims + y_dims

                        transformations = get_transformation(sdata.shapes[e_id], to_coordinate_system=cs_name)
                        transformations = _flatten_transformation_sequence(transformations)

                        for t in transformations:
                            if isinstance(t, sd.transformations.transformations.Translation):
                                for idx, ax in enumerate(t.axes):
                                    extent[cs_name][e_id][0] += t.translation[idx] if ax == "x" else 0  # type: ignore
                                    extent[cs_name][e_id][1] += t.translation[idx] if ax == "x" else 0  # type: ignore
                                    extent[cs_name][e_id][2] += t.translation[idx] if ax == "y" else 0  # type: ignore
                                    extent[cs_name][e_id][3] += t.translation[idx] if ax == "y" else 0  # type: ignore

                            else:
                                if isinstance(t, sd.transformations.transformations.Scale):
                                    for idx, ax in enumerate(t.axes):
                                        extent[cs_name][e_id][1] *= t.scale[idx] if ax == "x" else 1  # type: ignore
                                        extent[cs_name][e_id][3] *= t.scale[idx] if ax == "y" else 1  # type: ignore

                                elif isinstance(t, sd.transformations.transformations.Affine):
                                    pass

        if has_points and cs_contents.query(f"cs == '{cs_name}'")["has_points"][0]:
            for points_key in sdata.points:
                for e_id in element_ids:
                    if points_key == e_id:
                        tmp = sdata.points[points_key]
                        xmin = tmp["x"].min().compute()
                        xmax = tmp["x"].max().compute()
                        ymin = tmp["y"].min().compute()
                        ymax = tmp["y"].max().compute()
                        extent[cs_name][e_id] = [xmin, xmax, ymin, ymax]

    cswise_extent = {}
    for cs_name, cs_contents in extent.items():
        if len(cs_contents) > 0:
            xmin = min([v[0] for v in cs_contents.values()])
            xmax = max([v[1] for v in cs_contents.values()])
            ymin = min([v[2] for v in cs_contents.values()])
            ymax = max([v[3] for v in cs_contents.values()])
            cswise_extent[cs_name] = (xmin, xmax, ymin, ymax)

    if share_extent:
        global_extent = {}
        if len(cs_contents) > 0:
            xmin = min([v[0] for v in cswise_extent.values()])
            xmax = max([v[1] for v in cswise_extent.values()])
            ymin = min([v[2] for v in cswise_extent.values()])
            ymax = max([v[3] for v in cswise_extent.values()])
            for cs_name in cswise_extent:
                global_extent[cs_name] = (xmin, xmax, ymin, ymax)
        return global_extent

    return cswise_extent


def _panel_grid(
    num_panels: int,
    hspace: float,
    wspace: float,
    ncols: int,
    figsize: tuple[float, float],
    dpi: int | None = None,
) -> tuple[Figure, GridSpec]:
    n_panels_x = min(ncols, num_panels)
    n_panels_y = np.ceil(num_panels / n_panels_x).astype(int)

    fig = plt.figure(
        figsize=(figsize[0] * n_panels_x * (1 + wspace), figsize[1] * n_panels_y),
        dpi=dpi,
    )
    left = 0.2 / n_panels_x
    bottom = 0.13 / n_panels_y
    gs = GridSpec(
        nrows=n_panels_y,
        ncols=n_panels_x,
        left=left,
        right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
        bottom=bottom,
        top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
        hspace=hspace,
        wspace=wspace,
    )
    return fig, gs


def _get_scalebar(
    scalebar_dx: float | Sequence[float] | None = None,
    scalebar_units: str | Sequence[str] | None = None,
    len_lib: int | None = None,
) -> tuple[Sequence[float] | None, Sequence[str] | None]:
    if scalebar_dx is not None:
        _scalebar_dx = _get_list(scalebar_dx, _type=float, ref_len=len_lib, name="scalebar_dx")
        scalebar_units = "um" if scalebar_units is None else scalebar_units
        _scalebar_units = _get_list(scalebar_units, _type=str, ref_len=len_lib, name="scalebar_units")
    else:
        _scalebar_dx = None
        _scalebar_units = None

    return _scalebar_dx, _scalebar_units


@dataclass
class CmapParams:
    """Cmap params."""

    cmap: Colormap
    norm: Normalize
    na_color: str | tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


def _prepare_cmap_norm(
    cmap: Colormap | str | None = None,
    norm: Normalize | Sequence[Normalize] | None = None,
    na_color: str | tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    **kwargs: Any,
) -> CmapParams:
    cmap = copy(matplotlib.colormaps[rcParams["image.cmap"] if cmap is None else cmap])
    cmap.set_bad("lightgray" if na_color is None else na_color)

    if isinstance(norm, Normalize):
        pass  # TODO
    elif vcenter is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)

    return CmapParams(cmap, norm, na_color)


@dataclass
class OutlineParams:
    """Cmap params."""

    outline: bool
    outline_color: str | list[float]
    gap_size: float
    bg_size: float


def _set_outline(
    size: float,
    outline: bool = False,
    outline_width: tuple[float, float] = (0.3, 0.05),
    outline_color: str | list[float] = "#0000000ff",  # black, white
    **kwargs: Any,
) -> OutlineParams:
    bg_width, gap_width = outline_width
    point = np.sqrt(size)
    gap_size = (point + (point * gap_width) * 2) ** 2
    bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
    # the default black and white colors can be changes using the contour_config parameter

    if (len(outline_color) == 3 or len(outline_color) == 4) and all(isinstance(c, float) for c in outline_color):
        outline_color = matplotlib.colors.to_hex(outline_color)

    if outline:
        kwargs.pop("edgecolor", None)  # remove edge from kwargs if present
        kwargs.pop("alpha", None)  # remove alpha from kwargs if present

    return OutlineParams(outline, outline_color, gap_size, bg_size)


def _get_subplots(num_images: int, ncols: int = 4, width: int = 4, height: int = 3) -> plt.Figure | plt.Axes:
    """Set up the axs objects.

    Parameters
    ----------
    num_images
        Number of images to plot. Must be greater than 1.
    ncols
        Number of columns in the subplot grid, by default 4
    width
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
    """Return a list of random hex-color.

    Parameters
    ----------
    num_colors
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
    """Convert a series of continuous numerical values to hex color values using a colormap.

    Parameters
    ----------
    values
        The values to be converted to colors.
    cmap_name
        The name of the colormap to be used, by default 'viridis'.

    Returns
    -------
    pd.Series
        The converted color values as hex strings.
    """
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    colors = cmap(norm(values))

    return [colors.to_hex(color) for color in colors]


def _normalize(
    img: xr.DataArray,
    pmin: float | None = 3.0,
    pmax: float | None = 99.8,
    eps: float = 1e-20,
    clip: bool = False,
    name: str = "normed",
) -> xr.DataArray:
    """Perform a min max normalisation on the xr.DataArray.

    This function was adapted from the csbdeep package.

    Parameters
    ----------
    dataarray
        A xarray DataArray with an image field.
    pmin
        Lower quantile (min value) used to perform quantile normalization.
    pmax
        Upper quantile (max value) used to perform quantile normalization.
    eps
        Epsilon float added to prevent 0 division.
    clip
        Ensures that normed image array contains no values greater than 1.

    Returns
    -------
    xr.DataArray
        A min-max normalized image.
    """
    pmin = pmin or 0.0
    pmax = pmax or 100.0

    perc = np.percentile(img, [pmin, pmax])

    norm = (img - perc[0]) / (perc[1] - perc[0] + eps)

    if clip:
        norm = np.clip(norm, 0, 1)

    return norm


def _get_colors_for_categorical_obs(
    categories: Sequence[str | int],
    palette: ListedColormap | str | None = None,
    alpha: float = 1.0,
) -> list[str]:
    """
    Return a list of colors for a categorical observation.

    Parameters
    ----------
    adata
        AnnData object
    value_to_plot
        Name of a valid categorical observation
    categories
        categories of the categorical observation.

    Returns
    -------
    None
    """
    len_cat = len(categories)

    # check if default matplotlib palette has enough colors
    if palette is None:
        if len(rcParams["axes.prop_cycle"].by_key()["color"]) >= len_cat:
            cc = rcParams["axes.prop_cycle"]()
            palette = [next(cc)["color"] for _ in range(len_cat)]
        else:
            if len_cat <= 20:
                palette = default_20
            elif len_cat <= 28:
                palette = default_28
            elif len_cat <= len(default_102):  # 103 colors
                palette = default_102
            else:
                palette = ["grey" for _ in range(len_cat)]
                logging.info(
                    "input has more than 103 categories. Uniform " "'grey' color will be used for all categories."
                )

    # otherwise, single chanels turn out grey
    color_idx = np.linspace(0, 1, len_cat) if len_cat > 1 else [0.7]

    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        palette = [to_hex(x) for x in cmap(color_idx, alpha=alpha)]
    elif isinstance(palette, list):
        palette = [to_hex(x) for x in palette]
    elif isinstance(palette, ListedColormap):
        palette = [to_hex(x) for x in palette(color_idx, alpha=alpha)]
    elif isinstance(palette, LinearSegmentedColormap):
        palette = [to_hex(palette(x, alpha=alpha)) for x in color_idx]  # type: ignore[attr-defined]
    else:
        raise TypeError(f"Palette is {type(palette)} but should be string or `ListedColormap`.")

    return palette[:len_cat]  # type: ignore[return-value]


def _set_color_source_vec(
    adata: AnnData,
    value_to_plot: str | None,
    use_raw: bool | None = None,
    alt_var: str | None = None,
    layer: str | None = None,
    groups: Sequence[str] | str | None = None,
    palette: ListedColormap | str | None = None,
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

    color_map = dict(zip(categories, _get_colors_for_categorical_obs(categories)))
    # color_map = _get_palette(
    #     adata=adata, cluster_key=value_to_plot, categories=categories, palette=palette, alpha=alpha
    # )
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
        val_im: ArrayLike = map_array(seg, cell_id, color_vector.codes + 1)
        cols = colors.to_rgba_array(color_vector.categories)

    else:
        val_im = map_array(seg, cell_id, cell_id)  # replace with same seg id to remove missing segs

        try:
            cols = cmap_params.cmap(cmap_params.norm(color_vector))
        except TypeError:
            assert all(colors.is_color_like(c) for c in color_vector), "Not all values are color-like."
            cols = colors.to_rgba_array(color_vector)

    if seg_erosionpx is not None:
        val_im[val_im == erosion(val_im, square(seg_erosionpx))] = 0

    # check if no color is assigned, compute random colors
    unique_cols = np.unique(cols)
    if len(unique_cols) == 1 and unique_cols == 0:
        RNG = default_rng(42)
        cols = RNG.random((len(cols), 3))

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

    return np.dstack((seg_im, np.where(val_im > 0, 1, 0)))


def _get_palette(
    categories: Sequence[Any],
    adata: AnnData | None = None,
    cluster_key: None | str = None,
    palette: ListedColormap | str | None = None,
    alpha: float = 1.0,
) -> Mapping[str, str] | None:
    if adata is not None and palette is None:
        try:
            palette = adata.uns[f"{cluster_key}_colors"]  # type: ignore[arg-type]
            if len(palette) != len(categories):
                raise ValueError(
                    f"Expected palette to be of length `{len(categories)}`, found `{len(palette)}`. "
                    + f"Removing the colors in `adata.uns` with `adata.uns.pop('{cluster_key}_colors')` may help."
                )
            return {cat: to_hex(to_rgba(col)[:3]) for cat, col in zip(categories, palette)}
        except KeyError as e:
            logging.warning(e)
            return None

    len_cat = len(categories)

    if palette is None:
        if len_cat <= 20:
            palette = default_20
        elif len_cat <= 28:
            palette = default_28
        elif len_cat <= len(default_102):  # 103 colors
            palette = default_102
        else:
            palette = ["grey" for _ in range(len_cat)]
            logging.info("input has more than 103 categories. Uniform " "'grey' color will be used for all categories.")
        return {cat: to_hex(to_rgba(col)[:3]) for cat, col in zip(categories, palette[:len_cat])}

    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        palette = [to_hex(x) for x in cmap(np.linspace(0, 1, len_cat), alpha=alpha)]
    elif isinstance(palette, ListedColormap):
        palette = [to_hex(x) for x in palette(np.linspace(0, 1, len_cat), alpha=alpha)]
    else:
        raise TypeError(f"Palette is {type(palette)} but should be string or `ListedColormap`.")

    return dict(zip(categories, palette))


def _maybe_set_colors(
    source: AnnData, target: AnnData, key: str, palette: str | ListedColormap | Cycler | Sequence[Any] | None = None
) -> None:
    from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation

    color_key = f"{key}_colors"
    try:
        if palette is not None:
            raise KeyError("Unable to copy the palette when there was other explicitly specified.")
        target.uns[color_key] = source.uns[color_key]
    except KeyError:
        if isinstance(palette, ListedColormap):  # `scanpy` requires it
            palette = cycler(color=palette.colors)
        add_colors_for_categorical_sample_annotation(target, key=key, force_update_colors=True, palette=palette)


@dataclass
class LegendParams:
    """Legend params."""

    legend_fontsize: int | float | _FontSize | None = None
    legend_fontweight: int | _FontWeight = "bold"
    legend_loc: str | None = "right margin"
    legend_fontoutline: int | None = None
    na_in_legend: bool = True
    colorbar: bool = True


def _decorate_axs(
    ax: Axes,
    cax: PatchCollection,
    fig_params: FigParams,
    adata: AnnData,
    value_to_plot: str | None,
    color_source_vector: pd.Series[CategoricalDtype],
    palette: ListedColormap | str | None = None,
    alpha: float = 1.0,
    na_color: str | tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_loc: str | None = "right margin",
    legend_fontoutline: int | None = None,
    na_in_legend: bool = True,
    colorbar: bool = True,
    scalebar_dx: Sequence[float] | None = None,
    scalebar_units: Sequence[str] | None = None,
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Axes:
    if value_to_plot is not None:
        # if only dots were plotted without an associated value
        # there is not need to plot a legend or a colorbar

        if legend_fontoutline is not None:
            path_effect = [patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")]
        else:
            path_effect = []

        # Adding legends
        if is_categorical_dtype(color_source_vector):
            clusters = color_source_vector.categories
            palette = _get_palette(
                adata=adata, cluster_key=value_to_plot, categories=clusters, palette=palette, alpha=alpha
            )
            _add_categorical_legend(
                ax,
                color_source_vector,
                palette=palette,
                legend_loc=legend_loc,
                legend_fontweight=legend_fontweight,
                legend_fontsize=legend_fontsize,
                legend_fontoutline=path_effect,
                na_color=[na_color],
                na_in_legend=na_in_legend,
                multi_panel=fig_params.axs is not None,
            )
        elif colorbar:
            # TODO: na_in_legend should have some effect here
            plt.colorbar(cax, ax=ax, pad=0.01, fraction=0.08, aspect=30)

    # if img is not None:
    #     ax.imshow(img, cmap=img_cmap, alpha=img_alpha)
    # else:
    #     ax.set_aspect("equal")
    #     ax.invert_yaxis()

    if isinstance(scalebar_dx, list) and isinstance(scalebar_units, list):
        scalebar = ScaleBar(scalebar_dx, units=scalebar_units, **scalebar_kwargs)
        ax.add_artist(scalebar)

    return ax


def _get_list(
    var: Any,
    _type: type[Any] | tuple[type[Any], ...],
    ref_len: int | None = None,
    name: str | None = None,
) -> list[Any]:
    """
    Get a list from a variable.

    Parameters
    ----------
    var
        Variable to convert to a list.
    _type
        Type of the elements in the list.
    ref_len
        Reference length of the list.
    name
        Name of the variable.

    Returns
    -------
    List
    """
    if isinstance(var, _type):
        return [var] if ref_len is None else ([var] * ref_len)
    if isinstance(var, list):
        if ref_len is not None and ref_len != len(var):
            raise ValueError(
                f"Variable: `{name}` has length: {len(var)}, which is not equal to reference length: {ref_len}."
            )
        for v in var:
            if not isinstance(v, _type):
                raise ValueError(f"Variable: `{name}` has invalid type: {type(v)}, expected: {_type}.")
        return var

    raise ValueError(f"Can't make a list from variable: `{var}`")


def save_fig(fig: Figure, path: str | Path, make_dir: bool = True, ext: str = "png", **kwargs: Any) -> None:
    """
    Save a figure.

    Parameters
    ----------
    fig
        Figure to save.
    path
        Path where to save the figure. If path is relative, save it under :attr:`scanpy.settings.figdir`.
    make_dir
        Whether to try making the directory if it does not exist.
    ext
        Extension to use if none is provided.
    kwargs
        Keyword arguments for :func:`matplotlib.figure.Figure.savefig`.

    Returns
    -------
    None
        Just saves the plot.
    """
    if os.path.splitext(path)[1] == "":
        path = f"{path}.{ext}"

    path = Path(path)

    if not path.is_absolute():
        path = Path(settings.figdir) / path

    if make_dir:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.debug(f"Unable to create directory `{path.parent}`. Reason: `{e}`")

    logging.debug(f"Saving figure to `{path!r}`")

    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("transparent", True)

    fig.savefig(path, **kwargs)


def _get_cs_element_map(
    element: str | Sequence[str] | None,
    element_map: Mapping[str, Any],
) -> Mapping[str, str]:
    """Get the mapping between the coordinate system and the class."""
    # from spatialdata.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel, PointsModel, ShapesModel
    element = list(element_map.keys())[0] if element is None else element
    element = [element] if isinstance(element, str) else element
    d = {}
    for e in element:
        cs = list(element_map[e].attrs["transform"].keys())[0]
        d[cs] = e
        # model = get_model(element_map["blobs_labels"])
        # if model in [Image2DModel, Image3DModel, Labels2DModel, Labels3DModel]
    return d


def _multiscale_to_image(sdata: sd.SpatialData) -> sd.SpatialData:
    if sdata.images is None:
        raise ValueError("No images found in the SpatialData object.")

    for k, v in sdata.images.items():
        if isinstance(v, msi.multiscale_spatial_image.MultiscaleSpatialImage):
            sdata.images[k] = Image2DModel.parse(v["scale0"].ds.to_array().squeeze(axis=0))

    return sdata


def _get_linear_colormap(colors: list[str], background: str) -> list[LinearSegmentedColormap]:
    return [LinearSegmentedColormap.from_list(c, [background, c], N=256) for c in colors]


def _get_listed_colormap(color_dict: dict[str, str]) -> ListedColormap:
    sorted_labels = sorted(color_dict.keys())
    colors = [color_dict[k] for k in sorted_labels]

    return ListedColormap(["black"] + colors, N=len(colors) + 1)


def _translate_image(
    image: spatial_image.SpatialImage,
    translation: sd.transformations.transformations.Translation,
) -> spatial_image.SpatialImage:
    shifts: dict[str, int] = {}

    for idx, axis in enumerate(translation.axes):
        shifts[axis] = int(translation.translation[idx])

    img = image.values.copy()
    shifted_channels = []

    # split channels, shift axes individually, them recombine
    if len(image.shape) == 3:
        for c in range(image.shape[0]):
            channel = img[c, :, :]

            # iterates over [x, y]
            for axis, shift in shifts.items():
                pad_x, pad_y = (0, 0), (0, 0)
                if axis == "x" and shift > 0:
                    pad_x = (abs(shift), 0)
                elif axis == "x" and shift < 0:
                    pad_x = (0, abs(shift))

                if axis == "y" and shift > 0:
                    pad_y = (abs(shift), 0)
                elif axis == "y" and shift < 0:
                    pad_y = (0, abs(shift))

                channel = np.pad(channel, (pad_y, pad_x), mode="constant")

            shifted_channels.append(channel)

    return Image2DModel.parse(
        np.array(shifted_channels),
        dims=["c", "y", "x"],
        transformations=image.attrs["transform"],
    )


def _convert_polygon_to_linestrings(polygon: Polygon) -> list[LineString]:
    b = polygon.boundary.coords
    linestrings = [LineString(b[k : k + 2]) for k in range(len(b) - 1)]

    return [list(ls.coords) for ls in linestrings]


def _flatten_transformation_sequence(
    transformation_sequence: list[sd.transformations.transformations.Sequence],
) -> list[sd.transformations.transformations.Sequence]:
    if isinstance(transformation_sequence, sd.transformations.transformations.Sequence):
        transformations = list(transformation_sequence.transformations)
        found_bottom_of_tree = False
        while not found_bottom_of_tree:
            if all(not isinstance(t, sd.transformations.transformations.Sequence) for t in transformations):
                found_bottom_of_tree = True
            else:
                for idx, t in enumerate(transformations):
                    if isinstance(t, sd.transformations.transformations.Sequence):
                        transformations.pop(idx)
                        transformations += t.transformations

        return transformations

    if isinstance(transformation_sequence, sd.transformations.transformations.BaseTransformation):
        return [transformation_sequence]

    raise TypeError("Parameter 'transformation_sequence' must be a Sequence.")


def _robust_transform(element: Any, cs: str) -> Any:
    try:
        transformations = get_transformation(element, get_all=True)
        if cs not in transformations:
            return element
        transformations = transformations[cs]
        transformations = _flatten_transformation_sequence(transformations)
        for _, t in enumerate(transformations):
            if isinstance(t, sd.transformations.transformations.Translation):
                element = _translate_image(image=element, translation=t)

            elif isinstance(t, sd.transformations.transformations.Affine):
                # edge case, waiting for Luca to decompose affine into components
                # element = transform(element, t)
                # new_transformations = get_transformation(element, get_all=True)
                # new_transformations = new_transformations[cs]
                # new_transformations = _flatten_transformation_sequence(new_transformations)
                # seq = new_transformations[:len(new_transformations) - len(transformations)]
                # seq = sd.transformations.Sequence(seq)
                # set_transformation(element, seq, to_coordinate_system=cs)
                # element = _robust_transform(element, cs)
                # print(element.shape)
                pass

            else:
                element = transform(element, t)

    except ValueError as e:
        # hack, talk to Luca
        raise ValueError("Unable to transform element.") from e

    return element


def _mpl_ax_contains_elements(ax: Axes) -> bool:
    """Check if any objects have been plotted on the axes object.

    While extracting the extent, we need to know if the axes object has just been
    initialised and therefore has extent (0, 1), (0,1) or if it has been plotted on
    and therefore has a different extent.

    Based on: https://stackoverflow.com/a/71966295
    """
    return (
        len(ax.lines) > 0 or len(ax.collections) > 0 or len(ax.images) > 0 or len(ax.patches) > 0 or len(ax.tables) > 0
    )
