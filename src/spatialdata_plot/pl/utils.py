from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from copy import copy
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.patches as mplp
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import multiscale_spatial_image as msi
import numpy as np
import pandas as pd
import shapely
import spatial_image
import spatialdata as sd
import xarray as xr
from anndata import AnnData
from cycler import Cycler, cycler
from geopandas import GeoDataFrame
from matplotlib import colors, patheffects, rcParams
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import (
    ColorConverter,
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
    TwoSlopeNorm,
    to_rgba,
)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib_scalebar.scalebar import ScaleBar
from numpy.random import default_rng
from pandas.api.types import CategoricalDtype, is_categorical_dtype
from scanpy import settings
from scanpy.plotting._tools.scatterplots import _add_categorical_legend
from scanpy.plotting.palettes import default_20, default_28, default_102
from shapely.geometry import LineString, Polygon
from skimage.color import label2rgb
from skimage.morphology import erosion, square
from skimage.segmentation import find_boundaries
from skimage.util import map_array
from spatialdata._core.query.relational_query import _locate_value, get_values
from spatialdata._logging import logger as logging
from spatialdata._types import ArrayLike
from spatialdata.models import Image2DModel, SpatialElement

from spatialdata_plot.pl.render_params import (
    CmapParams,
    FigParams,
    OutlineParams,
    ScalebarParams,
    ShapesRenderParams,
    _FontSize,
    _FontWeight,
)
from spatialdata_plot.pp.utils import _get_coordinate_system_mapping

to_hex = partial(colors.to_hex, keep_alpha=True)


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
    # handle axes and size
    wspace = 0.75 / rcParams["figure.figsize"][0] + 0.02 if wspace is None else wspace
    figsize = rcParams["figure.figsize"] if figsize is None else figsize
    dpi = rcParams["figure.dpi"] if dpi is None else dpi
    if num_panels > 1 and ax is None:
        fig, grid = _panel_grid(
            num_panels=num_panels, hspace=hspace, wspace=wspace, ncols=ncols, dpi=dpi, figsize=figsize
        )
        axs: None | Sequence[Axes] = [plt.subplot(grid[c]) for c in range(num_panels)]
    elif num_panels > 1:
        if not isinstance(ax, Sequence):
            raise TypeError(f"Expected `ax` to be a `Sequence`, but got {type(ax).__name__}")
        if ax is not None and len(ax) != num_panels:
            raise ValueError(f"Len of `ax`: {len(ax)} is not equal to number of panels: {num_panels}.")
        if fig is None:
            raise ValueError(
                f"Invalid value of `fig`: {fig}. If a list of `Axes` is passed, a `Figure` must also be specified."
            )
        assert ax is None or isinstance(ax, Sequence), f"Invalid type of `ax`: {type(ax)}, expected `Sequence`."
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
        cs_has_images = any(e in sdata.images for e in element_ids)
        cs_has_labels = any(e in sdata.labels for e in element_ids)
        cs_has_points = any(e in sdata.points for e in element_ids)
        cs_has_shapes = any(e in sdata.shapes for e in element_ids)

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


def _get_collection_shape(
    shapes: list[GeoDataFrame],
    c: Any,
    s: float,
    norm: Any,
    render_params: ShapesRenderParams,
    fill_alpha: None | float = None,
    outline_alpha: None | float = None,
    **kwargs: Any,
) -> PatchCollection:
    """
    Get a PatchCollection for rendering given geometries with specified colors and outlines.

    Args:
    - shapes (list[GeoDataFrame]): List of geometrical shapes.
    - c: Color parameter.
    - s (float): Scale of the shape.
    - norm: Normalization for the color map.
    - fill_alpha (float, optional): Opacity for the fill color.
    - outline_alpha (float, optional): Opacity for the outline.
    - **kwargs: Additional keyword arguments.

    Returns
    -------
    - PatchCollection: Collection of patches for rendering.
    """
    cmap = kwargs["cmap"]

    try:
        # fails when numeric
        fill_c = ColorConverter().to_rgba_array(c)
    except ValueError:
        if norm is None:
            c = cmap(c)
        else:
            norm = colors.Normalize(vmin=min(c), vmax=max(c))
            c = cmap(norm(c))

    fill_c = ColorConverter().to_rgba_array(c)
    fill_c[..., -1] = render_params.fill_alpha

    if render_params.outline_params.outline:
        outline_c = ColorConverter().to_rgba_array(render_params.outline_params.outline_color)
        outline_c[..., -1] = render_params.outline_alpha
        outline_c = outline_c.tolist()
    else:
        outline_c = [None]
    outline_c = outline_c * fill_c.shape[0]

    shapes_df = pd.DataFrame(shapes, copy=True)

    # remove empty points/polygons
    shapes_df = shapes_df[shapes_df["geometry"].apply(lambda geom: not geom.is_empty)]

    # reset index of shapes_df for case of spatial query
    shapes_df = shapes_df.reset_index()

    rows = []

    def assign_fill_and_outline_to_row(
        shapes: list[GeoDataFrame], fill_c: list[Any], outline_c: list[Any], row: pd.Series, idx: int
    ) -> None:
        if len(shapes) > 1 and len(fill_c) == 1:
            row["fill_c"] = fill_c
            row["outline_c"] = outline_c
        else:
            row["fill_c"] = fill_c[idx]
            row["outline_c"] = outline_c[idx]

    # Match colors to the geometry, potentially expanding the row in case of
    # multipolygons
    for idx, row in shapes_df.iterrows():
        geom = row["geometry"]
        if geom.geom_type == "Polygon":
            row = row.to_dict()
            coords = np.array(geom.exterior.coords)
            centroid = np.mean(coords, axis=0)
            scaled_coords = [(centroid + (np.array(coord) - centroid) * s).tolist() for coord in geom.exterior.coords]
            row["geometry"] = mplp.Polygon(scaled_coords, closed=True)
            assign_fill_and_outline_to_row(shapes, fill_c, outline_c, row, idx)
            rows.append(row)

        elif geom.geom_type == "MultiPolygon":
            # mp = _make_patch_from_multipolygon(geom)
            for polygon in geom.geoms:
                mp_copy = row.to_dict()
                coords = np.array(polygon.exterior.coords)
                centroid = np.mean(coords, axis=0)
                scaled_coords = [(centroid + (coord - centroid) * s).tolist() for coord in coords]
                mp_copy["geometry"] = mplp.Polygon(scaled_coords, closed=True)
                assign_fill_and_outline_to_row(shapes, fill_c, outline_c, mp_copy, idx)
                rows.append(mp_copy)

        elif geom.geom_type == "Point":
            row = row.to_dict()
            scaled_radius = row["radius"] * s
            row["geometry"] = mplp.Circle(
                (geom.x, geom.y), radius=scaled_radius
            )  # Circle is always scaled from its center
            assign_fill_and_outline_to_row(shapes, fill_c, outline_c, row, idx)
            rows.append(row)

    patches = pd.DataFrame(rows)

    return PatchCollection(
        patches["geometry"].values.tolist(),
        snap=False,
        lw=render_params.outline_params.linewidth,
        facecolor=patches["fill_c"],
        edgecolor=None if all(outline is None for outline in outline_c) else outline_c,
        **kwargs,
    )


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


def _prepare_cmap_norm(
    cmap: Colormap | str | None = None,
    norm: Normalize | bool = False,
    na_color: str | tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    **kwargs: Any,
) -> CmapParams:
    is_default = cmap is None
    cmap = copy(matplotlib.colormaps[rcParams["image.cmap"] if cmap is None else cmap])
    cmap.set_bad("lightgray" if na_color is None else na_color)

    if isinstance(norm, Normalize) or not norm:
        pass  # TODO
    elif vcenter is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)

    return CmapParams(cmap, norm, na_color, is_default)


def _set_outline(
    outline: bool = False,
    outline_width: float = 1.5,
    outline_color: str | list[float] = "#0000000ff",  # black, white
    **kwargs: Any,
) -> OutlineParams:
    # Type checks for outline_width
    if isinstance(outline_width, int):
        outline_width = outline_width
    if not isinstance(outline_width, float):
        raise TypeError(f"Invalid type of `outline_width`: {type(outline_width)}, expected `float`.")
    if outline_width == 0.0:
        outline = False
    if outline_width < 0.0:
        logging.warning(f"Negative line widths are not allowed, changing {outline_width} to {(-1)*outline_width}")
        outline_width *= -1

    # the default black and white colors can be changed using the contour_config parameter
    if len(outline_color) in {3, 4} and all(isinstance(c, float) for c in outline_color):
        outline_color = matplotlib.colors.to_hex(outline_color)

    if outline:
        kwargs.pop("edgecolor", None)  # remove edge from kwargs if present
        kwargs.pop("alpha", None)  # remove alpha from kwargs if present

    return OutlineParams(outline, outline_color, outline_width)


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
    pmin: float | None = None,
    pmax: float | None = None,
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
    palette: ListedColormap | str | list[str] | None = None,
    alpha: float = 1.0,
    cmap_params: CmapParams | None = None,
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
        if cmap_params is not None and not cmap_params.is_default:
            palette = cmap_params.cmap
        elif len(rcParams["axes.prop_cycle"].by_key()["color"]) >= len_cat:
            cc = rcParams["axes.prop_cycle"]()
            palette = [next(cc)["color"] for _ in range(len_cat)]
        elif len_cat <= 20:
            palette = default_20
        elif len_cat <= 28:
            palette = default_28
        elif len_cat <= len(default_102):  # 103 colors
            palette = default_102
        else:
            palette = ["grey" for _ in range(len_cat)]
            logging.info("input has more than 103 categories. Uniform " "'grey' color will be used for all categories.")

    # otherwise, single channels turn out grey
    color_idx = np.linspace(0, 1, len_cat) if len_cat > 1 else [0.7]

    if isinstance(palette, str):
        palette = [to_hex(palette)]
    elif isinstance(palette, list):
        palette = [to_hex(x) for x in palette]
    elif isinstance(palette, ListedColormap):
        palette = [to_hex(x) for x in palette(color_idx, alpha=alpha)]
    elif isinstance(palette, LinearSegmentedColormap):
        palette = [to_hex(palette(x, alpha=alpha)) for x in color_idx]  # type: ignore[attr-defined]
    else:
        raise TypeError(f"Palette is {type(palette)} but should be string or list.")

    return palette[:len_cat]  # type: ignore[return-value]


def _set_color_source_vec(
    sdata: sd.SpatialData,
    element: SpatialElement | None,
    value_to_plot: str | None,
    element_name: list[str] | str | None = None,
    layer: str | None = None,
    groups: Sequence[str] | str | None = None,
    palette: str | list[str] | None = None,
    na_color: str | tuple[float, ...] | None = None,
    alpha: float = 1.0,
    cmap_params: CmapParams | None = None,
) -> tuple[ArrayLike | pd.Series | None, ArrayLike, bool]:
    if value_to_plot is None:
        color = np.full(len(element), to_hex(na_color))  # type: ignore[arg-type]
        return color, color, False

    # Figure out where to get the color from
    origins = _locate_value(value_key=value_to_plot, sdata=sdata, element_name=element_name)
    if len(origins) > 1:
        raise ValueError(
            f"Color key '{value_to_plot}' for element '{element_name}' been found in multiple locations: {origins}."
        )

    if len(origins) == 1:
        vals = get_values(value_key=value_to_plot, sdata=sdata, element_name=element_name)
        color_source_vector = vals[value_to_plot]

        # if all([isinstance(x, str) for x in color_source_vector]):
        #     raise TypeError(
        #         f"Color key '{value_to_plot}' for element '{element_name}' has string values, "
        #         f"but should be numerical or categorical."
        #     )

        # numerical case, return early
        if not is_categorical_dtype(color_source_vector):
            if palette is not None:
                logging.warning(
                    "Ignoring categorical palette which is given for a continuous variable. "
                    "Consider using `cmap` to pass a ColorMap."
                )
            return None, color_source_vector, False

        color_source_vector = pd.Categorical(color_source_vector)  # convert, e.g., `pd.Series`
        categories = color_source_vector.categories

        if groups is not None:
            color_source_vector = color_source_vector.remove_categories(categories.difference(groups))
            categories = groups

        color_map = dict(zip(categories, _get_colors_for_categorical_obs(categories, palette, cmap_params=cmap_params)))
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

    logging.warning(f"Color key '{value_to_plot}' for element '{element_name}' not been found, using default colors.")
    color = np.full(sdata.table.n_obs, to_hex(na_color))
    return color, color, False


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
        return np.dstack((seg_bound, np.where(val_im > 0, 1, 0)))  # add transparency here

    return np.dstack((seg_im, np.where(val_im > 0, 1, 0)))


def _get_palette(
    categories: Sequence[Any],
    adata: AnnData | None = None,
    cluster_key: None | str = None,
    palette: ListedColormap | str | list[str] | None = None,
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
        cmap = ListedColormap([palette])
    elif isinstance(palette, list):
        cmap = ListedColormap(palette)
    elif isinstance(palette, ListedColormap):
        cmap = palette
    else:
        raise TypeError(f"Palette is {type(palette)} but should be string or list.")
    palette = [to_hex(np.round(x, 5)) for x in cmap(np.linspace(0, 1, len_cat), alpha=alpha)]

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
        if isinstance(palette, str):
            palette = ListedColormap([palette])
        if isinstance(palette, ListedColormap):  # `scanpy` requires it
            palette = cycler(color=palette.colors)
        add_colors_for_categorical_sample_annotation(target, key=key, force_update_colors=True, palette=palette)


def _decorate_axs(
    ax: Axes,
    cax: PatchCollection,
    fig_params: FigParams,
    adata: AnnData,
    value_to_plot: str | None,
    color_source_vector: pd.Series[CategoricalDtype],
    palette: ListedColormap | str | list[str] | None = None,
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
            # order of clusters should agree to palette order
            clusters = color_source_vector.unique()
            clusters = clusters[~clusters.isnull()]
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
    shifts: dict[str, int] = {axis: int(translation.translation[idx]) for idx, axis in enumerate(translation.axes)}
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


def _split_multipolygon_into_outer_and_inner(mp: shapely.MultiPolygon):  # type: ignore
    # https://stackoverflow.com/a/21922058

    for geom in mp.geoms:
        if geom.geom_type == "MultiPolygon":
            exterior_coords = []
            interior_coords = []
            for part in geom:
                epc = _split_multipolygon_into_outer_and_inner(part)  # Recursive call
                exterior_coords += epc["exterior_coords"]
                interior_coords += epc["interior_coords"]
        elif geom.geom_type == "Polygon":
            exterior_coords = geom.exterior.coords[:]
            interior_coords = []
            for interior in geom.interiors:
                interior_coords += interior.coords[:]
        else:
            raise ValueError(f"Unhandled geometry type: {repr(geom.type)}")

    return interior_coords, exterior_coords


def _make_patch_from_multipolygon(mp: shapely.MultiPolygon) -> mpatches.PathPatch:
    # https://matplotlib.org/stable/gallery/shapes_and_collections/donut.html

    patches = []
    for geom in mp.geoms:
        if len(geom.interiors) == 0:
            # polygon has no holes
            patches += [mpatches.Polygon(geom.exterior.coords, closed=True)]
        else:
            inside, outside = _split_multipolygon_into_outer_and_inner(mp)
            if len(inside) > 0:
                codes = np.ones(len(inside), dtype=mpath.Path.code_type) * mpath.Path.LINETO
                codes[0] = mpath.Path.MOVETO
                all_codes = np.concatenate((codes, codes))
                vertices = np.concatenate((outside, inside[::-1]))
            else:
                all_codes = []
                vertices = np.concatenate(outside)
            patches += [mpatches.PathPatch(mpath.Path(vertices, all_codes))]

    return patches


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
