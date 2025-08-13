from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from copy import copy
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import dask
import datashader as ds
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import pandas as pd
import shapely
import spatialdata as sd
import xarray as xr
from anndata import AnnData
from cycler import Cycler, cycler
from datashader.core import Canvas
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
    to_rgba,
)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import CompositeGenericTransform
from matplotlib_scalebar.scalebar import ScaleBar
from numpy.ma.core import MaskedArray
from numpy.random import default_rng
from pandas.api.types import CategoricalDtype
from pandas.core.arrays.categorical import Categorical
from scanpy import settings
from scanpy.plotting._tools.scatterplots import _add_categorical_legend
from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation
from scanpy.plotting.palettes import default_20, default_28, default_102
from skimage.color import label2rgb
from skimage.morphology import erosion, square
from skimage.segmentation import find_boundaries
from skimage.util import map_array
from spatialdata import (
    SpatialData,
    get_element_annotators,
    get_extent,
    get_values,
    rasterize,
)
from spatialdata._core.query.relational_query import _locate_value
from spatialdata._types import ArrayLike
from spatialdata.models import Image2DModel, Labels2DModel, SpatialElement

# from spatialdata.transformations.transformations import Scale
from spatialdata.transformations import Affine, Identity, MapAxis, Scale, Translation
from spatialdata.transformations import Sequence as SDSequence
from spatialdata.transformations.operations import get_transformation
from xarray import DataArray, DataTree

from spatialdata_plot._logging import logger
from spatialdata_plot.pl.render_params import (
    CmapParams,
    FigParams,
    ImageRenderParams,
    LabelsRenderParams,
    OutlineParams,
    PointsRenderParams,
    ScalebarParams,
    ShapesRenderParams,
    _FontSize,
    _FontWeight,
)

to_hex = partial(colors.to_hex, keep_alpha=True)

# replace with
# from spatialdata._types import ColorLike
# once https://github.com/scverse/spatialdata/pull/689/ is in a release
ColorLike = tuple[float, ...] | str


def _verify_plotting_tree(sdata: SpatialData) -> SpatialData:
    """Verify that the plotting tree exists, and if not, create it."""
    if not hasattr(sdata, "plotting_tree"):
        sdata.plotting_tree = OrderedDict()

    return sdata


def _get_coordinate_system_mapping(sdata: SpatialData) -> dict[str, list[str]]:
    coordsys_keys = sdata.coordinate_systems
    image_keys = [] if sdata.images is None else sdata.images.keys()
    label_keys = [] if sdata.labels is None else sdata.labels.keys()
    shape_keys = [] if sdata.shapes is None else sdata.shapes.keys()
    point_keys = [] if sdata.points is None else sdata.points.keys()

    mapping: dict[str, list[str]] = {}

    if len(coordsys_keys) < 1:
        raise ValueError("SpatialData object must have at least one coordinate system to generate a mapping.")

    for key in coordsys_keys:
        mapping[key] = []

        for image_key in image_keys:
            transformations = get_transformation(sdata.images[image_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(image_key)

        for label_key in label_keys:
            transformations = get_transformation(sdata.labels[label_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(label_key)

        for shape_key in shape_keys:
            transformations = get_transformation(sdata.shapes[shape_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(shape_key)

        for point_key in point_keys:
            transformations = get_transformation(sdata.points[point_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(point_key)

    return mapping


def _is_color_like(color: Any) -> bool:
    """Check if a value is a valid color, returns False for pseudo-bools.

    For discussion, see: https://github.com/scverse/spatialdata-plot/issues/327.
    matplotlib accepts strings in [0, 1] as grey-scale values - therefore,
    "0" and "1" are considered valid colors. However, we won't do that
    so we're filtering these out.
    """
    if isinstance(color, bool):
        return False
    if isinstance(color, str):
        try:
            num_value = float(color)
            if 0 <= num_value <= 1:
                return False
        except ValueError:
            # we're not dealing with what matplotlib considers greyscale
            pass

    return bool(colors.is_color_like(color))


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
            num_panels=num_panels,
            hspace=hspace,
            wspace=wspace,
            ncols=ncols,
            dpi=dpi,
            figsize=figsize,
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
        elif isinstance(ax, Axes):
            # needed for rasterization if user provides Axes object
            fig = ax.get_figure()
            fig.set_dpi(dpi)

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


def _sanitise_na_color(na_color: ColorLike | None) -> tuple[str, bool]:
    """Return the color's hex value and a boolean indicating if the user changed the default color.

    Returns the hex representation of the color and a boolean indicating whether the
    color was changed by the user or not. Our default is "lightgray", but when we
    render labels, we give them random colors instead. However, the user could've
    manually specified "lightgray" as the color, so we need to check for that.

    Parameters
    ----------
        na_color (ColorLike | None): The color input specified by the user.

    Returns
    -------
        tuple[str, bool]: A tuple containing the hex color code and a boolean
        indicating if the color was user-specified.
    """
    if na_color == "default":
        # user kept the default
        return to_hex("lightgray"), False
    if na_color is None:
        # user wants to hide NAs
        return "#FFFFFF00", True  # zero alpha so it's hidden
    if colors.is_color_like(na_color):
        # user specified a color (including "lightgray")
        return to_hex(na_color), True

    # Handle unexpected values (optional)
    raise ValueError(f"Invalid na_color value: {na_color}")


def _get_centroid_of_pathpatch(pathpatch: mpatches.PathPatch) -> tuple[float, float]:
    # Extract the vertices from the PathPatch
    path = pathpatch.get_path()
    vertices = path.vertices
    x = vertices[:, 0]
    y = vertices[:, 1]

    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    # Calculate the centroid coordinates
    centroid_x = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * area)
    centroid_y = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * area)

    return centroid_x, centroid_y


def _scale_pathpatch_around_centroid(pathpatch: mpatches.PathPatch, scale_factor: float) -> None:
    centroid = _get_centroid_of_pathpatch(pathpatch)
    vertices = pathpatch.get_path().vertices
    scaled_vertices = np.array([centroid + (vertex - centroid) * scale_factor for vertex in vertices])
    pathpatch.get_path().vertices = scaled_vertices


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
        if len(c.shape) == 1 and c.shape[0] in [3, 4] and c.shape[0] == len(shapes) and c.dtype == float:
            if norm is None:
                c = cmap(c)
            else:
                try:
                    norm = colors.Normalize(vmin=min(c), vmax=max(c)) if norm is None else norm
                except ValueError as e:
                    raise ValueError(
                        "Could not convert values in the `color` column to float, if `color` column represents"
                        " categories, set the column to categorical dtype."
                    ) from e
                c = cmap(norm(c))
        else:
            fill_c = ColorConverter().to_rgba_array(c)
    except ValueError:
        if norm is None:
            c = cmap(c)
        else:
            try:
                norm = colors.Normalize(vmin=min(c), vmax=max(c)) if norm is None else norm
            except ValueError as e:
                raise ValueError(
                    "Could not convert values in the `color` column to float, if `color` column represents"
                    " categories, set the column to categorical dtype."
                ) from e
            c = cmap(norm(c))

    fill_c = ColorConverter().to_rgba_array(c)
    fill_c[..., -1] *= render_params.fill_alpha

    if render_params.outline_params.outline:
        outline_c = ColorConverter().to_rgba_array(render_params.outline_params.outline_color)
        outline_c[..., -1] = render_params.outline_alpha
        outline_c = outline_c.tolist()
    else:
        outline_c = [None]
    outline_c = outline_c * fill_c.shape[0]

    shapes_df = pd.DataFrame(shapes, copy=True)
    shapes_df = shapes_df[shapes_df["geometry"].apply(lambda geom: not geom.is_empty)]
    shapes_df = shapes_df.reset_index(drop=True)

    def _assign_fill_and_outline_to_row(
        fill_c: list[Any],
        outline_c: list[Any],
        row: dict[str, Any],
        idx: int,
        is_multiple_shapes: bool,
    ) -> None:
        try:
            if is_multiple_shapes and len(fill_c) == 1:
                row["fill_c"] = fill_c[0]
                row["outline_c"] = outline_c[0]
            else:
                row["fill_c"] = fill_c[idx]
                row["outline_c"] = outline_c[idx]
        except IndexError as e:
            raise IndexError("Could not assign fill and outline colors due to a mismatch in row numbers.") from e

    def _process_polygon(row: pd.Series, s: float) -> dict[str, Any]:
        coords = np.array(row["geometry"].exterior.coords)
        centroid = np.mean(coords, axis=0)
        scaled_coords = (centroid + (coords - centroid) * s).tolist()
        return {
            **row.to_dict(),
            "geometry": mpatches.Polygon(scaled_coords, closed=True),
        }

    def _process_multipolygon(row: pd.Series, s: float) -> list[dict[str, Any]]:
        mp = _make_patch_from_multipolygon(row["geometry"])
        row_dict = row.to_dict()
        for m in mp:
            _scale_pathpatch_around_centroid(m, s)

        return [{**row_dict, "geometry": m} for m in mp]

    def _process_point(row: pd.Series, s: float) -> dict[str, Any]:
        return {
            **row.to_dict(),
            "geometry": mpatches.Circle((row["geometry"].x, row["geometry"].y), radius=row["radius"] * s),
        }

    def _create_patches(shapes_df: GeoDataFrame, fill_c: list[Any], outline_c: list[Any], s: float) -> pd.DataFrame:
        rows = []
        is_multiple_shapes = len(shapes_df) > 1

        for idx, row in shapes_df.iterrows():
            geom_type = row["geometry"].geom_type
            processed_rows = []

            if geom_type == "Polygon":
                processed_rows.append(_process_polygon(row, s))
            elif geom_type == "MultiPolygon":
                processed_rows.extend(_process_multipolygon(row, s))
            elif geom_type == "Point":
                processed_rows.append(_process_point(row, s))

            for processed_row in processed_rows:
                _assign_fill_and_outline_to_row(fill_c, outline_c, processed_row, idx, is_multiple_shapes)
                rows.append(processed_row)

        return pd.DataFrame(rows)

    patches = _create_patches(shapes_df, fill_c, outline_c, s)
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
    norm: Normalize | None = None,
    na_color: ColorLike | None = None,
) -> CmapParams:
    # TODO: check refactoring norm out here as it gets overwritten later
    cmap_is_default = cmap is None
    if cmap is None:
        cmap = rcParams["image.cmap"]
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

    cmap = copy(cmap)

    assert isinstance(cmap, Colormap), f"Invalid type of `cmap`: {type(cmap)}, expected `Colormap`."

    if norm is None:
        norm = Normalize(vmin=None, vmax=None, clip=False)

    na_color, na_color_modified_by_user = _sanitise_na_color(na_color)
    cmap.set_bad(na_color)

    return CmapParams(
        cmap=cmap,
        norm=norm,
        na_color=na_color,
        cmap_is_default=cmap_is_default,
        na_color_modified_by_user=na_color_modified_by_user,
    )


def _set_outline(
    outline: bool = False,
    outline_width: float = 1.5,
    outline_color: str | list[float] = "#0000000ff",  # black, white
    **kwargs: Any,
) -> OutlineParams:
    if not isinstance(outline_width, int | float):
        raise TypeError(f"Invalid type of `outline_width`: {type(outline_width)}, expected `int` or `float`.")
    if outline_width == 0.0:
        outline = False
    if outline_width < 0.0:
        logger.warning(f"Negative line widths are not allowed, changing {outline_width} to {(-1) * outline_width}")
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

    # Ensure perc is an array of two elements
    if np.isscalar(perc):
        logger.warning(
            "Percentile range is too small, using the same percentile for both min "
            "and max. Consider using a larger percentile range."
        )
        perc = np.array([perc, perc])

    norm = (img - perc[0]) / (perc[1] - perc[0] + eps)  # type: ignore

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
        if cmap_params is not None and not cmap_params.cmap_is_default:
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
            logger.info("input has more than 103 categories. Uniform 'grey' color will be used for all categories.")
    else:
        # raise error when user didn't provide the right number of colors in palette
        if isinstance(palette, list) and len(palette) != len(categories):
            raise ValueError(
                f"The number of provided values in the palette ({len(palette)}) doesn't agree with the number of "
                f"categories that should be colored ({categories})."
            )

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
    na_color: ColorLike,
    element_name: list[str] | str | None = None,
    groups: list[str] | str | None = None,
    palette: list[str] | str | None = None,
    cmap_params: CmapParams | None = None,
    alpha: float = 1.0,
    table_name: str | None = None,
    table_layer: str | None = None,
    render_type: Literal["points"] | None = None,
) -> tuple[ArrayLike | pd.Series | None, ArrayLike, bool]:
    if value_to_plot is None and element is not None:
        color = np.full(len(element), na_color)
        return color, color, False

    # Figure out where to get the color from
    origins = _locate_value(
        value_key=value_to_plot,
        sdata=sdata,
        element_name=element_name,
        table_name=table_name,
    )

    if len(origins) > 1:
        raise ValueError(
            f"Color key '{value_to_plot}' for element '{element_name}' been found in multiple locations: {origins}."
        )

    if len(origins) == 1:
        color_source_vector = get_values(
            value_key=value_to_plot,
            sdata=sdata,
            element_name=element_name,
            table_name=table_name,
            table_layer=table_layer,
        )[value_to_plot]

        # numerical case, return early
        # TODO temporary split until refactor is complete
        if color_source_vector is not None and not isinstance(color_source_vector.dtype, pd.CategoricalDtype):
            if (
                not isinstance(element, GeoDataFrame)
                and isinstance(palette, list)
                and palette[0] is not None
                or isinstance(element, GeoDataFrame)
                and isinstance(palette, list)
            ):
                logger.warning(
                    "Ignoring categorical palette which is given for a continuous variable. "
                    "Consider using `cmap` to pass a ColorMap."
                )
            return None, color_source_vector, False

        color_source_vector = pd.Categorical(color_source_vector)  # convert, e.g., `pd.Series`

        color_mapping = _get_categorical_color_mapping(
            adata=sdata.table,
            cluster_key=value_to_plot,
            color_source_vector=color_source_vector,
            cmap_params=cmap_params,
            alpha=alpha,
            groups=groups,
            palette=palette,
            na_color=na_color,
            render_type=render_type,
        )

        color_source_vector = color_source_vector.set_categories(color_mapping.keys())
        if color_mapping is None:
            raise ValueError("Unable to create color palette.")

        # do not rename categories, as colors need not be unique
        color_vector = color_source_vector.map(color_mapping)

        return color_source_vector, color_vector, True

    logger.warning(f"Color key '{value_to_plot}' for element '{element_name}' not been found, using default colors.")
    color = np.full(sdata[table_name].n_obs, to_hex(na_color))
    return color, color, False


def _map_color_seg(
    seg: ArrayLike,
    cell_id: ArrayLike,
    color_vector: ArrayLike | pd.Series[CategoricalDtype],
    color_source_vector: pd.Series[CategoricalDtype],
    cmap_params: CmapParams,
    na_color: ColorLike,
    na_color_modified_by_user: bool = False,
    seg_erosionpx: int | None = None,
    seg_boundaries: bool = False,
) -> ArrayLike:
    cell_id = np.array(cell_id)

    if pd.api.types.is_categorical_dtype(color_vector.dtype):
        # Case A: users wants to plot a categorical column
        if np.any(color_source_vector.isna()):
            cell_id[color_source_vector.isna()] = 0
        val_im: ArrayLike = map_array(seg.copy(), cell_id, color_vector.codes + 1)
        cols = colors.to_rgba_array(color_vector.categories)
    elif pd.api.types.is_numeric_dtype(color_vector.dtype):
        # Case B: user wants to plot a continous column
        if isinstance(color_vector, pd.Series):
            color_vector = color_vector.to_numpy()
        cols = cmap_params.cmap(cmap_params.norm(color_vector))
        val_im = map_array(seg.copy(), cell_id, cell_id)
    else:
        # Case C: User didn't specify any colors
        if color_source_vector is not None and (
            set(color_vector) == set(color_source_vector)
            and len(set(color_vector)) == 1
            and set(color_vector) == {na_color}
            and not na_color_modified_by_user
        ):
            val_im = map_array(seg.copy(), cell_id, cell_id)
            RNG = default_rng(42)
            cols = RNG.random((len(color_vector), 3))
        else:
            # Case D: User didn't specify a column to color by, but modified the na_color
            val_im = map_array(seg.copy(), cell_id, cell_id)
            if "#" in str(color_vector[0]):
                # we have hex colors
                assert all(_is_color_like(c) for c in color_vector), "Not all values are color-like."
                cols = colors.to_rgba_array(color_vector)
            else:
                cols = cmap_params.cmap(cmap_params.norm(color_vector))

    if seg_erosionpx is not None:
        val_im[val_im == erosion(val_im, square(seg_erosionpx))] = 0

    seg_im: ArrayLike = label2rgb(
        label=val_im,
        colors=cols,
        bg_label=0,
        bg_color=(1, 1, 1),  # transparency doesn't really work
        image_alpha=0,
    )

    if seg_boundaries:
        if seg.shape[0] == 1:
            seg = np.squeeze(seg, axis=0)
        seg_bound: ArrayLike = np.clip(seg_im - find_boundaries(seg)[:, :, None], 0, 1)
        return np.dstack((seg_bound, np.where(val_im > 0, 1, 0)))  # add transparency here

    if len(val_im.shape) != len(seg_im.shape):
        val_im = np.expand_dims((val_im > 0).astype(int), axis=-1)
    return np.dstack((seg_im, val_im))


def _generate_base_categorial_color_mapping(
    adata: AnnData,
    cluster_key: str,
    color_source_vector: ArrayLike | pd.Series[CategoricalDtype],
    na_color: ColorLike,
    cmap_params: CmapParams | None = None,
) -> Mapping[str, str]:
    if adata is not None and cluster_key in adata.uns and f"{cluster_key}_colors" in adata.uns:
        colors = adata.uns[f"{cluster_key}_colors"]
        categories = color_source_vector.categories.tolist() + ["NaN"]
        if "#" not in na_color:
            # should be unreachable, but just for safety
            raise ValueError("Expected `na_color` to be a hex color, but got a non-hex color.")

        colors = [to_hex(to_rgba(color)[:3]) for color in colors]
        na_color = to_hex(to_rgba(na_color)[:3])

        if na_color and len(categories) > len(colors):
            return dict(zip(categories, colors + [na_color], strict=True))

        return dict(zip(categories, colors, strict=True))

    return _get_default_categorial_color_mapping(color_source_vector=color_source_vector, cmap_params=cmap_params)


def _modify_categorical_color_mapping(
    mapping: Mapping[str, str],
    groups: list[str] | str | None = None,
    palette: list[str] | str | None = None,
) -> Mapping[str, str]:
    if groups is None or isinstance(groups, list) and groups[0] is None:
        return mapping

    if palette is None or isinstance(palette, list) and palette[0] is None:
        # subset base mapping to only those specified in groups
        modified_mapping = {key: mapping[key] for key in mapping if key in groups or key == "NaN"}
    elif len(palette) == len(groups) and isinstance(groups, list) and isinstance(palette, list):
        modified_mapping = dict(zip(groups, palette, strict=True))
    else:
        raise ValueError(f"Expected palette to be of length `{len(groups)}`, found `{len(palette)}`.")

    return modified_mapping


def _get_default_categorial_color_mapping(
    color_source_vector: ArrayLike | pd.Series[CategoricalDtype],
    cmap_params: CmapParams | None = None,
) -> Mapping[str, str]:
    len_cat = len(color_source_vector.categories.unique())
    # Try to use provided colormap first
    if cmap_params is not None and cmap_params.cmap is not None and not cmap_params.cmap_is_default:
        # Generate evenly spaced indices for the colormap
        color_idx = np.linspace(0, 1, len_cat)
        if isinstance(cmap_params.cmap, ListedColormap):
            palette = [to_hex(x) for x in cmap_params.cmap(color_idx)]
        elif isinstance(cmap_params.cmap, LinearSegmentedColormap):
            palette = [to_hex(cmap_params.cmap(x)) for x in color_idx]
        else:
            # Fall back to default palettes if cmap is not of expected type
            palette = None
    else:
        palette = None

    # Fall back to default palettes if needed
    if palette is None:
        if len_cat <= 20:
            palette = default_20
        elif len_cat <= 28:
            palette = default_28
        elif len_cat <= len(default_102):  # 103 colors
            palette = default_102
        else:
            palette = ["grey"] * len_cat
            logger.info("input has more than 103 categories. Uniform 'grey' color will be used for all categories.")

    return dict(zip(color_source_vector.categories, palette[:len_cat], strict=True))


def _get_categorical_color_mapping(
    adata: AnnData,
    na_color: ColorLike,
    cluster_key: str | None = None,
    color_source_vector: ArrayLike | pd.Series[CategoricalDtype] | None = None,
    cmap_params: CmapParams | None = None,
    alpha: float = 1,
    groups: list[str] | str | None = None,
    palette: list[str] | str | None = None,
    render_type: Literal["points"] | None = None,
) -> Mapping[str, str]:
    if not isinstance(color_source_vector, Categorical):
        raise TypeError(f"Expected `categories` to be a `Categorical`, but got {type(color_source_vector).__name__}")

    if isinstance(groups, str):
        groups = [groups]

    if not palette and render_type == "points" and cmap_params is not None and not cmap_params.cmap_is_default:
        palette = cmap_params.cmap

        color_idx = color_idx = np.linspace(0, 1, len(color_source_vector.categories))
        if isinstance(palette, ListedColormap):
            palette = [to_hex(x) for x in palette(color_idx, alpha=alpha)]
        elif isinstance(palette, LinearSegmentedColormap):
            palette = [to_hex(palette(x, alpha=alpha)) for x in color_idx]  # type: ignore[attr-defined]
        return dict(zip(color_source_vector.categories, palette, strict=True))

    if isinstance(palette, str):
        palette = [palette]

    if cluster_key is None:
        # user didn't specify a column to use for coloring
        base_mapping = _get_default_categorial_color_mapping(
            color_source_vector=color_source_vector, cmap_params=cmap_params
        )
    else:
        base_mapping = _generate_base_categorial_color_mapping(
            adata=adata,
            cluster_key=cluster_key,
            color_source_vector=color_source_vector,
            na_color=na_color,
            cmap_params=cmap_params,
        )

    return _modify_categorical_color_mapping(mapping=base_mapping, groups=groups, palette=palette)


def _maybe_set_colors(
    source: AnnData,
    target: AnnData,
    key: str,
    palette: str | ListedColormap | Cycler | Sequence[Any] | None = None,
) -> None:
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
        palette = None
        add_colors_for_categorical_sample_annotation(target, key=key, force_update_colors=True, palette=palette)


def _decorate_axs(
    ax: Axes,
    cax: PatchCollection,
    fig_params: FigParams,
    value_to_plot: str | None,
    color_source_vector: pd.Series[CategoricalDtype] | Categorical,
    color_vector: pd.Series[CategoricalDtype] | Categorical,
    adata: AnnData | None = None,
    palette: ListedColormap | str | list[str] | None = None,
    alpha: float = 1.0,
    na_color: ColorLike | None = "#d3d3d3",  # lightgray
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
        if color_source_vector is not None and isinstance(color_source_vector.dtype, pd.CategoricalDtype):
            # order of clusters should agree to palette order
            clusters = color_source_vector.remove_unused_categories().unique()
            clusters = clusters[~clusters.isnull()]
            # derive mapping from color_source_vector and color_vector
            group_to_color_matching = pd.DataFrame(
                {
                    "cats": color_source_vector.remove_unused_categories(),
                    "color": color_vector,
                }
            )
            color_mapping = group_to_color_matching.drop_duplicates("cats").set_index("cats")["color"].to_dict()
            _add_categorical_legend(
                ax,
                pd.Categorical(values=color_source_vector, categories=clusters),
                palette=color_mapping,
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
            cb = plt.colorbar(cax, ax=ax, pad=0.01, fraction=0.08, aspect=30)
            cb.solids.set_alpha(alpha)

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


def save_fig(
    fig: Figure,
    path: str | Path,
    make_dir: bool = True,
    ext: str = "png",
    **kwargs: Any,
) -> None:
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
            logger.debug(f"Unable to create directory `{path.parent}`. Reason: `{e}`")

    logger.debug(f"Saving figure to `{path!r}`")

    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("transparent", True)

    fig.savefig(path, **kwargs)


def _get_linear_colormap(colors: list[str], background: str) -> list[LinearSegmentedColormap]:
    return [LinearSegmentedColormap.from_list(c, [background, c], N=256) for c in colors]


def _get_listed_colormap(color_dict: dict[str, str]) -> ListedColormap:
    sorted_labels = sorted(color_dict.keys())
    colors = [color_dict[k] for k in sorted_labels]

    return ListedColormap(["black"] + colors, N=len(colors) + 1)


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


def _get_valid_cs(
    sdata: sd.SpatialData,
    coordinate_systems: list[str],
    render_images: bool,
    render_labels: bool,
    render_points: bool,
    render_shapes: bool,
    elements: list[str],
) -> list[str]:
    """Get names of the valid coordinate systems.

    Valid cs are cs that contain elements to be rendered:
    1. In case the user specified elements:
        all cs that contain at least one of those elements
    2. Else:
        all cs that contain at least one element that should
        be rendered (depending on whether images/points/labels/...
        should be rendered)
    """
    cs_mapping = _get_coordinate_system_mapping(sdata)
    valid_cs = []
    for cs in coordinate_systems:
        if (
            elements
            and any(e in elements for e in cs_mapping[cs])
            or not elements
            and (
                (len(sdata.images.keys()) > 0 and render_images)
                or (len(sdata.labels.keys()) > 0 and render_labels)
                or (len(sdata.points.keys()) > 0 and render_points)
                or (len(sdata.shapes.keys()) > 0 and render_shapes)
            )
        ):  # not nice, but ruff wants it (SIM114)
            valid_cs.append(cs)
        else:
            logger.info(f"Dropping coordinate system '{cs}' since it doesn't have relevant elements.")
    return valid_cs


def _rasterize_if_necessary(
    image: DataArray,
    dpi: float,
    width: float,
    height: float,
    coordinate_system: str,
    extent: dict[str, tuple[float, float]],
) -> DataArray:
    """Ensure fast rendering by adapting the resolution if necessary.

    A DataArray is prepared for plotting. To improve performance, large images are rasterized.

    Parameters
    ----------
    image
        Input spatial image that should be rendered
    dpi
        Resolution of the figure
    width
        Width (in inches) of the figure
    height
        Height (in inches) of the figure
    coordinate_system
        name of the coordinate system the image belongs to
    extent
        extent of the (full size) image. Must be a dict containing a tuple with min and
        max extent for the keys "x" and "y".

    Returns
    -------
    DataArray
        Spatial image ready for rendering
    """
    has_c_dim = len(image.shape) == 3
    if has_c_dim:
        y_dims = image.shape[1]
        x_dims = image.shape[2]
    else:
        y_dims = image.shape[0]
        x_dims = image.shape[1]

    target_y_dims = dpi * height
    target_x_dims = dpi * width

    # TODO: when exactly do we want to rasterize?
    do_rasterization = y_dims > target_y_dims + 100 or x_dims > target_x_dims + 100
    if x_dims < 2000 and y_dims < 2000:
        do_rasterization = False

    if do_rasterization:
        logger.info("Rasterizing image for faster rendering.")
        # TODO: do we want min here?
        target_unit_to_pixels = min(target_y_dims / y_dims, target_x_dims / x_dims)
        image = rasterize(
            image,
            ("y", "x"),
            [extent["y"][0], extent["x"][0]],
            [extent["y"][1], extent["x"][1]],
            coordinate_system,
            target_unit_to_pixels=target_unit_to_pixels,
        )

    return image


def _multiscale_to_spatial_image(
    multiscale_image: DataTree,
    dpi: float,
    width: float,
    height: float,
    scale: str | None = None,
    is_label: bool = False,
) -> DataArray:
    """Extract the DataArray to be rendered from a multiscale image.

    From the `DataTree`, the scale that fits the given image size and dpi most is selected
    and returned. In case the lowest resolution is still too high, a rasterization step is added.

    Parameters
    ----------
    multiscale_image
        `DataTree` that should be rendered
    dpi
        dpi of the target image
    width
        width of the target image in inches
    height
        height of the target image in inches
    scale
        specific scale that the user chose, if None the heuristic is used
    is_label
        When True, the multiscale image contains labels which don't contain the `c` dimension

    Returns
    -------
    DataArray
        To be rendered, extracted from the DataTree respecting the dpi and size of the target image.
    """
    scales = [leaf.name for leaf in multiscale_image.leaves]
    x_dims = [multiscale_image[scale].dims["x"] for scale in scales]
    y_dims = [multiscale_image[scale].dims["y"] for scale in scales]

    if isinstance(scale, str):
        if scale not in scales and scale != "full":
            raise ValueError(f'Scale {scale} does not exist. Please select one of {scales} or set scale = "full"!')
        optimal_scale = scale
        if scale == "full":
            # use scale with highest resolution
            optimal_scale = scales[np.argmax(x_dims)]
    else:
        # ensure that lists are sorted
        order = np.argsort(x_dims)
        scales = [scales[i] for i in order]
        x_dims = [x_dims[i] for i in order]
        y_dims = [y_dims[i] for i in order]

        optimal_x = width * dpi
        optimal_y = height * dpi

        # get scale where the dimensions are close to the optimal values
        # when possible, pick higher resolution (worst case: downscaled afterwards)
        optimal_index_y = np.searchsorted(y_dims, optimal_y)
        if optimal_index_y == len(y_dims):
            optimal_index_y -= 1
        optimal_index_x = np.searchsorted(x_dims, optimal_x)
        if optimal_index_x == len(x_dims):
            optimal_index_x -= 1

        # pick the scale with higher resolution (worst case: downscaled afterwards)
        optimal_scale = scales[min(int(optimal_index_x), int(optimal_index_y))]

    # NOTE: problematic if there are cases with > 1 data variable
    data_var_keys = list(multiscale_image[optimal_scale].data_vars)
    image = multiscale_image[optimal_scale][data_var_keys[0]]

    return Labels2DModel.parse(image) if is_label else Image2DModel.parse(image, c_coords=image.coords["c"].values)


def _get_elements_to_be_rendered(
    render_cmds: list[
        tuple[
            str,
            ImageRenderParams | LabelsRenderParams | PointsRenderParams | ShapesRenderParams,
        ]
    ],
    cs_contents: pd.DataFrame,
    cs: str,
) -> list[str]:
    """
    Get the names of the elements to be rendered in the plot.

    Parameters
    ----------
    render_cmds
        List of tuples containing the commands and their respective parameters.
    cs_contents
        The dataframe indicating for each coordinate system which SpatialElements it contains.
    cs
        The name of the coordinate system to query cs_contents for.

    Returns
    -------
    List of names of the SpatialElements to be rendered in the plot.
    """
    elements_to_be_rendered: list[str] = []
    render_cmds_map = {
        "render_images": "has_images",
        "render_shapes": "has_shapes",
        "render_points": "has_points",
        "render_labels": "has_labels",
    }

    cs_query = cs_contents.query(f"cs == '{cs}'")

    for cmd, params in render_cmds:
        key = render_cmds_map.get(cmd)
        if key and cs_query[key][0]:
            elements_to_be_rendered += [params.element]

    return elements_to_be_rendered


def _validate_show_parameters(
    coordinate_systems: list[str] | str | None,
    legend_fontsize: int | float | _FontSize | None,
    legend_fontweight: int | _FontWeight,
    legend_loc: str | None,
    legend_fontoutline: int | None,
    na_in_legend: bool,
    colorbar: bool,
    wspace: float | None,
    hspace: float,
    ncols: int,
    frameon: bool | None,
    figsize: tuple[float, float] | None,
    dpi: int | None,
    fig: Figure | None,
    title: list[str] | str | None,
    share_extent: bool,
    pad_extent: int | float,
    ax: list[Axes] | Axes | None,
    return_ax: bool,
    save: str | Path | None,
) -> None:
    if coordinate_systems is not None and not isinstance(coordinate_systems, list | str):
        raise TypeError("Parameter 'coordinate_systems' must be a string or a list of strings.")

    font_weights = ["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
    if legend_fontweight is not None and (
        not isinstance(legend_fontweight, int | str)
        or (isinstance(legend_fontweight, str) and legend_fontweight not in font_weights)
    ):
        readable_font_weights = ", ".join(font_weights[:-1]) + ", or " + font_weights[-1]
        raise TypeError(
            "Parameter 'legend_fontweight' must be an integer or one of",
            f"the following strings: {readable_font_weights}.",
        )

    font_sizes = [
        "xx-small",
        "x-small",
        "small",
        "medium",
        "large",
        "x-large",
        "xx-large",
    ]

    if legend_fontsize is not None and (
        not isinstance(legend_fontsize, int | float | str)
        or (isinstance(legend_fontsize, str) and legend_fontsize not in font_sizes)
    ):
        readable_font_sizes = ", ".join(font_sizes[:-1]) + ", or " + font_sizes[-1]
        raise TypeError(
            "Parameter 'legend_fontsize' must be an integer, a float, or ",
            f"one of the following strings: {readable_font_sizes}.",
        )

    if legend_loc is not None and not isinstance(legend_loc, str):
        raise TypeError("Parameter 'legend_loc' must be a string.")

    if legend_fontoutline is not None and not isinstance(legend_fontoutline, int):
        raise TypeError("Parameter 'legend_fontoutline' must be an integer.")

    if not isinstance(na_in_legend, bool):
        raise TypeError("Parameter 'na_in_legend' must be a boolean.")

    if not isinstance(colorbar, bool):
        raise TypeError("Parameter 'colorbar' must be a boolean.")

    if wspace is not None and not isinstance(wspace, float):
        raise TypeError("Parameter 'wspace' must be a float.")

    if not isinstance(hspace, float):
        raise TypeError("Parameter 'hspace' must be a float.")

    if not isinstance(ncols, int):
        raise TypeError("Parameter 'ncols' must be an integer.")

    if frameon is not None and not isinstance(frameon, bool):
        raise TypeError("Parameter 'frameon' must be a boolean.")

    if figsize is not None and not isinstance(figsize, tuple):
        raise TypeError("Parameter 'figsize' must be a tuple of two floats.")

    if dpi is not None and not isinstance(dpi, int):
        raise TypeError("Parameter 'dpi' must be an integer.")

    if fig is not None and not isinstance(fig, Figure):
        raise TypeError("Parameter 'fig' must be a matplotlib.figure.Figure.")

    if title is not None and not isinstance(title, list | str):
        raise TypeError("Parameter 'title' must be a string or a list of strings.")

    if not isinstance(share_extent, bool):
        raise TypeError("Parameter 'share_extent' must be a boolean.")

    if not isinstance(pad_extent, int | float):
        raise TypeError("Parameter 'pad_extent' must be numeric.")

    if ax is not None and not isinstance(ax, Axes | list):
        raise TypeError("Parameter 'ax' must be a matplotlib.axes.Axes or a list of Axes.")

    if not isinstance(return_ax, bool):
        raise TypeError("Parameter 'return_ax' must be a boolean.")

    if save is not None and not isinstance(save, str | Path):
        raise TypeError("Parameter 'save' must be a string or a pathlib.Path.")


def _type_check_params(param_dict: dict[str, Any], element_type: str) -> dict[str, Any]:
    if (element := param_dict.get("element")) is not None and not isinstance(element, str):
        raise ValueError(
            "Parameter 'element' must be a string. If you want to display more elements, pass `element` "
            "as `None` or chain pl.render(...).pl.render(...).pl.show()"
        )
    if element_type == "images":
        param_dict["element"] = [element] if element is not None else list(param_dict["sdata"].images.keys())
    elif element_type == "labels":
        param_dict["element"] = [element] if element is not None else list(param_dict["sdata"].labels.keys())
    elif element_type == "points":
        param_dict["element"] = [element] if element is not None else list(param_dict["sdata"].points.keys())
    elif element_type == "shapes":
        param_dict["element"] = [element] if element is not None else list(param_dict["sdata"].shapes.keys())

    if (channel := param_dict.get("channel")) is not None and not isinstance(channel, list | str | int):
        raise TypeError("Parameter 'channel' must be a string, an integer, or a list of strings or integers.")
    if isinstance(channel, list):
        if not all(isinstance(c, str | int) for c in channel):
            raise TypeError("Each item in 'channel' list must be a string or an integer.")
        if not all(isinstance(c, type(channel[0])) for c in channel):
            raise TypeError("Each item in 'channel' list must be of the same type, either string or integer.")

    elif "channel" in param_dict:
        param_dict["channel"] = [channel] if channel is not None else None

    if (contour_px := param_dict.get("contour_px")) and not isinstance(contour_px, int):
        raise TypeError("Parameter 'contour_px' must be an integer.")

    if (color := param_dict.get("color")) and element_type in {
        "shapes",
        "points",
        "labels",
    }:
        if not isinstance(color, str):
            raise TypeError("Parameter 'color' must be a string.")
        if element_type in {"shapes", "points"}:
            if _is_color_like(color):
                logger.info("Value for parameter 'color' appears to be a color, using it as such.")
                param_dict["col_for_color"] = None
            else:
                param_dict["col_for_color"] = color
                param_dict["color"] = None
    elif "color" in param_dict and element_type != "labels":
        param_dict["col_for_color"] = None

    if outline_width := param_dict.get("outline_width"):
        if not isinstance(outline_width, float | int):
            raise TypeError("Parameter 'outline_width' must be numeric.")
        if outline_width < 0:
            raise ValueError("Parameter 'outline_width' cannot be negative.")

    if (outline_alpha := param_dict.get("outline_alpha")) and (
        not isinstance(outline_alpha, float | int) or not 0 <= outline_alpha <= 1
    ):
        raise TypeError("Parameter 'outline_alpha' must be numeric and between 0 and 1.")

    if contour_px is not None and contour_px <= 0:
        raise ValueError("Parameter 'contour_px' must be a positive number.")

    if (alpha := param_dict.get("alpha")) is not None:
        if not isinstance(alpha, float | int):
            raise TypeError("Parameter 'alpha' must be numeric.")
        if not 0 <= alpha <= 1:
            raise ValueError("Parameter 'alpha' must be between 0 and 1.")

    if (fill_alpha := param_dict.get("fill_alpha")) is not None:
        if not isinstance(fill_alpha, float | int):
            raise TypeError("Parameter 'fill_alpha' must be numeric.")
        if fill_alpha < 0:
            raise ValueError("Parameter 'fill_alpha' cannot be negative.")

    if (cmap := param_dict.get("cmap")) is not None and (palette := param_dict.get("palette")) is not None:
        raise ValueError("Both `palette` and `cmap` are specified. Please specify only one of them.")
    param_dict["cmap"] = cmap

    if (groups := param_dict.get("groups")) is not None:
        if not isinstance(groups, list | str):
            raise TypeError("Parameter 'groups' must be a string or a list of strings.")
        if isinstance(groups, str):
            param_dict["groups"] = [groups]
        elif not all(isinstance(g, str) for g in groups):
            raise TypeError("Each item in 'groups' must be a string.")

    palette = param_dict["palette"]

    if isinstance((palette := param_dict["palette"]), list):
        if not all(isinstance(p, str) for p in palette):
            raise ValueError("If specified, parameter 'palette' must contain only strings.")
    elif isinstance(palette, str | type(None)) and "palette" in param_dict:
        param_dict["palette"] = [palette] if palette is not None else None

    if element_type in ["shapes", "points", "labels"] and (palette := param_dict.get("palette")) is not None:
        groups = param_dict.get("groups")
        if groups is None:
            raise ValueError("When specifying 'palette', 'groups' must also be specified.")
        if len(groups) != len(palette):
            raise ValueError(
                f"The length of 'palette' and 'groups' must be the same, length is {len(palette)} and"
                f"{len(groups)} respectively."
            )

    if isinstance(cmap, list):
        if not all(isinstance(c, Colormap | str) for c in cmap):
            raise TypeError("Each item in 'cmap' list must be a string or a Colormap.")
    elif isinstance(cmap, Colormap | str | type(None)):
        if "cmap" in param_dict:
            param_dict["cmap"] = [cmap] if cmap is not None else None
    else:
        raise TypeError("Parameter 'cmap' must be a string, a Colormap, or a list of these types.")

    if (na_color := param_dict.get("na_color")) != "default" and (
        na_color is not None and not _is_color_like(na_color)
    ):
        raise ValueError("Parameter 'na_color' must be color-like.")

    if (norm := param_dict.get("norm")) is not None:
        if element_type in {"images", "labels"} and not isinstance(norm, Normalize):
            raise TypeError("Parameter 'norm' must be of type Normalize.")
        if element_type in ["shapes", "points"] and not isinstance(norm, bool | Normalize):
            raise TypeError("Parameter 'norm' must be a boolean or a mpl.Normalize.")

    if (scale := param_dict.get("scale")) is not None:
        if element_type in {"images", "labels"} and not isinstance(scale, str):
            raise TypeError("Parameter 'scale' must be a string if specified.")
        if element_type == "shapes":
            if not isinstance(scale, float | int):
                raise TypeError("Parameter 'scale' must be numeric.")
            if scale < 0:
                raise ValueError("Parameter 'scale' must be a positive number.")

    if size := param_dict.get("size"):
        if not isinstance(size, float | int):
            raise TypeError("Parameter 'size' must be numeric.")
        if size < 0:
            raise ValueError("Parameter 'size' must be a positive number.")

    table_name = param_dict.get("table_name")
    table_layer = param_dict.get("table_layer")
    if table_name and not isinstance(param_dict["table_name"], str):
        raise TypeError("Parameter 'table_name' must be a string.")

    if table_layer and not isinstance(param_dict["table_layer"], str):
        raise TypeError("Parameter 'table_layer' must be a string.")

    def _ensure_table_and_layer_exist_in_sdata(
        sdata: SpatialData, table_name: str | None, table_layer: str | None
    ) -> bool:
        """Ensure that table_name and table_layer are valid; throw error if not."""
        if table_name:
            if table_layer:
                if table_layer in sdata.tables[table_name].layers:
                    return True
                raise ValueError(f"Layer '{table_layer}' not found in table '{table_name}'.")
            return True  # using sdata.tables[table_name].X

        if table_layer:
            # user specified a layer but we have no tables => invalid
            if len(sdata.tables) == 0:
                raise ValueError("Trying to use 'table_layer' but no tables are present in the SpatialData object.")
            if len(sdata.tables) == 1:
                single_table_name = list(sdata.tables.keys())[0]
                if table_layer in sdata.tables[single_table_name].layers:
                    return True
                raise ValueError(f"Layer '{table_layer}' not found in table '{single_table_name}'.")
            # more than one tables, try to find which one has the given layer
            found_table = False
            for tname in sdata.tables:
                if table_layer in sdata.tables[tname].layers:
                    if found_table:
                        raise ValueError(
                            "Trying to guess 'table_name' based on 'table_layer', but found multiple matches."
                        )
                    found_table = True

            if found_table:
                return True

            raise ValueError(f"Layer '{table_layer}' not found in any table.")

        return True  # not using any table

    assert _ensure_table_and_layer_exist_in_sdata(param_dict.get("sdata"), table_name, table_layer)

    if (method := param_dict.get("method")) not in ["matplotlib", "datashader", None]:
        raise ValueError("If specified, parameter 'method' must be either 'matplotlib' or 'datashader'.")

    valid_ds_reduction_methods = [
        "sum",
        "mean",
        "any",
        "count",
        # "m2", -> not intended to be used alone (see https://datashader.org/api.html#datashader.reductions.m2)
        # "mode", -> not supported for points (see https://datashader.org/api.html#datashader.reductions.mode)
        "std",
        "var",
        "max",
        "min",
    ]
    if (ds_reduction := param_dict.get("ds_reduction")) and (ds_reduction not in valid_ds_reduction_methods):
        raise ValueError(f"Parameter 'ds_reduction' must be one of the following: {valid_ds_reduction_methods}.")

    if method == "datashader" and ds_reduction is None:
        param_dict["ds_reduction"] = "sum"

    return param_dict


def _validate_label_render_params(
    sdata: sd.SpatialData,
    element: str | None,
    cmap: list[Colormap | str] | Colormap | str | None,
    color: str | None,
    fill_alpha: float | int,
    contour_px: int | None,
    groups: list[str] | str | None,
    palette: list[str] | str | None,
    na_color: ColorLike | None,
    norm: Normalize | None,
    outline_alpha: float | int,
    scale: str | None,
    table_name: str | None,
    table_layer: str | None,
) -> dict[str, dict[str, Any]]:
    param_dict: dict[str, Any] = {
        "sdata": sdata,
        "element": element,
        "fill_alpha": fill_alpha,
        "contour_px": contour_px,
        "groups": groups,
        "palette": palette,
        "color": color,
        "na_color": na_color,
        "outline_alpha": outline_alpha,
        "cmap": cmap,
        "norm": norm,
        "scale": scale,
        "table_name": table_name,
        "table_layer": table_layer,
    }
    param_dict = _type_check_params(param_dict, "labels")

    element_params: dict[str, dict[str, Any]] = {}
    for el in param_dict["element"]:
        # ensure that the element exists in the SpatialData object
        _ = param_dict["sdata"][el]

        element_params[el] = {}
        element_params[el]["na_color"] = param_dict["na_color"]
        element_params[el]["cmap"] = param_dict["cmap"]
        element_params[el]["norm"] = param_dict["norm"]
        element_params[el]["fill_alpha"] = param_dict["fill_alpha"]
        element_params[el]["scale"] = param_dict["scale"]
        element_params[el]["outline_alpha"] = param_dict["outline_alpha"]
        element_params[el]["contour_px"] = param_dict["contour_px"]
        element_params[el]["table_layer"] = param_dict["table_layer"]

        element_params[el]["table_name"] = None
        element_params[el]["color"] = None
        if (color := param_dict["color"]) is not None:
            color, table_name = _validate_col_for_column_table(sdata, el, color, param_dict["table_name"], labels=True)
            element_params[el]["table_name"] = table_name
            element_params[el]["color"] = color

        element_params[el]["palette"] = param_dict["palette"] if element_params[el]["table_name"] is not None else None
        element_params[el]["groups"] = param_dict["groups"] if element_params[el]["table_name"] is not None else None

    return element_params


def _validate_points_render_params(
    sdata: sd.SpatialData,
    element: str | None,
    alpha: float | int,
    color: str | None,
    groups: list[str] | str | None,
    palette: list[str] | str | None,
    na_color: ColorLike | None,
    cmap: list[Colormap | str] | Colormap | str | None,
    norm: Normalize | None,
    size: float | int,
    table_name: str | None,
    table_layer: str | None,
    ds_reduction: str | None,
) -> dict[str, dict[str, Any]]:
    param_dict: dict[str, Any] = {
        "sdata": sdata,
        "element": element,
        "alpha": alpha,
        "color": color,
        "groups": groups,
        "palette": palette,
        "na_color": na_color,
        "cmap": cmap,
        "norm": norm,
        "size": size,
        "table_name": table_name,
        "table_layer": table_layer,
        "ds_reduction": ds_reduction,
    }
    param_dict = _type_check_params(param_dict, "points")

    element_params: dict[str, dict[str, Any]] = {}
    for el in param_dict["element"]:
        # ensure that the element exists in the SpatialData object
        _ = param_dict["sdata"][el]

        element_params[el] = {}
        element_params[el]["na_color"] = param_dict["na_color"]
        element_params[el]["cmap"] = param_dict["cmap"]
        element_params[el]["norm"] = param_dict["norm"]
        element_params[el]["color"] = param_dict["color"]
        element_params[el]["size"] = param_dict["size"]
        element_params[el]["alpha"] = param_dict["alpha"]
        element_params[el]["table_layer"] = param_dict["table_layer"]

        element_params[el]["table_name"] = None
        element_params[el]["col_for_color"] = None
        if (col_for_color := param_dict["col_for_color"]) is not None:
            col_for_color, table_name = _validate_col_for_column_table(
                sdata, el, col_for_color, param_dict["table_name"]
            )
            element_params[el]["table_name"] = table_name
            element_params[el]["col_for_color"] = col_for_color

        element_params[el]["palette"] = param_dict["palette"] if param_dict["col_for_color"] is not None else None
        element_params[el]["groups"] = param_dict["groups"] if param_dict["col_for_color"] is not None else None
        element_params[el]["ds_reduction"] = param_dict["ds_reduction"]

    return element_params


def _validate_shape_render_params(
    sdata: sd.SpatialData,
    element: str | None,
    fill_alpha: float | int,
    groups: list[str] | str | None,
    palette: list[str] | str | None,
    color: list[str] | str | None,
    na_color: ColorLike | None,
    outline_width: float | int,
    outline_color: str | list[float],
    outline_alpha: float | int,
    cmap: list[Colormap | str] | Colormap | str | None,
    norm: Normalize | None,
    scale: float | int,
    table_name: str | None,
    table_layer: str | None,
    method: str | None,
    ds_reduction: str | None,
) -> dict[str, dict[str, Any]]:
    param_dict: dict[str, Any] = {
        "sdata": sdata,
        "element": element,
        "fill_alpha": fill_alpha,
        "groups": groups,
        "palette": palette,
        "color": color,
        "na_color": na_color,
        "outline_width": outline_width,
        "outline_color": outline_color,
        "outline_alpha": outline_alpha,
        "cmap": cmap,
        "norm": norm,
        "scale": scale,
        "table_name": table_name,
        "table_layer": table_layer,
        "method": method,
        "ds_reduction": ds_reduction,
    }
    param_dict = _type_check_params(param_dict, "shapes")

    element_params: dict[str, dict[str, Any]] = {}
    for el in param_dict["element"]:
        # ensure that the element exists in the SpatialData object
        _ = param_dict["sdata"][el]

        element_params[el] = {}
        element_params[el]["fill_alpha"] = param_dict["fill_alpha"]
        element_params[el]["na_color"] = param_dict["na_color"]
        element_params[el]["outline_width"] = param_dict["outline_width"]
        element_params[el]["outline_color"] = param_dict["outline_color"]
        element_params[el]["outline_alpha"] = param_dict["outline_alpha"]
        element_params[el]["cmap"] = param_dict["cmap"]
        element_params[el]["norm"] = param_dict["norm"]
        element_params[el]["scale"] = param_dict["scale"]
        element_params[el]["table_layer"] = param_dict["table_layer"]

        element_params[el]["color"] = param_dict["color"]

        element_params[el]["table_name"] = None
        element_params[el]["col_for_color"] = None
        if (col_for_color := param_dict["col_for_color"]) is not None:
            col_for_color, table_name = _validate_col_for_column_table(
                sdata, el, col_for_color, param_dict["table_name"]
            )
            element_params[el]["table_name"] = table_name
            element_params[el]["col_for_color"] = col_for_color

        element_params[el]["palette"] = param_dict["palette"] if param_dict["col_for_color"] is not None else None
        element_params[el]["groups"] = param_dict["groups"] if param_dict["col_for_color"] is not None else None
        element_params[el]["method"] = param_dict["method"]
        element_params[el]["ds_reduction"] = param_dict["ds_reduction"]

    return element_params


def _validate_col_for_column_table(
    sdata: SpatialData,
    element_name: str,
    col_for_color: str | None,
    table_name: str | None,
    labels: bool = False,
) -> tuple[str | None, str | None]:
    if not labels and col_for_color in sdata[element_name].columns:
        table_name = None
    elif table_name is not None:
        tables = get_element_annotators(sdata, element_name)
        if table_name not in tables or (
            col_for_color not in sdata[table_name].obs.columns and col_for_color not in sdata[table_name].var_names
        ):
            table_name = None
            col_for_color = None
    else:
        tables = get_element_annotators(sdata, element_name)
        for table_name in tables.copy():
            if col_for_color not in sdata[table_name].obs.columns and col_for_color not in sdata[table_name].var_names:
                tables.remove(table_name)
        if len(tables) == 0:
            col_for_color = None
        elif len(tables) >= 1:
            table_name = next(iter(tables))
            if len(tables) > 1:
                warnings.warn(
                    f"Multiple tables contain color column, using {table_name}",
                    UserWarning,
                    stacklevel=2,
                )
    return col_for_color, table_name


def _validate_image_render_params(
    sdata: sd.SpatialData,
    element: str | None,
    channel: list[str] | list[int] | str | int | None,
    alpha: float | int | None,
    palette: list[str] | str | None,
    na_color: ColorLike | None,
    cmap: list[Colormap | str] | Colormap | str | None,
    norm: Normalize | None,
    scale: str | None,
) -> dict[str, dict[str, Any]]:
    param_dict: dict[str, Any] = {
        "sdata": sdata,
        "element": element,
        "channel": channel,
        "alpha": alpha,
        "palette": palette,
        "na_color": na_color,
        "cmap": cmap,
        "norm": norm,
        "scale": scale,
    }
    param_dict = _type_check_params(param_dict, "images")

    element_params: dict[str, dict[str, Any]] = {}
    for el in param_dict["element"]:
        element_params[el] = {}
        spatial_element = param_dict["sdata"][el]

        spatial_element_ch = (
            spatial_element.c if isinstance(spatial_element, DataArray) else spatial_element["scale0"].c
        )

        channel = param_dict["channel"]
        channel_list: list[str] | list[int] | None
        if isinstance(channel, list):
            type_ = type(channel[0])
            assert all(isinstance(ch, type_) for ch in channel), "All channels must be of the same type."
        # mypy complains that channel_list can be also of type list[str | int]
        channel_list = [channel] if isinstance(channel, int | str) else channel  # type: ignore[assignment]

        if channel_list is not None and (
            (isinstance(channel_list[0], int) and max([abs(ch) for ch in channel_list]) <= len(spatial_element_ch))  # type: ignore[arg-type]
            or all(ch in spatial_element_ch for ch in channel_list)
        ):
            element_params[el]["channel"] = channel_list
        else:
            element_params[el]["channel"] = None

        element_params[el]["alpha"] = param_dict["alpha"]

        if isinstance(palette := param_dict["palette"], list):
            if len(palette) == 1:
                palette_length = len(channel_list) if channel_list is not None else len(spatial_element_ch)
                palette = palette * palette_length
            if (channel_list is not None and len(palette) != len(channel_list)) and len(palette) != len(
                spatial_element_ch
            ):
                palette = None
        element_params[el]["palette"] = palette
        element_params[el]["na_color"] = param_dict["na_color"]

        if (cmap := param_dict["cmap"]) is not None:
            if len(cmap) == 1:
                cmap_length = len(channel_list) if channel_list is not None else len(spatial_element_ch)
                cmap = cmap * cmap_length
            if (channel_list is not None and len(cmap) != len(channel_list)) or len(cmap) != len(spatial_element_ch):
                cmap = None
        element_params[el]["cmap"] = cmap
        element_params[el]["norm"] = param_dict["norm"]
        if (scale := param_dict["scale"]) and isinstance(sdata[el], DataTree):
            if scale not in list(sdata[el].keys()) and scale != "full":
                element_params[el]["scale"] = None
            else:
                element_params[el]["scale"] = scale
        else:
            element_params[el]["scale"] = scale

    return element_params


def _get_wanted_render_elements(
    sdata: SpatialData,
    sdata_wanted_elements: list[str],
    params: (ImageRenderParams | LabelsRenderParams | PointsRenderParams | ShapesRenderParams),
    cs: str,
    element_type: Literal["images", "labels", "points", "shapes"],
) -> tuple[list[str], list[str], bool]:
    wants_elements = True
    if element_type in [
        "images",
        "labels",
        "points",
        "shapes",
    ]:  # Prevents eval security risk
        wanted_elements: list[str] = [params.element]
        wanted_elements_on_cs = [
            element for element in wanted_elements if cs in set(get_transformation(sdata[element], get_all=True).keys())
        ]

        sdata_wanted_elements.extend(wanted_elements_on_cs)
        return sdata_wanted_elements, wanted_elements_on_cs, wants_elements

    raise ValueError(f"Unknown element type {element_type}")


def _is_coercable_to_float(series: pd.Series) -> bool:
    numeric_series = pd.to_numeric(series, errors="coerce")
    return not numeric_series.isnull().any()


def _ax_show_and_transform(
    array: MaskedArray[tuple[int, ...], Any] | npt.NDArray[Any],
    trans_data: CompositeGenericTransform,
    ax: Axes,
    alpha: float | None = None,
    cmap: ListedColormap | LinearSegmentedColormap | None = None,
    zorder: int = 0,
    extent: list[float] | None = None,
    norm: Normalize | None = None,
) -> matplotlib.image.AxesImage:
    # default extent in mpl:
    image_extent = [-0.5, array.shape[1] - 0.5, array.shape[0] - 0.5, -0.5]
    if extent is not None:
        # make sure extent is [x_min, x_max, y_min, y_max]
        if extent[3] < extent[2]:
            extent[2], extent[3] = extent[3], extent[2]
        if extent[0] < 0:
            x_factor = array.shape[1] / (extent[1] - extent[0])
            image_extent[0] = image_extent[0] + (extent[0] * x_factor)
            image_extent[1] = image_extent[1] + (extent[0] * x_factor)
        if extent[2] < 0:
            y_factor = array.shape[0] / (extent[3] - extent[2])
            image_extent[2] = image_extent[2] + (extent[2] * y_factor)
            image_extent[3] = image_extent[3] + (extent[2] * y_factor)

    if not cmap and alpha is not None:
        im = ax.imshow(
            array,
            alpha=alpha,
            zorder=zorder,
            extent=tuple(image_extent),
            norm=norm,
        )
        im.set_transform(trans_data)
    else:
        im = ax.imshow(
            array,
            cmap=cmap,
            zorder=zorder,
            extent=tuple(image_extent),
            norm=norm,
        )
        im.set_transform(trans_data)
    return im


def set_zero_in_cmap_to_transparent(cmap: Colormap | str, steps: int | None = None) -> ListedColormap:
    """
    Modify colormap so that 0s are transparent.

    Parameters
    ----------
    cmap (Colormap | str): A matplotlib Colormap instance or a colormap name string.
    steps (int): The number of steps in the colormap.

    Returns
    -------
    ListedColormap: A new colormap instance with modified alpha values.
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    colors = cmap(np.arange(steps or cmap.N))
    colors[0, :] = [1.0, 1.0, 1.0, 0.0]

    return ListedColormap(colors)


def _get_extent_and_range_for_datashader_canvas(
    spatial_element: SpatialElement,
    coordinate_system: str,
    ax: Axes,
    fig_params: FigParams,
) -> tuple[Any, Any, list[Any], list[Any], Any]:
    extent = get_extent(spatial_element, coordinate_system=coordinate_system)
    x_ext = [min(0, extent["x"][0]), extent["x"][1]]
    y_ext = [min(0, extent["y"][0]), extent["y"][1]]
    previous_xlim = ax.get_xlim()
    previous_ylim = ax.get_ylim()
    # increase range if sth larger was rendered on the axis before
    if _mpl_ax_contains_elements(ax):
        x_ext = [min(x_ext[0], previous_xlim[0]), max(x_ext[1], previous_xlim[1])]
        y_ext = (
            [
                min(y_ext[0], previous_ylim[1]),
                max(y_ext[1], previous_ylim[0]),
            ]
            if ax.yaxis_inverted()
            else [
                min(y_ext[0], previous_ylim[0]),
                max(y_ext[1], previous_ylim[1]),
            ]
        )

    # compute canvas size in pixels close to the actual image size to speed up computation
    plot_width = x_ext[1] - x_ext[0]
    plot_height = y_ext[1] - y_ext[0]
    plot_width_px = int(round(fig_params.fig.get_size_inches()[0] * fig_params.fig.dpi))
    plot_height_px = int(round(fig_params.fig.get_size_inches()[1] * fig_params.fig.dpi))
    factor: float
    factor = np.min([plot_width / plot_width_px, plot_height / plot_height_px])
    plot_width = int(np.round(plot_width / factor))
    plot_height = int(np.round(plot_height / factor))

    return plot_width, plot_height, x_ext, y_ext, factor


def _create_image_from_datashader_result(
    ds_result: ds.transfer_functions.Image | np.ndarray[Any, np.dtype[np.uint8]],
    factor: float,
    ax: Axes,
) -> tuple[MaskedArray[tuple[int, ...], Any], matplotlib.transforms.Transform]:
    # create SpatialImage from datashader output to get it back to original size
    rgba_image_data = ds_result.copy() if isinstance(ds_result, np.ndarray) else ds_result.to_numpy().base
    rgba_image_data = np.transpose(rgba_image_data, (2, 0, 1))
    rgba_image = Image2DModel.parse(
        rgba_image_data,
        dims=("c", "y", "x"),
        transformations={"global": Scale([1, factor, factor], ("c", "y", "x"))},
    )

    _, trans_data = _prepare_transformation(rgba_image, "global", ax)

    rgba_image = np.transpose(rgba_image.data.compute(), (1, 2, 0))  # type: ignore[attr-defined]
    rgba_image = ma.masked_array(rgba_image)  # type conversion for mypy

    return rgba_image, trans_data


def _datashader_aggregate_with_function(
    reduction: (Literal["sum", "mean", "any", "count", "std", "var", "max", "min"] | None),
    cvs: Canvas,
    spatial_element: GeoDataFrame | dask.dataframe.core.DataFrame,
    col_for_color: str | None,
    element_type: Literal["points", "shapes"],
) -> DataArray:
    """
    When shapes or points are colored by a continuous value during rendering with datashader.

    This function performs the aggregation using the user-specified reduction method.

    Parameters
    ----------
    reduction: String specifying the datashader reduction method to be used.
        If None, "sum" is used as default.
    cvs: Canvas object previously created with ds.Canvas()
    spatial_element: geo or dask dataframe with the shapes or points to render
    col_for_color: name of the column containing the values by which to color
    element_type: tells us if this function is called from _render_shapes() or _render_points()
    """
    if reduction is None:
        reduction = "sum"

    reduction_function_map = {
        "sum": ds.sum,
        "mean": ds.mean,
        "any": ds.any,
        "count": ds.count,
        "std": ds.std,
        "var": ds.var,
        "max": ds.max,
        "min": ds.min,
    }

    try:
        reduction_function = reduction_function_map[reduction](column=col_for_color)
    except KeyError as e:
        raise ValueError(
            f"Reduction '{reduction}' is not supported. Please use one of: {', '.join(reduction_function_map.keys())}."
        ) from e

    element_function_map = {
        "points": cvs.points,
        "shapes": cvs.polygons,
    }

    try:
        element_function = element_function_map[element_type]
    except KeyError as e:
        raise ValueError(f"Element type '{element_type}' is not supported. Use 'points' or 'shapes'.") from e

    if element_type == "points":
        points_aggregate = element_function(spatial_element, "x", "y", agg=reduction_function)
        if reduction == "any":
            # replace False/True by nan/1
            points_aggregate = points_aggregate.astype(int)
            points_aggregate = points_aggregate.where(points_aggregate > 0)
        return points_aggregate

    # is shapes
    return element_function(spatial_element, geometry="geometry", agg=reduction_function)


def _datshader_get_how_kw_for_spread(
    reduction: (Literal["sum", "mean", "any", "count", "std", "var", "max", "min"] | None),
) -> str:
    # Get the best input for the how argument of ds.tf.spread(), needed for numerical values
    reduction = reduction or "sum"

    reduction_to_how_map = {
        "sum": "add",
        "mean": "source",
        "any": "source",
        "count": "add",
        "std": "source",
        "var": "source",
        "max": "max",
        "min": "min",
    }

    if reduction not in reduction_to_how_map:
        raise ValueError(
            f"Reduction {reduction} is not supported, please use one of the following: sum, mean, any, count"
            ", std, var, max, min."
        )

    return reduction_to_how_map[reduction]


def _prepare_transformation(
    element: DataArray | GeoDataFrame | dask.dataframe.core.DataFrame,
    coordinate_system: str,
    ax: Axes | None = None,
) -> tuple[
    matplotlib.transforms.Affine2D,
    matplotlib.transforms.CompositeGenericTransform | None,
]:
    trans = get_transformation(element, get_all=True)[coordinate_system]
    affine_trans = trans.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    trans = mtransforms.Affine2D(matrix=affine_trans)
    trans_data = trans + ax.transData if ax is not None else None

    return trans, trans_data


def _get_datashader_trans_matrix_of_single_element(
    trans: Identity | Scale | Affine | MapAxis | Translation,
) -> npt.NDArray[Any]:
    flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    tm: npt.NDArray[Any] = trans.to_affine_matrix(("x", "y"), ("x", "y"))

    if isinstance(trans, Identity):
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if isinstance(trans, (Scale | Affine)):
        # idea: "flip the y-axis", apply transformation, flip back
        flip_and_transform: npt.NDArray[Any] = flip_matrix @ tm @ flip_matrix
        return flip_and_transform
    if isinstance(trans, MapAxis):
        # no flipping needed
        return tm
    # for a Translation, we need the transposed transformation matrix
    tm_T = tm.T
    assert isinstance(tm_T, np.ndarray)
    return tm_T


def _get_transformation_matrix_for_datashader(
    trans: Scale | Identity | Affine | MapAxis | Translation | SDSequence,
) -> npt.NDArray[Any]:
    """Get the affine matrix needed to transform shapes for rendering with datashader."""
    if isinstance(trans, SDSequence):
        tm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for x in trans.transformations:
            tm = tm @ _get_datashader_trans_matrix_of_single_element(x)
        return tm
    return _get_datashader_trans_matrix_of_single_element(trans)


def _datashader_map_aggregate_to_color(
    agg: DataArray,
    cmap: str | list[str] | ListedColormap,
    color_key: None | list[str] = None,
    min_alpha: float = 40,
    span: None | list[float] = None,
    clip: bool = True,
) -> ds.tf.Image | np.ndarray[Any, np.dtype[np.uint8]]:
    """ds.tf.shade() part, ensuring correct clipping behavior.

    If necessary (norm.clip=False), split shading in 3 parts and in the end, stack results.
    This ensures the correct clipping behavior, because else datashader would always automatically clip.
    """
    if not clip and isinstance(cmap, Colormap) and span is not None:
        # in case we use datashader together with a Normalize object where clip=False
        # why we need this is documented in https://github.com/scverse/spatialdata-plot/issues/372
        agg_in = agg.where((agg >= span[0]) & (agg <= span[1]))
        img_in = ds.tf.shade(
            agg_in,
            cmap=cmap,
            span=(span[0], span[1]),
            how="linear",
            color_key=color_key,
            min_alpha=min_alpha,
        )

        agg_under = agg.where(agg < span[0])
        img_under = ds.tf.shade(
            agg_under,
            cmap=[to_hex(cmap.get_under())[:7]],
            min_alpha=min_alpha,
            color_key=color_key,
        )

        agg_over = agg.where(agg > span[1])
        img_over = ds.tf.shade(
            agg_over,
            cmap=[to_hex(cmap.get_over())[:7]],
            min_alpha=min_alpha,
            color_key=color_key,
        )

        # stack the 3 arrays manually: go from under, through in to over and always overlay the values where alpha=0
        stack = img_under.to_numpy().base
        if stack is None:
            stack = img_in.to_numpy().base
        else:
            stack[stack[:, :, 3] == 0] = img_in.to_numpy().base[stack[:, :, 3] == 0]
        img_over = img_over.to_numpy().base
        if img_over is not None:
            stack[stack[:, :, 3] == 0] = img_over[stack[:, :, 3] == 0]
        return stack

    return ds.tf.shade(
        agg,
        cmap=cmap,
        color_key=color_key,
        min_alpha=min_alpha,
        span=span,
        how="linear",
    )


def _hex_no_alpha(hex: str) -> str:
    """
    Return a hex color string without an alpha component.

    Parameters
    ----------
    hex : str
        The input hex color string. Must be in one of the following formats:
        - "#RRGGBB": a hex color without an alpha channel.
        - "#RRGGBBAA": a hex color with an alpha channel that will be removed.

    Returns
    -------
    str
        The hex color string in "#RRGGBB" format.
    """
    if not isinstance(hex, str):
        raise TypeError("Input must be a string")
    if not hex.startswith("#"):
        raise ValueError("Invalid hex color: must start with '#'")

    hex_digits = hex[1:]
    length = len(hex_digits)

    if length == 6:
        if not all(c in "0123456789abcdefABCDEF" for c in hex_digits):
            raise ValueError("Invalid hex color: contains non-hex characters")
        return hex  # Already in #RRGGBB format.

    if length == 8:
        if not all(c in "0123456789abcdefABCDEF" for c in hex_digits):
            raise ValueError("Invalid hex color: contains non-hex characters")
        # Return only the first 6 characters, stripping the alpha.
        return "#" + hex_digits[:6]

    raise ValueError("Invalid hex color length: must be either '#RRGGBB' or '#RRGGBBAA'")
