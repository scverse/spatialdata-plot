from __future__ import annotations

import os
import warnings
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal

import dask
import datashader as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata as sd
from anndata import AnnData
from dask.array.core import slices_from_chunks
from matplotlib import colors, patheffects, rcParams
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import (
    Colormap,
    ListedColormap,
    Normalize,
)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib_scalebar.scalebar import ScaleBar
from pandas.api.types import CategoricalDtype, is_numeric_dtype
from pandas.core.arrays.categorical import Categorical
from scanpy import settings
from scanpy.plotting._tools.scatterplots import _add_categorical_legend
from spatialdata import (
    SpatialData,
    get_element_annotators,
    get_extent,
    join_spatialelement_table,
    rasterize,
)
from spatialdata import (
    deepcopy as sd_deepcopy,
)
from spatialdata._types import ArrayLike
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    get_model,
    get_table_keys,
)
from spatialdata.transformations.operations import get_transformation
from xarray import DataArray, DataTree

from spatialdata_plot._logging import logger
from spatialdata_plot.pl.render_params import (
    CmapParams,
    Color,
    ColorbarSpec,
    ColorLike,
    FigParams,
    GraphRenderParams,
    ImageRenderParams,
    LabelsRenderParams,
    PointsRenderParams,
    ScalebarParams,
    ShapesRenderParams,
    _FontSize,
    _FontWeight,
)

to_hex = partial(colors.to_hex, keep_alpha=True)

_GROUPS_IGNORED_WARNING = "Parameter 'groups' is ignored when 'color' is a literal color, not a column name."

_RENDER_CMD_TO_CS_FLAG: dict[str, str] = {
    "render_images": "has_images",
    "render_shapes": "has_shapes",
    "render_points": "has_points",
    "render_labels": "has_labels",
}


def _check_obs_var_shadow(
    sdata: SpatialData | None,
    element_name: str | None,
    value_to_plot: str | None,
    table_name: str | None,
) -> None:
    """Raise if ``value_to_plot`` exists in both ``table.obs.columns`` and ``table.var_names``.

    Upstream ``_get_table_origins`` uses an ``elif`` chain, so a key that lives in
    both locations is silently resolved to ``obs`` — masking the user's likely
    intent of plotting gene expression. Catch this here before any value fetch.
    Any ``None`` parameter short-circuits the check.
    """
    if (
        value_to_plot is None
        or table_name is None
        or element_name is None
        or sdata is None
        or table_name not in sdata.tables
    ):
        return
    if table_name not in get_element_annotators(sdata, element_name):
        return
    table = sdata.tables[table_name]
    if value_to_plot in table.obs.columns and value_to_plot in table.var_names:
        raise ValueError(
            f"`color={value_to_plot!r}` is ambiguous: it exists in both "
            f"`table[{table_name!r}].obs.columns` and `table[{table_name!r}].var_names`. "
            "Rename one of them (or drop the obs column) so the intended source is unambiguous."
        )


def _gate_palette_and_groups(
    element_params: dict[str, Any],
    param_dict: dict[str, Any],
) -> None:
    """Set palette/groups on element_params only when col_for_color is present, else warn."""
    has_col = element_params.get("col_for_color") is not None
    element_params["palette"] = param_dict["palette"] if has_col else None
    if not has_col and param_dict["groups"] is not None:
        logger.warning(_GROUPS_IGNORED_WARNING)
    element_params["groups"] = param_dict["groups"] if has_col else None


def _extract_scalar_value(value: Any, default: float = 0.0) -> float:
    """
    Extract a scalar float value from various data types.

    Handles pandas Series, arrays, lists, and other iterables by taking the first element.
    Converts non-numeric values to the default value.

    Parameters
    ----------
    value : Any
        The value to extract a scalar from
    default : float, default 0.0
        Default value to return if conversion fails

    Returns
    -------
    float
        The extracted scalar value
    """
    try:
        # Handle pandas Series or similar objects with iloc
        if hasattr(value, "iloc"):
            if len(value) > 0:
                value = value.iloc[0]
            else:
                return default

        # Handle other array-like objects
        elif hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            if len(value) > 0:
                value = value[0]
            else:
                return default

        # Convert to float, handling NaN values
        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError, IndexError):
        return default


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


_MPL_SINGLE_LETTER_COLORS = frozenset("bgrcmykw")


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
    scalebar_kwargs: Mapping[str, Any] | None = None,
) -> tuple[FigParams, ScalebarParams]:
    # handle axes and size
    wspace = 0.75 / rcParams["figure.figsize"][0] + 0.02 if wspace is None else wspace
    figsize = rcParams["figure.figsize"] if figsize is None else figsize
    # When creating a new figure, fall back to rcParams; when the user provides
    # their own axes, preserve the figure's existing DPI (only override if
    # the user explicitly passed dpi= to show()).
    resolved_dpi = rcParams["figure.dpi"] if dpi is None else dpi
    if num_panels > 1 and ax is None:
        fig, grid = _panel_grid(
            num_panels=num_panels,
            hspace=hspace,
            wspace=wspace,
            ncols=ncols,
            dpi=resolved_dpi,
            figsize=figsize,
        )
        axs: None | Sequence[Axes] = [plt.subplot(grid[c]) for c in range(num_panels)]
    elif num_panels > 1:
        if not isinstance(ax, Sequence):
            raise TypeError(f"Expected `ax` to be a `Sequence`, but got {type(ax).__name__}")
        if len(ax) != num_panels:
            raise ValueError(f"Len of `ax`: {len(ax)} is not equal to number of panels: {num_panels}.")
        if fig is None:
            fig = ax[0].get_figure()
        axs = ax
        if dpi is not None:
            fig.set_dpi(dpi)
    else:
        axs = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=resolved_dpi, constrained_layout=True)
        else:
            if isinstance(ax, Sequence):
                if len(ax) != 1:
                    raise ValueError(f"Len of `ax`: {len(ax)} is not equal to number of panels: {num_panels}.")
                ax = ax[0]
            if not isinstance(ax, Axes):
                raise TypeError(f"Expected `ax` to be an `Axes` or a `Sequence` of `Axes`, but got {type(ax).__name__}")
            fig = ax.get_figure()
            if dpi is not None:
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
    scalebar_params = ScalebarParams(
        scalebar_dx=scalebar_dx,
        scalebar_units=scalebar_units,
        scalebar_kwargs=dict(scalebar_kwargs) if scalebar_kwargs else {},
    )

    return fig_params, scalebar_params


def _draw_scalebar(ax: Axes, scalebar_params: ScalebarParams, panel_idx: int) -> None:
    """Attach a single :class:`matplotlib_scalebar.scalebar.ScaleBar` to ``ax``.

    No-op when ``scalebar_dx`` is ``None``. ``scalebar_dx`` and ``scalebar_units`` are
    broadcast lists indexed by the panel position; ``scalebar_kwargs`` is forwarded
    verbatim to :class:`~matplotlib_scalebar.scalebar.ScaleBar`.
    """
    if scalebar_params.scalebar_dx is None or scalebar_params.scalebar_units is None:
        return
    dx = scalebar_params.scalebar_dx[panel_idx]
    units = scalebar_params.scalebar_units[panel_idx]
    ax.add_artist(ScaleBar(dx, units=units, **scalebar_params.scalebar_kwargs))


def _get_cs_contents(sdata: sd.SpatialData) -> pd.DataFrame:
    """Check which coordinate systems contain which elements and return that info."""
    cs_mapping = _get_coordinate_system_mapping(sdata)
    content_flags = ["has_images", "has_labels", "has_points", "has_shapes"]

    rows = []
    for cs_name, element_ids in cs_mapping.items():
        rows.append(
            {
                "cs": cs_name,
                "has_images": any(e in sdata.images for e in element_ids),
                "has_labels": any(e in sdata.labels for e in element_ids),
                "has_points": any(e in sdata.points for e in element_ids),
                "has_shapes": any(e in sdata.shapes for e in element_ids),
            }
        )

    cs_contents = pd.DataFrame(rows, columns=["cs"] + content_flags)
    cs_contents[content_flags] = cs_contents[content_flags].astype("bool")
    return cs_contents


def _join_table_for_element(
    sdata: sd.SpatialData,
    element: str,
    table_name: str,
) -> tuple[Any, AnnData]:
    """Left-join ``element`` with its annotating ``table_name``.

    A left join keeps every shape, including those without a table row (they get no color value and
    are rendered with ``na_color``), matching the points/labels behaviour instead of silently dropping
    unannotated shapes.

    Wraps the workaround for scverse/spatialdata#1099: ``join_spatialelement_table``
    calls ``table.obs.reset_index()`` which fails when the obs index name matches
    an existing column (e.g. "EntityID" in Merfish data). When that collision is
    present, the obs index may also be a non-RangeIndex of int dtype, which
    AnnData's ``_normalize_index`` rejects when the join indexes back into the
    table. Temporarily swap to a clean RangeIndex / drop the conflicting name;
    restore on exit.

    Also patches ``joined_table.uns["spatialdata_attrs"]["region"]`` to the
    actual unique regions after the join so downstream lookups see consistent
    metadata.
    """
    _obs = sdata[table_name].obs
    _saved_index_name = _obs.index.name
    _saved_index: pd.Index | None = None
    _name_collides = _saved_index_name is not None and _saved_index_name in _obs.columns
    if _name_collides and not isinstance(_obs.index, pd.RangeIndex):
        _saved_index = _obs.index
        _obs.index = pd.RangeIndex(len(_obs))
    elif _name_collides:
        _obs.index.name = None

    try:
        element_dict, joined_table = join_spatialelement_table(
            sdata, spatial_element_names=element, table_name=table_name, how="left"
        )
    finally:
        if _saved_index is not None:
            _obs.index = _saved_index
        _obs.index.name = _saved_index_name

    joined_table.uns["spatialdata_attrs"]["region"] = (
        joined_table.obs[joined_table.uns["spatialdata_attrs"]["region_key"]].unique().tolist()
    )
    return element_dict[element], joined_table


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


def _build_alignment_dtype_hint(
    sdata: sd.SpatialData | None,
    element: object,
    color_series: pd.Series,
    table_name: str | None,
) -> str:
    """Build a diagnostic hint string for dtype mismatches between element and table indices."""
    el_dtype = getattr(getattr(element, "index", None), "dtype", None)
    if el_dtype is None or table_name is None or sdata is None or table_name not in sdata.tables:
        return ""
    try:
        _, _, instance_key = get_table_keys(sdata.tables[table_name])
    except (KeyError, ValueError):
        return ""
    tbl_dtype = sdata.tables[table_name].obs[instance_key].dtype
    if el_dtype != tbl_dtype:
        return f" (hint: element index dtype is {el_dtype}, '{instance_key}' dtype is {tbl_dtype})"
    return ""


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
    na_color: Color = Color("default"),
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_loc: str | None = "right margin",
    legend_fontoutline: int | None = None,
    na_in_legend: bool = True,
    colorbar: bool = True,
    colorbar_params: dict[str, object] | None = None,
    colorbar_requests: list[ColorbarSpec] | None = None,
    colorbar_label: str | None = None,
    legend_title: str | None = None,
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
                na_color=[na_color.get_hex()],
                na_in_legend=na_in_legend,
                multi_panel=fig_params.axs is not None,
            )
            # scanpy's helper doesn't accept a title; set it post-hoc so the user can
            # disambiguate fill vs outline when both legends are drawn.
            if legend_title is not None and (legend := ax.get_legend()) is not None:
                legend.set_title(legend_title)
        elif colorbar and colorbar_requests is not None and cax is not None:
            colorbar_requests.append(
                ColorbarSpec(
                    ax=ax,
                    mappable=cax,
                    params=colorbar_params,
                    label=colorbar_label,
                    alpha=alpha,
                )
            )

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

    # Rasterize when the source image is substantially larger than what the
    # current figure DPI × size requires.  The +100 margin avoids rasterizing
    # when the image is only slightly larger than the target.
    do_rasterization = y_dims > target_y_dims + 100 or x_dims > target_x_dims + 100

    if do_rasterization:
        logger.info("Rasterizing image for faster rendering.")
        # ``rasterize`` interprets ``target_unit_to_pixels`` in world units, not
        # intrinsic pixels. Dividing by world extent keeps the result correct
        # for any transformation (translation, scale, etc.).
        world_x = float(extent["x"][1]) - float(extent["x"][0])
        world_y = float(extent["y"][1]) - float(extent["y"][0])
        target_unit_to_pixels = min(target_y_dims / world_y, target_x_dims / world_x)
        image = rasterize(
            image,
            ("y", "x"),
            [extent["y"][0], extent["x"][0]],
            [extent["y"][1], extent["x"][1]],
            coordinate_system,
            target_unit_to_pixels=target_unit_to_pixels,
        )
        if hasattr(image.data, "compute"):
            # rasterize is lazy; downstream reads the result once per channel (NaN check,
            # compositing, draw), so materialize once instead of re-running the warp each time.
            image = image.copy(data=image.data.compute())

    return image


def _rasterize_if_necessary_datashader(
    image: DataArray,
    dpi: float,
    width: float,
    height: float,
    coordinate_system: str,
    extent: dict[str, tuple[float, float]],
    downsample_method: str,
) -> DataArray:
    """Downsample to canvas resolution with a configurable datashader reduction.

    Used by ``render_images(method='datashader')`` so sparse images (mostly
    zeros, rare non-zero pixels) survive the downsample step instead of
    being averaged away by the default mean aggregation.
    """
    has_c_dim = len(image.shape) == 3
    y_dims, x_dims = (image.shape[1], image.shape[2]) if has_c_dim else image.shape

    target_y_dims = int(dpi * height)
    target_x_dims = int(dpi * width)

    if y_dims <= target_y_dims and x_dims <= target_x_dims:
        return image

    # spatialdata.rasterize is invoked solely to inherit the output coords and
    # spatial transformation; its mean-aggregated values are overwritten below.
    # TODO: this wastes a full per-channel resample pass. A future refactor can
    # construct the target DataArray + transformation directly once spatialdata
    # exposes a public geometry-only helper.
    world_x = float(extent["x"][1]) - float(extent["x"][0])
    world_y = float(extent["y"][1]) - float(extent["y"][0])
    target_unit_to_pixels = min(target_y_dims / world_y, target_x_dims / world_x)
    base = rasterize(
        image,
        ("y", "x"),
        [extent["y"][0], extent["x"][0]],
        [extent["y"][1], extent["x"][1]],
        coordinate_system,
        target_unit_to_pixels=target_unit_to_pixels,
    )

    out_y, out_x = (base.shape[1], base.shape[2]) if has_c_dim else base.shape
    # Materialize once: per-chunk reductions across channels would otherwise
    # trigger repeated dask graph evaluations on the same source array.
    src = image.compute() if hasattr(image.data, "compute") else image
    cvs = ds.Canvas(
        plot_width=out_x,
        plot_height=out_y,
        x_range=(float(extent["x"][0]), float(extent["x"][1])),
        y_range=(float(extent["y"][0]), float(extent["y"][1])),
    )
    base.values = np.asarray(cvs.raster(src, downsample_method=downsample_method).values).astype(base.dtype, copy=False)
    return base


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
        # sort scales ascending by x resolution
        order = np.argsort(x_dims)
        scales = [scales[i] for i in order]
        x_dims = [x_dims[i] for i in order]
        y_dims = [y_dims[i] for i in order]

        optimal_x = width * dpi
        optimal_y = height * dpi

        # Pick the lowest-resolution scale where both x and y are >= the
        # target pixel count.  Falls back to highest available resolution.
        optimal_scale = scales[-1]
        for i, (xd, yd) in enumerate(zip(x_dims, y_dims, strict=True)):
            if xd >= optimal_x and yd >= optimal_y:
                optimal_scale = scales[i]
                break

    # NOTE: problematic if there are cases with > 1 data variable
    data_var_keys = list(multiscale_image[optimal_scale].data_vars)
    image = multiscale_image[optimal_scale][data_var_keys[0]]

    return Labels2DModel.parse(image) if is_label else Image2DModel.parse(image, c_coords=image.coords["c"].values)


def _get_elements_to_be_rendered(
    render_cmds: list[
        tuple[
            str,
            ImageRenderParams | LabelsRenderParams | PointsRenderParams | ShapesRenderParams | GraphRenderParams,
        ]
    ],
    cs_index: pd.DataFrame,
    cs: str,
) -> list[str]:
    """
    Get the names of the elements to be rendered in the plot.

    Parameters
    ----------
    render_cmds
        List of tuples containing the commands and their respective parameters.
    cs_index
        The cs_contents dataframe indexed by the "cs" column.
    cs
        The name of the coordinate system to query cs_index for.

    Returns
    -------
    List of names of the SpatialElements to be rendered in the plot.
    """
    elements_to_be_rendered: list[str] = []

    cs_row = cs_index.loc[cs] if cs in cs_index.index else None

    for cmd, params in render_cmds:
        if cmd == "render_graph":
            # Graph doesn't have its own CS flag; include its element so
            # _get_valid_cs keeps the coordinate system alive.
            elements_to_be_rendered.append(params.element)
        else:
            key = _RENDER_CMD_TO_CS_FLAG.get(cmd)
            if key and cs_row is not None and cs_row[key]:
                elements_to_be_rendered.append(params.element)

    return elements_to_be_rendered


def _validate_show_parameters(
    coordinate_systems: list[str] | str | None,
    legend_fontsize: int | float | _FontSize | None,
    legend_fontweight: int | _FontWeight,
    legend_loc: str | None,
    legend_fontoutline: int | None,
    na_in_legend: bool,
    colorbar: bool,
    colorbar_params: dict[str, object] | None,
    wspace: float | None,
    hspace: float,
    ncols: int,
    frameon: bool | None,
    figsize: tuple[float, float] | None,
    dpi: int | None,
    fig: Figure | None,
    title: list[str] | str | None,
    pad_extent: int | float,
    ax: list[Axes] | Axes | None,
    return_ax: bool,
    save: str | Path | None,
    show: bool | None,
    scalebar_dx: float | None,
    scalebar_units: str,
    scalebar_params: dict[str, Any] | None,
    legend_params: dict[str, Any] | None,
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

    if colorbar_params is not None and not isinstance(colorbar_params, dict):
        raise TypeError("Parameter 'colorbar_params' must be a dictionary or None.")

    if wspace is not None and not isinstance(wspace, float):
        raise TypeError("Parameter 'wspace' must be a float.")

    if not isinstance(hspace, float):
        raise TypeError("Parameter 'hspace' must be a float.")

    if not isinstance(ncols, int):
        raise TypeError("Parameter 'ncols' must be an integer.")

    if frameon is not None and not isinstance(frameon, bool):
        raise TypeError("Parameter 'frameon' must be a boolean.")

    if figsize is not None and (
        not isinstance(figsize, tuple | list | np.ndarray)
        or len(figsize) != 2
        or not all(isinstance(x, int | float) and not isinstance(x, bool) for x in figsize)
    ):
        raise TypeError("Parameter 'figsize' must be a tuple, list, or numpy array of two numbers.")

    if dpi is not None and not isinstance(dpi, int):
        raise TypeError("Parameter 'dpi' must be an integer.")

    if fig is not None and not isinstance(fig, Figure):
        raise TypeError("Parameter 'fig' must be a matplotlib.figure.Figure.")

    if title is not None and not isinstance(title, list | str):
        raise TypeError("Parameter 'title' must be a string or a list of strings.")

    if not isinstance(pad_extent, int | float):
        raise TypeError("Parameter 'pad_extent' must be numeric.")

    if ax is not None and not isinstance(ax, Axes | list):
        raise TypeError("Parameter 'ax' must be a matplotlib.axes.Axes or a list of Axes.")

    if not isinstance(return_ax, bool):
        raise TypeError("Parameter 'return_ax' must be a boolean.")

    if save is not None and not isinstance(save, str | Path):
        raise TypeError("Parameter 'save' must be a string or a pathlib.Path.")

    if show is not None and not isinstance(show, bool):
        raise TypeError("Parameter 'show' must be a boolean or None.")

    if scalebar_dx is not None:
        if not isinstance(scalebar_dx, int | float) or isinstance(scalebar_dx, bool):
            raise TypeError("Parameter 'scalebar_dx' must be a number or None.")
        if scalebar_dx <= 0:
            raise ValueError("Parameter 'scalebar_dx' must be > 0.")
        if not isinstance(scalebar_units, str):
            raise TypeError("Parameter 'scalebar_units' must be a string.")

    if scalebar_params is not None and not isinstance(scalebar_params, dict):
        raise TypeError("Parameter 'scalebar_params' must be a dictionary or None.")

    if legend_params is not None:
        if not isinstance(legend_params, dict):
            raise TypeError("Parameter 'legend_params' must be a dictionary or None.")
        # `loc` is matplotlib.Legend's native key; `location` aligns with colorbar_params / scalebar_params.
        allowed_legend_keys = {"loc", "location", "fontsize", "fontweight", "fontoutline", "na_in_legend"}
        unknown = set(legend_params) - allowed_legend_keys
        if unknown:
            raise ValueError(
                f"Unknown legend_params key(s): {sorted(unknown)}. Allowed keys: {sorted(allowed_legend_keys)}."
            )


def _check_color_column_collision(
    sdata: SpatialData,
    elements: list[str],
    color: str,
    element_type: str,
) -> None:
    """Raise if ``color`` is a color-like string that also names a column in the element or its tables."""
    matches: list[str] = []
    for el in elements:
        if element_type in {"shapes", "points"}:
            try:
                el_cols = sdata[el].columns
            except (KeyError, AttributeError):
                el_cols = ()
            if color in el_cols:
                matches.append(f"element '{el}'")
                continue
        try:
            tables = get_element_annotators(sdata, el)
        except (KeyError, ValueError):
            tables = set()
        for t in tables:
            adata = sdata[t]
            if color in adata.obs.columns or color in adata.var_names:
                matches.append(f"table '{t}' (annotating '{el}')")
                break
    if matches:
        locations = ", ".join(matches)
        raise ValueError(
            f"`color={color!r}` is ambiguous: it is a valid matplotlib color name AND a column "
            f"name in {locations}. Disambiguate by either passing an unambiguous color form "
            f"(hex string like '#ffa500' or an RGB(A) tuple), or by renaming the column."
        )


def _type_check_params(param_dict: dict[str, Any], element_type: str) -> dict[str, Any]:
    from spatialdata_plot.pl._color import _is_color_like

    colorbar = param_dict.get("colorbar", "auto")
    if colorbar not in {True, False, None, "auto"}:
        raise TypeError("Parameter 'colorbar' must be one of True, False or 'auto'.")

    colorbar_params = param_dict.get("colorbar_params")
    if colorbar_params is not None and not isinstance(colorbar_params, dict):
        raise TypeError("Parameter 'colorbar_params' must be a dictionary or None.")

    element = param_dict.get("element")
    if element is not None and not isinstance(element, str):
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

    channel = param_dict.get("channel")
    if channel is not None and not isinstance(channel, list | str | int):
        raise TypeError("Parameter 'channel' must be a string, an integer, or a list of strings or integers.")
    if isinstance(channel, list):
        if not all(isinstance(c, str | int) for c in channel):
            raise TypeError("Each item in 'channel' list must be a string or an integer.")
        if not all(isinstance(c, type(channel[0])) for c in channel):
            raise TypeError("Each item in 'channel' list must be of the same type, either string or integer.")

    elif "channel" in param_dict:
        param_dict["channel"] = [channel] if channel is not None else None

    contour_px = param_dict.get("contour_px")
    if contour_px and not isinstance(contour_px, int):
        raise TypeError("Parameter 'contour_px' must be an integer.")

    color = param_dict.get("color")
    if color and element_type in {
        "shapes",
        "points",
        "labels",
        "graph",
    }:
        if not isinstance(color, str | tuple | list):
            raise TypeError("Parameter 'color' must be a string or a tuple/list of floats.")
        if _is_color_like(color):
            if isinstance(color, str):
                _check_color_column_collision(param_dict["sdata"], param_dict["element"], color, element_type)
            param_dict["col_for_color"] = None
            param_dict["color"] = Color(color)
            if param_dict["color"].alpha_is_user_defined():
                if element_type == "points" and param_dict.get("alpha") is None:
                    param_dict["alpha"] = param_dict["color"].get_alpha_as_float()
                elif element_type in {"shapes", "labels"} and param_dict.get("fill_alpha") is None:
                    param_dict["fill_alpha"] = param_dict["color"].get_alpha_as_float()
                else:
                    logger.info(
                        f"Alpha implied by color '{color}' is ignored since the parameter 'alpha' or 'fill_alpha' "
                        "is set and its value takes precedence."
                    )
        elif isinstance(color, str):
            param_dict["col_for_color"] = color
            param_dict["color"] = None
        else:
            raise ValueError(f"{color} is not a valid RGB(A) array and therefore can't be used as 'color' value.")
    elif "color" in param_dict and element_type != "images":
        param_dict["col_for_color"] = None

    outline_width = param_dict.get("outline_width")
    if outline_width:
        # outline_width only exists for shapes at the moment
        if isinstance(outline_width, tuple):
            for ow in outline_width:
                if isinstance(ow, float | int):
                    if ow < 0:
                        raise ValueError("Parameter 'outline_width' cannot contain negative values.")
                else:
                    raise TypeError("Parameter 'outline_width' must contain only numerics when it is a tuple.")
        elif not isinstance(outline_width, float | int):
            raise TypeError("Parameter 'outline_width' must be numeric or a tuple of two numerics.")
        if isinstance(outline_width, float | int) and outline_width < 0:
            raise ValueError("Parameter 'outline_width' cannot be negative.")

    outline_alpha = param_dict.get("outline_alpha")
    if outline_alpha:
        if isinstance(outline_alpha, tuple):
            if element_type != "shapes":
                raise ValueError("Parameter 'outline_alpha' must be a single numeric.")
            if len(outline_alpha) == 1:
                if not isinstance(outline_alpha[0], float | int) or not 0 <= outline_alpha[0] <= 1:
                    raise TypeError("Parameter 'outline_alpha' must be numeric and between 0 and 1.")
                param_dict["outline_alpha"] = outline_alpha[0]
            elif len(outline_alpha) < 1:
                raise ValueError("Empty tuple is not supported as input for outline_alpha!")
            else:
                if len(outline_alpha) > 2:
                    logger.warning(
                        f"Tuple of length {len(outline_alpha)} was passed for outline_alpha, only first two positions "
                        "are used since more than 2 outlines are not supported!"
                    )
                if (
                    not isinstance(outline_alpha[0], float | int)
                    or not isinstance(outline_alpha[1], float | int)
                    or not 0 <= outline_alpha[0] <= 1
                    or not 0 <= outline_alpha[1] <= 1
                ):
                    raise TypeError("Parameter 'outline_alpha' must contain numeric values between 0 and 1.")
                param_dict["outline_alpha"] = (outline_alpha[0], outline_alpha[1])
        elif not isinstance(outline_alpha, float | int) or not 0 <= outline_alpha <= 1:
            raise TypeError("Parameter 'outline_alpha' must be numeric and between 0 and 1.")

    outline_color = param_dict.get("outline_color")
    if "outline_color" in param_dict and element_type in {"shapes", "labels"}:
        param_dict["col_for_outline_color"] = None
    if outline_color:
        if not isinstance(outline_color, str | tuple | list):
            raise TypeError("Parameter 'outline_color' must be a string or a tuple/list of floats or colors.")
        if isinstance(outline_color, tuple | list):
            if len(outline_color) < 1:
                raise ValueError("Empty tuple is not supported as input for outline_color!")
            if len(outline_color) == 1:
                param_dict["outline_color"] = Color(outline_color[0])
            elif len(outline_color) == 2:
                # assuming the case of 2 outlines
                param_dict["outline_color"] = (Color(outline_color[0]), Color(outline_color[1]))
            elif len(outline_color) in [3, 4]:
                # assuming RGB(A) array
                param_dict["outline_color"] = Color(outline_color)
            else:
                raise ValueError(
                    f"Tuple/List of length {len(outline_color)} was passed for outline_color. Valid options would be: "
                    "tuple of 2 colors (for 2 outlines) or an RGB(A) array, aka a list/tuple of 3-4 floats."
                )
        elif isinstance(outline_color, str) and element_type in {"shapes", "labels"}:
            if _is_color_like(outline_color):
                _check_color_column_collision(param_dict["sdata"], param_dict["element"], outline_color, element_type)
                param_dict["outline_color"] = Color(outline_color)
            else:
                if isinstance(param_dict.get("outline_width"), tuple):
                    raise ValueError(
                        "Coloring outlines by a column is not supported with two outlines. "
                        "Pass a scalar `outline_width` or a literal color for `outline_color`."
                    )
                param_dict["col_for_outline_color"] = outline_color
                param_dict["outline_color"] = None
        else:
            param_dict["outline_color"] = Color(outline_color)

    if contour_px is not None and contour_px < 2:
        raise ValueError(
            "Parameter 'contour_px' must be >= 2; values below 2 produce no visible outline "
            "(a 1x1 erosion is the identity transformation)."
        )

    alpha = param_dict.get("alpha")
    if alpha is not None:
        if not isinstance(alpha, float | int):
            raise TypeError("Parameter 'alpha' must be numeric.")
        if not 0 <= alpha <= 1:
            raise ValueError("Parameter 'alpha' must be between 0 and 1.")
    elif element_type == "points":
        # set default alpha for points if not given by user explicitly or implicitly (as part of color)
        param_dict["alpha"] = 1.0

    fill_alpha = param_dict.get("fill_alpha")
    if fill_alpha is not None:
        if not isinstance(fill_alpha, float | int):
            raise TypeError("Parameter 'fill_alpha' must be numeric.")
        if fill_alpha < 0:
            raise ValueError("Parameter 'fill_alpha' cannot be negative.")
    elif element_type == "shapes":
        # set default fill_alpha for shapes if not given by user explicitly or implicitly (as part of color)
        param_dict["fill_alpha"] = 1.0
    elif element_type == "labels":
        # set default fill_alpha for labels if not given by user explicitly or implicitly (as part of color)
        param_dict["fill_alpha"] = 0.4

    cmap = param_dict.get("cmap")
    palette = param_dict.get("palette")
    if cmap is not None and palette is not None:
        raise ValueError("Both `palette` and `cmap` are specified. Please specify only one of them.")
    param_dict["cmap"] = cmap

    groups = param_dict.get("groups")
    if groups is not None:
        if not isinstance(groups, list | str):
            raise TypeError("Parameter 'groups' must be a string or a list of strings.")
        if isinstance(groups, str):
            param_dict["groups"] = [groups]
        elif not all(isinstance(g, str) for g in groups):
            raise TypeError("Each item in 'groups' must be a string.")

    palette = param_dict["palette"]

    # dict palettes (e.g. from make_palette_from_data) bypass groups validation
    if isinstance(palette, dict):
        from matplotlib.colors import is_color_like

        invalid = [f"'{k}': '{v}'" for k, v in palette.items() if not is_color_like(v)]
        if invalid:
            raise ValueError(f"Dict palette contains invalid color values: {', '.join(invalid)}.")
    elif isinstance(palette, list):
        if not all(isinstance(p, str) for p in palette):
            raise ValueError("If specified, parameter 'palette' must contain only strings.")
    elif isinstance(palette, str | type(None)) and "palette" in param_dict and element_type != "graph":
        param_dict["palette"] = [palette] if palette is not None else None

    palette_group = param_dict.get("palette")
    if element_type in ["shapes", "points", "labels"] and palette_group is not None and not isinstance(palette, dict):
        groups = param_dict.get("groups")
        if groups is not None and len(groups) != len(palette_group):
            raise ValueError(
                f"The length of 'palette' and 'groups' must be the same, length is {len(palette_group)} and"
                f"{len(groups)} respectively."
            )

    if isinstance(cmap, list):
        if not all(isinstance(c, Colormap | str) for c in cmap):
            raise TypeError("Each item in 'cmap' list must be a string or a Colormap.")
    elif isinstance(cmap, Colormap | str | type(None)):
        if "cmap" in param_dict and element_type != "graph":
            param_dict["cmap"] = [cmap] if cmap is not None else None
    else:
        raise TypeError("Parameter 'cmap' must be a string, a Colormap, or a list of these types.")

    # validation happens within Color constructor (images don't use na_color)
    if "na_color" in param_dict:
        param_dict["na_color"] = Color(param_dict.get("na_color"))

    norm = param_dict.get("norm")
    if norm is not None:
        if element_type == "images":
            if isinstance(norm, list):
                if not norm:
                    raise ValueError("Parameter 'norm' list must not be empty.")
                if not all(isinstance(n, Normalize) for n in norm):
                    raise TypeError("Every item in 'norm' list must be a Normalize instance.")
            elif not isinstance(norm, Normalize):
                raise TypeError("Parameter 'norm' must be a Normalize or a list of Normalize instances.")
        elif element_type == "labels" and not isinstance(norm, Normalize):
            raise TypeError("Parameter 'norm' must be of type Normalize.")
        if element_type in {"shapes", "points"} and not isinstance(norm, bool | Normalize):
            raise TypeError("Parameter 'norm' must be a boolean or a mpl.Normalize.")
        if element_type == "graph" and not isinstance(norm, Normalize):
            raise TypeError("Parameter 'norm' must be a Normalize instance.")

    scale = param_dict.get("scale")
    if scale is not None:
        if element_type in {"images", "labels"} and not isinstance(scale, str):
            raise TypeError("Parameter 'scale' must be a string if specified.")
        if element_type == "shapes":
            if not isinstance(scale, float | int):
                raise TypeError("Parameter 'scale' must be numeric.")
            if scale < 0:
                raise ValueError("Parameter 'scale' must be a positive number.")

    size = param_dict.get("size")
    if size:
        if not isinstance(size, float | int):
            raise TypeError("Parameter 'size' must be numeric.")
        if size < 0:
            raise ValueError("Parameter 'size' must be a positive number.")

    shape = param_dict.get("shape")
    if element_type == "shapes" and shape is not None:
        valid_shapes = {"circle", "hex", "visium_hex", "square"}
        if not isinstance(shape, str):
            raise TypeError(f"Parameter 'shape' must be a String from {valid_shapes} if not None.")
        if shape not in valid_shapes:
            raise ValueError(f"'{shape}' is not supported for 'shape', please choose from {valid_shapes}.")

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

    _ensure_table_and_layer_exist_in_sdata(param_dict.get("sdata"), table_name, table_layer)

    method = param_dict.get("method")
    if method not in ["matplotlib", "datashader", None]:
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
    ds_reduction = param_dict.get("ds_reduction")
    if ds_reduction and (ds_reduction not in valid_ds_reduction_methods):
        raise ValueError(f"Parameter 'ds_reduction' must be one of the following: {valid_ds_reduction_methods}.")

    if element_type == "graph":
        for key in ("connectivity_key",):
            val = param_dict.get(key)
            if val is not None and not isinstance(val, str):
                raise TypeError(f"Parameter '{key}' must be a string.")

        for key in ("obsp_key", "weight_key", "group_key"):
            val = param_dict.get(key)
            if val is not None and not isinstance(val, str):
                raise TypeError(f"Parameter '{key}' must be a string or None.")

        for key in ("edge_width", "edge_alpha"):
            val = param_dict.get(key)
            if val == "weight":
                continue
            if not isinstance(val, float | int):
                raise TypeError(f"Parameter '{key}' must be numeric or the literal string 'weight'.")
            if val < 0:
                raise ValueError(f"Parameter '{key}' cannot be negative.")

        linestyle = param_dict.get("linestyle")
        if linestyle is not None and not isinstance(linestyle, str | list | tuple):
            raise TypeError("Parameter 'linestyle' must be a string or a sequence of strings.")

        for key in ("include_self_loops", "rasterize"):
            val = param_dict.get(key)
            if val is not None and not isinstance(val, bool):
                raise TypeError(f"Parameter '{key}' must be a boolean.")

    return param_dict


def _resolve_color_panels(color: Any) -> tuple[Any, list[str] | None]:
    """Split a ``color`` argument into a scalar color and an optional multi-panel key list.

    Returns ``(scalar_color, panel_keys)``. When ``panel_keys`` is ``None`` the call is a
    normal single-color render and ``scalar_color`` is the (unchanged) color to use. When
    ``panel_keys`` is a list, the render must be expanded into one panel per key.

    A list of all-strings is treated as multi-panel keys; a length-1 list normalizes to a
    scalar color; an all-numeric list stays a single RGB(A) color. Empty, duplicate, or
    mixed str/number lists raise ``ValueError``.
    """
    if not isinstance(color, list):
        return color, None
    if all(isinstance(c, str) for c in color):
        if len(color) == 0:
            raise ValueError("`color` was given an empty list; provide at least one column/key name.")
        duplicate_keys = sorted(k for k, n in Counter(color).items() if n > 1)
        if duplicate_keys:
            raise ValueError(f"`color` contains duplicate keys {duplicate_keys}; each multi-panel key must be unique.")
        if len(color) == 1:
            return color[0], None
        return None, list(color)
    if any(isinstance(c, str) for c in color):
        raise ValueError(
            "`color` list must be either all column/key names (str) for a multi-panel plot, "
            "or 3-4 floats for a single RGB(A) color, not a mix of both."
        )
    return color, None


def _expand_color_panels(
    sdata: SpatialData,
    color: Any,
    render_fn_name: str,
    validate: Callable[[Any], dict[str, Any]],
) -> list[tuple[str | None, dict[str, Any]]]:
    """Resolve ``color`` into validated per-panel render params for the multi-panel ``color=[...]`` feature.

    ``validate`` is a callback that runs the render function's own parameter validation for a single
    color value and returns its per-element ``params_dict``. Returns a list of ``(panel_key, params_dict)``
    pairs: a single ``(None, params_dict)`` for the scalar case, or one entry per key for a key list.

    Enforces that only one ``render_*`` call per figure may pass a color list, and aggregates per-key
    validation errors into a single message. Used by ``render_shapes`` and ``render_labels``.
    """
    color, panel_keys = _resolve_color_panels(color)
    if panel_keys is not None and any(
        getattr(params, "panel_key", None) is not None for params in getattr(sdata, "plotting_tree", {}).values()
    ):
        raise ValueError(
            "Only one `render_*` call may use a list of color keys per figure. Other chained render "
            "calls must use a single (scalar) color; they are drawn into every panel as a shared layer."
        )

    color_specs = [(None, color)] if panel_keys is None else [(key, key) for key in panel_keys]
    panel_param_dicts: list[tuple[str | None, dict[str, Any]]] = []
    key_errors: dict[str, str] = {}
    for panel_key, color_value in color_specs:
        try:
            params_dict = validate(color_value)
        except (KeyError, ValueError) as e:
            if panel_keys is None:
                raise
            key_errors[panel_key] = str(e)  # type: ignore[index]
            continue
        panel_param_dicts.append((panel_key, params_dict))
    if key_errors:
        details = "\n".join(f"  - {key!r}: {msg}" for key, msg in key_errors.items())
        raise ValueError(f"Invalid color key(s) for multi-panel `{render_fn_name}`:\n{details}")
    return panel_param_dicts


def _validate_as_points_size(size: float) -> None:
    """Validate the centroid marker `size` used by ``render_shapes``/``render_labels`` with ``as_points=True``."""
    if isinstance(size, bool) or not isinstance(size, (int, float)):
        raise TypeError("Parameter 'size' must be numeric.")
    if size <= 0:
        raise ValueError("Parameter 'size' must be a positive number.")


def _validate_label_render_params(
    sdata: sd.SpatialData,
    element: str | None,
    cmap: list[Colormap | str] | Colormap | str | None,
    color: ColorLike | None,
    fill_alpha: float | int | None,
    contour_px: int | None,
    groups: list[str] | str | None,
    palette: dict[str, str] | list[str] | str | None,
    na_color: ColorLike | None,
    norm: Normalize | None,
    outline_alpha: float | int,
    outline_color: ColorLike | None,
    scale: str | None,
    table_name: str | None,
    table_layer: str | None,
    colorbar: bool | str | None,
    colorbar_params: dict[str, object] | None,
    gene_symbols: str | None = None,
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
        "outline_color": outline_color,
        "cmap": cmap,
        "norm": norm,
        "scale": scale,
        "table_name": table_name,
        "table_layer": table_layer,
        "colorbar": colorbar,
        "colorbar_params": colorbar_params,
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
        element_params[el]["outline_color"] = param_dict["outline_color"]
        element_params[el]["contour_px"] = param_dict["contour_px"]
        element_params[el]["table_layer"] = param_dict["table_layer"]

        element_params[el]["table_name"] = None
        element_params[el]["color"] = param_dict["color"]  # literal Color or None
        element_params[el]["col_for_color"] = None
        if (col_for_color := param_dict["col_for_color"]) is not None:
            col_for_color, table_name = _validate_col_for_column_table(
                sdata, el, col_for_color, param_dict["table_name"], labels=True, gene_symbols=gene_symbols
            )
            element_params[el]["table_name"] = table_name
            element_params[el]["col_for_color"] = col_for_color

        element_params[el]["col_for_outline_color"] = None
        element_params[el]["outline_table_name"] = None
        if (col_for_outline_color := param_dict.get("col_for_outline_color")) is not None:
            col_for_outline_color, outline_table_name = _validate_col_for_column_table(
                sdata,
                el,
                col_for_outline_color,
                param_dict["table_name"],
                labels=True,
                gene_symbols=gene_symbols,
            )
            element_params[el]["col_for_outline_color"] = col_for_outline_color
            element_params[el]["outline_table_name"] = outline_table_name

        _gate_palette_and_groups(element_params[el], param_dict)
        element_params[el]["colorbar"] = param_dict["colorbar"]
        element_params[el]["colorbar_params"] = param_dict["colorbar_params"]

    return element_params


def _validate_points_render_params(
    sdata: sd.SpatialData,
    element: str | None,
    alpha: float | int | None,
    color: ColorLike | None,
    groups: list[str] | str | None,
    palette: dict[str, str] | list[str] | str | None,
    na_color: ColorLike | None,
    cmap: list[Colormap | str] | Colormap | str | None,
    norm: Normalize | None,
    size: float | int,
    table_name: str | None,
    table_layer: str | None,
    ds_reduction: str | None,
    colorbar: bool | str | None,
    colorbar_params: dict[str, object] | None,
    gene_symbols: str | None = None,
    density: bool = False,
    density_how: Literal["linear", "log", "cbrt", "eq_hist"] = "linear",
    transfunc: Callable[[float], float] | None = None,
    method: str | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(density, bool):
        raise TypeError("Parameter 'density' must be a bool.")
    allowed_how = ("linear", "log", "cbrt", "eq_hist")
    if density_how not in allowed_how:
        raise ValueError(f"Parameter 'density_how' must be one of {allowed_how}; got {density_how!r}.")

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
        "colorbar": colorbar,
        "colorbar_params": colorbar_params,
    }
    param_dict = _type_check_params(param_dict, "points")

    if density:
        if method == "matplotlib":
            raise ValueError(
                "density=True requires the datashader backend; got method='matplotlib'. "
                "Either drop method= or set method='datashader'."
            )
        # Literal color (resolved into param_dict["color"] as a Color instance, with
        # col_for_color set to None) is ambiguous with density: it could mean a
        # single-hue cmap or a one-entry palette. Force the user to choose.
        if param_dict["color"] is not None and param_dict["col_for_color"] is None:
            raise ValueError(
                "density=True with a literal color is ambiguous. Pass cmap= to recolor the "
                "density, or palette= to assign a categorical color, but not color=<literal>."
            )
        # Warn-and-ignore: these parameters do not interact meaningfully with a
        # count-based density and are silently dropped to keep the API consistent.
        if size != 1.0:
            warnings.warn(
                "size is ignored when density=True; spreading would distort the count signal.",
                UserWarning,
                stacklevel=3,
            )
        if transfunc is not None:
            warnings.warn(
                "transfunc is ignored when density=True (no continuous color vector to transform).",
                UserWarning,
                stacklevel=3,
            )
        if isinstance(norm, Normalize) and (norm.vmin is not None or norm.vmax is not None):
            warnings.warn(
                "norm.vmin/vmax are ignored when density=True; use density_how= to control intensity mapping.",
                UserWarning,
                stacklevel=3,
            )
        if ds_reduction is not None:
            warnings.warn(
                "datashader_reduction is ignored when density=True; counts are forced.",
                UserWarning,
                stacklevel=3,
            )

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
        col_for_color = param_dict["col_for_color"]
        if col_for_color is not None:
            col_for_color, table_name = _validate_col_for_column_table(
                sdata, el, col_for_color, param_dict["table_name"], gene_symbols=gene_symbols
            )
            element_params[el]["table_name"] = table_name
            element_params[el]["col_for_color"] = col_for_color

        _gate_palette_and_groups(element_params[el], param_dict)
        element_params[el]["ds_reduction"] = param_dict["ds_reduction"]
        element_params[el]["colorbar"] = param_dict["colorbar"]
        element_params[el]["colorbar_params"] = param_dict["colorbar_params"]

    return element_params


def _validate_shape_render_params(
    sdata: sd.SpatialData,
    element: str | None,
    fill_alpha: float | int | None,
    groups: list[str] | str | None,
    palette: dict[str, str] | list[str] | str | None,
    color: ColorLike | None,
    na_color: ColorLike | None,
    outline_width: float | int | tuple[float | int, float | int] | None,
    outline_color: ColorLike | tuple[ColorLike] | None,
    outline_alpha: float | int | tuple[float | int, float | int] | None,
    cmap: list[Colormap | str] | Colormap | str | None,
    norm: Normalize | None,
    scale: float | int,
    table_name: str | None,
    table_layer: str | None,
    shape: Literal["circle", "hex", "visium_hex", "square"] | None,
    method: str | None,
    ds_reduction: str | None,
    colorbar: bool | str | None,
    colorbar_params: dict[str, object] | None,
    gene_symbols: str | None = None,
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
        "shape": shape,
        "method": method,
        "ds_reduction": ds_reduction,
        "colorbar": colorbar,
        "colorbar_params": colorbar_params,
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
        element_params[el]["shape"] = param_dict["shape"]

        element_params[el]["color"] = param_dict["color"]

        element_params[el]["table_name"] = None
        element_params[el]["col_for_color"] = None
        col_for_color = param_dict["col_for_color"]
        if col_for_color is not None:
            col_for_color, table_name = _validate_col_for_column_table(
                sdata, el, col_for_color, param_dict["table_name"], gene_symbols=gene_symbols
            )
            element_params[el]["table_name"] = table_name
            element_params[el]["col_for_color"] = col_for_color

        element_params[el]["col_for_outline_color"] = None
        element_params[el]["outline_table_name"] = None
        col_for_outline_color = param_dict.get("col_for_outline_color")
        if col_for_outline_color is not None:
            col_for_outline_color, outline_table_name = _validate_col_for_column_table(
                sdata, el, col_for_outline_color, param_dict["table_name"], gene_symbols=gene_symbols
            )
            element_params[el]["col_for_outline_color"] = col_for_outline_color
            element_params[el]["outline_table_name"] = outline_table_name

        _gate_palette_and_groups(element_params[el], param_dict)
        element_params[el]["method"] = param_dict["method"]
        element_params[el]["ds_reduction"] = param_dict["ds_reduction"]
        element_params[el]["colorbar"] = param_dict["colorbar"]
        element_params[el]["colorbar_params"] = param_dict["colorbar_params"]

    return element_params


def _resolve_gene_symbols(
    adata: AnnData,
    col_for_color: str,
    gene_symbols: str,
) -> str:
    """Resolve a gene symbol to its var_name using an alternate var column.

    Mimics scanpy's ``gene_symbols`` behaviour: look up *col_for_color* in
    ``adata.var[gene_symbols]`` and return the corresponding ``var_name``
    (i.e. the var index value).
    """
    if gene_symbols not in adata.var.columns:
        raise KeyError(f"Column '{gene_symbols}' not found in `adata.var`. Cannot use it as `gene_symbols` lookup.")
    mask = adata.var[gene_symbols] == col_for_color
    if not mask.any():
        raise KeyError(f"'{col_for_color}' not found in `adata.var['{gene_symbols}']`.")
    n_matches = mask.sum()
    if n_matches > 1:
        logger.warning(
            f"Gene symbol '{col_for_color}' maps to {n_matches} var_names in column '{gene_symbols}'. "
            f"Using the first match: '{adata.var.index[mask][0]}'."
        )
    return str(adata.var.index[mask][0])


def _validate_graph_render_params(
    sdata: SpatialData,
    element: str | None,
    connectivity_key: str,
    table_name: str | None,
    color: ColorLike | None,
    edge_width: float | Literal["weight"],
    edge_alpha: float | Literal["weight"],
    groups: list[str] | str | None,
    group_key: str | None,
    obsp_key: str | None = None,
    weight_key: str | None = None,
    palette: dict[str, str] | list[str] | str | None = None,
    na_color: ColorLike | None = "default",
    cmap: Colormap | str | None = None,
    norm: Normalize | None = None,
    linestyle: str | Sequence[str] = "solid",
    include_self_loops: bool = False,
    rasterize: bool = True,
) -> dict[str, Any]:
    """Validate and resolve parameters for render_graph."""
    from spatialdata_plot.pl._color import _get_colors_for_categorical_obs, _prepare_cmap_norm

    param_dict: dict[str, Any] = {
        "sdata": sdata,
        "element": element,
        "color": color,
        "groups": groups,
        "palette": palette,
        "na_color": na_color,
        "cmap": cmap,
        "norm": norm if norm is not None else Normalize(clip=False),
        "table_name": table_name,
        "connectivity_key": connectivity_key,
        "obsp_key": obsp_key,
        "weight_key": weight_key,
        "group_key": group_key,
        "edge_width": edge_width,
        "edge_alpha": edge_alpha,
        "linestyle": linestyle,
        "include_self_loops": include_self_loops,
        "rasterize": rasterize,
    }
    param_dict = _type_check_params(param_dict, "graph")

    if param_dict["table_name"] is None:
        candidates = [tname for tname in sdata.tables if _resolve_obsp_key(sdata[tname], connectivity_key) is not None]
        if len(candidates) == 0:
            raise ValueError(
                f"No table found with connectivity key '{connectivity_key}' in obsp. "
                f"Available tables: {list(sdata.tables.keys())}."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple tables contain connectivity key '{connectivity_key}': {candidates}. "
                "Please specify `table_name` explicitly."
            )
        param_dict["table_name"] = candidates[0]

    if param_dict["table_name"] not in sdata.tables:
        raise KeyError(f"Table '{param_dict['table_name']}' not found. Available: {list(sdata.tables.keys())}.")

    table = sdata[param_dict["table_name"]]
    connectivity_obsp_key = _require_obsp_key(table, connectivity_key, param_name="connectivity_key")

    _, region_key, _ = get_table_keys(table)
    if region_key is None:
        raise ValueError(
            f"Table '{param_dict['table_name']}' has no `region_key`; cannot associate its observations "
            "with a spatial element. Re-parse the table with `TableModel.parse(..., region_key=...)`."
        )

    if param_dict["element"] is None:
        regions = table.obs[region_key].unique().tolist()
        spatial_regions = [r for r in regions if r in sdata.shapes or r in sdata.points or r in sdata.labels]
        if len(spatial_regions) == 0:
            raise ValueError(
                f"Table '{param_dict['table_name']}' does not annotate any spatial element. Region values: {regions}."
            )
        if len(spatial_regions) > 1:
            raise ValueError(
                f"Table '{param_dict['table_name']}' annotates multiple spatial elements: {spatial_regions}. "
                "Please specify `element` explicitly."
            )
        param_dict["element"] = spatial_regions[0]
    elif not (
        param_dict["element"] in sdata.shapes
        or param_dict["element"] in sdata.points
        or param_dict["element"] in sdata.labels
    ):
        raise KeyError(
            f"Element '{param_dict['element']}' not found in shapes, points, or labels. "
            f"Available: shapes={list(sdata.shapes.keys())}, "
            f"points={list(sdata.points.keys())}, labels={list(sdata.labels.keys())}."
        )

    # _type_check_params normalised string groups → list; renormalise the working set here.
    if param_dict["groups"] is not None and param_dict["group_key"] is None:
        raise ValueError("`groups` requires `group_key` to be specified.")
    if param_dict["group_key"] is not None and param_dict["group_key"] not in table.obs.columns:
        raise KeyError(
            f"`group_key='{param_dict['group_key']}'` not found in table obs columns. "
            f"Available: {list(table.obs.columns)}."
        )
    if param_dict["groups"] is not None and param_dict["group_key"] is not None:
        groups_set: set[Any] = set(param_dict["groups"])
        available_groups = set(table.obs[param_dict["group_key"]].dropna().unique())
        missing_groups = groups_set - available_groups
        if missing_groups:
            try:
                missing_str = str(sorted(missing_groups))
            except TypeError:
                missing_str = str(list(missing_groups))
            if missing_groups == groups_set:
                logger.warning(
                    f"None of the requested groups {missing_str} were found in column "
                    f"'{param_dict['group_key']}'. Resulting plot will contain no edges."
                )
            else:
                logger.warning(
                    f"Groups {missing_str} not found in column '{param_dict['group_key']}' and will be ignored."
                )

    # After _type_check_params: col_for_color is the non-color string user passed via `color=`;
    # color is either a Color (user gave a real color) or None (user gave a column name or nothing).
    col_for_color = param_dict.get("col_for_color")
    if col_for_color is not None and col_for_color not in table.obs.columns:
        raise ValueError(
            f"`color='{col_for_color}'` is not a matplotlib color and was not found in "
            f"`table.obs` columns. Available obs columns: {list(table.obs.columns)}."
        )

    color_is_obs_col = col_for_color is not None
    if obsp_key is not None and color_is_obs_col:
        raise ValueError(
            "Cannot set both `color` (as an obs column) and `obsp_key` for edge coloring. "
            "Pick one source: scalar color, obs-column color, or obsp-matrix color."
        )
    if obsp_key is not None and param_dict["color"] is not None:
        raise ValueError(
            "Cannot set both `color` and `obsp_key` for edge coloring. "
            "Use `obsp_key` for matrix-driven coloring with `cmap`/`norm`, "
            "or `color` for a scalar / obs-column-driven coloring."
        )

    color_obsp_key: str | None = None
    obs_col: str | None = None
    color_source: str = "scalar"
    cmap_params: CmapParams | None = None
    palette_map: dict[str, str] | None = None

    if obsp_key is not None:
        color_obsp_key = _require_obsp_key(table, obsp_key, param_name="obsp_key")
        color_source = "obsp"
        cmap_params = _prepare_cmap_norm(cmap=cmap, norm=param_dict["norm"])
    elif color_is_obs_col:
        obs_col = col_for_color
        obs_values = table.obs[obs_col]
        if isinstance(obs_values.dtype, pd.CategoricalDtype) or obs_values.dtype == object:
            color_source = "obs_categorical"
            categories = (
                obs_values.cat.categories.tolist()
                if isinstance(obs_values.dtype, pd.CategoricalDtype)
                else sorted(obs_values.dropna().unique().tolist())
            )
            if isinstance(palette, dict):
                missing = [c for c in categories if c not in palette]
                if missing:
                    raise KeyError(
                        f"Palette dict is missing entries for categories: {missing}. "
                        f"Available categories: {categories}."
                    )
                palette_map = {c: palette[c] for c in categories}
            else:
                cat_colors = _get_colors_for_categorical_obs(categories=categories, palette=palette)
                palette_map = dict(zip(categories, cat_colors, strict=True))
        else:
            color_source = "obs_continuous"
            cmap_params = _prepare_cmap_norm(cmap=cmap, norm=param_dict["norm"])

    # When edge_width/edge_alpha="weight" but weight_key isn't given, fall back to the
    # connectivity matrix so binary graphs still produce a per-edge array.
    resolved_weight_key: str | None = None
    if edge_width == "weight" or edge_alpha == "weight":
        resolved_weight_key = _require_obsp_key(
            table, weight_key if weight_key is not None else connectivity_key, param_name="weight_key"
        )

    edge_color = param_dict["color"] if param_dict["color"] is not None else Color("grey")
    parsed_na_color = param_dict["na_color"]

    return {
        "element": param_dict["element"],
        "connectivity_key": connectivity_key,
        "connectivity_obsp_key": connectivity_obsp_key,
        "obsp_key": color_obsp_key,
        "obs_col": obs_col,
        "cmap_params": cmap_params,
        "palette_map": palette_map,
        "na_color": parsed_na_color,
        "color_source": color_source,
        "table_name": param_dict["table_name"],
        "weight_key": resolved_weight_key,
        "color": edge_color,
        "edge_width": edge_width,
        "edge_alpha": edge_alpha,
        "groups": param_dict["groups"],
        "group_key": param_dict["group_key"],
    }


def _resolve_obsp_key(table: AnnData, connectivity_key: str) -> str | None:
    """Resolve connectivity_key to an actual obsp key. Accepts full key or prefix."""
    if connectivity_key in table.obsp:
        return connectivity_key
    suffixed = f"{connectivity_key}_connectivities"
    if suffixed in table.obsp:
        return suffixed
    return None


def _require_obsp_key(table: AnnData, key: str, *, param_name: str) -> str:
    """Resolve key (with prefix fallback) or raise KeyError."""
    resolved = _resolve_obsp_key(table, key)
    if resolved is None:
        raise KeyError(
            f"`{param_name}='{key}'` not found in `table.obsp`. "
            f"Tried '{key}' and '{key}_connectivities'. "
            f"Available obsp keys: {list(table.obsp.keys())}."
        )
    return resolved


def _validate_col_for_column_table(
    sdata: SpatialData,
    element_name: str,
    col_for_color: str | None,
    table_name: str | None,
    labels: bool = False,
    gene_symbols: str | None = None,
) -> tuple[str | None, str | None]:
    if col_for_color is None:
        return None, None

    if not labels and col_for_color in sdata[element_name].columns and table_name is None:
        return col_for_color, None
    if table_name is not None:
        tables = get_element_annotators(sdata, element_name)
        if table_name not in tables:
            logger.warning(f"Table '{table_name}' does not annotate element '{element_name}'.")
            raise KeyError(f"Table '{table_name}' does not annotate element '{element_name}'.")
        if col_for_color not in sdata[table_name].obs.columns and col_for_color not in sdata[table_name].var_names:
            if gene_symbols is not None:
                col_for_color = _resolve_gene_symbols(sdata[table_name], col_for_color, gene_symbols)
            else:
                raise KeyError(
                    f"Column '{col_for_color}' not found in obs/var of table '{table_name}' "
                    f"for element '{element_name}'."
                )
    else:
        tables = get_element_annotators(sdata, element_name)
        if len(tables) == 0:
            raise KeyError(
                f"Element '{element_name}' has no annotating tables. "
                f"Cannot use column '{col_for_color}' for coloring. "
                "Please ensure the element is annotated by at least one table."
            )
        # Now check which tables contain the column
        resolved_var_name: str | None = None
        if gene_symbols is not None and not any(gene_symbols in sdata[t].var.columns for t in tables):
            available = sorted({c for t in tables for c in sdata[t].var.columns})
            raise KeyError(
                f"Column '{gene_symbols}' specified in `gene_symbols=` was not found in "
                f"`adata.var` of any table annotating element '{element_name}'. "
                f"Available var columns: {available}"
            )
        for annotates in tables.copy():
            if col_for_color not in sdata[annotates].obs.columns and col_for_color not in sdata[annotates].var_names:
                if gene_symbols is not None:
                    try:
                        resolved_var_name = _resolve_gene_symbols(sdata[annotates], col_for_color, gene_symbols)
                    except KeyError:
                        tables.remove(annotates)
                else:
                    tables.remove(annotates)
        if len(tables) == 0:
            raise KeyError(
                f"Unable to locate color key '{col_for_color}' for element '{element_name}'. "
                "Please ensure the key exists in a table annotating this element."
            )
        table_name = next(iter(tables))
        if len(tables) > 1:
            logger.warning(f"Multiple tables contain column '{col_for_color}', using table '{table_name}'.")
        if resolved_var_name is not None:
            col_for_color = resolved_var_name
    return col_for_color, table_name


def _validate_image_render_params(
    sdata: sd.SpatialData,
    element: str | None,
    channel: list[str] | list[int] | str | int | None,
    alpha: float | int | None,
    palette: list[str] | str | None,
    cmap: list[Colormap | str] | Colormap | str | None,
    norm: list[Normalize] | Normalize | None,
    scale: str | None,
    colorbar: bool | str | None,
    colorbar_params: dict[str, object] | None,
) -> dict[str, dict[str, Any]]:
    param_dict: dict[str, Any] = {
        "sdata": sdata,
        "element": element,
        "channel": channel,
        "alpha": alpha,
        "palette": palette,
        "cmap": cmap,
        "norm": norm,
        "scale": scale,
        "colorbar": colorbar,
        "colorbar_params": colorbar_params,
    }
    param_dict = _type_check_params(param_dict, "images")

    element_params: dict[str, dict[str, Any]] = {}
    for el in param_dict["element"]:
        element_params[el] = {}
        spatial_element = param_dict["sdata"][el]

        # robustly get channel names from image or multiscale image
        spatial_element_ch = (
            spatial_element.c.values if isinstance(spatial_element, DataArray) else spatial_element["scale0"].c.values
        )
        channel = param_dict["channel"]
        if channel is not None:
            # Normalize channel to always be a list of str or a list of int
            if isinstance(channel, str):
                channel = [channel]

            if isinstance(channel, int):
                channel = [channel]

            # If channel is a list, ensure all elements are the same type
            if not (isinstance(channel, list) and channel and all(isinstance(c, type(channel[0])) for c in channel)):
                raise TypeError("Each item in 'channel' list must be of the same type, either string or integer.")

            invalid = [c for c in channel if c not in spatial_element_ch]
            if invalid:
                raise ValueError(
                    f"Invalid channel(s): {', '.join(str(c) for c in invalid)}. Valid choices are: {spatial_element_ch}"
                )
            element_params[el]["channel"] = channel
        else:
            element_params[el]["channel"] = None

        element_params[el]["alpha"] = param_dict["alpha"]

        palette = param_dict["palette"]
        assert isinstance(palette, list | type(None))  # if present, was converted to list, just to make sure

        if isinstance(palette, list):
            # case A: single palette for all channels
            if len(palette) == 1:
                palette_length = len(channel) if channel is not None else len(spatial_element_ch)
                palette = palette * palette_length
            # case B: one palette per channel (either given or derived from channel length)
            channels_to_use = spatial_element_ch if element_params[el]["channel"] is None else channel
            if channels_to_use is not None and len(palette) != len(channels_to_use):
                raise ValueError(
                    f"Palette length ({len(palette)}) does not match channel length "
                    f"({', '.join(str(c) for c in channels_to_use)})."
                )
        element_params[el]["palette"] = palette

        expected_len = len(channel) if channel is not None else len(spatial_element_ch)

        cmap = param_dict["cmap"]
        if cmap is not None:
            if len(cmap) == 1:
                cmap = cmap * expected_len
            if len(cmap) != expected_len:
                raise ValueError(
                    f"Length of 'cmap' list ({len(cmap)}) must match the number of channels ({expected_len})."
                )
        element_params[el]["cmap"] = cmap

        norm = param_dict["norm"]
        if isinstance(norm, list) and len(norm) > 1 and len(norm) != expected_len:
            raise ValueError(f"Length of 'norm' list ({len(norm)}) must match the number of channels ({expected_len}).")
        element_params[el]["norm"] = norm
        scale = param_dict["scale"]
        if scale and isinstance(param_dict["sdata"][el], DataTree):
            valid_scales = list(param_dict["sdata"][el].keys())
            if scale not in valid_scales and scale != "full":
                raise ValueError(
                    f"Scale '{scale}' does not exist in image '{el}'. Valid scales: {valid_scales + ['full']}."
                )
        element_params[el]["scale"] = scale
        element_params[el]["colorbar"] = param_dict["colorbar"]
        element_params[el]["colorbar_params"] = param_dict["colorbar_params"]

    return element_params


def _get_wanted_render_elements(
    sdata: SpatialData,
    sdata_wanted_elements: list[str],
    params: ImageRenderParams | LabelsRenderParams | PointsRenderParams | ShapesRenderParams,
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


# --- Per-cell measurements into the annotating table (centroid / area / equivalent diameter) ---

# Destination keys (measurements are stored intrinsic, coordinate-system independent).
# obsm["spatial"] is the squidpy/scanpy convention for per-cell coordinates (an N x 2 array).
_CENTROID_OBSM_KEY = "spatial"
_AREA_OBS_KEY = "area"
_DIAMETER_OBS_KEY = "equivalent_diameter"


def _pixel_to_coord(idx: ArrayLike, coord: ArrayLike) -> ArrayLike:
    """Map fractional pixel indices to intrinsic coordinates along one axis (handles non-unit spacing)."""
    spacing = (coord[1] - coord[0]) if len(coord) > 1 else 1.0
    return coord[0] + np.asarray(idx) * spacing


def _stream_label_centroid_stats(data: Any) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Per-label ``(labels, mean_x_index, mean_y_index, area)`` via a streaming bincount aggregator.

    Streams the raster block by block — one chunk in memory at a time for a dask array, a
    bounded row-block at a time for a numpy array — accumulating per-label ``count`` (= area),
    ``sum_x`` and ``sum_y``. Labels are relabelled to a dense ``0..k-1`` range, so memory is
    O(number of distinct labels), independent of the raster size *and* of the label-id magnitude
    (sparse/global ids do not blow it up). The reduction is additive, so it is exact across block
    boundaries. Background label 0 is excluded.
    """
    n_rows, n_cols = data.shape
    if hasattr(data, "chunks"):  # dask
        block_slices = slices_from_chunks(data.chunks)
    else:
        data = np.asarray(data)
        # bound the per-block coordinate-weight arrays to ~8M pixels
        step = max(1, min(n_rows, (8 << 20) // max(1, n_cols)))
        block_slices = [(slice(r0, min(r0 + step, n_rows)), slice(0, n_cols)) for r0 in range(0, n_rows, step)]

    def _load(row_sl: slice, col_sl: slice) -> np.ndarray:
        block = data[row_sl, col_sl]
        block = np.asarray(block.compute() if hasattr(block, "compute") else block)
        # label rasters are integer-valued even when stored as float; cast so np.unique/searchsorted
        # stay integer (and the dense bincount indices below are always int).
        return block.astype(np.int64) if block.dtype.kind == "f" else block

    # Pass 1: the sorted set of present label values -> dense relabelling (keeps memory O(n_labels)).
    uniq = np.zeros(0, dtype=np.int64)
    for row_sl, col_sl in block_slices:
        uniq = np.union1d(uniq, np.unique(_load(row_sl, col_sl)))
    k = uniq.size

    # Pass 2: additive per-(dense-)label count / sum_x / sum_y.
    count = np.zeros(k)
    sum_x = np.zeros(k)
    sum_y = np.zeros(k)
    for row_sl, col_sl in block_slices:
        block = _load(row_sl, col_sl)
        block_rows, block_cols = block.shape
        idx = np.searchsorted(uniq, block.reshape(-1))  # dense 0..k-1 indices (always int)
        cols = np.tile(np.arange(col_sl.start, col_sl.start + block_cols, dtype=np.float64), block_rows)
        rows = np.repeat(np.arange(row_sl.start, row_sl.start + block_rows, dtype=np.float64), block_cols)
        count += np.bincount(idx, minlength=k)
        sum_x += np.bincount(idx, weights=cols, minlength=k)
        sum_y += np.bincount(idx, weights=rows, minlength=k)

    keep = uniq != 0  # drop background; every kept label has count >= 1
    return uniq[keep], sum_x[keep] / count[keep], sum_y[keep] / count[keep], count[keep]


def _compute_element_measurements(sdata: SpatialData, element_name: str) -> pd.DataFrame:
    """One row per instance with intrinsic ``["x", "y", "area"]``, indexed by instance id.

    Shapes use shapely's vectorized centroid; circles (``Point`` geometry + ``radius``) have
    ``area = pi*r**2`` (shapely ``.area`` is 0 for them), polygons use the geometric area. 2D labels
    use the streaming bincount aggregator (``area`` = pixel count) — it holds one chunk plus
    O(n_labels) accumulators, so it scales to Xenium-size masks where a whole-array ``regionprops``
    table would run out of memory. Area meaning differs across element types but is consistent within
    one element.
    """
    element = sdata[element_name]
    model = get_model(element)
    if model is ShapesModel:
        geometry = element.geometry
        centroids = geometry.centroid
        # Dispatch on geometry TYPE, not a column name: circles are Point geometries (shapely .area
        # is 0 for them) with a radius -> pi*r**2; everything else uses the true geometric area.
        if (geometry.geom_type == "Point").all():
            area = np.pi * np.asarray(element["radius"], dtype=float) ** 2
        else:
            area = geometry.area.to_numpy()
        return pd.DataFrame(
            {"x": centroids.x.to_numpy(), "y": centroids.y.to_numpy(), "area": area}, index=element.index
        )
    if model is Labels2DModel:
        # multiscale rasters carry their data on the scale0 level
        raster = next(iter(element["scale0"].values())) if isinstance(element, DataTree) else element
        labels, x_idx, y_idx, area = _stream_label_centroid_stats(raster.data)
        # bincount gives mean 0-based pixel indices; map them onto the raster's intrinsic coords.
        return pd.DataFrame(
            {
                "x": _pixel_to_coord(x_idx, raster.coords["x"].values),
                "y": _pixel_to_coord(y_idx, raster.coords["y"].values),
                "area": np.asarray(area, dtype=float),
            },
            index=labels,
        )
    raise NotImplementedError(
        f"Measurement is only supported for shapes and 2D labels; element {element_name!r} is a {model.__name__}."
    )


def _valid_spatial_obsm(arr: ArrayLike, n_obs: int) -> bool:
    """Whether ``arr`` is a usable ``obsm["spatial"]``: a 2D ``(n_obs, 2)`` coordinate grid."""
    return bool(arr.ndim == 2 and arr.shape == (n_obs, 2))


def _obsm_region_finite(table: AnnData, key: str, mask: ArrayLike) -> bool:
    """Whether ``obsm[key]`` already holds finite coords for every ``mask`` row (already populated)."""
    if key not in table.obsm:
        return False
    arr = np.asarray(table.obsm[key])
    if not _valid_spatial_obsm(arr, table.n_obs):
        return False
    region = arr[mask].astype(float)
    return bool(region.size and np.isfinite(region).all())


def _check_obs_numeric(table: AnnData, key: str) -> None:
    """Raise if ``obs[key]`` exists but is non-numeric, before any mutation (avoids half-writes)."""
    if key in table.obs and not is_numeric_dtype(table.obs[key]):
        raise ValueError(
            f"Cannot write measurements into obs[{key!r}]: the existing column is "
            f"{table.obs[key].dtype} (not numeric). Drop or rename the column first."
        )


def _write_region(table: AnnData, mask: ArrayLike, key: str, values: ArrayLike, *, obsm: bool) -> None:
    """Write ``values`` into ``obsm[key]`` (2D) or ``obs[key]`` (1D) at ``mask`` rows; others stay/NaN.

    Refuses to overwrite an incompatible existing ``obsm[key]`` (e.g. a 3-column xyz array) rather
    than silently dropping data.
    """
    store = table.obsm if obsm else table.obs
    if key in store:
        existing = np.asarray(store[key])
        if obsm and not _valid_spatial_obsm(existing, table.n_obs):
            raise ValueError(
                f"Refusing to overwrite obsm[{key!r}] with shape {existing.shape}; expected "
                f"({table.n_obs}, 2). Remove it first if you want it replaced."
            )
        arr = existing.astype(float, copy=True)
    else:
        arr = np.full((table.n_obs, 2) if obsm else table.n_obs, np.nan)
    arr[mask] = np.asarray(values, dtype=float)
    store[key] = arr


def _measure_into_table(
    sdata: SpatialData, element_name: str, table_name: str, *, centroids: bool, area: bool, diameter: bool
) -> None:
    """Compute and write the requested measurements for one element into its annotating table.

    Only the rows belonging to ``element_name`` are touched (a table may annotate several elements).
    Centroids already present for those rows (e.g. reader-provided ``obsm["spatial"]``) are not
    overwritten; ``area``/``diameter`` overwrite our own columns. All targets are validated before
    the first write, so a bad column never leaves the table half-written.
    """
    table = sdata.tables[table_name]
    _, region_key, instance_key = get_table_keys(table)
    mask = (table.obs[region_key].astype(str) == str(element_name)).to_numpy()
    if not mask.any():
        raise ValueError(f"Table {table_name!r} does not annotate element {element_name!r} (no matching rows).")

    # #1: never clobber centroids already populated for this element's rows (reader/prior-call coords).
    if centroids and _obsm_region_finite(table, _CENTROID_OBSM_KEY, mask):
        warnings.warn(
            f"obsm[{_CENTROID_OBSM_KEY!r}] is already populated for element {element_name!r}; not "
            f"overwriting its centroids (remove it to recompute).",
            UserWarning,
            stacklevel=3,
        )
        centroids = False
    if not (centroids or area or diameter):
        return

    keys = table.obs[instance_key].to_numpy()[mask]
    meas = _compute_element_measurements(sdata, element_name).reindex(keys)
    # #2: instance ids annotated in the table but absent from the element reindex to NaN -> warn.
    missing = int(meas[["x", "y"]].isna().any(axis=1).sum())
    if missing:
        warnings.warn(
            f"{missing}/{len(keys)} instances annotated for {element_name!r} have no match in the "
            f"element (instance-id dtype mismatch, e.g. str vs int?); writing NaN for them.",
            UserWarning,
            stacklevel=3,
        )

    # #4: validate obs targets up front so an existing non-numeric column raises before any write.
    if area:
        _check_obs_numeric(table, _AREA_OBS_KEY)
    if diameter:
        _check_obs_numeric(table, _DIAMETER_OBS_KEY)

    area_vals = meas["area"].to_numpy()
    if centroids:
        _write_region(table, mask, _CENTROID_OBSM_KEY, meas[["x", "y"]].to_numpy(), obsm=True)
    if area:
        _write_region(table, mask, _AREA_OBS_KEY, area_vals, obsm=False)
    if diameter:
        _write_region(table, mask, _DIAMETER_OBS_KEY, 2.0 * np.sqrt(area_vals / np.pi), obsm=False)


def _resolve_measure_table(sdata: SpatialData, element_name: str, table_name: str | None) -> str:
    """Resolve the single annotating table for ``element_name`` (where measurements are written)."""
    if table_name is not None:
        if table_name not in sdata.tables:
            raise KeyError(f"Table {table_name!r} not found in `sdata.tables`.")
        return table_name
    annotators = sorted(get_element_annotators(sdata, element_name))
    if not annotators:
        raise ValueError(
            f"Element {element_name!r} has no annotating table; per-cell measurements need a table "
            f"to write into. Pass `table_name=` or annotate the element first."
        )
    if len(annotators) > 1:
        raise ValueError(
            f"Element {element_name!r} is annotated by multiple tables ({', '.join(annotators)}); "
            f"pass `table_name=` to pick one."
        )
    return annotators[0]


def measure_obs(
    sdata: SpatialData,
    element: str | None = None,
    *,
    table_name: str | None = None,
    centroids: bool = True,
    area: bool = True,
    diameter: bool = True,
    inplace: bool = True,
) -> SpatialData | None:
    """Measure per-cell centroids, area and equivalent diameter into an element's annotating table.

    Computes one centroid, area and equivalent diameter per instance of a shapes or 2D-labels
    element and writes them, squidpy-style, into the annotating :class:`~anndata.AnnData` table:
    centroids go to ``table.obsm["spatial"]`` (an ``(n_obs, 2)`` array, the squidpy convention),
    area and diameter to ``table.obs["area"]`` and ``table.obs["equivalent_diameter"]``. Values are
    stored in the element's *intrinsic* pixel coordinates/units (which align directly with the
    element's own raster/geometry).
    Labels area is the pixel count; shapes area is ``geometry.area`` (``pi*r**2`` for circles);
    equivalent diameter is ``2 * sqrt(area / pi)``. Persisting them once lets later renders (and
    downstream tools such as squidpy) reuse them instead of recomputing.

    Centroids already present for an element's rows (e.g. a reader-provided ``obsm["spatial"]``) are
    **not** overwritten — a warning is emitted and that element's centroid write is skipped (remove
    the key to recompute); ``area``/``diameter`` overwrite our own columns. Instances annotated in
    the table but absent from the element are written as NaN with a warning. Per-cell measurements
    need a table to write into, so an element without an annotating table cannot be measured (this
    raises). The label path never densifies the raster — it streams it block by block with memory
    O(n_labels), scaling to Xenium-size masks.

    Parameters
    ----------
    sdata
        The ``SpatialData`` object holding the element and its annotating table.
    element
        Name of the shapes/2D-labels element to measure. If ``None``, every shapes/2D-labels
        element that has exactly one annotating table is measured.
    table_name
        Name of the annotating table to write into. If ``None``, it is inferred from the element's
        annotators (an error is raised when there are zero or several).
    centroids, area, diameter
        Which measurements to compute/write. At least one must be ``True``. They are written to
        ``obsm["spatial"]``, ``obs["area"]`` and ``obs["equivalent_diameter"]`` respectively.
    inplace
        If ``True`` (default), mutate ``sdata``'s table in place and return ``None``. If ``False``,
        operate on a deep copy and return the modified ``SpatialData``.

    Returns
    -------
    ``None`` if ``inplace`` is ``True``, otherwise the modified deep-copied ``SpatialData``.
    """
    if not (centroids or area or diameter):
        raise ValueError("Nothing to measure: set at least one of `centroids`, `area`, `diameter` to True.")
    target = sdata if inplace else sd_deepcopy(sdata)
    if element is None:
        # measure every shapes / 2D-labels element that has exactly one annotating table
        names = [
            n
            for n in list(target.shapes) + list(target.labels)
            if get_model(target[n]) in (ShapesModel, Labels2DModel) and len(get_element_annotators(target, n)) == 1
        ]
        if not names:
            raise ValueError(
                "No shapes/2D-labels element with a single annotating table was found; nothing to "
                "measure. Pass an explicit `element=` (and `table_name=` if ambiguous)."
            )
    else:
        names = [element]
    for name in names:
        table = _resolve_measure_table(target, name, table_name)
        _measure_into_table(target, name, table, centroids=centroids, area=area, diameter=diameter)
    return None if inplace else target


# --- Fast extent for axis-aligned transforms ------------------------------------------------------
# spatialdata's `get_extent(..., exact=True)` transforms every shapes/points geometry (O(N)) just to
# take a bounding box. For an axis-aligned transform (scale/flip/90deg-rotation/axis-swap + translation)
# the exact extent equals the bbox of the *transformed corners*, so we transform 4 corners instead;
# rotation/shear and other element types fall back to `get_extent`.


def _is_axis_aligned(linear2x2: ArrayLike, *, rtol: float = 1e-9) -> bool:
    """Whether a 2x2 linear map sends axis-aligned boxes to axis-aligned boxes.

    True for a *monomial matrix* (at most one non-zero per row and per column): scale, axis flips,
    90/180/270-degree rotations and axis swaps. For such maps the exact extent equals the bounding box
    of the transformed corners. A relative tolerance ignores floating-point noise in the affine matrix.
    """
    m = np.abs(np.asarray(linear2x2, dtype=float))
    nz = m > rtol * (m.max() or 1.0)
    return bool((nz.sum(0) <= 1).all() and (nz.sum(1) <= 1).all() and int(nz.sum()) == m.shape[0])


def _element_extent_fast(
    element: Any, coordinate_system: str, *, transformations: Mapping[str, Any] | None = None
) -> dict[str, tuple[float, float]] | None:
    """Extent of one shapes/points element in ``coordinate_system`` via corner-transform.

    Returns ``None`` to fall back to ``get_extent`` for an unsupported type, a non-axis-aligned
    transform, or an anisotropically-scaled circle (an ellipse, but spatialdata stores one radius, so
    cheap and exact agree only under isotropic scale). ``transformations`` may pass a pre-fetched
    ``get_transformation(element, get_all=True)`` to avoid re-reading it.
    """
    model = get_model(element)
    if model not in (ShapesModel, PointsModel):
        return None
    if transformations is None:
        transformations = get_transformation(element, get_all=True)
    matrix = transformations[coordinate_system].to_affine_matrix(("x", "y"), ("x", "y"))
    affine = matrix[:2, :2]
    if not _is_axis_aligned(affine):
        return None

    if model is PointsModel:
        x, y = element["x"], element["y"]
        xmin, ymin, xmax, ymax = (float(v) for v in dask.compute(x.min(), y.min(), x.max(), y.max()))
    else:  # ShapesModel
        geom = element.geometry
        if (geom.geom_type == "Point").all():  # circles
            a = np.abs(affine)
            nz = a[a > 1e-9 * (a.max() or 1.0)]
            if not np.allclose(nz, nz[0]):  # anisotropic -> radius handling diverges from spatialdata
                return None
            x, y = geom.x.to_numpy(), geom.y.to_numpy()
            r = np.asarray(element["radius"], dtype=float)
            xmin = float(np.nanmin(x - r))
            ymin = float(np.nanmin(y - r))
            xmax = float(np.nanmax(x + r))
            ymax = float(np.nanmax(y + r))
        else:  # polygons / multipolygons
            xmin, ymin, xmax, ymax = (float(v) for v in geom.total_bounds)  # C-level union; skips empties

    if not np.isfinite((xmin, ymin, xmax, ymax)).all():  # all-empty element: defer to get_extent's clear error
        return None
    corners = np.array([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])
    tc = corners @ affine.T + matrix[:2, 2]
    return {"x": (float(tc[:, 0].min()), float(tc[:, 0].max())), "y": (float(tc[:, 1].min()), float(tc[:, 1].max()))}


def _fast_extent(element: Any, coordinate_system: str) -> dict[str, tuple[float, float]]:
    """Element extent via the fast corner-transform; identical to ``get_extent`` but avoids transforming
    every geometry for axis-aligned transforms (falls back to ``get_extent`` for rotation/shear).
    """
    return _element_extent_fast(element, coordinate_system) or get_extent(element, coordinate_system=coordinate_system)


def _get_extent_fast(
    sdata: SpatialData,
    coordinate_system: str,
    *,
    has_images: bool = True,
    has_labels: bool = True,
    has_points: bool = True,
    has_shapes: bool = True,
    elements: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Drop-in replacement for spatialdata ``get_extent(sdata, ...)`` with a fast path for shapes/points.

    Shapes/points with axis-aligned transforms get the corner-transform extent (identical result, no
    per-geometry transform); everything else (rotation/shear, images, labels) delegates to spatialdata's
    ``get_extent``. The union semantics match spatialdata's ``get_extent``.
    """
    include = {"images": has_images, "labels": has_labels, "points": has_points, "shapes": has_shapes}
    element_dicts = {"images": sdata.images, "labels": sdata.labels, "points": sdata.points, "shapes": sdata.shapes}
    mins: dict[str, list[float]] = {"x": [], "y": []}
    maxs: dict[str, list[float]] = {"x": [], "y": []}
    for etype, edict in element_dicts.items():
        if not include[etype]:
            continue
        for name, element in edict.items():
            if elements is not None and name not in elements:
                continue
            transformations = get_transformation(element, get_all=True)
            if coordinate_system not in transformations:
                continue
            ext = (
                _element_extent_fast(element, coordinate_system, transformations=transformations)
                if etype in ("shapes", "points")
                else None
            )
            if ext is None:  # rotation/shear, image/label (already cheap), or unsupported
                ext = get_extent(element, coordinate_system=coordinate_system)
            for ax in ("x", "y"):
                mins[ax].append(ext[ax][0])
                maxs[ax].append(ext[ax][1])
    if not mins["x"]:  # nothing matched -> defer to spatialdata (preserves its error behaviour)
        return get_extent(
            sdata,
            coordinate_system=coordinate_system,
            has_images=has_images,
            has_labels=has_labels,
            has_points=has_points,
            has_shapes=has_shapes,
            elements=elements,
        )
    return {ax: (min(mins[ax]), max(maxs[ax])) for ax in ("x", "y")}
