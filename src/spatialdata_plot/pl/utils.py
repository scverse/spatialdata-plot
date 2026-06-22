from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
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
from geopandas import GeoDataFrame
from matplotlib import colors, patheffects, rcParams
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import (
    Colormap,
    ListedColormap,
)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.legend import Legend
from matplotlib_scalebar.scalebar import ScaleBar
from pandas.api.types import CategoricalDtype
from pandas.core.arrays.categorical import Categorical
from scanpy import settings
from scanpy.plotting import palettes
from scanpy.plotting._tools.scatterplots import _add_categorical_legend
from spatialdata import (
    SpatialData,
    get_extent,
    join_spatialelement_table,
    rasterize,
)
from spatialdata._types import ArrayLike
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    SpatialElement,
    get_model,
    get_table_keys,
)
from spatialdata.transformations.operations import get_transformation
from xarray import DataArray, DataTree

from spatialdata_plot._logging import logger
from spatialdata_plot.pl.render_params import (
    Color,
    ColorbarSpec,
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

_RENDER_CMD_TO_CS_FLAG: dict[str, str] = {
    "render_images": "has_images",
    "render_shapes": "has_shapes",
    "render_points": "has_points",
    "render_labels": "has_labels",
}


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


def _legend_ncol(n: int) -> int:
    """Column count for a categorical legend with ``n`` entries."""
    return 1 if n <= 14 else 2 if n <= 30 else 3


def _categorical_legend_handles(ax: Axes, color_map: Mapping[Any, Any], na_hex: str | None = None) -> list[Any]:
    """Empty-scatter handles (colored dots) for a categorical legend, with an optional NA entry."""
    handles = [ax.scatter([], [], c=color, label=str(cat)) for cat, color in color_map.items()]
    if na_hex is not None:
        handles.append(ax.scatter([], [], c=na_hex, label="NA"))
    return handles


def _stack_categorical_legend(
    ax: Axes,
    color_mapping: Mapping[Any, Any],
    *,
    na_hex: str | None,
    title: str | None,
    column: str | None,
    legend_fontsize: int | float | _FontSize | None,
) -> None:
    """Build the 2nd+ categorical legend on a shared axes without dropping existing ones (#364).

    Placement and the column auto-title are finalized later by ``_setup_stacked_legends``.
    """
    handles = _categorical_legend_handles(ax, color_mapping, na_hex)
    if (cur := ax.get_legend()) is not None:
        ax.add_artist(cur)  # else ax.legend() below drops it
    new_leg = ax.legend(
        handles=handles,
        title=title,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=legend_fontsize,
        ncol=_legend_ncol(len(handles)),
    )
    new_leg._sdata_column = column  # type: ignore[attr-defined]


# A per-entry legend past this many categories is unreadable, and scanpy builds it in O(categories^2)
# (one autoscaling artist each), dominating the render — so skip it with a warning. Tied to scanpy's
# default_102 palette, beyond which its *default* colors also stop being distinguishable (uniform grey).
_MAX_LEGEND_CATEGORIES = len(palettes.default_102)


def _first_color_per_category(source: pd.Categorical, color_vector: Any) -> dict[Any, Any]:
    """Map each *used* category to the colour at its first occurrence, in appearance order.

    One vectorised pass over the codes (no per-point Python loop, see #379), shared by the legend and
    datashader colour-key builders so the "first colour per category" logic lives once. ``source`` is the
    length-N categorical; ``color_vector`` is the aligned colour vector. Categories never present are
    omitted (callers add any na fallback); appearance order matches the old ``drop_duplicates`` first row.
    """
    cats = source.categories
    at = color_vector.iloc if isinstance(color_vector, pd.Series) else color_vector  # positional
    unique_codes, first_indices = np.unique(np.asarray(source.codes), return_index=True)
    return {
        cats[code]: at[idx]
        for idx, code in sorted(zip(first_indices, unique_codes, strict=True))  # appearance order
        if code >= 0 and idx < len(color_vector)
    }


def _decorate_axs(
    ax: Axes,
    cax: PatchCollection,
    fig_params: FigParams,
    value_to_plot: str | None,
    color_source_vector: pd.Series[CategoricalDtype] | Categorical,
    color_vector: pd.Series[CategoricalDtype] | Categorical,
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
            color_mapping = _first_color_per_category(color_source_vector.remove_unused_categories(), color_vector)
            color_mapping = {k: v for k, v in color_mapping.items() if not pd.isnull(k)}  # NA handled separately
            # A 2nd categorical render would make scanpy's bare `ax.legend()` merge every labeled
            # artist into one legend and drop the first (#364), so route 2nd+ legends (i.e. when a
            # tagged legend already exists) through a helper that keeps them separate.
            tagged = (getattr(c, "_sdata_column", None) is not None for c in ax.get_children() if isinstance(c, Legend))
            already = any(tagged)
            if legend_loc in (None, "none"):
                pass  # legend suppressed
            elif len(clusters) > _MAX_LEGEND_CATEGORIES:
                # A per-entry legend this large is unreadable and scanpy builds it in O(categories^2)
                # (one autoscaling artist each), dominating the render. Skip it.
                logger.warning(
                    f"Skipping the categorical legend for '{value_to_plot}': {len(clusters)} categories "
                    f"exceed the {_MAX_LEGEND_CATEGORIES}-entry limit (unreadable and very slow to build). "
                    f"Pass a `groups` subset to get a legend."
                )
            elif already:
                na_hex = na_color.get_hex() if (na_in_legend and pd.isnull(color_source_vector).any()) else None
                _stack_categorical_legend(
                    ax,
                    color_mapping,
                    na_hex=na_hex,
                    title=legend_title,
                    column=value_to_plot,
                    legend_fontsize=legend_fontsize,
                )
            else:
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
                # Tag with the column; the column auto-title (when 2+ legends) is applied in
                # `_setup_stacked_legends`. An explicit title wins now.
                if (legend := ax.get_legend()) is not None:
                    legend._sdata_column = value_to_plot  # type: ignore[attr-defined]
                    if legend_title is not None:
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


def _format_element_names(element_name: list[str] | str | None) -> str:
    if element_name is None:
        return "the requested element"
    if isinstance(element_name, str):
        return f"'{element_name}'"
    return ", ".join(f"'{name}'" for name in element_name)


def _format_element_name(element_name: list[str] | str | None) -> str:
    if isinstance(element_name, str):
        return element_name
    if isinstance(element_name, list) and len(element_name) > 0:
        return ", ".join(element_name)
    return "<unknown>"


def _preview_values(values: Sequence[Any], limit: int = 5) -> str:
    values = list(values)
    preview = ", ".join(map(str, values[:limit]))
    if len(values) > limit:
        preview += ", ..."
    return preview


def _ensure_one_to_one_mapping(
    sdata: SpatialData,
    element: SpatialElement | None,
    element_name: list[str] | str | None,
    table_name: str | None,
) -> None:
    if table_name is None or element_name is None:
        return

    table = sdata.get(table_name, None)
    if table is None:
        return

    _validate_table_instance_uniqueness(table, element_name, table_name)
    _validate_shape_index_uniqueness(element, element_name, table_name)


def _validate_shape_index_uniqueness(
    element: SpatialElement | None,
    element_name: list[str] | str | None,
    table_name: str,
) -> None:
    if not isinstance(element, GeoDataFrame):
        return

    duplicates = element.index[element.index.duplicated(keep=False)]
    if duplicates.empty:
        return

    element_label = _format_element_names(element_name)
    preview = _preview_values(pd.Index(duplicates).unique())
    raise ValueError(
        f"{element_label} contains duplicate index values ({preview}) while table '{table_name}' "
        "requires a one-to-one mapping between shapes and annotations. "
        "Please ensure each spatial element has a unique index."
    )


def _validate_table_instance_uniqueness(
    table: AnnData,
    element_name: list[str] | str | None,
    table_name: str,
) -> None:
    try:
        _, region_key, instance_key = get_table_keys(table)
    except (AttributeError, KeyError, ValueError):
        return

    if instance_key is None or instance_key not in table.obs.columns:
        return

    obs = table.obs
    if region_key is not None and region_key in obs.columns and element_name is not None:
        element_names = [element_name] if isinstance(element_name, str) else list(element_name)
        obs = obs[obs[region_key].isin(element_names)]

    if obs.empty:
        return

    duplicates_mask = obs[instance_key].duplicated(keep=False)
    if not duplicates_mask.any():
        return

    element_label = _format_element_names(element_name)
    preview = _preview_values(obs.loc[duplicates_mask, instance_key].astype(str).unique())
    raise ValueError(
        f"Table '{table_name}' contains duplicate '{instance_key}' values for {element_label}: {preview}. "
        "Each observation must annotate a single spatial element. Please deduplicate the table or subset it "
        "before plotting."
    )


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
    """Element extent via the fast corner-transform.

    Identical to ``get_extent`` but avoids transforming every geometry for axis-aligned
    transforms (falls back to ``get_extent`` for rotation/shear).
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
        full_extent: dict[str, tuple[float, float]] = get_extent(
            sdata,
            coordinate_system=coordinate_system,
            has_images=has_images,
            has_labels=has_labels,
            has_points=has_points,
            has_shapes=has_shapes,
            elements=elements,
        )
        return full_extent
    return {ax: (min(mins[ax]), max(maxs[ax])) for ax in ("x", "y")}
