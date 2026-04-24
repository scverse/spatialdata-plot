from __future__ import annotations

import dataclasses
from collections import abc
from collections.abc import Sequence
from copy import copy
from typing import Any

import dask
import dask.dataframe as dd
import datashader as ds
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import xarray as xr
from anndata import AnnData
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from scanpy._settings import settings as sc_settings
from scanpy.plotting._tools.scatterplots import _add_categorical_legend
from spatialdata import get_extent, get_values, join_spatialelement_table
from spatialdata._core.query.relational_query import match_table_to_element
from spatialdata.models import PointsModel, ShapesModel, get_table_keys
from spatialdata.transformations import set_transformation
from spatialdata.transformations.transformations import Identity
from xarray import DataTree

from spatialdata_plot._logging import _log_context, logger
from spatialdata_plot.pl._datashader import (
    _apply_ds_norm,
    _build_color_key,
    _build_ds_colorbar,
    _ds_aggregate,
    _ds_shade_categorical,
    _ds_shade_continuous,
    _DsReduction,
    _render_ds_image,
    _render_ds_outlines,
)
from spatialdata_plot.pl.render_params import (
    ChannelLegendEntry,
    CmapParams,
    Color,
    ColorbarSpec,
    FigParams,
    ImageRenderParams,
    LabelsRenderParams,
    LegendParams,
    PointsRenderParams,
    ScalebarParams,
    ShapesRenderParams,
)
from spatialdata_plot.pl.utils import (
    _ax_show_and_transform,
    _convert_shapes,
    _datashader_canvas_from_dataframe,
    _decorate_axs,
    _get_collection_shape,
    _get_colors_for_categorical_obs,
    _get_extent_and_range_for_datashader_canvas,
    _get_linear_colormap,
    _hex_no_alpha,
    _map_color_seg,
    _maybe_set_colors,
    _mpl_ax_contains_elements,
    _multiscale_to_spatial_image,
    _prepare_cmap_norm,
    _prepare_transformation,
    _rasterize_if_necessary,
    _set_color_source_vec,
    _validate_polygons,
)

_Normalize = Normalize | abc.Sequence[Normalize]


def _want_decorations(color_vector: Any, na_color: Color) -> bool:
    """Return whether legend/colorbar decorations should be shown.

    Decorations are suppressed when all colors equal the NA color
    (i.e., nothing informative to display).
    """
    if color_vector is None:
        return False
    cv = np.asarray(color_vector)
    if cv.size == 0:
        return False
    first = cv.flat[0]
    if not (cv == first).all():
        return True
    na_hex = na_color.get_hex()
    if isinstance(first, str) and first.startswith("#") and na_hex.startswith("#"):
        return _hex_no_alpha(first) != _hex_no_alpha(na_hex)
    return bool(first != na_hex)


def _log_datashader_method(method: str, ds_reduction: _DsReduction | None, default: _DsReduction) -> None:
    """Log the datashader backend and effective reduction being used."""
    effective = ds_reduction if ds_reduction is not None else default
    logger.info(
        f"Using '{method}' backend with '{effective}' as reduction"
        " method to speed up plotting. Depending on the reduction method, the value"
        " range of the plot might change. Set method to 'matplotlib' to disable"
        " this behaviour."
    )


def _reparse_points(
    sdata_filt: sd.SpatialData,
    element: str,
    df: pd.DataFrame,
    transformation: Any,
    coordinate_system: str,
) -> None:
    """Re-register a points DataFrame in *sdata_filt* with its transformation."""
    dd_frame = dask.dataframe.from_pandas(df, npartitions=1)
    sdata_filt.points[element] = PointsModel.parse(dd_frame, coordinates={"x": "x", "y": "y"})
    set_transformation(
        element=sdata_filt.points[element],
        transformation=transformation,
        to_coordinate_system=coordinate_system,
    )


def _warn_groups_ignored_continuous(
    groups: str | list[str] | None,
    color_source_vector: pd.Categorical | None,
    col_for_color: str | None,
) -> None:
    """Warn when ``groups`` is set but coloring is continuous (no categorical source)."""
    if groups is not None and color_source_vector is None and col_for_color is not None:
        logger.warning(
            f"`groups` is ignored when coloring by continuous column '{col_for_color}'. "
            "`groups` filters categories of the column specified via `color`; "
            "it has no effect on continuous data."
        )


def _warn_missing_groups(
    groups: str | list[str],
    color_source_vector: pd.Categorical,
    col_for_color: str | None = None,
) -> None:
    """Warn when ``groups`` contains values absent from the color column's categories."""
    groups_set = {groups} if isinstance(groups, str) else set(groups)
    missing = groups_set - set(color_source_vector.categories)
    if not missing:
        return
    col_label = f" '{col_for_color}'" if col_for_color else " the color column"
    try:
        missing_str = str(sorted(missing))
    except TypeError:
        missing_str = str(list(missing))
    if missing == groups_set:
        logger.warning(
            f"None of the requested groups {missing_str} were found in{col_label}. "
            "This usually means `groups` refers to values from a different column than `color`. "
            "The `groups` parameter selects categories of the column specified via `color`."
        )
    else:
        try:
            cats_str = str(sorted(color_source_vector.categories))
        except TypeError:
            cats_str = str(list(color_source_vector.categories))
        logger.warning(
            f"Groups {missing_str} were not found in{col_label} and will be ignored. Available categories: {cats_str}."
        )


def _filter_groups_transparent_na(
    groups: str | list[str],
    color_source_vector: pd.Categorical,
    color_vector: pd.Series | np.ndarray | list[str],
) -> tuple[np.ndarray, pd.Categorical, np.ndarray]:
    """Return a boolean mask and filtered color vectors for groups filtering.

    Used when ``na_color=None`` (fully transparent) so that non-matching
    elements are removed entirely instead of rendered invisibly.
    """
    keep = color_source_vector.isin(groups)
    filtered_csv = color_source_vector[keep]
    filtered_cv = np.asarray(color_vector)[keep]
    return keep, filtered_csv, filtered_cv


def _split_colorbar_params(
    params: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object], str | None]:
    """Split colorbar params into layout hints, Matplotlib kwargs, and label override."""
    layout: dict[str, object] = {}
    cbar_kwargs: dict[str, object] = {}
    label_override: str | None = None
    for key, value in (params or {}).items():
        key_lower = key.lower()
        if key_lower in {"loc", "location"}:
            layout["location"] = value
        elif key_lower == "width" or key_lower == "fraction":
            layout["fraction"] = value
        elif key_lower == "pad":
            layout["pad"] = value
        elif key_lower == "label":
            label_override = None if value is None else str(value)
        else:
            cbar_kwargs[key] = value
    return layout, cbar_kwargs, label_override


def _resolve_colorbar_label(
    colorbar_params: dict[str, object] | None,
    fallback: str | None,
    *,
    is_default_channel_name: bool = False,
) -> str | None:
    """Pick a colorbar label from params or fall back to provided value."""
    _, _, label = _split_colorbar_params(colorbar_params)
    if label is not None:
        return label
    if is_default_channel_name:
        return None
    return fallback


def _should_request_colorbar(
    colorbar: bool | str | None,
    *,
    has_mappable: bool,
    is_continuous: bool,
    auto_condition: bool = True,
) -> bool:
    """Resolve colorbar setting to a final boolean request."""
    if not has_mappable or not is_continuous:
        return False
    if colorbar is True:
        return True
    if colorbar in {False, None}:
        return False
    return bool(auto_condition)


def _make_palette(
    color_source_vector: pd.Series | None,
    color_vector: Any,
) -> ListedColormap:
    """Build a ListedColormap from a color vector, filtering out NaN entries when categorical."""
    if color_source_vector is None:
        return ListedColormap(dict.fromkeys(color_vector))
    return ListedColormap(dict.fromkeys(color_vector[~pd.Categorical(color_source_vector).isnull()]))


def _add_legend_and_colorbar(
    ax: matplotlib.axes.SubplotBase,
    cax: ScalarMappable | None,
    fig_params: FigParams,
    adata: AnnData | None,
    col_for_color: str | None,
    color_source_vector: pd.Series | None,
    color_vector: Any,
    palette: ListedColormap | list[str] | None,
    alpha: float,
    na_color: Color,
    legend_params: LegendParams,
    colorbar: bool | str | None,
    colorbar_params: dict[str, object] | None,
    colorbar_requests: list[ColorbarSpec] | None,
    scalebar_params: ScalebarParams,
) -> None:
    """Add legend, colorbar, and scalebar decorations if the color vector warrants them."""
    if not _want_decorations(color_vector, na_color):
        return

    if palette is None:
        palette = _make_palette(color_source_vector, color_vector)

    if color_source_vector is not None and hasattr(color_source_vector, "remove_unused_categories"):
        color_source_vector = color_source_vector.remove_unused_categories()

    wants_colorbar = _should_request_colorbar(
        colorbar,
        has_mappable=cax is not None,
        is_continuous=col_for_color is not None and color_source_vector is None,
    )

    _decorate_axs(
        ax=ax,
        cax=cax,
        fig_params=fig_params,
        adata=adata,
        value_to_plot=col_for_color,
        color_source_vector=color_source_vector,
        color_vector=color_vector,
        palette=palette,
        alpha=alpha,
        na_color=na_color,
        legend_fontsize=legend_params.legend_fontsize,
        legend_fontweight=legend_params.legend_fontweight,
        legend_loc=legend_params.legend_loc,
        legend_fontoutline=legend_params.legend_fontoutline,
        na_in_legend=legend_params.na_in_legend,
        colorbar=wants_colorbar and legend_params.colorbar,
        colorbar_params=colorbar_params,
        colorbar_requests=colorbar_requests,
        colorbar_label=_resolve_colorbar_label(
            colorbar_params,
            col_for_color if isinstance(col_for_color, str) else None,
        ),
        scalebar_dx=scalebar_params.scalebar_dx,
        scalebar_units=scalebar_params.scalebar_units,
    )


def _render_shapes(
    sdata: sd.SpatialData,
    render_params: ShapesRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
    colorbar_requests: list[ColorbarSpec] | None = None,
) -> None:
    _log_context.set("render_shapes")
    element = render_params.element
    col_for_color = render_params.col_for_color
    groups = render_params.groups
    table_layer = render_params.table_layer

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_tables=bool(render_params.table_name),
    )

    table_name = render_params.table_name
    if table_name is None:
        table = None
        shapes = sdata_filt[element]
    else:
        # Workaround for upstream spatialdata bug (scverse/spatialdata#1099):
        # join_spatialelement_table calls table.obs.reset_index() which fails when
        # the obs index name matches an existing column (e.g. "EntityID" in Merfish data).
        # Temporarily drop the conflicting index name for the join, then restore it.
        _obs = sdata[table_name].obs
        _saved_index_name = _obs.index.name
        if _saved_index_name is not None and _saved_index_name in _obs.columns:
            _obs.index.name = None

        try:
            element_dict, joined_table = join_spatialelement_table(
                sdata, spatial_element_names=element, table_name=table_name, how="inner"
            )
        finally:
            _obs.index.name = _saved_index_name
        sdata_filt[element] = shapes = element_dict[element]
        joined_table.uns["spatialdata_attrs"]["region"] = (
            joined_table.obs[joined_table.uns["spatialdata_attrs"]["region_key"]].unique().tolist()
        )
        sdata_filt[table_name] = table = joined_table

    shapes = sdata_filt[element]

    # Capture the transformation *before* any groups filtering that may strip
    # coordinate-system metadata from the element (see #420, #447).
    trans, trans_data = _prepare_transformation(sdata_filt.shapes[element], coordinate_system)

    # get color vector (categorical or continuous)
    color_source_vector, color_vector, _ = _set_color_source_vec(
        sdata=sdata_filt,
        element=sdata_filt[element],
        element_name=element,
        value_to_plot=col_for_color,
        groups=groups,
        palette=render_params.palette,
        na_color=(render_params.color if render_params.color is not None else render_params.cmap_params.na_color),
        cmap_params=render_params.cmap_params,
        table_name=table_name,
        table_layer=table_layer,
        coordinate_system=coordinate_system,
    )

    values_are_categorical = color_source_vector is not None

    _warn_groups_ignored_continuous(groups, color_source_vector, col_for_color)

    if groups is not None and color_source_vector is not None:
        _warn_missing_groups(groups, color_source_vector, col_for_color)

    # When groups are specified, filter out non-matching elements by default.
    # Only show non-matching elements if the user explicitly sets na_color.
    _na = render_params.cmap_params.na_color
    if groups is not None and color_source_vector is not None and (_na.default_color_set or _na.is_fully_transparent()):
        keep, color_source_vector, color_vector = _filter_groups_transparent_na(
            groups, color_source_vector, color_vector
        )
        shapes = shapes[keep].reset_index(drop=True)
        if len(shapes) == 0:
            return
        sdata_filt[element] = shapes

    # color_source_vector is None when the values aren't categorical
    if values_are_categorical and render_params.transfunc is not None:
        color_vector = render_params.transfunc(color_vector)

    norm = copy(render_params.cmap_params.norm)

    if len(color_vector) == 0:
        color_vector = [render_params.cmap_params.na_color.get_hex_with_alpha()]

    # continuous case: leave NaNs as NaNs; utils maps them to na_color during draw
    if color_source_vector is None and not values_are_categorical:
        _series = color_vector if isinstance(color_vector, pd.Series) else pd.Series(color_vector)

        try:
            color_vector = np.asarray(_series, dtype=float)
        except (TypeError, ValueError):
            nan_count = int(_series.isna().sum())
            if nan_count:
                logger.warning(
                    f"Found {nan_count} NaN values in color data. "
                    "These observations will be colored with the 'na_color'."
                )
            color_vector = _series.to_numpy()
        else:
            if np.isnan(color_vector).any():
                nan_count = int(np.isnan(color_vector).sum())
                logger.warning(
                    f"Found {nan_count} NaN values in color data. "
                    "These observations will be colored with the 'na_color'."
                )

    palette = _make_palette(color_source_vector, color_vector)

    has_valid_color = (
        len(set(color_vector)) != 1
        or list(set(color_vector))[0] != render_params.cmap_params.na_color.get_hex_with_alpha()
    )
    if has_valid_color and color_source_vector is not None and col_for_color is not None:
        # necessary in case different shapes elements are annotated with one table
        color_source_vector = color_source_vector.remove_unused_categories()

    shapes = gpd.GeoDataFrame(shapes, geometry="geometry")
    # convert shapes if necessary
    if render_params.shape is not None:
        current_type = shapes["geometry"].type
        if not (render_params.shape == "circle" and (current_type == "Point").all()):
            logger.info(f"Converting {shapes.shape[0]} shapes to {render_params.shape}.")
            max_extent = np.max(
                [
                    shapes.total_bounds[2] - shapes.total_bounds[0],
                    shapes.total_bounds[3] - shapes.total_bounds[1],
                ]
            )
            shapes = _convert_shapes(shapes, render_params.shape, max_extent)

    shapes = _validate_polygons(shapes)

    # Determine which method to use for rendering
    method = render_params.method

    if method is None:
        method = "datashader" if len(shapes) > 10000 else "matplotlib"

    _default_reduction: _DsReduction = "max"

    if method != "matplotlib":
        _log_datashader_method(method, render_params.ds_reduction, _default_reduction)

    if method == "datashader":
        _geometry = shapes["geometry"]
        is_point = _geometry.type == "Point"

        # Handle circles encoded as points with radius
        if is_point.any():
            radius_values = shapes[is_point]["radius"]
            # Convert to numeric, replacing non-numeric values with NaN
            radius_numeric = pd.to_numeric(radius_values, errors="coerce")
            scale = radius_numeric * render_params.scale
            shapes.loc[is_point, "geometry"] = _geometry[is_point].buffer(scale.to_numpy())

        # Handle polygon/multipolygon scaling
        is_polygon = _geometry.type.isin(["Polygon", "MultiPolygon"])
        if is_polygon.any() and render_params.scale != 1.0:
            from shapely import affinity

            shapes.loc[is_polygon, "geometry"] = _geometry[is_polygon].apply(
                lambda geom: affinity.scale(geom, xfact=render_params.scale, yfact=render_params.scale)
            )

        # apply transformations to the individual points
        tm = trans.get_matrix()
        transformed_geometry = shapes["geometry"].transform(
            lambda x: (np.hstack([x, np.ones((x.shape[0], 1))]) @ tm.T)[:, :2]
        )
        transformed_element = ShapesModel.parse(
            gpd.GeoDataFrame(
                data=shapes.drop("geometry", axis=1),
                geometry=transformed_geometry,
            )
        )

        plot_width, plot_height, x_ext, y_ext, factor = _get_extent_and_range_for_datashader_canvas(
            transformed_element, "global", ax, fig_params
        )

        cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_ext, y_range=y_ext)

        # in case we are coloring by a column in table
        if col_for_color is not None and col_for_color not in transformed_element.columns:
            # Ensure color vector length matches the number of shapes
            if len(color_vector) != len(transformed_element):
                if len(color_vector) == 1:
                    # If single color, broadcast to all shapes
                    color_vector = [color_vector[0]] * len(transformed_element)
                else:
                    logger.warning(
                        f"Color vector length ({len(color_vector)}) does not match element count "
                        f"({len(transformed_element)}). This may indicate a bug."
                    )
                    if len(color_vector) > len(transformed_element):
                        color_vector = color_vector[: len(transformed_element)]
                    else:
                        na_color = render_params.cmap_params.na_color.get_hex_with_alpha()
                        color_vector = list(color_vector) + [na_color] * (len(transformed_element) - len(color_vector))

            transformed_element[col_for_color] = color_vector if color_source_vector is None else color_source_vector
        # Render shapes with datashader
        color_by_categorical = col_for_color is not None and color_source_vector is not None
        if color_by_categorical:
            cat_series = transformed_element[col_for_color]
            if not isinstance(cat_series.dtype, pd.CategoricalDtype):
                cat_series = cat_series.astype("category")
            transformed_element[col_for_color] = cat_series

        agg, reduction_bounds, nan_agg = _ds_aggregate(
            cvs,
            transformed_element,
            col_for_color,
            color_by_categorical,
            render_params.ds_reduction,
            _default_reduction,
            "shapes",
        )

        agg, color_span = _apply_ds_norm(agg, norm)
        na_color_hex = _hex_no_alpha(render_params.cmap_params.na_color.get_hex())
        if render_params.cmap_params.na_color.is_fully_transparent():
            nan_agg = None
        color_key = _build_color_key(
            transformed_element,
            col_for_color,
            color_by_categorical,
            color_vector,
            na_color_hex,
        )

        nan_shaded = None
        if color_by_categorical or col_for_color is None:
            shaded = _ds_shade_categorical(
                agg,
                color_key,
                color_vector,
                render_params.fill_alpha,
            )
        else:
            shaded, nan_shaded, reduction_bounds = _ds_shade_continuous(
                agg,
                color_span,
                norm,
                render_params.cmap_params.cmap,
                render_params.fill_alpha,
                reduction_bounds,
                nan_agg,
                na_color_hex,
            )

        _render_ds_outlines(
            cvs,
            transformed_element,
            render_params,
            fig_params,
            ax,
            factor,
            x_ext + y_ext,
        )

        _cax = _render_ds_image(
            ax,
            shaded,
            factor,
            render_params.zorder,
            x_ext + y_ext,
            nan_result=nan_shaded,
        )

        cax = _build_ds_colorbar(reduction_bounds, norm, render_params.cmap_params.cmap)

    elif method == "matplotlib":
        # render outlines separately to ensure they are always underneath the shape
        if render_params.outline_alpha[0] > 0 and isinstance(render_params.outline_params.outer_outline_color, Color):
            _cax = _get_collection_shape(
                shapes=shapes,
                s=render_params.scale,
                c=np.array(["white"]),  # hack, will be invisible bc fill_alpha=0
                render_params=render_params,
                rasterized=sc_settings._vector_friendly,
                cmap=None,
                norm=None,
                fill_alpha=0.0,
                outline_alpha=render_params.outline_alpha[0],
                outline_color=render_params.outline_params.outer_outline_color.get_hex(),
                linewidth=render_params.outline_params.outer_outline_linewidth,
                zorder=render_params.zorder,
                # **kwargs,
            )
            cax = ax.add_collection(_cax)
            # Transform the paths in PatchCollection
            for path in _cax.get_paths():
                path.vertices = trans.transform(path.vertices)
        if render_params.outline_alpha[1] > 0 and isinstance(render_params.outline_params.inner_outline_color, Color):
            _cax = _get_collection_shape(
                shapes=shapes,
                s=render_params.scale,
                c=np.array(["white"]),  # hack, will be invisible bc fill_alpha=0
                render_params=render_params,
                rasterized=sc_settings._vector_friendly,
                cmap=None,
                norm=None,
                fill_alpha=0.0,
                outline_alpha=render_params.outline_alpha[1],
                outline_color=render_params.outline_params.inner_outline_color.get_hex(),
                linewidth=render_params.outline_params.inner_outline_linewidth,
                zorder=render_params.zorder,
                # **kwargs,
            )
            cax = ax.add_collection(_cax)
            # Transform the paths in PatchCollection
            for path in _cax.get_paths():
                path.vertices = trans.transform(path.vertices)

        _cax = _get_collection_shape(
            shapes=shapes,
            s=render_params.scale,
            c=color_vector.copy(),  # copy bc c is modified in _get_collection_shape
            render_params=render_params,
            rasterized=sc_settings._vector_friendly,
            cmap=render_params.cmap_params.cmap,
            norm=norm,
            fill_alpha=render_params.fill_alpha,
            outline_alpha=0.0,
            zorder=render_params.zorder,
            # **kwargs,
        )
        cax = ax.add_collection(_cax)

        # Transform the paths in PatchCollection
        for path in _cax.get_paths():
            path.vertices = trans.transform(path.vertices)

    if not values_are_categorical:
        # Respect explicit vmin/vmax; otherwise derive from finite numeric values, falling back to [0, 1] if unavailable
        vmin = render_params.cmap_params.norm.vmin
        vmax = render_params.cmap_params.norm.vmax
        if vmin is None or vmax is None:
            numeric_values = pd.to_numeric(np.asarray(color_vector), errors="coerce")
            finite_mask = np.isfinite(numeric_values)
            if finite_mask.any():
                data_min = float(np.nanmin(numeric_values[finite_mask]))
                data_max = float(np.nanmax(numeric_values[finite_mask]))
                if vmin is None:
                    vmin = data_min
                if vmax is None:
                    vmax = data_max
            else:
                if vmin is None:
                    vmin = 0.0
                if vmax is None:
                    vmax = 1.0
        _cax.set_clim(vmin=vmin, vmax=vmax)

    _add_legend_and_colorbar(
        ax=ax,
        cax=cax,
        fig_params=fig_params,
        adata=table,
        col_for_color=col_for_color,
        color_source_vector=color_source_vector,
        color_vector=color_vector,
        palette=palette,
        alpha=render_params.fill_alpha,
        na_color=render_params.cmap_params.na_color,
        legend_params=legend_params,
        colorbar=render_params.colorbar,
        colorbar_params=render_params.colorbar_params,
        colorbar_requests=colorbar_requests,
        scalebar_params=scalebar_params,
    )


def _render_points(
    sdata: sd.SpatialData,
    render_params: PointsRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
    colorbar_requests: list[ColorbarSpec] | None = None,
) -> None:
    _log_context.set("render_points")
    element = render_params.element
    col_for_color = render_params.col_for_color
    table_name = render_params.table_name
    table_layer = render_params.table_layer
    color = render_params.color.get_hex() if render_params.color else None
    groups = render_params.groups
    palette = render_params.palette

    if isinstance(groups, str):
        groups = [groups]

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        # keep tables intact; we pick the right rows ourselves via the table metadata
        filter_tables=False,
    )

    points = sdata.points[element]
    coords = ["x", "y"]

    if table_name is not None and col_for_color not in points.columns:
        logger.warning(
            f"Annotating points with {col_for_color} which is stored in the table `{table_name}`. "
            f"To improve performance, it is advisable to store point annotations directly in the .parquet file."
        )

    if col_for_color is None or (
        table_name is not None
        and (col_for_color in sdata_filt[table_name].obs.columns or col_for_color in sdata_filt[table_name].var_names)
    ):
        points = points[coords].compute()
    else:
        coords += [col_for_color]
        points = points[coords].compute()

    added_color_from_table = False
    if col_for_color is not None and col_for_color not in points.columns:
        color_values = get_values(
            value_key=col_for_color,
            sdata=sdata_filt,
            element_name=element,
            table_name=table_name,
            table_layer=table_layer,
        )
        points = points.merge(
            color_values[[col_for_color]],
            how="left",
            left_index=True,
            right_index=True,
        )
        added_color_from_table = True

    # Reset to sequential index so row order matches after _reparse_points round-trip (#358).
    points = points.reset_index(drop=True)

    n_points = len(points)
    points_pd_with_color = points
    # When we pull colors from a table, keep the raw points (with color) for later,
    # but strip the color column from the model we register in sdata so color lookup
    # keeps using the table instead of seeing duplicates on the points dataframe.
    points_for_model = (
        points_pd_with_color.drop(columns=[col_for_color], errors="ignore")
        if added_color_from_table and col_for_color is not None
        else points_pd_with_color
    )

    # we construct an anndata to hack the plotting functions
    if table_name is None:
        adata = AnnData(
            X=points[["x", "y"]].values,
            obs=points[coords],
            dtype=points[["x", "y"]].values.dtype,
        )
    else:
        matched_table = match_table_to_element(sdata=sdata, element_name=element, table_name=table_name)
        adata_obs = matched_table.obs.copy()
        # if the points are colored by values in X (or a different layer), add the values to obs
        if col_for_color in matched_table.var_names:
            if table_layer is None:
                adata_obs[col_for_color] = matched_table[:, col_for_color].X.flatten().copy()
            else:
                adata_obs[col_for_color] = matched_table[:, col_for_color].layers[table_layer].flatten().copy()
        adata = AnnData(
            X=points[["x", "y"]].values,
            obs=adata_obs,
            dtype=points[["x", "y"]].values.dtype,
            uns=matched_table.uns,
        )
        sdata_filt[table_name] = adata

    # we can modify the sdata because of dealing with a copy

    # Convert back to dask dataframe to modify sdata
    transformation_in_cs = sdata_filt.points[element].attrs["transform"][coordinate_system]
    _reparse_points(sdata_filt, element, points_for_model, transformation_in_cs, coordinate_system)

    if col_for_color is not None:
        assert isinstance(col_for_color, str)
        cols = sc.get.obs_df(adata, [col_for_color])
        # maybe set color based on type
        if isinstance(cols[col_for_color].dtype, pd.CategoricalDtype):
            uns_color_key = f"{col_for_color}_colors"
            if uns_color_key in adata.uns:
                _maybe_set_colors(
                    source=adata,
                    target=adata,
                    key=col_for_color,
                    palette=palette,
                )

    # when user specified a single color, we emulate the form of `na_color` and use it
    default_color = (
        render_params.color if col_for_color is None and color is not None else render_params.cmap_params.na_color
    )
    assert isinstance(default_color, Color)  # shut up mypy

    color_element = sdata_filt.points[element]
    # Always pass the table through to color resolution; dropping the color column
    # from the registered points (see above) avoids duplicate-origin ambiguities.
    color_table_name = table_name

    # Reuse color data already loaded from the table to avoid a redundant get_values() call.
    _preloaded = points_pd_with_color[col_for_color] if added_color_from_table and col_for_color is not None else None

    color_source_vector, color_vector, _ = _set_color_source_vec(
        sdata=sdata_filt,
        element=color_element,
        element_name=element,
        value_to_plot=col_for_color,
        groups=groups,
        palette=palette,
        na_color=default_color,
        cmap_params=render_params.cmap_params,
        alpha=render_params.alpha,
        table_name=color_table_name,
        render_type="points",
        coordinate_system=coordinate_system,
        preloaded_color_data=_preloaded,
    )

    if added_color_from_table and col_for_color is not None:
        _reparse_points(
            sdata_filt,
            element,
            points_pd_with_color,
            transformation_in_cs,
            coordinate_system,
        )

    _warn_groups_ignored_continuous(groups, color_source_vector, col_for_color)

    if groups is not None and color_source_vector is not None:
        _warn_missing_groups(groups, color_source_vector, col_for_color)

    # When groups are specified, filter out non-matching elements by default.
    # Only show non-matching elements if the user explicitly sets na_color.
    _na = render_params.cmap_params.na_color
    if groups is not None and color_source_vector is not None and (_na.default_color_set or _na.is_fully_transparent()):
        keep, color_source_vector, color_vector = _filter_groups_transparent_na(
            groups, color_source_vector, color_vector
        )
        n_points = int(keep.sum())
        if n_points == 0:
            return
        # filter the materialized points, adata, and re-register in sdata_filt
        points = points[keep].reset_index(drop=True)
        adata = adata[keep]
        _reparse_points(sdata_filt, element, points, transformation_in_cs, coordinate_system)

    # color_source_vector is None when the values aren't categorical
    if color_source_vector is None and render_params.transfunc is not None:
        color_vector = render_params.transfunc(color_vector)

    trans, trans_data = _prepare_transformation(sdata.points[element], coordinate_system, ax)

    norm = copy(render_params.cmap_params.norm)

    method = render_params.method

    if method is None:
        method = "datashader" if n_points > 10000 else "matplotlib"

    _default_reduction: _DsReduction = "sum"

    if method == "datashader":
        _log_datashader_method(method, render_params.ds_reduction, _default_reduction)

        # NOTE: s in matplotlib is in units of points**2
        # use dpi/100 as a factor for cases where dpi!=100
        px = int(np.round(np.sqrt(render_params.size) * (fig_params.fig.dpi / 100)))

        # Apply transformations and materialize to pandas immediately so
        # datashader aggregates without dask scheduler overhead.  See #379.
        transformed_element = PointsModel.parse(
            trans.transform(sdata_filt.points[element][["x", "y"]]),
            annotation=sdata_filt.points[element][sdata_filt.points[element].columns.drop(["x", "y"])],
            transformations={coordinate_system: Identity()},
        ).compute()

        plot_width, plot_height, x_ext, y_ext, factor = _datashader_canvas_from_dataframe(
            transformed_element, ax, fig_params
        )

        # use datashader for the visualization of points
        cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_ext, y_range=y_ext)

        # ensure color column exists on the transformed element with positional alignment
        if col_for_color is not None and col_for_color not in transformed_element.columns:
            series_index = transformed_element.index
            if color_source_vector is not None:
                if isinstance(color_source_vector, dd.Series):
                    color_source_vector = color_source_vector.compute()
                source_series = (
                    color_source_vector.reindex(series_index)
                    if isinstance(color_source_vector, pd.Series)
                    else pd.Series(color_source_vector, index=series_index)
                )
                transformed_element[col_for_color] = source_series
            else:
                if isinstance(color_vector, dd.Series):
                    color_vector = color_vector.compute()
                color_series = (
                    color_vector.reindex(series_index)
                    if isinstance(color_vector, pd.Series)
                    else pd.Series(color_vector, index=series_index)
                )
                transformed_element[col_for_color] = color_series

        color_dtype = transformed_element[col_for_color].dtype if col_for_color is not None else None
        color_by_categorical = col_for_color is not None and (
            color_source_vector is not None
            or isinstance(color_dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(color_dtype)
            or pd.api.types.is_string_dtype(color_dtype)
        )
        if color_by_categorical and not isinstance(color_dtype, pd.CategoricalDtype):
            transformed_element[col_for_color] = transformed_element[col_for_color].astype("category")

        agg, reduction_bounds, nan_agg = _ds_aggregate(
            cvs,
            transformed_element,
            col_for_color,
            color_by_categorical,
            render_params.ds_reduction,
            _default_reduction,
            "points",
        )

        agg, color_span = _apply_ds_norm(agg, norm)
        na_color_hex = _hex_no_alpha(render_params.cmap_params.na_color.get_hex())
        if render_params.cmap_params.na_color.is_fully_transparent():
            nan_agg = None
        color_key = _build_color_key(
            transformed_element,
            col_for_color,
            color_by_categorical,
            color_vector,
            na_color_hex,
        )

        if (
            color_vector is not None
            and len(color_vector) > 0
            and isinstance(color_vector[0], str)
            and color_vector[0].startswith("#")
        ):
            color_vector = np.asarray([_hex_no_alpha(c) for c in color_vector])

        nan_shaded = None
        if color_by_categorical or col_for_color is None:
            shaded = _ds_shade_categorical(
                agg,
                color_key,
                color_vector,
                render_params.alpha,
                spread_px=px,
            )
        else:
            shaded, nan_shaded, reduction_bounds = _ds_shade_continuous(
                agg,
                color_span,
                norm,
                render_params.cmap_params.cmap,
                render_params.alpha,
                reduction_bounds,
                nan_agg,
                na_color_hex,
                spread_px=px,
                ds_reduction=render_params.ds_reduction,
            )

        _render_ds_image(
            ax,
            shaded,
            factor,
            render_params.zorder,
            x_ext + y_ext,
            nan_result=nan_shaded,
        )

        cax = _build_ds_colorbar(reduction_bounds, norm, render_params.cmap_params.cmap)

    elif method == "matplotlib":
        # update axis limits if plot was empty before (necessary if datashader comes after)
        update_parameters = not _mpl_ax_contains_elements(ax)
        _cax = ax.scatter(
            adata[:, 0].X.flatten(),
            adata[:, 1].X.flatten(),
            s=render_params.size,
            c=color_vector,
            rasterized=sc_settings._vector_friendly,
            cmap=render_params.cmap_params.cmap,
            norm=norm,
            alpha=render_params.alpha,
            transform=trans_data,
            zorder=render_params.zorder,
            plotnonfinite=True,  # nan points should be rendered as well
        )
        cax = _cax
        if update_parameters:
            # necessary if points are plotted with mpl first and then with datashader
            extent = get_extent(sdata_filt.points[element], coordinate_system=coordinate_system)
            ax.set_xbound(extent["x"])
            ax.set_ybound(extent["y"])

    _add_legend_and_colorbar(
        ax=ax,
        cax=cax,
        fig_params=fig_params,
        adata=adata,
        col_for_color=col_for_color,
        color_source_vector=color_source_vector,
        color_vector=color_vector,
        palette=None,
        alpha=render_params.alpha,
        na_color=render_params.cmap_params.na_color,
        legend_params=legend_params,
        colorbar=render_params.colorbar,
        colorbar_params=render_params.colorbar_params,
        colorbar_requests=colorbar_requests,
        scalebar_params=scalebar_params,
    )


_LUMINANCE_WEIGHTS = np.array([0.2989, 0.5870, 0.1140])


def _grayscale_transform(img_cyx: np.ndarray) -> np.ndarray:
    """Convert a (3, y, x) RGB image to (1, y, x) luminance."""
    return np.tensordot(_LUMINANCE_WEIGHTS, img_cyx, axes=([0], [0]))[np.newaxis]


def _normalize_dtype_to_float(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to float64 in [0, 1] for display with matplotlib.

    Intended for RGB/RGBA image data where negative values are not meaningful.

    - uint8 → divide by 255
    - other unsigned int → divide by dtype max
    - signed int → divide by dtype max, clip negatives to 0
    - float already in [0, 1] → pass through
    - float outside [0, 1] → global auto-range (preserves relative balance across channels)
    """
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    if arr.dtype.kind == "u":
        return arr.astype(np.float64) / np.iinfo(arr.dtype).max
    if arr.dtype.kind == "i":
        return np.clip(arr.astype(np.float64) / np.iinfo(arr.dtype).max, 0, 1)
    # Float: if already in [0, 1], keep as-is; otherwise auto-range globally
    arr_f: np.ndarray = arr.astype(np.float64)
    vmin, vmax = arr_f.min(), arr_f.max()
    if vmin >= 0.0 and vmax <= 1.0:
        return arr_f
    if vmin == vmax:
        return np.zeros_like(arr_f)
    logger.info(
        "Float RGB image has values outside [0, 1] (range [%.3f, %.3f]); "
        "auto-ranging globally. Pass an explicit 'norm' to control contrast.",
        vmin,
        vmax,
    )
    result: np.ndarray = (arr_f - vmin) / (vmax - vmin)
    return result


def _is_rgb_image(channel_coords: list[Any]) -> tuple[bool, bool]:
    """Check if channel coordinates indicate an RGB(A) image.

    Checks case-insensitively whether channel names are {r, g, b} or {r, g, b, a}.

    Parameters
    ----------
    channel_coords
        The channel coordinate values from the image.

    Returns
    -------
    tuple[bool, bool]
        (is_rgb, has_alpha) — whether the image is RGB and whether it includes an alpha channel.
    """
    names = {str(c).lower() for c in channel_coords}
    if names == {"r", "g", "b", "a"} and len(channel_coords) == 4:
        return True, True
    if names == {"r", "g", "b"} and len(channel_coords) == 3:
        return True, False
    return False, False


def _collect_channel_legend_entries(
    channels: Sequence[str | int],
    seed_colors: Sequence[str | tuple[float, ...]],
    channel_legend_entries: list[ChannelLegendEntry],
) -> None:
    """Accumulate channel-to-color mappings for a deferred combined legend."""
    channel_names = [str(ch) for ch in channels]
    if len(set(channel_names)) != len(channel_names):
        logger.warning("channels_as_legend: duplicate channel names detected; skipping legend entries.")
        return

    color_hexes = [matplotlib.colors.to_hex(c, keep_alpha=False) for c in seed_colors]
    for name, color in zip(channel_names, color_hexes, strict=True):
        channel_legend_entries.append(ChannelLegendEntry(channel_name=name, color_hex=color))


def _draw_channel_legend(
    ax: matplotlib.axes.SubplotBase,
    entries: list[ChannelLegendEntry],
    legend_params: LegendParams,
    fig_params: FigParams,
) -> None:
    """Draw a single combined categorical legend from accumulated channel entries.

    Because ``_add_categorical_legend`` adds invisible labeled scatter artists,
    calling it here automatically merges with any earlier legend entries
    (e.g. from labels or shapes) on the same axes via ``ax.legend()``.

    ``multi_panel`` is only set when no prior legend exists on the axis,
    to avoid shrinking the axes twice (once for labels/shapes, once for
    channels).
    """
    # Deduplicate: if the same channel name appears twice, keep the last color
    palette_dict: dict[str, str] = {}
    for entry in entries:
        palette_dict[entry.channel_name] = entry.color_hex

    legend_loc = legend_params.legend_loc
    if legend_loc == "on data":
        logger.warning(
            "legend_loc='on data' is not supported for channel legends (no scatter coordinates); "
            "falling back to 'right margin'."
        )
        legend_loc = "right margin"

    categories = pd.Categorical(list(palette_dict))

    path_effect = (
        [patheffects.withStroke(linewidth=legend_params.legend_fontoutline, foreground="w")]
        if legend_params.legend_fontoutline is not None
        else []
    )

    # Only apply multi_panel shrink if no legend already exists on this axis
    # (labels/shapes draw their legend during the render loop and already shrink).
    has_existing_legend = ax.get_legend() is not None
    needs_multi_panel = fig_params.axs is not None and not has_existing_legend

    _add_categorical_legend(
        ax,
        categories,
        palette=palette_dict,
        legend_loc=legend_loc,
        legend_fontweight=legend_params.legend_fontweight,
        legend_fontsize=legend_params.legend_fontsize,
        legend_fontoutline=path_effect,
        na_color=["lightgray"],
        na_in_legend=False,
        multi_panel=needs_multi_panel,
    )


def _render_images(
    sdata: sd.SpatialData,
    render_params: ImageRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
    rasterize: bool,
    colorbar_requests: list[ColorbarSpec] | None = None,
    channel_legend_entries: list[ChannelLegendEntry] | None = None,
) -> None:
    _log_context.set("render_images")
    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_tables=False,
    )

    palette = render_params.palette
    img = sdata_filt[render_params.element]
    extent = get_extent(img, coordinate_system=coordinate_system)
    scale = render_params.scale

    # get best scale out of multiscale image
    if isinstance(img, DataTree):
        img = _multiscale_to_spatial_image(
            multiscale_image=img,
            dpi=fig_params.fig.dpi,
            width=fig_params.fig.get_size_inches()[0],
            height=fig_params.fig.get_size_inches()[1],
            scale=scale,
        )
    # rasterize spatial image if necessary to speed up performance
    if rasterize:
        img = _rasterize_if_necessary(
            image=img,
            dpi=fig_params.fig.dpi,
            width=fig_params.fig.get_size_inches()[0],
            height=fig_params.fig.get_size_inches()[1],
            coordinate_system=coordinate_system,
            extent=extent,
        )

    channels = img.coords["c"].values.tolist() if render_params.channel is None else render_params.channel

    # the channel parameter has been previously validated, so when not None, render_params.channel is a list
    assert isinstance(channels, list)

    _, trans_data = _prepare_transformation(img, coordinate_system, ax)

    # --- Apply image transforms ---
    transfunc = render_params.transfunc
    needs_transform = transfunc is not None or render_params.grayscale

    if needs_transform:
        raw = np.stack([img.sel(c=ch).values for ch in channels], axis=0)

        # 1) Apply transfunc (before grayscale)
        if isinstance(transfunc, list):
            if len(transfunc) != raw.shape[0]:
                raise ValueError(
                    f"Length of transfunc list ({len(transfunc)}) must match the number of channels ({raw.shape[0]})."
                )
            raw = np.stack([fn(raw[i]) for i, fn in enumerate(transfunc)], axis=0)
        elif transfunc is not None:
            raw = transfunc(raw)

        # 2) Apply grayscale (after transfunc)
        if render_params.grayscale:
            if raw.shape[0] != 3:
                raise ValueError(
                    f"grayscale=True requires exactly 3 channels"
                    f"{' after transfunc' if transfunc is not None else ''}, "
                    f"got {raw.shape[0]}. Select 3 channels via the 'channel' parameter."
                )
            raw = _grayscale_transform(raw)

        # Rebuild image with new channel coords
        new_channels = list(range(raw.shape[0]))
        img = xr.DataArray(
            data=raw,
            dims=("c", "y", "x"),
            coords={"c": new_channels, "y": img.coords["y"], "x": img.coords["x"]},
        )
        channels = new_channels

    n_channels = len(channels)

    # When grayscale was applied and user didn't provide an explicit cmap,
    # default to "gray" for intuitive single-channel rendering.
    got_multiple_cmaps = isinstance(render_params.cmap_params, list)
    if (
        render_params.grayscale
        and not got_multiple_cmaps
        and isinstance(render_params.cmap_params, CmapParams)
        and render_params.cmap_params.cmap_is_default
    ):
        render_params = dataclasses.replace(
            render_params,
            cmap_params=_prepare_cmap_norm(
                cmap="gray",
                norm=render_params.cmap_params.norm,
                na_color=render_params.cmap_params.na_color,
            ),
        )

    # True if user gave n cmaps for n channels
    got_multiple_cmaps = isinstance(render_params.cmap_params, list)
    if got_multiple_cmaps:
        logger.warning(
            "You're blending multiple cmaps. "
            "If the plot doesn't look like you expect, it might be because your "
            "cmaps go from a given color to 'white', and not to 'transparent'. "
            "Therefore, the 'white' of higher layers will overlay the lower layers. "
            "Consider using 'palette' instead."
        )

    # not using got_multiple_cmaps here because of ruff :(
    if isinstance(render_params.cmap_params, list) and len(render_params.cmap_params) != n_channels:
        raise ValueError("If 'cmap' is provided, its length must match the number of channels.")

    # Detect RGB(A) images by channel names — skip when user overrides with palette/cmap
    is_rgb, has_alpha = _is_rgb_image(channels)
    has_explicit_cmap = (
        isinstance(render_params.cmap_params, CmapParams) and not render_params.cmap_params.cmap_is_default
    )
    if is_rgb and palette is None and not got_multiple_cmaps and not has_explicit_cmap:
        coord_map = {str(c).lower(): c for c in channels}
        ordered = [coord_map[ch] for ch in ("r", "g", "b")]

        # Apply norm per channel if user provided one, otherwise normalize by dtype
        user_norm = (
            render_params.cmap_params.norm
            if isinstance(render_params.cmap_params, CmapParams)
            and isinstance(render_params.cmap_params.norm, Normalize)
            and (render_params.cmap_params.norm.vmin is not None or render_params.cmap_params.norm.vmax is not None)
            else None
        )

        if user_norm is not None:
            rgb_layers = []
            for ch in ordered:
                ch_norm = copy(user_norm)
                rgb_layers.append(np.clip(ch_norm(img.sel(c=ch).values).astype(np.float64), 0, 1))
            stacked = np.stack(rgb_layers, axis=-1)
        else:
            stacked = _normalize_dtype_to_float(np.moveaxis(img.sel(c=ordered).values, 0, -1))

        show_kwargs: dict[str, Any] = {"zorder": render_params.zorder}

        if has_alpha and render_params.alpha == 1.0:
            alpha_layer = _normalize_dtype_to_float(img.sel(c=coord_map["a"]).values)
            stacked = np.concatenate([stacked, alpha_layer[..., np.newaxis]], axis=-1)
        else:
            show_kwargs["alpha"] = render_params.alpha
            if has_alpha:
                logger.info(
                    "Image has an alpha channel, but an explicit 'alpha' value was provided. "
                    "Using the user-specified alpha=%.2f instead of the per-pixel alpha from the data.",
                    render_params.alpha,
                )

        _ax_show_and_transform(stacked, trans_data, ax, **show_kwargs)
        return

    # 1) Image has only 1 channel
    if n_channels == 1 and not isinstance(render_params.cmap_params, list):
        layer = img.sel(c=channels[0]).squeeze() if isinstance(channels[0], str) else img.isel(c=channels[0]).squeeze()

        cmap = (
            _get_linear_colormap(palette, "k")[0]
            if isinstance(palette, list) and all(isinstance(p, str) for p in palette)
            else render_params.cmap_params.cmap
        )

        # Overwrite alpha in cmap: https://stackoverflow.com/a/10127675
        cmap._init()
        cmap._lut[:, -1] = render_params.alpha

        # norm needs to be passed directly to ax.imshow(). If we normalize before, that method would always clip.
        _ax_show_and_transform(
            layer,
            trans_data,
            ax,
            cmap=cmap,
            zorder=render_params.zorder,
            norm=render_params.cmap_params.norm,
        )

        wants_colorbar = _should_request_colorbar(
            render_params.colorbar,
            has_mappable=n_channels == 1,
            is_continuous=True,
            auto_condition=n_channels == 1,
        )
        if wants_colorbar and legend_params.colorbar and colorbar_requests is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=render_params.cmap_params.norm)
            colorbar_requests.append(
                ColorbarSpec(
                    ax=ax,
                    mappable=sm,
                    params=render_params.colorbar_params,
                    label=_resolve_colorbar_label(
                        render_params.colorbar_params,
                        str(channels[0]),
                        is_default_channel_name=isinstance(channels[0], (int, np.integer)),
                    ),
                )
            )

    # 2) Image has any number of channels but 1
    else:
        if n_channels >= 5 and render_params.colorbar == "auto":
            logger.info(
                "Colorbars are not shown by default for images with 5+ channels. "
                "To show individual channel colorbars, render channels separately "
                "with `channel=<name>` and `colorbar=True`."
            )
        layers = {}
        for ch_idx, ch in enumerate(channels):
            layers[ch] = img.sel(c=ch).copy(deep=True).squeeze()
            if isinstance(render_params.cmap_params, list):
                ch_norm = render_params.cmap_params[ch_idx].norm
            else:
                ch_norm = render_params.cmap_params.norm

            # Normalize objects are stateful — always copy to prevent cross-channel mutation
            if isinstance(ch_norm, Normalize):
                ch_norm = copy(ch_norm)

            layers[ch] = ch_norm(layers[ch])

        # Colors for the channel legend (set by each branch if applicable)
        legend_colors: list[str] | None = None

        # 2A) Image has 3 channels, no palette info, and no/only one cmap was given
        if palette is None and n_channels == 3 and not isinstance(render_params.cmap_params, list):
            if render_params.cmap_params.cmap_is_default:  # -> use RGB
                stacked = np.clip(np.stack([layers[ch] for ch in layers], axis=-1), 0, 1)
                legend_colors = ["red", "green", "blue"]
            else:  # -> use given cmap for each channel
                channel_cmaps = [render_params.cmap_params.cmap] * n_channels
                stacked = (
                    np.stack(
                        [channel_cmaps[ind](layers[ch]) for ind, ch in enumerate(channels)],
                        0,
                    ).sum(0)
                    / n_channels
                )
                stacked = stacked[:, :, :3]
                logger.warning(
                    "One cmap was given for multiple channels and is now used for each channel. "
                    "You're blending multiple cmaps. "
                    "If the plot doesn't look like you expect, it might be because your "
                    "cmaps go from a given color to 'white', and not to 'transparent'. "
                    "Therefore, the 'white' of higher layers will overlay the lower layers. "
                    "Consider using 'palette' instead."
                )

            _ax_show_and_transform(
                stacked,
                trans_data,
                ax,
                render_params.alpha,
                zorder=render_params.zorder,
            )

        # 2B) Image has n channels, no palette/cmap info -> sample n categorical colors
        elif palette is None and not got_multiple_cmaps:
            # overwrite if n_channels == 2 for intuitive result
            if n_channels == 2:
                seed_colors = ["#ff0000ff", "#00ff00ff"]
                channel_cmaps = [_get_linear_colormap([c], "k")[0] for c in seed_colors]
                colored = np.stack(
                    [channel_cmaps[ch_ind](layers[ch]) for ch_ind, ch in enumerate(channels)],
                    0,
                ).sum(0)
                colored = np.clip(colored[:, :, :3], 0, 1)
            elif n_channels == 3:
                seed_colors = _get_colors_for_categorical_obs(list(range(n_channels)))
                channel_cmaps = [_get_linear_colormap([c], "k")[0] for c in seed_colors]
                colored = np.stack(
                    [channel_cmaps[ind](layers[ch]) for ind, ch in enumerate(channels)],
                    0,
                ).sum(0)
                colored = np.clip(colored[:, :, :3], 0, 1)
            else:
                if isinstance(render_params.cmap_params, list):
                    cmap_is_default = render_params.cmap_params[0].cmap_is_default
                else:
                    cmap_is_default = render_params.cmap_params.cmap_is_default

                if cmap_is_default:
                    seed_colors = _get_colors_for_categorical_obs(list(range(n_channels)))
                else:
                    # Sample n_channels colors evenly from the colormap
                    if isinstance(render_params.cmap_params, list):
                        seed_colors = [
                            render_params.cmap_params[i].cmap(i / (n_channels - 1)) for i in range(n_channels)
                        ]
                    else:
                        seed_colors = [render_params.cmap_params.cmap(i / (n_channels - 1)) for i in range(n_channels)]
                channel_cmaps = [_get_linear_colormap([c], "k")[0] for c in seed_colors]

                # Stack (n_channels, height, width) -> (height*width, n_channels)
                H, W = next(iter(layers.values())).shape
                comp_rgb = np.zeros((H, W, 3), dtype=float)

                # For each channel: map to RGBA, apply constant alpha, then add
                for ch_idx, ch in enumerate(channels):
                    layer_arr = layers[ch]
                    rgba = channel_cmaps[ch_idx](layer_arr)
                    rgba[..., 3] = render_params.alpha
                    comp_rgb += rgba[..., :3] * rgba[..., 3][..., None]

                colored = np.clip(comp_rgb, 0, 1)
                logger.info(
                    f"Your image has {n_channels} channels. Sampling categorical colors and using "
                    f"multichannel strategy 'stack' to render."
                )  # TODO: update when pca is added as strategy

            legend_colors = seed_colors

            _ax_show_and_transform(
                colored,
                trans_data,
                ax,
                render_params.alpha,
                zorder=render_params.zorder,
            )

        # 2C) Image has n channels and palette info
        elif palette is not None and not got_multiple_cmaps:
            if len(palette) != n_channels:
                raise ValueError("If 'palette' is provided, its length must match the number of channels.")

            channel_cmaps = [_get_linear_colormap([c], "k")[0] for c in palette if isinstance(c, str)]
            colored = np.stack([channel_cmaps[i](layers[c]) for i, c in enumerate(channels)], 0).sum(0)
            colored = np.clip(colored[:, :, :3], 0, 1)

            legend_colors = list(palette)

            _ax_show_and_transform(
                colored,
                trans_data,
                ax,
                render_params.alpha,
                zorder=render_params.zorder,
            )

        elif palette is None and got_multiple_cmaps:
            channel_cmaps = [cp.cmap for cp in render_params.cmap_params]  # type: ignore[union-attr]
            colored = (
                np.stack(
                    [channel_cmaps[ind](layers[ch]) for ind, ch in enumerate(channels)],
                    0,
                ).sum(0)
                / n_channels
            )
            colored = colored[:, :, :3]

            legend_colors = [matplotlib.colors.to_hex(cm(0.75)) for cm in channel_cmaps]

            _ax_show_and_transform(
                colored,
                trans_data,
                ax,
                render_params.alpha,
                zorder=render_params.zorder,
            )

        # 2D) Image has n channels, no palette but cmap info
        elif palette is not None and got_multiple_cmaps:
            raise ValueError("If 'palette' is provided, 'cmap' must be None.")

        # Collect channel legend entries (single point for all multi-channel paths)
        if render_params.channels_as_legend and channel_legend_entries is not None:
            if legend_colors is not None:
                _collect_channel_legend_entries(channels, legend_colors, channel_legend_entries)
            else:
                logger.warning(
                    "channels_as_legend requires distinct per-channel colors; "
                    "ignored when a single cmap is shared across channels. "
                    "Use 'palette' or a list of cmaps instead."
                )


def _render_labels(
    sdata: sd.SpatialData,
    render_params: LabelsRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
    rasterize: bool,
    colorbar_requests: list[ColorbarSpec] | None = None,
) -> None:
    _log_context.set("render_labels")
    element = render_params.element
    table_name = render_params.table_name
    table_layer = render_params.table_layer
    palette = render_params.palette
    col_for_color = render_params.col_for_color
    groups = render_params.groups
    scale = render_params.scale

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_tables=bool(table_name),
    )

    label = sdata_filt.labels[element]
    extent = get_extent(label, coordinate_system=coordinate_system)

    # get best scale out of multiscale label
    if isinstance(label, DataTree):
        label = _multiscale_to_spatial_image(
            multiscale_image=label,
            dpi=fig_params.fig.dpi,
            width=fig_params.fig.get_size_inches()[0],
            height=fig_params.fig.get_size_inches()[1],
            scale=scale,
            is_label=True,
        )

    # rasterize spatial image if necessary to speed up performance
    if rasterize:
        label = _rasterize_if_necessary(
            image=label,
            dpi=fig_params.fig.dpi,
            width=fig_params.fig.get_size_inches()[0],
            height=fig_params.fig.get_size_inches()[1],
            coordinate_system=coordinate_system,
            extent=extent,
        )

        # the above adds a useless c dimension of 1 (y, x) -> (1, y, x)
        label = label.squeeze()

    if table_name is None:
        instance_id = np.unique(label)
        table = None
    else:
        _, region_key, instance_key = get_table_keys(sdata[table_name])
        table = sdata[table_name][sdata[table_name].obs[region_key].isin([element])]

        # get instance id based on subsetted table
        instance_id = np.unique(table.obs[instance_key].values)

    _, trans_data = _prepare_transformation(label, coordinate_system, ax)

    na_color = (
        render_params.color
        if col_for_color is None and render_params.color is not None
        else render_params.cmap_params.na_color
    )
    color_source_vector, color_vector, categorical = _set_color_source_vec(
        sdata=sdata_filt,
        element=label,
        element_name=element,
        value_to_plot=col_for_color,
        groups=groups,
        palette=palette,
        na_color=na_color,
        cmap_params=render_params.cmap_params,
        table_name=table_name,
        table_layer=table_layer,
        render_type="labels",
        coordinate_system=coordinate_system,
    )

    # rasterize could have removed labels from label
    # only problematic if color is specified
    if rasterize and col_for_color is not None:
        labels_in_rasterized_image = np.unique(label.values)
        mask = np.isin(instance_id, labels_in_rasterized_image)
        instance_id = instance_id[mask]
        color_vector = color_vector[mask]
        if isinstance(color_vector.dtype, pd.CategoricalDtype):
            color_vector = color_vector.remove_unused_categories()
            assert color_source_vector is not None
            color_source_vector = color_source_vector[mask]
        else:
            assert color_source_vector is None

    _warn_groups_ignored_continuous(groups, color_source_vector, col_for_color)

    if groups is not None and color_source_vector is not None:
        _warn_missing_groups(groups, color_source_vector, col_for_color)

    # When groups are specified, zero out non-matching label IDs so they render as background.
    # Only show non-matching labels if the user explicitly sets na_color.
    _na = render_params.cmap_params.na_color
    if (
        groups is not None
        and categorical
        and color_source_vector is not None
        and (_na.default_color_set or _na.is_fully_transparent())
    ):
        keep_vec = color_source_vector.isin(groups)
        matching_ids = instance_id[keep_vec]
        keep_mask = np.isin(label.values, matching_ids)
        label = label.copy()
        label.values[~keep_mask] = 0
        instance_id = instance_id[keep_vec]
        color_source_vector = color_source_vector[keep_vec]
        color_vector = color_vector[keep_vec]
        if isinstance(color_vector.dtype, pd.CategoricalDtype):
            color_vector = color_vector.remove_unused_categories()

    def _draw_labels(
        seg_erosionpx: int | None,
        seg_boundaries: bool,
        alpha: float,
        outline_color: Color | None = None,
    ) -> matplotlib.image.AxesImage:
        labels = _map_color_seg(
            seg=label.values,
            cell_id=instance_id,
            color_vector=color_vector,
            color_source_vector=color_source_vector,
            cmap_params=render_params.cmap_params,
            seg_erosionpx=seg_erosionpx,
            seg_boundaries=seg_boundaries,
            na_color=na_color,
            outline_color=outline_color,
        )

        _cax = ax.imshow(
            labels,
            rasterized=True,
            cmap=None if categorical else render_params.cmap_params.cmap,
            norm=None if categorical else render_params.cmap_params.norm,
            alpha=alpha,
            origin="lower",
            zorder=render_params.zorder,
        )
        _cax.set_transform(trans_data)
        return _cax

    # When color is a literal (col_for_color is None) and no explicit outline_color,
    # use the literal color for outlines so they are visible (e.g., color='white' on
    # a dark background). When color is data-driven, outlines inherit the per-label
    # colors from label2rgb (outline_color stays None).
    effective_outline_color = render_params.outline_color
    if effective_outline_color is None and col_for_color is None and render_params.color is not None:
        effective_outline_color = render_params.color

    # default case: no contour, just fill
    # since contour_px is passed to skimage.morphology.erosion to create the contour,
    # any border thickness is only within the label, not outside. Therefore, the case
    # of fill_alpha == outline_alpha is equivalent to fill-only
    if (render_params.fill_alpha > 0.0 and render_params.outline_alpha == 0.0) or (
        render_params.fill_alpha == render_params.outline_alpha
    ):
        cax = _draw_labels(seg_erosionpx=None, seg_boundaries=False, alpha=render_params.fill_alpha)
        alpha_to_decorate_ax = render_params.fill_alpha

    # outline-only case
    elif render_params.fill_alpha == 0.0 and render_params.outline_alpha > 0.0:
        cax = _draw_labels(
            seg_erosionpx=render_params.contour_px,
            seg_boundaries=True,
            alpha=render_params.outline_alpha,
            outline_color=effective_outline_color,
        )
        alpha_to_decorate_ax = render_params.outline_alpha

    # pretty case: both outline and infill
    elif render_params.fill_alpha > 0.0 and render_params.outline_alpha > 0.0:
        # first plot the infill ...
        cax_infill = _draw_labels(seg_erosionpx=None, seg_boundaries=False, alpha=render_params.fill_alpha)

        # ... then overlay the contour
        cax_contour = _draw_labels(
            seg_erosionpx=render_params.contour_px,
            seg_boundaries=True,
            alpha=render_params.outline_alpha,
            outline_color=effective_outline_color,
        )

        # pass the less-transparent _cax for the legend
        cax = cax_infill if render_params.fill_alpha > render_params.outline_alpha else cax_contour
        alpha_to_decorate_ax = max(render_params.fill_alpha, render_params.outline_alpha)

    else:
        raise ValueError("Parameters 'fill_alpha' and 'outline_alpha' cannot both be 0.")

    colorbar_requested = _should_request_colorbar(
        render_params.colorbar,
        has_mappable=cax is not None,
        is_continuous=col_for_color is not None and color_source_vector is None and not categorical,
    )

    _ = _decorate_axs(
        ax=ax,
        cax=cax,
        fig_params=fig_params,
        adata=table,
        value_to_plot=col_for_color,
        color_source_vector=color_source_vector,
        color_vector=color_vector,
        palette=palette,
        alpha=alpha_to_decorate_ax,
        na_color=render_params.cmap_params.na_color,
        legend_fontsize=legend_params.legend_fontsize,
        legend_fontweight=legend_params.legend_fontweight,
        legend_loc=legend_params.legend_loc,
        legend_fontoutline=legend_params.legend_fontoutline,
        na_in_legend=(legend_params.na_in_legend if groups is None else len(groups) == len(set(color_vector))),
        colorbar=colorbar_requested and legend_params.colorbar,
        colorbar_params=render_params.colorbar_params,
        colorbar_requests=colorbar_requests,
        colorbar_label=_resolve_colorbar_label(
            render_params.colorbar_params,
            col_for_color if isinstance(col_for_color, str) else None,
        ),
        scalebar_dx=scalebar_params.scalebar_dx,
        scalebar_units=scalebar_params.scalebar_units,
        # scalebar_kwargs=scalebar_params.scalebar_kwargs,
    )
