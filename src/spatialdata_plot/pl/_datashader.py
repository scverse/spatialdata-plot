"""Datashader aggregation, shading, and rendering helpers.

Shared by ``_render_shapes`` and ``_render_points`` in ``render.py``.
"""

from __future__ import annotations

from copy import copy
from typing import Any, Literal

import dask.dataframe as dd
import datashader as ds
import matplotlib
import matplotlib.colors
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from spatialdata_plot._logging import logger
from spatialdata_plot.pl.render_params import Color, FigParams, ShapesRenderParams
from spatialdata_plot.pl.utils import (
    _DS_REDUCTION_FUNCS,
    _ax_show_and_transform,
    _convert_alpha_to_datashader_range,
    _create_image_from_datashader_result,
    _datashader_aggregate_with_function,
    _datashader_map_aggregate_to_color,
    _datshader_get_how_kw_for_spread,
    _hex_no_alpha,
)

# ---------------------------------------------------------------------------
# Type aliases and constants
# ---------------------------------------------------------------------------

_DsReduction = Literal["sum", "mean", "any", "count", "std", "var", "max", "min"]

# Sentinel category name used in datashader categorical paths to represent
# missing (NaN) values.  Must not collide with realistic user category names.
_DS_NAN_CATEGORY = "ds_nan"

# Private column name under which the outline color vector is attached to the
# datashader rasterizer element. Must not collide with a real user column;
# the leading/trailing dunders are deliberate.
_OUTLINE_INTERNAL_COL = "__sdp_outline_col__"

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _apply_user_alpha(result: ds.tf.Image | np.ndarray, alpha: float) -> ds.tf.Image | np.ndarray:
    """Scale the alpha channel of a datashader shade result by ``alpha``.

    ``ds.tf.shade(min_alpha=...)`` is a floor, not a scale, so user alpha
    must be applied post-hoc. See #617.
    """
    if alpha >= 1.0 or result is None:
        return result
    arr = result if isinstance(result, np.ndarray) else result.to_numpy().base
    if arr is None or arr.ndim != 3 or arr.shape[-1] != 4:
        return result
    arr[..., 3] = (arr[..., 3].astype(np.float32) * alpha).astype(np.uint8)
    return result


def _coerce_categorical_source(series: pd.Series | dd.Series) -> pd.Categorical:
    """Return a ``pd.Categorical`` from a pandas or dask Series."""
    if isinstance(series, dd.Series):
        if isinstance(series.dtype, pd.CategoricalDtype) and getattr(series.cat, "known", True) is False:
            series = series.cat.as_known()
        series = series.compute()
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.array
    return pd.Categorical(series)


def _build_datashader_color_key(
    cat_series: pd.Categorical,
    color_vector: Any,
    na_color_hex: str,
) -> dict[str, str]:
    """Build a datashader ``color_key`` dict from a categorical series and its color vector."""
    na_hex = _hex_no_alpha(na_color_hex) if na_color_hex.startswith("#") else na_color_hex
    colors_arr = np.asarray(color_vector, dtype=object)
    categories = np.asarray(cat_series.categories, dtype=str)
    codes = np.asarray(cat_series.codes)

    if len(colors_arr) != len(codes):
        logger.warning(
            f"color_vector length ({len(color_vector)}) does not match categorical series length "
            f"({len(codes)}); some categories may receive the na_color fallback."
        )

    # Use np.unique to find the first occurrence of each category in one pass,
    # avoiding a Python loop over all points.  See #379.
    unique_codes, first_indices = np.unique(codes, return_index=True)

    first_color: dict[str, str] = {}
    for code, idx in zip(unique_codes, first_indices, strict=True):
        if code < 0 or idx >= len(colors_arr):
            continue
        c = colors_arr[idx]
        first_color[categories[code]] = _hex_no_alpha(c) if isinstance(c, str) and c.startswith("#") else c

    return {cat: first_color.get(cat, na_hex) for cat in categories}


def _inject_ds_nan_sentinel(series: pd.Series, sentinel: str = _DS_NAN_CATEGORY) -> pd.Series:
    """Add a sentinel category for NaN values in a categorical series.

    Safely handles series that are not yet categorical, dask-backed
    categoricals that need ``as_known()``, and series that already
    contain the sentinel.
    """
    if not isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype("category")
    if hasattr(series.cat, "as_known"):
        series = series.cat.as_known()
    if sentinel not in series.cat.categories:
        series = series.cat.add_categories(sentinel)
    return series.fillna(sentinel)


# ---------------------------------------------------------------------------
# Pipeline helpers (aggregate -> norm -> shade -> render)
# ---------------------------------------------------------------------------


def _ds_aggregate(
    cvs: Any,
    transformed_element: Any,
    col_for_color: str | None,
    color_by_categorical: bool,
    ds_reduction: _DsReduction | None,
    default_reduction: _DsReduction,
    geom_type: Literal["points", "shapes"],
) -> tuple[Any, tuple[Any, Any] | None, Any | None]:
    """Aggregate spatial elements with datashader.

    Dispatches between categorical (ds.by), continuous (reduction function),
    and no-color (ds.count) aggregation modes.

    Returns (agg, reduction_bounds, nan_agg).
    """
    reduction_bounds = None
    nan_agg = None

    def _agg_call(element: Any, agg_func: Any) -> Any:
        if geom_type == "shapes":
            return cvs.polygons(element, geometry="geometry", agg=agg_func)
        return cvs.points(element, "x", "y", agg=agg_func)

    if col_for_color is not None:
        if color_by_categorical:
            if ds_reduction is not None:
                logger.warning(
                    f'ds_reduction="{ds_reduction}" is ignored for categorical data; '
                    "categorical aggregation always uses count."
                )
            transformed_element[col_for_color] = _inject_ds_nan_sentinel(transformed_element[col_for_color])
            agg = _agg_call(transformed_element, ds.by(col_for_color, ds.count()))
        else:
            reduction_name = ds_reduction if ds_reduction is not None else default_reduction
            logger.info(
                f'Using the datashader reduction "{reduction_name}". "max" will give an output '
                "very close to the matplotlib result."
            )
            agg = _datashader_aggregate_with_function(
                reduction_name, cvs, transformed_element, col_for_color, geom_type
            )
            reduction_bounds = (agg.min(), agg.max())

            nan_elements = transformed_element[transformed_element[col_for_color].isnull()]
            if len(nan_elements) > 0:
                nan_agg = _datashader_aggregate_with_function("any", cvs, nan_elements, None, geom_type)
    else:
        agg = _agg_call(transformed_element, ds.count())

    return agg, reduction_bounds, nan_agg


def _apply_ds_norm(
    agg: Any,
    norm: Normalize,
) -> tuple[Any, list[float] | None]:
    """Apply norm vmin/vmax to a datashader aggregate.

    When vmin == vmax, maps the value to 0.5 using an artificial [0, 1] span.
    Returns (agg, color_span) where color_span is None if no norm was set.
    """
    if norm.vmin is None and norm.vmax is None:
        return agg, None
    norm.vmin = np.min(agg) if norm.vmin is None else norm.vmin
    norm.vmax = np.max(agg) if norm.vmax is None else norm.vmax
    color_span: list[float] = [norm.vmin, norm.vmax]
    if norm.vmin == norm.vmax:
        color_span = [0, 1]
        if norm.clip:
            agg = (agg - agg) + 0.5
        else:
            agg = agg.where((agg >= norm.vmin) | (np.isnan(agg)), other=-1)
            agg = agg.where((agg <= norm.vmin) | (np.isnan(agg)), other=2)
            agg = agg.where((agg != norm.vmin) | (np.isnan(agg)), other=0.5)
    return agg, color_span


def _build_color_key(
    transformed_element: Any,
    col_for_color: str | None,
    color_by_categorical: bool,
    color_vector: Any,
    na_color_hex: str,
) -> dict[str, str] | None:
    """Build a datashader color key mapping categories to hex colors.

    Returns None when not coloring by a categorical column.
    """
    if not color_by_categorical or col_for_color is None:
        return None
    cat_series = _coerce_categorical_source(transformed_element[col_for_color])
    return _build_datashader_color_key(cat_series, color_vector, na_color_hex)


def _ds_shade_continuous(
    agg: Any,
    color_span: list[float] | None,
    norm: Normalize,
    cmap: Any,
    alpha: float,
    reduction_bounds: tuple[Any, Any] | None,
    nan_agg: Any | None,
    na_color_hex: str,
    spread_px: int | None = None,
    ds_reduction: _DsReduction | None = None,
    how: str = "linear",
) -> tuple[Any, Any | None, tuple[Any, Any] | None]:
    """Shade a continuous datashader aggregate, optionally applying spread and NaN coloring.

    Returns (shaded, nan_shaded, reduction_bounds).
    """
    if spread_px is not None:
        spread_how = _datshader_get_how_kw_for_spread(ds_reduction)
        agg = ds.tf.spread(agg, px=spread_px, how=spread_how)
        reduction_bounds = (agg.min(), agg.max())

    ds_cmap = cmap
    if (
        reduction_bounds is not None
        and reduction_bounds[0] == reduction_bounds[1]
        and (color_span is None or color_span != [0, 1])
    ):
        ds_cmap = matplotlib.colors.to_hex(cmap(0.0), keep_alpha=False)
        reduction_bounds = (
            reduction_bounds[0],
            reduction_bounds[0] + 1,
        )

    shaded = _datashader_map_aggregate_to_color(
        agg,
        cmap=ds_cmap,
        min_alpha=_convert_alpha_to_datashader_range(alpha),
        span=color_span,
        clip=norm.clip,
        how=how,
    )
    shaded = _apply_user_alpha(shaded, alpha)

    nan_shaded = None
    if nan_agg is not None:
        shade_kwargs: dict[str, Any] = {"cmap": na_color_hex, "how": "linear"}
        if spread_px is not None:
            nan_agg = ds.tf.spread(nan_agg, px=spread_px, how="max")
        else:
            # only shapes (no spread) pass min_alpha for NaN shading
            shade_kwargs["min_alpha"] = _convert_alpha_to_datashader_range(alpha)
        nan_shaded = ds.tf.shade(nan_agg, **shade_kwargs)
        nan_shaded = _apply_user_alpha(nan_shaded, alpha)

    return shaded, nan_shaded, reduction_bounds


def _ds_shade_categorical(
    agg: Any,
    color_key: dict[str, str] | None,
    color_vector: Any,
    alpha: float,
    spread_px: int | None = None,
    how: str = "linear",
    density: bool = False,
) -> Any:
    """Shade a categorical or no-color datashader aggregate."""
    ds_cmap = None
    if color_key is None and color_vector is not None:
        ds_cmap = color_vector[0]
        if isinstance(ds_cmap, str) and ds_cmap[0] == "#":
            ds_cmap = _hex_no_alpha(ds_cmap)

    # The default min_alpha (~254) is a near-full-opacity floor — right for scatter
    # plots, but it collapses the count-driven alpha range and makes categorical
    # density read as a flat hue cloud. Drop the floor under density so per-pixel
    # alpha can actually encode count. A small non-zero floor (~15%) keeps the
    # sparse edges visible under density_how="linear" instead of vanishing.
    min_alpha = 40.0 if density else _convert_alpha_to_datashader_range(alpha)

    agg_to_shade = ds.tf.spread(agg, px=spread_px) if spread_px is not None else agg
    shaded = _datashader_map_aggregate_to_color(
        agg_to_shade,
        cmap=ds_cmap,
        color_key=color_key,
        min_alpha=min_alpha,
        how=how,
    )
    return _apply_user_alpha(shaded, alpha)


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------


def _render_ds_image(
    ax: matplotlib.axes.SubplotBase,
    shaded: Any,
    factor: float,
    zorder: int,
    x_min: float = 0.0,
    y_min: float = 0.0,
    nan_result: Any | None = None,
) -> Any:
    """Render a shaded datashader image onto matplotlib axes, with optional NaN overlay.

    Alpha is NOT passed to ``ax.imshow`` because it is already encoded in
    the RGBA channels produced by ``ds.tf.shade(min_alpha=...)``.  Passing
    it again would apply transparency twice (see #367).
    """
    if nan_result is not None:
        rgba_nan, trans_nan = _create_image_from_datashader_result(nan_result, factor, ax, x_min, y_min)
        _ax_show_and_transform(rgba_nan, trans_nan, ax, zorder=zorder)
    rgba_image, trans_data = _create_image_from_datashader_result(shaded, factor, ax, x_min, y_min)
    return _ax_show_and_transform(rgba_image, trans_data, ax, zorder=zorder)


def _render_ds_outlines(
    cvs: Any,
    transformed_element: Any,
    render_params: ShapesRenderParams,
    fig_params: FigParams,
    ax: matplotlib.axes.SubplotBase,
    factor: float,
    x_min: float = 0.0,
    y_min: float = 0.0,
    outline_color_vector: Any | None = None,
    outline_color_source_vector: pd.Series | None = None,
) -> None:
    """Aggregate, shade, and render shape outlines (outer and inner) with datashader.

    When ``outline_color_vector`` is provided, the outer outline is colored per-shape
    via ``ds.by`` (categorical) or a numeric reduction (continuous) instead of a
    single literal color. The two-outline form is rejected at validation, so this
    only affects the outer outline.
    """
    ds_lw_factor = fig_params.fig.dpi / 72
    assert len(render_params.outline_alpha) == 2  # noqa: S101

    for idx, (outline_color_obj, linewidth) in enumerate(
        [
            (render_params.outline_params.outer_outline_color, render_params.outline_params.outer_outline_linewidth),
            (render_params.outline_params.inner_outline_color, render_params.outline_params.inner_outline_linewidth),
        ]
    ):
        alpha = render_params.outline_alpha[idx]
        if alpha <= 0:
            continue
        if idx == 0 and outline_color_vector is not None:
            _render_ds_outline_by_column(
                cvs=cvs,
                transformed_element=transformed_element,
                outline_color_vector=outline_color_vector,
                outline_color_source_vector=outline_color_source_vector,
                cmap_params=render_params.cmap_params,
                ds_reduction=render_params.ds_reduction,
                line_width=linewidth * ds_lw_factor,
                alpha=alpha,
                fig_params=fig_params,
                ax=ax,
                factor=factor,
                x_min=x_min,
                y_min=y_min,
                zorder=render_params.zorder,
            )
            continue
        agg_outline = cvs.line(
            transformed_element,
            geometry="geometry",
            line_width=linewidth * ds_lw_factor,
        )
        if isinstance(outline_color_obj, Color):
            shaded = ds.tf.shade(
                agg_outline,
                cmap=outline_color_obj.get_hex(),
                min_alpha=_convert_alpha_to_datashader_range(alpha),
                how="linear",
            )
            shaded = _apply_user_alpha(shaded, alpha)
            rgba, trans = _create_image_from_datashader_result(shaded, factor, ax, x_min, y_min)
            _ax_show_and_transform(rgba, trans, ax, zorder=render_params.zorder)


def _render_ds_outline_by_column(
    cvs: Any,
    transformed_element: Any,
    outline_color_vector: Any | None,
    outline_color_source_vector: pd.Series | None,
    cmap_params: Any,
    ds_reduction: _DsReduction | None,
    line_width: float,
    alpha: float,
    fig_params: FigParams,
    ax: matplotlib.axes.SubplotBase,
    factor: float,
    x_min: float,
    y_min: float,
    zorder: int,
) -> None:
    """Aggregate + shade an outline colored by an obs column via datashader.

    Two-outline form is not supported for column-driven outline coloring,
    so this only renders the outer outline.
    """
    color_by_categorical = outline_color_source_vector is not None
    na_color_hex = _hex_no_alpha(cmap_params.na_color.get_hex())

    # Attach the outline vector under a private column name so a fill column with
    # the same key never gets overwritten. Use .assign() to avoid mutating the
    # caller's dataframe.
    if color_by_categorical:
        cat_series = pd.Categorical(outline_color_source_vector)
        attach_value: Any = _inject_ds_nan_sentinel(pd.Series(cat_series))
    else:
        attach_value = np.asarray(outline_color_vector)
    transformed_element = transformed_element.assign(**{_OUTLINE_INTERNAL_COL: attach_value})

    if color_by_categorical:
        agg_outline = cvs.line(
            transformed_element,
            geometry="geometry",
            agg=ds.by(_OUTLINE_INTERNAL_COL, ds.count()),
            line_width=line_width,
        )
        color_key = _build_datashader_color_key(
            _coerce_categorical_source(transformed_element[_OUTLINE_INTERNAL_COL]),
            outline_color_vector,
            na_color_hex,
        )
        shaded = ds.tf.shade(
            agg_outline,
            color_key=color_key,
            min_alpha=_convert_alpha_to_datashader_range(alpha),
            how="linear",
        )
    else:
        reduction_name = ds_reduction if ds_reduction is not None else "max"
        try:
            reduction_function = _DS_REDUCTION_FUNCS[reduction_name](column=_OUTLINE_INTERNAL_COL)
        except KeyError as e:
            raise ValueError(
                f"Reduction '{reduction_name}' is not supported. Use one of: {', '.join(_DS_REDUCTION_FUNCS.keys())}."
            ) from e
        agg_outline = cvs.line(
            transformed_element,
            geometry="geometry",
            agg=reduction_function,
            line_width=line_width,
        )
        # Apply the user-provided norm (vmin/vmax) the same way the fill path does so
        # an explicit Normalize takes effect for the outline cmap.
        norm = copy(cmap_params.norm)
        agg_outline, color_span = _apply_ds_norm(agg_outline, norm)
        shaded = ds.tf.shade(
            agg_outline,
            cmap=cmap_params.cmap,
            span=color_span,
            min_alpha=_convert_alpha_to_datashader_range(alpha),
            how="linear",
        )

    shaded = _apply_user_alpha(shaded, alpha)
    rgba, trans = _create_image_from_datashader_result(shaded, factor, ax, x_min, y_min)
    _ax_show_and_transform(rgba, trans, ax, zorder=zorder)


def _build_ds_colorbar(
    reduction_bounds: tuple[Any, Any] | None,
    norm: Normalize,
    cmap: Any,
) -> ScalarMappable | None:
    """Create a ScalarMappable for the colorbar from datashader reduction bounds.

    Returns None if there is no continuous reduction.
    """
    if reduction_bounds is None:
        return None
    vmin = reduction_bounds[0].values if norm.vmin is None else norm.vmin
    vmax = reduction_bounds[1].values if norm.vmax is None else norm.vmax
    if (norm.vmin is not None or norm.vmax is not None) and norm.vmin == norm.vmax:
        assert norm.vmin is not None
        assert norm.vmax is not None
        vmin = norm.vmin - 0.5
        vmax = norm.vmin + 0.5
    return ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap=cmap,
    )
