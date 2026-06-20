"""Datashader aggregation, shading, and rendering helpers.

Shared by ``_render_shapes`` and ``_render_points`` in ``render.py``.
"""

from __future__ import annotations

from typing import Any, Literal

import dask
import dask.dataframe as dd
import datashader as ds
import matplotlib
import matplotlib.colors
import matplotlib.image
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import pandas as pd
from datashader.core import Canvas
from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)
from matplotlib.transforms import CompositeGenericTransform
from numpy.ma.core import MaskedArray
from spatialdata.models import Image2DModel, SpatialElement
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import Scale, Translation
from spatialdata.transformations.transformations import Sequence as TransformSequence
from xarray import DataArray

from spatialdata_plot._logging import logger
from spatialdata_plot.pl._color import (
    _make_continuous_mappable,
)
from spatialdata_plot.pl.render_params import Color, FigParams, ShapesRenderParams, _DsReduction
from spatialdata_plot.pl.utils import (
    _fast_extent,
    to_hex,
)

# ---------------------------------------------------------------------------
# Type aliases and constants
# ---------------------------------------------------------------------------

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
    uniform_alpha: bool = False,
) -> tuple[Any, Any | None, tuple[Any, Any] | None]:
    """Shade a continuous datashader aggregate, optionally applying spread and NaN coloring.

    Returns (shaded, nan_shaded, reduction_bounds). ``uniform_alpha`` (as_points markers) uses a full
    alpha floor so each dot is one flat colour at ``alpha`` instead of fading by per-pixel count.
    """
    if spread_px is not None:
        # markers overlay (don't accumulate): spread with "max" so overlapping dots keep the true
        # value range instead of summing and inflating the colorbar (see as_points).
        spread_how = "max" if uniform_alpha else _datashader_get_how_kw_for_spread(ds_reduction)
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
        min_alpha=254.0 if uniform_alpha else _convert_alpha_to_datashader_range(alpha),
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
    uniform_alpha: bool = False,
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
    # uniform_alpha (as_points markers): full floor so every dot is one flat colour at
    # `alpha`, matching matplotlib's markers instead of fading single-cell pixels.
    min_alpha = 40.0 if density else 254.0 if uniform_alpha else _convert_alpha_to_datashader_range(alpha)

    agg_to_shade = ds.tf.spread(agg, px=spread_px) if spread_px is not None else agg
    shaded = _datashader_map_aggregate_to_color(
        agg_to_shade,
        cmap=ds_cmap,
        color_key=color_key,
        min_alpha=min_alpha,
        how=how,
    )
    return _apply_user_alpha(shaded, alpha)


def _shade_datashader_aggregate(
    cvs: ds.Canvas,
    frame: Any,
    *,
    col_for_color: str | None,
    color_vector: Any,
    color_by_categorical: bool,
    norm: Normalize | None,
    cmap: Any,
    na_color: Color,
    alpha: float,
    ds_reduction: _DsReduction | None,
    default_reduction: _DsReduction,
    kind: Literal["shapes", "points"],
    spread_px: int | None = None,
    shade_how: str = "linear",
    density: bool = False,
    uniform_alpha: bool = False,
    strip_alpha_hex: bool = False,
) -> tuple[Any, Any | None, tuple[Any, Any] | None, Any]:
    """Aggregate ``frame`` over ``cvs`` and shade it; the core shared by shapes and points.

    Runs the steps every datashader caller repeats: ``_ds_aggregate`` -> ``_apply_ds_norm`` ->
    transparent-NA guard -> ``_build_color_key`` -> categorical/continuous shade dispatch. Returns
    ``(shaded, nan_shaded, reduction_bounds, color_vector)``; ``color_vector`` is returned because the
    points path strips hex alpha off it and the caller's legend must match. Callers keep their
    element-specific prep (geometry transform / point parse), outline rendering, ``_render_ds_image``
    and ``_build_ds_colorbar``.
    """
    agg, reduction_bounds, nan_agg = _ds_aggregate(
        cvs, frame, col_for_color, color_by_categorical, ds_reduction, default_reduction, kind
    )
    agg, color_span = _apply_ds_norm(agg, norm)
    na_color_hex = _hex_no_alpha(na_color.get_hex())
    if na_color.is_fully_transparent():
        nan_agg = None
    color_key = _build_color_key(frame, col_for_color, color_by_categorical, color_vector, na_color_hex)

    if (
        strip_alpha_hex
        and color_vector is not None
        and len(color_vector) > 0
        and isinstance(color_vector[0], str)
        and color_vector[0].startswith("#")
    ):
        # color_vector usually holds only a few distinct hex strings (one per category), so strip
        # alpha on the unique values and map back rather than parsing once per point.
        unique_hex, inverse = np.unique(color_vector, return_inverse=True)
        color_vector = np.asarray([_hex_no_alpha(c) for c in unique_hex])[inverse]

    # density without a color column collapses to a sequential count gradient; everything else with no
    # explicit continuous value (categorical or no color) goes through the categorical shader.
    plain_density = density and col_for_color is None
    nan_shaded = None
    if not plain_density and (color_by_categorical or col_for_color is None):
        shaded = _ds_shade_categorical(
            agg,
            color_key,
            color_vector,
            alpha,
            spread_px=spread_px,
            how=shade_how,
            density=density,
            uniform_alpha=uniform_alpha,
        )
    else:
        shaded, nan_shaded, reduction_bounds = _ds_shade_continuous(
            agg,
            color_span,
            norm,
            cmap,
            alpha,
            reduction_bounds,
            nan_agg,
            na_color_hex,
            spread_px=spread_px,
            ds_reduction=ds_reduction,
            how=shade_how,
            uniform_alpha=uniform_alpha,
        )

    return shaded, nan_shaded, reduction_bounds, color_vector


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

    # Attach the outline vector under a private column name so a fill column with the
    # same key never gets overwritten. Assign positionally (via a Series indexed to the
    # element) — `.assign(col=series)` aligns by index, which silently inserts NaN when
    # the element's index is non-contiguous (e.g. after an inner-join). The NaNs would
    # then be lifted to the `ds_nan` sentinel and one polygon's outline would render as
    # `na_color` instead of its real category.
    transformed_element = transformed_element.copy()
    if color_by_categorical:
        cat = pd.Categorical(outline_color_source_vector)
        attach_cat = _inject_ds_nan_sentinel(pd.Series(cat))
        transformed_element[_OUTLINE_INTERNAL_COL] = pd.Categorical(
            attach_cat.to_numpy(), categories=attach_cat.cat.categories
        )
    else:
        transformed_element[_OUTLINE_INTERNAL_COL] = np.asarray(outline_color_vector)

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
        norm = cmap_params.fresh_norm()
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
    return _make_continuous_mappable(vmin, vmax, cmap)


# ---------------------------------------------------------------------------
# Datashader reduction constants
# ---------------------------------------------------------------------------


_DS_REDUCTION_FUNCS: dict[str, Any] = {
    "sum": ds.sum,
    "mean": ds.mean,
    "any": ds.any,
    "count": ds.count,
    "std": ds.std,
    "var": ds.var,
    "max": ds.max,
    "min": ds.min,
}


# ---------------------------------------------------------------------------
# Color / alpha helpers
# ---------------------------------------------------------------------------


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


def _convert_alpha_to_datashader_range(alpha: float) -> float:
    """Convert alpha from the range [0, 1] to the range [0, 255] used in datashader."""
    # prevent a value of 255, bc that led to fully colored test plots instead of just colored points/shapes
    return min([254, alpha * 255])


# ---------------------------------------------------------------------------
# Canvas geometry helpers
# ---------------------------------------------------------------------------


def _ax_show_and_transform(
    array: MaskedArray[tuple[int, ...], Any] | npt.NDArray[Any],
    trans_data: CompositeGenericTransform,
    ax: Axes,
    alpha: float | None = None,
    cmap: ListedColormap | LinearSegmentedColormap | None = None,
    zorder: int = 0,
    norm: Normalize | None = None,
    interpolation: str | None = None,
) -> matplotlib.image.AxesImage:
    # ``extent`` uses mpl's pixel-grid convention; world placement happens via
    # ``set_transform(trans_data)`` afterwards.
    image_extent = (-0.5, array.shape[1] - 0.5, array.shape[0] - 0.5, -0.5)
    # ``alpha`` is applied only when no cmap is set, so RGBA arrays already
    # carrying per-pixel alpha (e.g. datashader output) are not double-attenuated.
    imshow_kwargs: dict[str, Any] = {"zorder": zorder, "extent": image_extent, "norm": norm}
    if not cmap and alpha is not None:
        imshow_kwargs["alpha"] = alpha
    else:
        imshow_kwargs["cmap"] = cmap
    if interpolation is not None:
        imshow_kwargs["interpolation"] = interpolation
    im = ax.imshow(array, **imshow_kwargs)
    im.set_transform(trans_data)
    return im


def _pad_degenerate_extent(ext: list[Any]) -> list[Any]:
    """Pad a zero-width extent to a unit window centered on its value; pass others through."""
    return [ext[0] - 0.5, ext[1] + 0.5] if ext[1] == ext[0] else ext


def _circle_quad_segs(max_radius_px: float) -> int:
    """Segments-per-quadrant for buffering circles, chosen by the largest disc's on-screen radius.

    Circles are stored as ``Point + radius`` and buffered to polygons before datashader rasterizes
    them; shapely's default (``resolution=16`` = 65 vertices) is far more than a small disc needs and
    dominates the render cost for large circle sets. Use fewer vertices when discs are small (their
    extra vertices are sub-pixel and invisible) and only step up to the full default once discs are
    large enough to show facets. ``NaN`` (e.g. all-NaN radius) falls through to the faithful default.
    """
    if max_radius_px <= 8:
        return 4
    if max_radius_px <= 32:
        return 8
    return 16


def _circle_buffer_quad_segs(
    centroids_xy: np.ndarray,
    max_radius: float,
    tm: np.ndarray,
    fig_params: FigParams,
) -> int:
    """Pick the circle-buffer ``resolution`` from the largest circle's on-screen pixel radius.

    Estimates the same world-units-per-pixel ``factor`` the datashader canvas will use (mirrors
    ``_compute_datashader_canvas_params``), computed *before* buffering from the transformed centroids
    expanded by the (major-axis) radius. ``tm`` is the coordinate-system affine; an anisotropic/shear
    transform turns the circle into an ellipse, so size to its largest stretch (major axis).
    """
    linear = tm[:2, :2]
    stretch = float(np.linalg.svd(linear, compute_uv=False).max())  # circle -> ellipse major-axis scale
    r_t = float(max_radius) * stretch
    xy_t = centroids_xy @ linear.T + tm[:2, 2]
    ext_w = (xy_t[:, 0].max() + r_t) - (xy_t[:, 0].min() - r_t)
    ext_h = (xy_t[:, 1].max() + r_t) - (xy_t[:, 1].min() - r_t)
    fig = fig_params.fig
    fig_px_w = fig.get_size_inches()[0] * fig.dpi
    fig_px_h = fig.get_size_inches()[1] * fig.dpi
    factor = max(ext_w / fig_px_w, ext_h / fig_px_h)
    return _circle_quad_segs(r_t / factor if factor > 0 else 0.0)


def _compute_datashader_canvas_params(
    x_ext: list[Any],
    y_ext: list[Any],
    fig_params: FigParams,
) -> tuple[Any, Any, list[Any], list[Any], Any]:
    """Compute datashader canvas dimensions from spatial extents.

    Shared logic used by both the dask-based and pandas-based entry points.
    """
    # A zero-width extent (single point, coincident points, axis-aligned line) has no scale to
    # build a canvas from; pad it so the factor below doesn't divide by zero.
    x_ext, y_ext = _pad_degenerate_extent(x_ext), _pad_degenerate_extent(y_ext)

    # Compute canvas size in pixels, capped at the figure's display resolution.
    # Using np.max ensures the canvas never exceeds display pixels on either axis,
    # preventing pixel-based operations (spread, line_width) from being downscaled
    # to sub-pixel size when the data aspect ratio differs from the figure's.
    plot_width = x_ext[1] - x_ext[0]
    plot_height = y_ext[1] - y_ext[0]
    plot_width_px = int(round(fig_params.fig.get_size_inches()[0] * fig_params.fig.dpi))
    plot_height_px = int(round(fig_params.fig.get_size_inches()[1] * fig_params.fig.dpi))
    factor: float
    factor = np.max([plot_width / plot_width_px, plot_height / plot_height_px])
    plot_width = int(np.round(plot_width / factor))
    plot_height = int(np.round(plot_height / factor))

    return plot_width, plot_height, x_ext, y_ext, factor


def _get_extent_and_range_for_datashader_canvas(
    spatial_element: SpatialElement,
    coordinate_system: str,
    fig_params: FigParams,
) -> tuple[Any, Any, list[Any], list[Any], Any]:
    extent = _fast_extent(spatial_element, coordinate_system)
    x_ext = [float(extent["x"][0]), float(extent["x"][1])]
    y_ext = [float(extent["y"][0]), float(extent["y"][1])]
    return _compute_datashader_canvas_params(x_ext, y_ext, fig_params)


def _datashader_canvas_from_dataframe(
    df: pd.DataFrame,
    fig_params: FigParams,
) -> tuple[Any, Any, list[Any], list[Any], Any]:
    """Compute datashader canvas params directly from a pandas DataFrame.

    Avoids the overhead of ``get_extent()`` (which requires a dask-backed
    SpatialElement) by reading min/max from the already-materialised data.
    """
    if len(df) == 0:
        # Empty input (e.g., a bounding_box_query with no overlap) — caller
        # should short-circuit; return zero-sized canvas params as a sentinel.
        return 0, 0, [0.0, 0.0], [0.0, 0.0], 1.0
    x_ext = [float(df["x"].min()), float(df["x"].max())]
    y_ext = [float(df["y"].min()), float(df["y"].max())]
    return _compute_datashader_canvas_params(x_ext, y_ext, fig_params)


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


def _create_image_from_datashader_result(
    ds_result: ds.transfer_functions.Image | np.ndarray[Any, np.dtype[np.uint8]],
    factor: float,
    ax: Axes,
    x_min: float = 0.0,
    y_min: float = 0.0,
) -> tuple[MaskedArray[tuple[int, ...], Any], matplotlib.transforms.Transform]:
    # create SpatialImage from datashader output to get it back to original size
    rgba_image_data = ds_result.copy() if isinstance(ds_result, np.ndarray) else ds_result.to_numpy().base
    rgba_image_data = np.transpose(rgba_image_data, (2, 0, 1))
    transformation: Scale | TransformSequence = Scale([1, factor, factor], ("c", "y", "x"))
    if x_min != 0.0 or y_min != 0.0:
        # Canvas pixel (0, 0) corresponds to world (x_min, y_min). Without this
        # translation the rgba would render at the world origin instead of at
        # the element's actual position.
        transformation = TransformSequence([transformation, Translation([x_min, y_min], ("x", "y"))])
    rgba_image = Image2DModel.parse(
        rgba_image_data,
        dims=("c", "y", "x"),
        transformations={"global": transformation},
    )

    _, trans_data = _prepare_transformation(rgba_image, "global", ax)

    rgba_image = np.transpose(rgba_image.data.compute(), (1, 2, 0))  # type: ignore[attr-defined]
    rgba_image = ma.masked_array(rgba_image)  # type conversion for mypy

    return rgba_image, trans_data


# ---------------------------------------------------------------------------
# Aggregation / shading helpers
# ---------------------------------------------------------------------------


def _datashader_aggregate_with_function(
    reduction: _DsReduction | None,
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

    try:
        reduction_function = _DS_REDUCTION_FUNCS[reduction](column=col_for_color)
    except KeyError as e:
        raise ValueError(
            f"Reduction '{reduction}' is not supported. Please use one of: {', '.join(_DS_REDUCTION_FUNCS.keys())}."
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


def _datashader_get_how_kw_for_spread(
    reduction: _DsReduction | None,
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


def _apply_cmap_alpha_to_datashader_result(
    result: Any,
    agg: DataArray,
    cmap: str | list[str] | Colormap,
    span: list[float] | tuple[float, float] | None,
) -> Any:
    """Apply the colormap's alpha channel to a datashader RGBA result.

    Datashader ignores the per-entry alpha channel of matplotlib colormaps,
    so pixels that the cmap marks as transparent (alpha=0) are rendered
    opaque.  This function post-processes the shaded RGBA output to restore
    the cmap's intended transparency.  See :issue:`376`.
    """
    if not isinstance(cmap, Colormap):
        return result

    # Quick check: does this cmap have any transparent entries?
    test_vals = np.linspace(0, 1, min(cmap.N, 256))
    cmap_alphas = cmap(test_vals)[:, 3]
    if np.all(cmap_alphas >= 1.0):
        return result

    # Get or ensure we have an (H, W, 4) uint8 array
    if hasattr(result, "values"):
        # datashader Image — uint32 packed, convert via to_numpy()
        rgba = result.to_numpy().base
        if rgba is None:
            return result
    else:
        rgba = result

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return result

    # Normalise aggregate values to [0, 1] using the same span datashader used
    agg_vals = agg.values.astype(np.float64)
    valid = np.isfinite(agg_vals)
    if not valid.any():
        return result

    if span is not None:
        lo, hi = float(span[0]), float(span[1])
    else:
        lo = float(np.nanmin(agg_vals))
        hi = float(np.nanmax(agg_vals))

    if hi <= lo or not np.isfinite(lo) or not np.isfinite(hi):
        return result

    normed = np.clip((agg_vals - lo) / (hi - lo), 0.0, 1.0)

    # Look up cmap alpha for each pixel
    desired_alpha = cmap(normed)[:, :, 3]

    # Zero out pixels where the cmap wants transparency
    transparent = valid & (desired_alpha < 1.0)
    if transparent.any():
        # Scale the existing alpha by the cmap's alpha
        rgba[transparent, 3] = (rgba[transparent, 3].astype(np.float32) * desired_alpha[transparent]).astype(np.uint8)

    return result


def _datashader_map_aggregate_to_color(
    agg: DataArray,
    cmap: str | list[str] | ListedColormap,
    color_key: list[str] | dict[str, str] | None = None,
    min_alpha: float = 40,
    span: None | list[float] = None,
    clip: bool = True,
    how: str = "linear",
) -> ds.tf.Image | np.ndarray[Any, np.dtype[np.uint8]]:
    """ds.tf.shade() part, ensuring correct clipping behavior.

    If necessary (norm.clip=False), split shading in 3 parts and in the end, stack results.
    This ensures the correct clipping behavior, because else datashader would always automatically clip.

    ``how`` controls the count-to-color mapping passed to :func:`datashader.transfer_functions.shade`
    (``"linear"`` by default; ``"log"``/``"cbrt"``/``"eq_hist"`` compress dynamic range). The split-shade
    branch used for ``norm.clip=False`` always uses ``"linear"`` since per-segment shading would otherwise
    interact poorly with rank-based mappings.
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

        return _apply_cmap_alpha_to_datashader_result(stack, agg, cmap, span)

    result = ds.tf.shade(
        agg,
        cmap=cmap,
        color_key=color_key,
        min_alpha=min_alpha,
        span=span,
        how=how,
    )
    return _apply_cmap_alpha_to_datashader_result(result, agg, cmap, span)
