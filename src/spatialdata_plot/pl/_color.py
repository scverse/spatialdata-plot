"""Color resolution, palettes, and colormap helpers (extracted from utils.py, see #696)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib
import numpy as np
import pandas as pd
import spatialdata as sd
from anndata import AnnData
from cycler import Cycler, cycler
from geopandas import GeoDataFrame
from matplotlib import colors, rcParams
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    ColorConverter,
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
    to_rgba,
)
from numpy.random import default_rng
from pandas.api.types import CategoricalDtype, is_bool_dtype, is_numeric_dtype, is_string_dtype
from pandas.core.arrays.categorical import Categorical
from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation
from scanpy.plotting.palettes import default_20, default_28, default_102
from skimage.color import label2rgb
from skimage.morphology import erosion, footprint_rectangle
from skimage.util import map_array
from spatialdata import (
    get_values,
)
from spatialdata._core.query.relational_query import _locate_value
from spatialdata._types import ArrayLike
from spatialdata.models import (
    SpatialElement,
)

from spatialdata_plot._logging import logger
from spatialdata_plot.pl.render_params import (
    CmapParams,
    Color,
    ColorLike,
    OutlineParams,
)
from spatialdata_plot.pl.utils import (
    _MPL_SINGLE_LETTER_COLORS,
    _build_alignment_dtype_hint,
    _ensure_one_to_one_mapping,
    _format_element_name,
    to_hex,
)


def _is_color_like(color: Any) -> bool:
    """Check if a value is a valid color.

    We reject several matplotlib shorthand notations that are likely to collide
    with column or gene names. For discussion, see:

    - https://github.com/scverse/spatialdata-plot/issues/211
    - https://github.com/scverse/spatialdata-plot/issues/327

    Rejected shorthands:

    - Greyscale strings: ``"0"``, ``"0.5"``, ``"1"`` (floats in [0, 1])
    - Short hex: ``"#RGB"`` / ``"#RGBA"`` (only ``#RRGGBB`` / ``#RRGGBBAA`` accepted)
    - Single-letter colors: ``"b"``, ``"g"``, ``"r"``, ``"c"``, ``"m"``, ``"y"``, ``"k"``, ``"w"``
    - CN cycle notation: ``"C0"``, ``"C1"``, …
    - ``tab:`` prefixed colors: ``"tab:blue"``, ``"tab:orange"``, …
    - ``xkcd:`` prefixed colors: ``"xkcd:sky blue"``, …
    """
    if isinstance(color, str):
        # greyscale strings
        try:
            num_value = float(color)
            if 0 <= num_value <= 1:
                return False
        except ValueError:
            pass

        # short hex
        if color.startswith("#") and len(color) not in [7, 9]:
            return False

        # single-letter color shortcuts
        if color in _MPL_SINGLE_LETTER_COLORS:
            return False

        # CN cycle notation (C0, C1, …)
        if len(color) >= 2 and color[0] == "C" and color[1:].isdigit():
            return False

        # tab: and xkcd: prefixed colors
        if color.startswith(("tab:", "xkcd:")):
            return False

    return bool(colors.is_color_like(color))


def _make_continuous_mappable(vmin: float, vmax: float, cmap: Any) -> ScalarMappable:
    """Build a ``ScalarMappable`` for a continuous colorbar, with a ±0.5 fallback when ``vmin == vmax``."""
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)


def _resolve_continuous_norm(values: Any, cmap_params: CmapParams) -> Normalize:
    """Resolve a concrete ``Normalize`` for continuous coloring.

    Honor explicit ``norm`` vmin/vmax, else the finite-value data range of ``values``, else
    ``[0, 1]``. Shared by the pixel and colorbar sites so both derive the same range. A degenerate
    ``vmin == vmax`` is left as-is (matplotlib expands it downstream), not reset to ``[0, 1]``.
    """
    base = cmap_params.norm
    vmin, vmax = base.vmin, base.vmax
    if vmin is None or vmax is None:
        arr = np.asarray(values)
        if not np.issubdtype(arr.dtype, np.number):
            arr = pd.to_numeric(arr.ravel(), errors="coerce")
        finite = np.isfinite(arr)
        data_min = float(np.nanmin(arr[finite])) if finite.any() else 0.0
        data_max = float(np.nanmax(arr[finite])) if finite.any() else 1.0
        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max
    return Normalize(vmin=vmin, vmax=vmax, clip=base.clip)


def _apply_mask_to_outline_vectors(
    outline_color_vector: Any,
    outline_color_source_vector: pd.Series | None,
    mask: Any,
) -> tuple[Any, pd.Series | None]:
    """Apply a boolean ``keep`` mask to outline color vector(s).

    Used to keep outline data aligned with the fill data after a ``groups``
    or rasterize-based filter is applied to the rendered element.
    """
    arr = np.asarray(mask)
    if outline_color_source_vector is not None:
        outline_color_source_vector = outline_color_source_vector[arr]
    return outline_color_vector[arr], outline_color_source_vector


def _align_outline_vector_to_length(
    outline_color_vector: Any,
    outline_color_source_vector: pd.Series | None,
    n: int,
) -> tuple[Any, pd.Series | None]:
    """Pad or truncate the outline color vector(s) to length ``n``.

    Used when the outline column annotates a different row count than the rendered
    element (cross-table case, or rasterize-induced label drop). Missing entries
    are padded with NaN so downstream code maps them to ``na_color``.
    """
    if outline_color_vector is None or len(outline_color_vector) == n:
        return outline_color_vector, outline_color_source_vector
    if len(outline_color_vector) > n:
        if outline_color_source_vector is not None:
            outline_color_source_vector = outline_color_source_vector[:n]
        return outline_color_vector[:n], outline_color_source_vector
    pad = n - len(outline_color_vector)
    if outline_color_source_vector is not None:
        # Categorical: downstream picks one hex per category from rows that *have* a
        # category. NaN-padded rows contribute no category, so the per-row hex pad is
        # immaterial; pad with NaN to skip the allocation.
        padded_vec = np.concatenate([np.asarray(outline_color_vector), np.full(pad, np.nan, dtype=object)])
        outline_color_source_vector = pd.Categorical(
            list(outline_color_source_vector) + [None] * pad,
            categories=outline_color_source_vector.categories,
        )
    else:
        # Continuous: numeric vector, pad with NaN so cmap maps padded rows to na_color.
        padded_vec = np.concatenate([np.asarray(outline_color_vector, dtype=float), np.full(pad, np.nan)])
    return padded_vec, outline_color_source_vector


def _color_vector_to_rgba(
    color_vector: Any | None,
    color_source_vector: pd.Series | None,
    cmap_params: CmapParams,
    n_rows: int,
) -> np.ndarray:
    """Convert a fill/outline `color_vector` (categorical hex strings or continuous numerics) to (N, 4) RGBA.

    Mirrors the per-row mapping done inside :func:`_get_collection_shape` so that
    callers can pre-materialize an outline-color array. NaN/non-finite entries are
    painted with ``cmap_params.na_color``.
    """
    na_rgba = colors.to_rgba(cmap_params.na_color.get_hex_with_alpha())
    if color_vector is None:
        rgba = np.empty((n_rows, 4), dtype=float)
        rgba[:] = na_rgba
        return rgba

    if color_source_vector is not None:
        # Categorical: color_vector contains hex strings aligned to color_source_vector
        return np.asarray(ColorConverter().to_rgba_array(list(color_vector)))

    arr = np.asarray(color_vector)
    if arr.ndim == 2 and arr.shape[1] in (3, 4) and np.issubdtype(arr.dtype, np.number):
        return np.asarray(ColorConverter().to_rgba_array(arr))

    rgba = np.empty((len(arr), 4), dtype=float)
    rgba[:] = na_rgba
    if np.issubdtype(arr.dtype, np.number):
        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            used_norm = _resolve_continuous_norm(arr, cmap_params)
            rgba[finite_mask] = cmap_params.cmap(used_norm(arr[finite_mask]))
        return rgba

    # Object dtype: mix of numerics and color-like specs (apply cmap to the numeric subset only)
    series = pd.Series(arr, copy=False)
    num = pd.to_numeric(series, errors="coerce").to_numpy()
    is_num = np.isfinite(num)
    if is_num.any():
        used_norm = _resolve_continuous_norm(num, cmap_params)
        rgba[is_num] = cmap_params.cmap(used_norm(num[is_num]))
    color_mask = (~is_num) & series.notna().to_numpy()
    if color_mask.any():
        rgba[color_mask] = ColorConverter().to_rgba_array(series[color_mask].tolist())
    return rgba


def _prepare_cmap_norm(
    cmap: Colormap | str | None = None,
    norm: Normalize | None = None,
    na_color: Color = Color(),
) -> CmapParams:
    # TODO: check refactoring norm out here as it gets overwritten later
    cmap_is_default = cmap is None
    if cmap is None:
        cmap = rcParams["image.cmap"]
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

    cmap = copy(cmap)

    assert isinstance(cmap, Colormap), f"Invalid type of `cmap`: {type(cmap)}, expected `Colormap`."

    norm = Normalize(vmin=None, vmax=None, clip=False) if norm is None else copy(norm)

    cmap.set_bad(na_color.get_hex_with_alpha())

    return CmapParams(
        cmap=cmap,
        norm=norm,
        na_color=na_color,
        cmap_is_default=cmap_is_default,
    )


def _set_outline(
    outline_alpha: float | int | tuple[float | int, float | int] | None,
    outline_width: int | float | tuple[float | int, float | int] | None,
    outline_color: Color | tuple[Color, Color | None] | None,
    **kwargs: Any,
) -> tuple[tuple[float, float], OutlineParams]:
    """Create OutlineParams object for shapes, including possibility of double outline.

    Rules for outline rendering:
    1) outline_alpha always takes precedence if given by the user.
    In absence of outline_alpha:
    2) If outline_color is specified and implying an alpha (e.g. RGBA array or #RRGGBBAA): that alpha is used
    3) If outline_color (w/o implying an alpha) and/or outline_width is specified: alpha of outlines set to 1.0
    """
    # A) User doesn't want to see outlines
    if (
        outline_alpha == 0.0
        or (isinstance(outline_alpha, tuple) and np.all(np.array(outline_alpha) == 0.0))
        or not (outline_alpha or outline_width or outline_color)
    ):
        return (0.0, 0.0), OutlineParams(None, 1.5, None, 0.5)

    # B) User wants to see at least 1 outline
    if isinstance(outline_width, tuple):
        if len(outline_width) != 2:
            raise ValueError(
                f"Tuple of length {len(outline_width)} was passed for outline_width. When specifying multiple outlines,"
                " please pass a tuple of exactly length 2."
            )
        if not outline_color:
            outline_color = (Color("#000000"), Color("#ffffff"))
        elif not isinstance(outline_color, tuple):
            raise ValueError(
                "No tuple was passed for outline_color, while two outlines were specified by using the outline_width "
                "argument. Please specify the outline colors in a tuple of length two."
            )

    if isinstance(outline_color, tuple):
        if len(outline_color) != 2:
            raise ValueError(
                f"Tuple of length {len(outline_color)} was passed for outline_color. When specifying multiple outlines,"
                " please pass a tuple of exactly length 2."
            )
        if not outline_width:
            outline_width = (1.5, 0.5)
        elif not isinstance(outline_width, tuple):
            raise ValueError(
                "No tuple was passed for outline_width, while two outlines were specified by using the outline_color "
                "argument. Please specify the outline widths in a tuple of length two."
            )

    if isinstance(outline_width, float | int):
        outline_width = (outline_width, 0.0)
    elif not outline_width:
        outline_width = (1.5, 0.0)
    if isinstance(outline_color, Color):
        outline_color = (outline_color, None)
    elif not outline_color:
        outline_color = (Color("#000000ff"), None)

    assert isinstance(outline_color, tuple), "outline_color is not a tuple"  # shut up mypy
    assert isinstance(outline_width, tuple), "outline_width is not a tuple"

    for ow in outline_width:
        if not isinstance(ow, int | float):
            raise TypeError(f"Invalid type of `outline_width`: {type(ow)}, expected `int` or `float`.")

    if outline_alpha:
        if isinstance(outline_alpha, int | float):
            # for a single outline: second width value is 0.0
            outline_alpha = (outline_alpha, 0.0) if outline_width[1] == 0.0 else (outline_alpha, outline_alpha)
    else:
        # if alpha wasn't explicitly specified by the user
        outer_ol_alpha = outline_color[0].get_alpha_as_float() if isinstance(outline_color[0], Color) else 1.0
        inner_ol_alpha = outline_color[1].get_alpha_as_float() if isinstance(outline_color[1], Color) else 1.0
        outline_alpha = (outer_ol_alpha, inner_ol_alpha)

    # handle possible linewidths of 0.0 => outline won't be rendered in the first place
    if outline_width[0] == 0.0:
        outline_alpha = (0.0, outline_alpha[1])
    if outline_width[1] == 0.0:
        outline_alpha = (outline_alpha[0], 0.0)

    if outline_alpha[0] > 0.0 or outline_alpha[1] > 0.0:
        kwargs.pop("edgecolor", None)  # remove edge from kwargs if present
        kwargs.pop("alpha", None)  # remove alpha from kwargs if present

    return outline_alpha, OutlineParams(
        outline_color[0],
        outline_width[0],
        outline_color[1],
        outline_width[1],
    )


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


def _infer_color_data_kind(
    series: pd.Series,
    value_to_plot: str,
    element_name: list[str] | str | None,
    table_name: str | None,
    warn_on_object_to_categorical: bool = False,
) -> tuple[Literal["numeric", "categorical"], pd.Series | pd.Categorical]:
    element_label = _format_element_name(element_name)

    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical", pd.Categorical(series)

    if is_bool_dtype(series.dtype):
        return "numeric", series.astype(float)

    if is_numeric_dtype(series.dtype):
        return "numeric", pd.to_numeric(series, errors="coerce")

    if is_string_dtype(series.dtype) or series.dtype == object:
        non_na = series[~pd.isna(series)]
        if len(non_na) == 0:
            return "numeric", pd.to_numeric(series, errors="coerce")

        numeric_like = pd.to_numeric(non_na, errors="coerce")
        has_numeric = numeric_like.notna().any()
        has_non_numeric = numeric_like.isna().any()

        if has_numeric and has_non_numeric:
            invalid_examples = non_na[numeric_like.isna()].astype(str).unique()[:3]
            location = f" in table '{table_name}'" if table_name is not None else ""
            raise TypeError(
                f"Column '{value_to_plot}' for element '{element_label}'{location} contains both numeric and "
                f"non-numeric values (e.g. {', '.join(invalid_examples)}). "
                "Please ensure that the column stores consistent data."
            )

        if has_numeric:
            return "numeric", pd.to_numeric(series, errors="coerce")

        if warn_on_object_to_categorical:
            logger.warning(
                f"Converting copy of '{value_to_plot}' column to categorical dtype for categorical plotting. "
                "Consider converting before plotting."
            )

        return "categorical", pd.Categorical(series)

    return "numeric", pd.to_numeric(series, errors="coerce")


def _extract_color_column(
    table: AnnData,
    value_key: str,
    *,
    origin: str,
    element: GeoDataFrame,
    element_name: str,
    table_layer: str | None = None,
) -> pd.Series:
    """Read one color column from ``table`` aligned to ``element`` order, without copying the table.

    Equivalent to ``get_values(value_key, sdata=..., element_name=..., table_name=...)[value_key]`` but
    skips the table->element join, whose ``table[indices, :].copy()`` does an expensive out-of-order
    sparse CSR row-gather. Restricts to rows annotating ``element_name`` (via ``region_key``), then
    reindexes to the element's instance order (``NaN`` for instances with no table row), preserving the
    categorical dtype of ``obs`` columns so the downstream legend path is unchanged.
    """
    attrs = table.uns["spatialdata_attrs"]
    region_key, instance_key = attrs["region_key"], attrs["instance_key"]
    mask = table.obs[region_key].to_numpy() == element_name
    inst = table.obs[instance_key].to_numpy()[mask]
    if origin == "var":
        source = table.layers[table_layer] if table_layer is not None else table.X
        col = source[:, table.var_names.get_loc(value_key)]
        col = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
        values = pd.Series(col[mask], index=inst)
    else:  # obs column; .values keeps a Categorical categorical so the legend path still sees one
        values = pd.Series(table.obs[value_key].values[mask], index=inst)
    return values.reindex(element.index)


def _set_color_source_vec(
    sdata: sd.SpatialData,
    element: SpatialElement | None,
    value_to_plot: str | None,
    na_color: Color,
    element_name: list[str] | str | None = None,
    groups: list[str] | str | None = None,
    palette: dict[str, str] | list[str] | str | None = None,
    cmap_params: CmapParams | None = None,
    alpha: float = 1.0,
    table_name: str | None = None,
    table_layer: str | None = None,
    render_type: Literal["points", "labels"] | None = None,
    coordinate_system: str | None = None,
    preloaded_color_data: pd.Series | None = None,
) -> tuple[ArrayLike | pd.Series | None, ArrayLike, bool]:
    if value_to_plot is None and element is not None:
        color = np.full(len(element), na_color.get_hex_with_alpha())
        return color, color, False

    # Figure out where to get the color from
    origins = _locate_value(
        value_key=value_to_plot,
        sdata=sdata,
        element_name=element_name,
        table_name=table_name,
    )

    # When both the element's own dataframe and the chosen table contain a
    # column with this name, an explicit `table_name=` resolves the ambiguity —
    # keep only the table origin and skip the multi-origin error below.
    explicit_table_shadows_df = table_name is not None and any(o.origin == "df" for o in origins)
    if explicit_table_shadows_df:
        origins = [o for o in origins if o.origin != "df"]

    if len(origins) > 1:
        raise ValueError(
            f"Color key '{value_to_plot}' for element '{element_name}' was found in multiple locations: {origins}. "
            "Please keep it in exactly one place (preferably on the points parquet for speed) to avoid ambiguity."
        )

    if len(origins) == 1 and value_to_plot is not None:
        if table_name is not None:
            _ensure_one_to_one_mapping(
                sdata=sdata,
                element=element,
                element_name=element_name,
                table_name=table_name,
            )
        if preloaded_color_data is not None:
            color_source_vector = preloaded_color_data
        elif (
            isinstance(element, GeoDataFrame)
            and isinstance(element_name, str)
            and table_name is not None
            and table_name in sdata.tables
            and origins[0].origin in ("obs", "var")
        ):
            # Fast path: read the single aligned column directly instead of joining/copying the
            # whole annotating table (the join's out-of-order sparse row-gather dominates large renders).
            color_source_vector = _extract_color_column(
                sdata[table_name],
                value_to_plot,
                origin=origins[0].origin,
                element=element,
                element_name=element_name,
                table_layer=table_layer,
            )
        elif explicit_table_shadows_df:
            # Pass the table as `element` so upstream `get_values` skips the
            # element-column lookup and avoids the multi-origin error.
            color_source_vector = get_values(
                value_key=value_to_plot,
                element=sdata[table_name],
                element_name=element_name,
                table_layer=table_layer,
            )[value_to_plot]
        else:
            color_source_vector = get_values(
                value_key=value_to_plot,
                sdata=sdata,
                element_name=element_name,
                table_name=table_name,
                table_layer=table_layer,
            )[value_to_plot]

        color_series = (
            color_source_vector if isinstance(color_source_vector, pd.Series) else pd.Series(color_source_vector)
        )

        if color_series.isna().all():
            element_label = _format_element_name(element_name)
            dtype_hint = _build_alignment_dtype_hint(sdata, element, color_series, table_name)
            hint_suffix = f" {dtype_hint.strip()}" if dtype_hint else ""
            logger.warning(
                f"Column '{value_to_plot}' for element '{element_label}' contains only NaN values; "
                f"rendering with na_color.{hint_suffix}"
            )
            na_color_arr = np.full(len(color_series), na_color.get_hex_with_alpha())
            return na_color_arr, na_color_arr, False

        kind, processed = _infer_color_data_kind(
            series=color_series,
            value_to_plot=value_to_plot,
            element_name=element_name,
            table_name=table_name,
            warn_on_object_to_categorical=table_name is not None,
        )

        if kind == "numeric":
            numeric_vector = processed
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
            return None, numeric_vector, False

        assert isinstance(processed, pd.Categorical)
        if not processed.ordered:
            # ensure deterministic category order when the source is unordered (e.g., from a Python set)
            processed = processed.reorder_categories(sorted(processed.categories))
        color_source_vector = processed  # convert, e.g., `pd.Series`

        # When the value lives on the element's own DataFrame (origin="df"),
        # there is no reason to look up a table for .uns colors.
        value_from_element = origins[0].origin == "df"

        # Use the provided table_name parameter, fall back to only one present
        table_to_use: str | None
        if value_from_element:
            table_to_use = None
        elif table_name is not None and table_name in sdata.tables:
            table_to_use = table_name
        elif table_name is not None and table_name not in sdata.tables:
            logger.warning(f"Table '{table_name}' not found in `sdata.tables`. Falling back to default behavior.")
            table_to_use = None
        else:
            table_keys = list(sdata.tables.keys())
            if len(table_keys) == 1:
                table_to_use = table_keys[0]
            elif len(table_keys) > 1:
                table_to_use = table_keys[0]
                logger.warning(f"No table name provided, using '{table_to_use}' as fallback for color mapping.")
            else:
                table_to_use = None

        adata_for_mapping = sdata[table_to_use] if table_to_use is not None else None

        # Check if custom colors exist in the resolved table's .uns slot
        if (
            value_to_plot is not None
            and table_to_use is not None
            and _has_colors_in_uns(sdata, table_to_use, value_to_plot)
        ):
            # Extract colors directly from the table's .uns slot
            # Convert Color to ColorLike (str) for the function
            na_color_like: ColorLike = na_color.get_hex() if isinstance(na_color, Color) else na_color
            color_mapping = _extract_colors_from_table_uns(
                sdata=sdata,
                table_name=table_to_use,
                col_to_colorby=value_to_plot,
                color_source_vector=color_source_vector,
                na_color=na_color_like,
            )
            if color_mapping is not None:
                if isinstance(palette, str):
                    palette = [palette]
                color_mapping = _modify_categorical_color_mapping(
                    mapping=color_mapping,
                    groups=groups,
                    palette=palette,
                )
            else:
                logger.warning(f"Failed to extract colors for '{value_to_plot}', falling back to default mapping.")
                # Fall back to the existing method if extraction fails
                color_mapping = _get_categorical_color_mapping(
                    adata=sdata[table_to_use],
                    cluster_key=value_to_plot,
                    color_source_vector=color_source_vector,
                    cmap_params=cmap_params,
                    alpha=alpha,
                    groups=groups,
                    palette=palette,
                    na_color=na_color,
                    render_type=render_type,
                )
        else:
            color_mapping = None

        if color_mapping is None:
            # Use the existing color mapping method
            color_mapping = _get_categorical_color_mapping(
                adata=adata_for_mapping,
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
        # pd.Categorical.map() demotes to object dtype when mapped values aren't unique
        # (e.g. two categories share a color). Wrapping back in pd.Categorical ensures
        # downstream consumers always receive a Categorical for categorical data.
        color_vector = pd.Categorical(color_source_vector.map(color_mapping, na_action="ignore"))
        # nan handling: only add the NA category if needed, and store it as a hex string
        na_color_hex = na_color.get_hex_with_alpha() if isinstance(na_color, Color) else str(na_color)
        if color_vector.isna().any():
            if na_color_hex not in color_vector.categories:
                color_vector = color_vector.add_categories(na_color_hex)
            color_vector[pd.isna(color_vector)] = na_color_hex

        return color_source_vector, color_vector, True

    if table_name is None:
        raise KeyError(
            f"Unable to locate color key '{value_to_plot}' for element '{element_name}'. "
            "Please ensure the key exists in a table annotating this element."
        )
    raise KeyError(
        f"Unable to locate color key '{value_to_plot}' in table '{table_name}' for element '{element_name}'."
    )


ColorType = Literal["categorical", "continuous", "none"]


@dataclass(frozen=True)
class ColorSpec:
    """Resolved color for one element layer: the explicit color-state the renderers consume.

    ``colortype`` names the three states the renderers used to infer implicitly from
    ``source_vector``/``categorical``: ``categorical`` (``source_vector`` is the Categorical),
    ``continuous`` (``source_vector`` is None, ``color_vector`` is numeric), ``none`` (no/uncolorable
    value -> ``color_vector`` is the na_color, ``source_vector`` is a non-None na array).
    """

    colortype: ColorType
    source_vector: ArrayLike | pd.Series | None
    color_vector: ArrayLike

    # Predicates on the invariant ``colortype`` — safe to read anywhere (unlike the vectors, which
    # the renderers mutate after resolution). They replace scattered, typo-prone ``colortype == "..."``.
    @property
    def is_categorical(self) -> bool:
        return self.colortype == "categorical"

    @property
    def is_continuous(self) -> bool:
        return self.colortype == "continuous"

    @property
    def is_none(self) -> bool:
        return self.colortype == "none"


def resolve_color(*args: Any, **kwargs: Any) -> ColorSpec:
    """Resolve an element's color into a typed :class:`ColorSpec` (the #700 IR's color layer).

    Pass-through wrapper over :func:`_set_color_source_vec` that classifies its result into an
    explicit ``colortype`` so callers stop re-deriving the state from ``source is None``/``categorical``.
    """
    source_vector, color_vector, categorical = _set_color_source_vec(*args, **kwargs)
    if categorical:
        colortype: ColorType = "categorical"
    elif source_vector is None:
        colortype = "continuous"
    else:
        colortype = "none"
    return ColorSpec(colortype=colortype, source_vector=source_vector, color_vector=color_vector)


def _map_color_seg(
    seg: ArrayLike,
    cell_id: ArrayLike,
    color_vector: ArrayLike | pd.Series[CategoricalDtype],
    color_source_vector: pd.Series[CategoricalDtype],
    cmap_params: CmapParams,
    na_color: Color,
    seg_erosionpx: int | None = None,
    seg_boundaries: bool = False,
    outline_color: Color | None = None,
    outline_color_vector: ArrayLike | pd.Series[CategoricalDtype] | None = None,
    outline_color_source_vector: pd.Series[CategoricalDtype] | None = None,
) -> ArrayLike:
    cell_id = np.array(cell_id)

    if isinstance(color_vector.dtype, pd.CategoricalDtype):
        # Case A: users wants to plot a categorical column
        val_im: ArrayLike = map_array(seg, cell_id, color_vector.codes + 1)
        cols = colors.to_rgba_array(color_vector.categories)
    elif pd.api.types.is_numeric_dtype(color_vector.dtype):
        # Case B: user wants to plot a continous column
        if isinstance(color_vector, pd.Series):
            color_vector = color_vector.to_numpy()
        # normalize only the not nan values, else the whole array would contain only nan values
        normed_color_vector = color_vector.copy().astype(float)
        used_norm = _resolve_continuous_norm(normed_color_vector, cmap_params)
        normed_color_vector[~np.isnan(normed_color_vector)] = used_norm(
            normed_color_vector[~np.isnan(normed_color_vector)]
        )
        cols = cmap_params.cmap(normed_color_vector)
        val_im = map_array(seg, cell_id, cell_id)
    else:
        # Case C: User didn't specify any colors
        if color_source_vector is not None and (
            set(color_vector) == set(color_source_vector)
            and len(set(color_vector)) == 1
            and set(color_vector) == {na_color.get_hex_with_alpha()}
            and not na_color.color_modified_by_user()
        ):
            val_im = map_array(seg, cell_id, cell_id)
            RNG = default_rng(42)
            cols = RNG.random((len(color_vector), 3))
        else:
            # Case D: User didn't specify a column to color by, but modified the na_color
            val_im = map_array(seg, cell_id, cell_id)
            first_value = color_vector.iloc[0] if isinstance(color_vector, pd.Series) else color_vector[0]
            if _is_color_like(first_value):
                # we have color-like values (e.g., hex or named colors)
                assert all(_is_color_like(c) for c in color_vector), "Not all values are color-like."
                cols = colors.to_rgba_array(color_vector)
            else:
                used_norm = _resolve_continuous_norm(color_vector, cmap_params)
                cols = cmap_params.cmap(used_norm(color_vector))

    if seg_erosionpx is not None:
        val_im[val_im == erosion(val_im, footprint_rectangle((seg_erosionpx, seg_erosionpx)))] = 0

    if seg_boundaries and outline_color_vector is not None:
        # Column-driven outline: build per-label colors from the outline vector and overlay
        # on the eroded ring. Two cases (mirroring _set_color_source_vec's return contract):
        #  - categorical: outline_color_source_vector is the source Categorical; outline_color_vector
        #    holds hex strings aligned to cells.
        #  - continuous: outline_color_source_vector is None; outline_color_vector is numeric.
        if outline_color_source_vector is not None:
            cat = pd.Categorical(outline_color_source_vector)
            cat_codes = cat.codes
            outline_val_im: ArrayLike = map_array(seg, cell_id, cat_codes + 1)
            color_arr = np.asarray(outline_color_vector, dtype=object)
            # Pick the first per-cell hex for each category in one vectorized pass
            # (avoids `K × O(N)` Python loops on large label sets).
            cat_colors: list[Any] = [na_color.get_hex_with_alpha()] * len(cat.categories)
            unique_codes, first_indices = np.unique(cat_codes, return_index=True)
            for code, idx in zip(unique_codes, first_indices, strict=True):
                if code >= 0:
                    cat_colors[code] = color_arr[idx]
            outline_cols = colors.to_rgba_array(cat_colors)
        else:
            # Continuous: numeric values normalized via cmap
            ov = (
                outline_color_vector.to_numpy()
                if isinstance(outline_color_vector, pd.Series)
                else np.asarray(outline_color_vector)
            )
            normed = ov.copy().astype(float)
            finite = ~np.isnan(normed)
            if finite.any():
                normed[finite] = _resolve_continuous_norm(ov, cmap_params)(normed[finite])
            outline_cols = cmap_params.cmap(normed)
            outline_val_im = map_array(seg, cell_id, cell_id)
        if seg_erosionpx is not None:
            outline_val_im[
                outline_val_im == erosion(outline_val_im, footprint_rectangle((seg_erosionpx, seg_erosionpx)))
            ] = 0
        outline_seg_im = label2rgb(
            label=outline_val_im,
            colors=outline_cols,
            bg_label=0,
            bg_color=(1, 1, 1),
            image_alpha=0,
        )
        outline_mask = val_im > 0
        alpha_channel = outline_mask.astype(float)
        return np.dstack((outline_seg_im, alpha_channel))

    if seg_boundaries and outline_color is not None:
        # Uniform outline color requested: skip label2rgb, build RGBA directly
        outline_rgba = colors.to_rgba(outline_color.get_hex_with_alpha())
        outline_mask = val_im > 0
        rgba = np.zeros((*val_im.shape, 4), dtype=float)
        rgba[outline_mask, :3] = outline_rgba[:3]
        rgba[outline_mask, 3] = outline_rgba[3]
        return rgba

    seg_im: ArrayLike = label2rgb(
        label=val_im,
        colors=cols,
        bg_label=0,
        bg_color=(1, 1, 1),  # transparency doesn't really work
        image_alpha=0,
    )

    if seg_boundaries:
        # Data-driven outline: use seg_im colors on the eroded ring, transparent elsewhere
        outline_mask = val_im > 0
        alpha_channel = outline_mask.astype(float)
        return np.dstack((seg_im, alpha_channel))

    if len(val_im.shape) != len(seg_im.shape):
        val_im = np.expand_dims((val_im > 0).astype(int), axis=-1)
    return np.dstack((seg_im, val_im))


def _generate_base_categorial_color_mapping(
    adata: AnnData | None,
    cluster_key: str,
    color_source_vector: ArrayLike | pd.Series[CategoricalDtype],
    na_color: Color,
    cmap_params: CmapParams | None = None,
) -> Mapping[str, str]:
    if adata is not None and cluster_key in adata.uns and f"{cluster_key}_colors" in adata.uns:
        all_colors = adata.uns[f"{cluster_key}_colors"]

        # When plotting per-coordinate-system, the color_source_vector may carry
        # categories from other coordinate systems that aren't present in the
        # current subset.  Drop them so that categories and colors stay aligned.
        color_source_vector = color_source_vector.remove_unused_categories()

        # The stored colors in .uns correspond 1-to-1 to the *full* set of
        # categories in adata.obs[cluster_key].  Subset to the categories that
        # are still present after removing unused ones.
        if cluster_key in adata.obs and hasattr(adata.obs[cluster_key], "cat"):
            all_cats = adata.obs[cluster_key].cat.categories.tolist()
            keep_idx = [i for i, c in enumerate(all_cats) if c in color_source_vector.categories]
            colors = [to_hex(to_rgba(all_colors[i])[:3]) for i in keep_idx]
        else:
            colors = [to_hex(to_rgba(c)[:3]) for c in all_colors]

        categories = color_source_vector.categories.tolist() + ["NaN"]

        if len(categories) > len(colors):
            return dict(zip(categories, colors + [na_color.get_hex_with_alpha()], strict=True))

        return dict(zip(categories, colors, strict=True))

    return _get_default_categorial_color_mapping(color_source_vector=color_source_vector, cmap_params=cmap_params)


def _has_colors_in_uns(
    sdata: sd.SpatialData,
    table_name: str | None,
    col_to_colorby: str,
) -> bool:
    """
    Check if <column_name>_colors exists in the specified table's .uns slot.

    Parameters
    ----------
    sdata
        SpatialData object containing tables
    table_name
        Name of the table to check. If None, uses the first available table.
    col_to_colorby
        Name of the categorical column (e.g., "celltype")

    Returns
    -------
    True if <col_to_colorby>_colors exists in the table's .uns, False otherwise
    """
    color_key = f"{col_to_colorby}_colors"

    # Determine which table to use
    if table_name is not None:
        if table_name not in sdata.tables:
            return False
        table_to_use = table_name
    else:
        if len(sdata.tables.keys()) == 0:
            return False
        # When no table is specified, check all tables for the color key
        return any(color_key in adata.uns for adata in sdata.tables.values())

    adata = sdata.tables[table_to_use]
    return color_key in adata.uns


def _extract_colors_from_table_uns(
    sdata: sd.SpatialData,
    table_name: str | None,
    col_to_colorby: str,
    color_source_vector: ArrayLike | pd.Series[CategoricalDtype],
    na_color: ColorLike,
) -> Mapping[str, str] | None:
    """
    Extract categorical colors from the <column_name>_colors pattern in adata.uns.

    This function looks for colors stored in the format <col_to_colorby>_colors in the
    specified table's .uns slot and creates a mapping from categories to colors.

    Parameters
    ----------
    sdata
        SpatialData object containing tables
    table_name
        Name of the table to look in. If None, uses the first available table.
    col_to_colorby
        Name of the categorical column (e.g., "celltype")
    color_source_vector
        Categorical vector containing the categories to map
    na_color
        Color to use for NaN/missing values

    Returns
    -------
    Mapping from category names to hex colors, or None if colors not found
    """
    color_key = f"{col_to_colorby}_colors"

    # Determine which table to use
    if table_name is not None:
        if table_name not in sdata.tables:
            logger.warning(f"Table '{table_name}' not found in sdata. Available tables: {list(sdata.tables.keys())}")
            return None
        table_to_use = table_name
    else:
        if len(sdata.tables) == 0:
            logger.warning("No tables found in sdata.")
            return None
        # No explicit table provided: search all tables for the color key
        candidate_tables: list[str] = [
            name
            for name, ad in sdata.tables.items()
            if color_key in ad.uns  # type: ignore[union-attr]
        ]
        if not candidate_tables:
            logger.debug(f"Color key '{color_key}' not found in any table uns.")
            return None
        table_to_use = candidate_tables[0]
        if len(candidate_tables) > 1:
            logger.warning(
                f"Color key '{color_key}' found in multiple tables {candidate_tables}; using table '{table_to_use}'."
            )
        logger.info(f"No table name provided, using '{table_to_use}' for color extraction.")

    adata = sdata.tables[table_to_use]

    # Check if the color pattern exists
    if color_key not in adata.uns:
        logger.debug(f"Color key '{color_key}' not found in table '{table_to_use}' uns.")
        return None

    # Extract colors and categories
    stored_colors = adata.uns[color_key]
    # Drop categories not present in the current subset (e.g. when plotting
    # per-coordinate-system) so that positional color lookups stay aligned.
    color_source_vector = color_source_vector.remove_unused_categories()
    categories = color_source_vector.categories.tolist()

    # Validate na_color format and convert to hex string
    if isinstance(na_color, Color):
        na_color_hex = na_color.get_hex()
    else:
        na_color_str = str(na_color)
        if "#" not in na_color_str:
            logger.warning("Expected `na_color` to be a hex color, converting...")
            na_color_hex = to_hex(to_rgba(na_color)[:3])
        else:
            na_color_hex = na_color_str

    # Strip alpha channel from na_color if present
    if len(na_color_hex) == 9:  # #rrggbbaa format
        na_color_hex = na_color_hex[:7]  # Keep only #rrggbb

    def _to_hex_no_alpha(color_value: Any) -> str | None:
        try:
            rgba = to_rgba(color_value)[:3]
            hex_color: str = to_hex(rgba)
            if len(hex_color) == 9:
                hex_color = hex_color[:7]
            return hex_color
        except (TypeError, ValueError) as e:
            logger.warning(f"Error converting color '{color_value}' to hex format: {e}")
            return None

    color_mapping: dict[str, str] = {}

    if isinstance(stored_colors, Mapping):
        for category in categories:
            raw_color = stored_colors.get(category)
            if raw_color is None:
                logger.warning(f"No color specified for '{category}' in '{color_key}', using na_color.")
                color_mapping[category] = na_color_hex
                continue
            hex_color = _to_hex_no_alpha(raw_color)
            color_mapping[category] = hex_color if hex_color is not None else na_color_hex
        logger.info(f"Successfully extracted {len(color_mapping)} colors from '{color_key}' in table '{table_to_use}'.")
    else:
        try:
            hex_colors = [_to_hex_no_alpha(color) for color in stored_colors]
        except TypeError:
            logger.warning(f"Unsupported color storage for '{color_key}'. Expected sequence or mapping.")
            return None

        # Map by the category's position in the *full* table, not in the
        # (possibly subset) color_source_vector, so colors stay consistent
        # across coordinate systems.
        all_cats = (
            adata.obs[col_to_colorby].cat.categories.tolist()
            if col_to_colorby in adata.obs and hasattr(adata.obs[col_to_colorby], "cat")
            else categories
        )
        # Map category -> index once (O(K)) instead of a per-category list scan
        # (was O(K^2) via list.index). all_cats comes from pandas .categories,
        # which is unique, so a plain dict comprehension is sufficient.
        cat_to_idx: dict[Any, int] = {c: i for i, c in enumerate(all_cats)}
        for category in categories:
            idx = cat_to_idx.get(category)
            if idx is not None and idx < len(hex_colors) and hex_colors[idx] is not None:
                hex_color = hex_colors[idx]
                assert hex_color is not None  # type narrowing for mypy
                color_mapping[category] = hex_color
            else:
                logger.warning(f"Not enough colors provided for category '{category}', using na_color.")
                color_mapping[category] = na_color_hex
        logger.info(f"Successfully extracted {len(hex_colors)} colors from '{color_key}' in table '{table_to_use}'.")

    color_mapping["NaN"] = na_color_hex
    return color_mapping


def _modify_categorical_color_mapping(
    mapping: Mapping[str, str],
    groups: list[str] | str | None = None,
    palette: dict[str, str] | list[str] | str | None = None,
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
    adata: AnnData | None,
    na_color: Color,
    cluster_key: str | None = None,
    color_source_vector: ArrayLike | pd.Series[CategoricalDtype] | None = None,
    cmap_params: CmapParams | None = None,
    alpha: float = 1,
    groups: list[str] | str | None = None,
    palette: dict[str, str] | list[str] | str | None = None,
    render_type: Literal["points", "labels"] | None = None,
) -> Mapping[str, str]:
    if not isinstance(color_source_vector, Categorical):
        raise TypeError(f"Expected `categories` to be a `Categorical`, but got {type(color_source_vector).__name__}")

    # Dict palette (e.g. from make_palette_from_data): use directly as category→color mapping
    if isinstance(palette, dict):
        na_color_hex = na_color.get_hex_with_alpha() if isinstance(na_color, Color) else str(na_color)
        if isinstance(groups, str):
            groups = [groups]
        if groups is not None:
            mapping = {cat: palette.get(cat, na_color_hex) for cat in groups if cat in color_source_vector.categories}
        else:
            mapping = {cat: palette.get(cat, na_color_hex) for cat in color_source_vector.categories}
        mapping["NaN"] = na_color_hex
        return mapping

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


def _get_linear_colormap(colors: list[str], background: str) -> list[LinearSegmentedColormap]:
    return [LinearSegmentedColormap.from_list(c, [background, c], N=256) for c in colors]
