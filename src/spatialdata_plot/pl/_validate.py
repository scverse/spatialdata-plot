"""Parameter validation and type checking for render_* / show (extracted from utils.py, see #696)."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import spatialdata as sd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from spatialdata import (
    SpatialData,
    get_element_annotators,
)
from spatialdata.models import get_table_keys
from xarray import DataArray, DataTree

from spatialdata_plot._logging import logger
from spatialdata_plot.pl._color import _get_colors_for_categorical_obs, _is_color_like, _prepare_cmap_norm
from spatialdata_plot.pl.render_params import (
    CmapParams,
    Color,
    ColorLike,
    _FontSize,
    _FontWeight,
)

_GROUPS_IGNORED_WARNING = "Parameter 'groups' is ignored when 'color' is a literal color, not a column name."


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


def _type_check_params(param_dict: dict[str, Any], element_type: str) -> dict[str, Any]:
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
