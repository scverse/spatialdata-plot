"""Palette generation utilities.

Two public functions:

- :func:`make_palette` — produce *n* colours, optionally reordered for
  maximum perceptual contrast or colourblind accessibility.
- :func:`make_palette_from_data` — like :func:`make_palette` but derives
  the number of colours from a :class:`~spatialdata.SpatialData` element.

Both share the same *palette* / *method* vocabulary.  The *palette*
parameter controls **which** colours are used (the source), while
*method* controls **how** they are ordered or assigned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_hex, to_rgb
from matplotlib.pyplot import colormaps as mpl_colormaps
from scanpy.plotting.palettes import default_20, default_28, default_102

if TYPE_CHECKING:
    import spatialdata as sd

# ---------------------------------------------------------------------------
# Built-in named palettes
# ---------------------------------------------------------------------------

# Okabe & Ito (2008) — designed for universal colour-vision accessibility.
# Hex values from https://jfly.uni-koeln.de/color/
_OKABE_ITO: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

_NAMED_PALETTES: dict[str, list[str]] = {
    "okabe_ito": _OKABE_ITO,
}

# ---------------------------------------------------------------------------
# Color-space helpers
# ---------------------------------------------------------------------------

# Oklab conversion (Björn Ottosson, public domain)
# https://bottosson.github.io/posts/oklab/


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """SRGB [0,1] → linear RGB."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear RGB → sRGB [0,1]."""
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert Nx3 sRGB [0,1] array to Oklab."""
    lin = _srgb_to_linear(rgb)
    l = 0.4122214708 * lin[:, 0] + 0.5363325363 * lin[:, 1] + 0.0514459929 * lin[:, 2]
    m = 0.2119034982 * lin[:, 0] + 0.6806995451 * lin[:, 1] + 0.1073969566 * lin[:, 2]
    s = 0.0883024619 * lin[:, 0] + 0.2817188376 * lin[:, 1] + 0.6299787005 * lin[:, 2]
    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)
    return np.column_stack(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ]
    )


# ---------------------------------------------------------------------------
# Color-vision-deficiency simulation (Brettel, Viénot & Mollon 1997)
# ---------------------------------------------------------------------------

# Simulation matrices for dichromacy in linear RGB space.
# Source: libDaltonLens / DaltonLens-Python (MIT licensed constants).
_CVD_MATRICES: dict[str, np.ndarray] = {
    "protanopia": np.array(
        [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]
    ),
    "deuteranopia": np.array(
        [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]
    ),
    "tritanopia": np.array(
        [[-0.006540, 0.975530, 0.031010], [0.016270, 0.943972, 0.039758], [-0.244708, 0.759930, 0.484778]]
    ),
}


def _simulate_cvd(rgb: np.ndarray, cvd_type: str) -> np.ndarray:
    """Simulate color vision deficiency on Nx3 sRGB [0,1] array.

    For ``"general"``, returns the element-wise minimum distinctness across
    all three deficiency types (worst-case).
    """
    if cvd_type == "general":
        return np.stack([_simulate_cvd(rgb, t) for t in ("protanopia", "deuteranopia", "tritanopia")])

    mat = _CVD_MATRICES[cvd_type]
    lin = _srgb_to_linear(rgb)
    sim = lin @ mat.T
    return np.clip(_linear_to_srgb(np.clip(sim, 0, 1)), 0, 1)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Shared optimization core
# ---------------------------------------------------------------------------


def _perceptual_distance_matrix(
    rgb: np.ndarray,
    colorblind_type: str | None = None,
) -> np.ndarray:
    """Pairwise Oklab Euclidean distance between colors.

    If *colorblind_type* is set, distances are computed on CVD-simulated
    colors.  For ``"general"``, the minimum distance across all three
    deficiency types is used (worst-case optimization).
    """
    if colorblind_type is not None:
        sim = _simulate_cvd(rgb, colorblind_type)
        if colorblind_type == "general":
            mats = [_pairwise_oklab_dist(_rgb_to_oklab(s)) for s in sim]
            return np.minimum.reduce(mats)  # type: ignore[no-any-return]
        rgb = sim

    lab = _rgb_to_oklab(rgb)
    return _pairwise_oklab_dist(lab)


def _pairwise_oklab_dist(lab: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance in Oklab space."""
    diff = lab[:, np.newaxis, :] - lab[np.newaxis, :, :]
    return np.sqrt((diff**2).sum(axis=-1))  # type: ignore[no-any-return]


def _optimize_assignment(
    weight_matrix: np.ndarray,
    color_dist: np.ndarray,
    n_random: int = 5000,
    n_swaps: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Find a permutation that maximizes ``sum(weights * color_dist[perm, perm])``.

    Returns an index array: ``perm[category_idx] = color_idx``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = weight_matrix.shape[0]
    if n <= 2:
        # For n<=2 there are at most 2 permutations; just try both.
        if n <= 1:
            return np.arange(n)
        id_perm = np.arange(n)
        sw_perm = np.array([1, 0])
        s_id = float(np.sum(weight_matrix * color_dist[np.ix_(id_perm, id_perm)]))
        s_sw = float(np.sum(weight_matrix * color_dist[np.ix_(sw_perm, sw_perm)]))
        return sw_perm if s_sw > s_id else id_perm

    def _score(perm: np.ndarray) -> float:
        return float(np.sum(weight_matrix * color_dist[np.ix_(perm, perm)]))

    best_perm = np.arange(n)
    best_score = _score(best_perm)

    for _ in range(n_random):
        perm = rng.permutation(n)
        s = _score(perm)
        if s > best_score:
            best_score = s
            best_perm = perm.copy()

    for _ in range(n_swaps):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        best_perm[i], best_perm[j] = best_perm[j], best_perm[i]
        s = _score(best_perm)
        if s > best_score:
            best_score = s
        else:
            best_perm[i], best_perm[j] = best_perm[j], best_perm[i]

    return best_perm


def _optimized_order(
    colors_list: list[str],
    *,
    colorblind_type: str | None = None,
    n_random: int = 5000,
    n_swaps: int = 10000,
    seed: int = 0,
) -> list[str]:
    """Reorder *colors_list* to maximize pairwise perceptual spread."""
    n = len(colors_list)
    if n <= 2:
        return colors_list

    rgb = np.array([to_rgb(c) for c in colors_list])
    cdist = _perceptual_distance_matrix(rgb, colorblind_type=colorblind_type)

    # Uniform weight matrix: all off-diagonal pairs equally important
    weights = np.ones((n, n)) - np.eye(n)

    rng = np.random.default_rng(seed)
    perm = _optimize_assignment(weights, cdist, n_random=n_random, n_swaps=n_swaps, rng=rng)
    return [to_hex(rgb[perm[i]]) for i in range(n)]


# ---------------------------------------------------------------------------
# Palette resolution
# ---------------------------------------------------------------------------


def _resolve_palette(palette: list[str] | str | None, n: int) -> list[str]:
    """Resolve *n* colours from an explicit list, a named palette, or scanpy defaults."""
    if isinstance(palette, list):
        if len(palette) < n:
            raise ValueError(f"Palette has {len(palette)} colors but {n} are needed.")
        return list(palette[:n])

    if isinstance(palette, str):
        if palette in _NAMED_PALETTES:
            colors = _NAMED_PALETTES[palette]
            if len(colors) < n:
                raise ValueError(
                    f"Named palette '{palette}' has {len(colors)} colors but {n} are needed. "
                    f"Please provide a palette with at least {n} colors."
                )
            return list(colors[:n])

        if palette in mpl_colormaps:
            cmap = mpl_colormaps[palette]
            if isinstance(cmap, ListedColormap):
                # Qualitative colormaps (tab10, Set1, etc.): sample by index
                if n > cmap.N:
                    raise ValueError(f"Colormap '{palette}' has {cmap.N} colors but {n} are needed.")
                return [to_hex(cmap(i)) for i in range(n)]
            indices = np.linspace(0, 1, n)
            return [to_hex(cmap(i)) for i in indices]

        raise ValueError(
            f"Unknown palette name '{palette}'. Use a list of colors, a matplotlib colormap name, "
            f"or one of: {', '.join(sorted(_NAMED_PALETTES))}."
        )

    # palette is None — use scanpy defaults
    if n <= 20:
        return list(default_20[:n])
    if n <= 28:
        return list(default_28[:n])
    if n <= len(default_102):
        return list(default_102[:n])

    raise ValueError(
        f"{n} colors needed but no palette was provided and the default palette only has "
        f"{len(default_102)} colors. Please provide a palette."
    )


def _resolve_element(
    sdata: sd.SpatialData,
    element: str,
    color: str,
    table_name: str | None = None,
) -> pd.Categorical:
    """Extract categorical labels from a SpatialData element.

    Labels come from a column on the element itself, or from a linked
    table (joined on the instance key to guarantee alignment).
    """
    if element in sdata.shapes:
        gdf = sdata.shapes[element]
        if color in gdf.columns:
            labels_series = gdf[color]
        else:
            labels_series, _matched_indices = _get_labels_from_table(sdata, element, color, table_name)
    elif element in sdata.points:
        ddf = sdata.points[element]
        if color in ddf.columns:
            labels_series = ddf[[color]].compute()[color]
        else:
            labels_series, _matched_indices = _get_labels_from_table(sdata, element, color, table_name)
    else:
        available = list(sdata.shapes.keys()) + list(sdata.points.keys())
        raise KeyError(
            f"Element '{element}' not found in sdata.shapes or sdata.points. "
            f"Available elements: {available}. Note: labels (raster) elements are not yet supported."
        )

    is_categorical = isinstance(getattr(labels_series, "dtype", None), pd.CategoricalDtype)
    return labels_series.values if is_categorical else pd.Categorical(labels_series)


def _get_labels_from_table(
    sdata: sd.SpatialData,
    element: str,
    color: str,
    table_name: str | None = None,
) -> tuple[pd.Series, np.ndarray]:
    """Extract a column from the table linked to an element.

    Returns (labels_series, element_indices) where element_indices maps
    each table row to its position in the element, ensuring coord-label
    alignment.
    """
    from spatialdata.models import get_table_keys

    matches: list[str] = []
    for name in sdata.tables:
        table = sdata.tables[name]
        region = table.uns.get("spatialdata_attrs", {}).get("region")
        if region is not None:
            regions = [region] if isinstance(region, str) else region
            if element in regions and color in table.obs.columns:
                matches.append(name)

    if not matches:
        raise KeyError(
            f"Column '{color}' not found for element '{element}'. Looked in the element itself and all linked tables."
        )

    if table_name is not None:
        if table_name not in matches:
            raise KeyError(
                f"Table '{table_name}' does not annotate element '{element}' or does not contain column '{color}'."
            )
        resolved_name = table_name
    elif len(matches) == 1:
        resolved_name = matches[0]
    else:
        raise ValueError(
            f"Multiple tables annotate element '{element}' with column '{color}': {matches}. "
            f"Please specify table_name= to disambiguate."
        )

    table = sdata.tables[resolved_name]
    _, _, instance_key = get_table_keys(table)

    # Join on instance key to align table rows with element positions
    instance_ids = table.obs[instance_key].values
    element_index = sdata.shapes[element].index if element in sdata.shapes else sdata.points[element].compute().index

    # Map each table instance_id to its position in the element index
    element_idx_map = {val: i for i, val in enumerate(element_index)}
    matched_indices = []
    valid_mask = []
    for iid in instance_ids:
        if iid in element_idx_map:
            matched_indices.append(element_idx_map[iid])
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_mask_arr = np.array(valid_mask)
    if not any(valid_mask):
        raise ValueError(f"No matching instance keys between table '{resolved_name}' and element '{element}'.")

    labels = table.obs.loc[valid_mask_arr, color]
    return labels.reset_index(drop=True), np.array(matched_indices)


# ---------------------------------------------------------------------------
# Method lookup tables
# ---------------------------------------------------------------------------

# Maps non-spatial contrast methods → CVD type (None = normal vision).
_CONTRAST_CVD_TYPES: dict[str, str | None] = {
    "contrast": None,
    "colorblind": "general",
    "protanopia": "protanopia",
    "deuteranopia": "deuteranopia",
    "tritanopia": "tritanopia",
}

_ALL_METHODS = sorted({"default", *_CONTRAST_CVD_TYPES})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

Method = Literal[
    "default",
    "contrast",
    "colorblind",
    "protanopia",
    "deuteranopia",
    "tritanopia",
]


def make_palette(
    n: int,
    *,
    palette: list[str] | str | None = None,
    method: Method = "default",
    n_random: int = 5000,
    n_swaps: int = 10000,
    seed: int = 0,
) -> list[str]:
    """Generate a list of *n* colours.

    The *palette* parameter controls **which** colours are sampled, while
    *method* controls **how** they are ordered.

    Parameters
    ----------
    n
        Number of colours to produce.
    palette
        Source colours.  Can be:

        - ``None`` — scanpy default palettes.
        - A **list** of colour strings (hex or named).
        - A **named palette**: ``"okabe_ito"`` (8 colourblind-safe
          colours).
        - A **matplotlib colormap name**: ``"tab10"``, ``"Set2"``, etc.
    method
        Ordering strategy:

        - ``"default"`` — take the first *n* colours in source order.
        - ``"contrast"`` — reorder to maximise pairwise perceptual
          distance (Oklab space).
        - ``"colorblind"`` — reorder to maximise pairwise distance
          under worst-case colour-vision deficiency.
        - ``"protanopia"`` / ``"deuteranopia"`` / ``"tritanopia"`` —
          reorder for a specific colour-vision deficiency.
    n_random
        Random permutations to try (optimisation methods only).
    n_swaps
        Pairwise swap iterations (optimisation methods only).
    seed
        Random seed for reproducibility (optimisation methods only).

    Returns
    -------
    list[str]
        List of *n* hex colour strings.

    Examples
    --------
    >>> sdp.pl.make_palette(5)
    >>> sdp.pl.make_palette(8, palette="okabe_ito")
    >>> sdp.pl.make_palette(10, palette="tab10", method="contrast")
    >>> sdp.pl.make_palette(6, palette="tab10", method="colorblind")
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")

    colors = _resolve_palette(palette, n)

    if method == "default":
        return [to_hex(to_rgb(c)) for c in colors]

    if method in _CONTRAST_CVD_TYPES:
        cvd_type = _CONTRAST_CVD_TYPES[method]
        return _optimized_order(colors, colorblind_type=cvd_type, n_random=n_random, n_swaps=n_swaps, seed=seed)

    valid = ", ".join(f"'{m}'" for m in _ALL_METHODS)
    raise ValueError(f"Unknown method '{method}'. Choose from {valid}.")


def make_palette_from_data(
    sdata: sd.SpatialData,
    element: str,
    color: str,
    *,
    palette: list[str] | str | None = None,
    method: Method = "default",
    table_name: str | None = None,
    n_random: int = 5000,
    n_swaps: int = 10000,
    seed: int = 0,
) -> dict[str, str]:
    """Generate a categorical colour palette for a spatial element.

    The *palette* parameter controls **which** colours are used (the source),
    while *method* controls **how** they are assigned to categories.

    Parameters
    ----------
    sdata
        A :class:`spatialdata.SpatialData` object.
    element
        Name of a shapes or points element in *sdata*.
    color
        Column name containing categorical labels.  The column is first
        looked up directly on the element (both for shapes and points);
        if not found there, it falls back to the linked AnnData table.
    palette
        Source colours.  Accepts the same values as
        :func:`make_palette` (*None*, a list, a named palette, or a
        matplotlib colormap name).
    table_name
        Name of the table to use when *color* is looked up from a linked
        table.  Required when multiple tables annotate the same element.
    method
        Strategy for assigning colours to categories:

        - ``"default"`` — assign in sorted category order (reproduces
          the current render-pipeline behaviour).
        - ``"contrast"`` / ``"colorblind"`` / ``"protanopia"`` /
          ``"deuteranopia"`` / ``"tritanopia"`` — reorder to maximise
          perceptual spread.
    n_random
        Random permutations to try (optimisation methods only).
    n_swaps
        Pairwise swap iterations (optimisation methods only).
    seed
        Random seed for reproducibility (optimisation methods only).

    Returns
    -------
    dict[str, str]
        Mapping from category name to hex colour string.  Can be passed
        directly as ``palette=`` to any render function.

    Examples
    --------
    >>> palette = sdp.pl.make_palette_from_data(sdata, "cells", "cell_type")
    >>> palette = sdp.pl.make_palette_from_data(sdata, "cells", "cell_type", palette="tab10")
    >>> palette = sdp.pl.make_palette_from_data(sdata, "cells", "cell_type", method="contrast")
    >>> palette = sdp.pl.make_palette_from_data(sdata, "cells", "cell_type", method="colorblind")
    >>> sdata.pl.render_shapes("cells", color="cell_type", palette=palette).pl.show()
    """
    labels_cat = _resolve_element(sdata, element, color, table_name=table_name)

    categories = list(labels_cat.categories)
    n_cat = len(categories)
    if n_cat == 0:
        raise ValueError(f"No categories found in column '{color}'.")

    colors_list = _resolve_palette(palette, n_cat)

    if method == "default":
        return {cat: to_hex(to_rgb(c)) for cat, c in zip(categories, colors_list, strict=True)}

    if method in _CONTRAST_CVD_TYPES:
        cvd_type = _CONTRAST_CVD_TYPES[method]
        reordered = _optimized_order(
            colors_list, colorblind_type=cvd_type, n_random=n_random, n_swaps=n_swaps, seed=seed
        )
        return dict(zip(categories, reordered, strict=True))

    valid = ", ".join(f"'{m}'" for m in _ALL_METHODS)
    raise ValueError(f"Unknown method '{method}'. Choose from {valid}.")
