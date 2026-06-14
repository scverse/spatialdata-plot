"""Shape geometry and matplotlib patch construction (extracted from utils.py, see #696)."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.colors import ColorConverter
from scipy.spatial import ConvexHull
from shapely.errors import GEOSException

from spatialdata_plot._logging import logger
from spatialdata_plot.pl._color import _resolve_continuous_norm
from spatialdata_plot.pl.render_params import ShapesRenderParams
from spatialdata_plot.pl.utils import _extract_scalar_value


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
    scale_value = _extract_scalar_value(scale_factor, default=1.0)
    centroid = _get_centroid_of_pathpatch(pathpatch)
    vertices = pathpatch.get_path().vertices
    scaled_vertices = np.array([centroid + (vertex - centroid) * scale_value for vertex in vertices])
    pathpatch.get_path().vertices = scaled_vertices


def _normalize_geom(geom: Any) -> Any:
    """Canonicalize ring orientation so matplotlib's fill rules render holes correctly.

    ``shapely.normalize`` (shapely>=2) is preferred; falls back to ``geom.normalize()``.
    None/empty geometries and geometries that fail to normalize are returned unchanged.
    """
    if geom is None or getattr(geom, "is_empty", False):
        return geom
    normalize_func = getattr(shapely, "normalize", None)
    if callable(normalize_func):
        try:
            return normalize_func(geom)
        except (GEOSException, TypeError, ValueError):
            return geom
    if hasattr(geom, "normalize"):
        try:
            return geom.normalize()
        except (GEOSException, TypeError, ValueError):
            return geom
    return geom


def _make_patch_from_multipolygon(mp: shapely.MultiPolygon) -> list[mpatches.PathPatch]:
    """
    Create PathPatches from a MultiPolygon, preserving holes robustly.

    This follows the same strategy as GeoPandas' internal Polygon plotting:
    each (multi)polygon part becomes a compound Path composed of the exterior
    ring and all interior rings. Orientation is handled by prior geometry
    normalization rather than manual ring reversal.
    """
    patches: list[mpatches.PathPatch] = []

    for poly in mp.geoms:
        if poly.is_empty:
            continue

        # Ensure 2D vertices in case geometries carry Z
        exterior = np.asarray(poly.exterior.coords)[..., :2]
        interiors = [np.asarray(ring.coords)[..., :2] for ring in poly.interiors]

        if len(interiors) == 0:
            # Simple polygon without holes
            patches.append(mpatches.Polygon(exterior, closed=True))
            continue

        # Build a compound path: exterior + all interior rings
        compound_path = mpath.Path.make_compound_path(
            mpath.Path(exterior, closed=True),
            *[mpath.Path(ring, closed=True) for ring in interiors],
        )
        patches.append(mpatches.PathPatch(compound_path))

    return patches


def _build_shape_patches(
    shapes: GeoDataFrame,
    scale: float,
) -> tuple[list[mpatches.Patch], list[int], int]:
    """Build matplotlib patches from shape geometries, once.

    Patch geometry is independent of colour/alpha, so it can be built a single time and
    shared across the fill and outline ``PatchCollection``s in :func:`_render_shapes`
    instead of being rebuilt per layer (the dominant cost for shape elements).

    Returns
    -------
    patches
        The matplotlib patches (a MultiPolygon expands to several patches).
    patch_row_idx
        For each patch, the index into the empty-filtered, re-indexed shapes — used to
        look up the per-shape colour.
    n_shapes
        Number of shapes after empty filtering (used for the single-colour broadcast rule).
    """
    df: GeoDataFrame | pd.DataFrame = shapes if isinstance(shapes, GeoDataFrame) else pd.DataFrame(shapes)
    if "geometry" not in df.columns:
        return [], [], 0

    # Normalize ring orientation, then drop empty geometries (both vectorized; fall
    # back to per-geometry normalization only if the bulk call rejects an input).
    geom_array = df["geometry"].to_numpy()
    try:
        geom_array = shapely.normalize(geom_array)
    except (GEOSException, TypeError, ValueError):
        geom_array = np.array([_normalize_geom(g) for g in geom_array], dtype=object)
    keep = ~shapely.is_empty(geom_array)
    geoms = geom_array[keep]
    radii = df["radius"].to_numpy()[keep] if "radius" in df.columns else None

    # Resolve the scale scalar once instead of per shape.
    scale_value = _extract_scalar_value(scale, default=1.0)

    patches: list[mpatches.Patch] = []
    patch_row_idx: list[int] = []
    for i, geom in enumerate(geoms):
        geom_type = geom.geom_type
        if geom_type == "Polygon":
            coords = np.asarray(geom.exterior.coords)
            centroid = np.mean(coords, axis=0)
            scaled = centroid + (coords - centroid) * scale_value
            patches.append(mpatches.Polygon(scaled, closed=True))
            patch_row_idx.append(i)
        elif geom_type == "MultiPolygon":
            for m in _make_patch_from_multipolygon(geom):
                _scale_pathpatch_around_centroid(m, scale_value)
                patches.append(m)
                patch_row_idx.append(i)
        elif geom_type == "Point":
            radius_value = _extract_scalar_value(radii[i], default=0.0) if radii is not None else 0.0
            patches.append(mpatches.Circle((geom.x, geom.y), radius=radius_value * scale_value))
            patch_row_idx.append(i)

    return patches, patch_row_idx, len(geoms)


def _get_collection_shape(
    shapes: list[GeoDataFrame],
    c: Any,
    s: float,
    render_params: ShapesRenderParams,
    fill_alpha: None | float = None,
    outline_alpha: None | float = None,
    outline_color: None | str | list[float] | np.ndarray = "white",
    linewidth: float = 0.0,
    prebuilt_patches: tuple[list[mpatches.Patch], list[int], int] | None = None,
    **kwargs: Any,
) -> PatchCollection:
    """
    Build a PatchCollection for shapes with correct handling of.

      - continuous numeric vectors with NaNs,
      - per-row RGBA arrays,
      - a single color or a list of color specs.

    Only NaNs are painted with na_color; finite values are mapped via norm+cmap.

    .. note::
       When ``outline_color`` is passed as an ``(N, 4)`` RGBA array of dtype ``float``,
       its alpha channel is mutated in place to apply ``outline_alpha``. Pass a copy
       if you need to retain the original buffer.
    """
    cmap = kwargs["cmap"]

    # Resolve na color once
    na_rgba = colors.to_rgba(render_params.cmap_params.na_color.get_hex_with_alpha())

    # Try to interpret c as numpy array
    c_arr = np.asarray(c)
    fill_c: np.ndarray

    def _as_rgba_array(x: Any) -> np.ndarray:
        return np.asarray(ColorConverter().to_rgba_array(x))

    # Case A: per-row numeric colors given as Nx3 or Nx4 float array
    if (
        c_arr.ndim == 2
        and c_arr.shape[0] == len(shapes)
        and c_arr.shape[1] in (3, 4)
        and np.issubdtype(c_arr.dtype, np.number)
    ):
        fill_c = _as_rgba_array(c_arr)

    # Case B: continuous numeric vector len == n_shapes (possibly with NaNs)
    elif c_arr.ndim == 1 and len(c_arr) == len(shapes) and np.issubdtype(c_arr.dtype, np.number):
        finite_mask = np.isfinite(c_arr)

        # Map finite values through cmap(norm(.)); NaNs get na_color.
        fill_c = np.empty((len(c_arr), 4), dtype=float)
        fill_c[:] = na_rgba
        if finite_mask.any():
            used_norm = _resolve_continuous_norm(c_arr, render_params.cmap_params)
            fill_c[finite_mask] = cmap(used_norm(c_arr[finite_mask]))

    elif c_arr.ndim == 1 and len(c_arr) == len(shapes) and c_arr.dtype == object:
        # Split into numeric vs color-like
        c_series = pd.Series(c_arr, copy=False)
        num = pd.to_numeric(c_series, errors="coerce").to_numpy()
        is_num = np.isfinite(num)

        # init with na color
        fill_c = np.empty((len(c_series), 4), dtype=float)
        fill_c[:] = na_rgba

        # numeric entries via cmap(norm)
        if is_num.any():
            used_norm = _resolve_continuous_norm(num, render_params.cmap_params)
            fill_c[is_num] = cmap(used_norm(num[is_num]))

        # non-numeric, non-NaN entries as explicit colors
        non_numeric_color_mask = (~is_num) & c_series.notna().to_numpy()
        if non_numeric_color_mask.any():
            fill_c[non_numeric_color_mask] = ColorConverter().to_rgba_array(c_series[non_numeric_color_mask].tolist())

    # Case C: single color or list of color-like specs (strings or tuples)
    else:
        fill_c = _as_rgba_array(c)

    # Apply optional fill alpha without destroying existing transparency
    if fill_alpha is not None:
        nonzero_alpha = fill_c[..., -1] > 0
        fill_c[nonzero_alpha, -1] = fill_alpha

    # Outline handling
    if outline_alpha and outline_alpha > 0.0:
        outline_arr = np.asarray(outline_color) if not isinstance(outline_color, str) else None
        if outline_arr is not None and outline_arr.ndim == 2 and outline_arr.shape == (len(shapes), 4):
            # Per-shape RGBA array. Mutate in place when already float so we don't allocate twice
            # on the hot path; otherwise upcast to a fresh float buffer.
            outline_c_array = outline_arr if outline_arr.dtype == float else outline_arr.astype(float)
        else:
            outline_c_array = _as_rgba_array(outline_color)
        outline_c_array[..., -1] = outline_alpha
        outline_c = outline_c_array.tolist()
    else:
        outline_c = [None] * fill_c.shape[0]

    # Build (or reuse) the matplotlib patches. Geometry is colour-independent, so the
    # caller can build it once via `_build_shape_patches` and share it across the fill
    # and outline collections instead of rebuilding it on every call.
    patches, patch_row_idx, n_shapes = (
        prebuilt_patches if prebuilt_patches is not None else _build_shape_patches(shapes, s)
    )

    if not patches:
        return PatchCollection([])

    # Expand the per-shape fill colours to per-patch (a MultiPolygon owns several
    # patches). Preserve the single-colour broadcast used for multi-shape elements.
    broadcast_single = n_shapes > 1 and len(fill_c) == 1
    patch_fill = np.repeat(fill_c, len(patches), axis=0) if broadcast_single else fill_c[patch_row_idx]

    return PatchCollection(
        patches,
        snap=False,
        lw=linewidth,
        facecolor=patch_fill,
        edgecolor=None if all(o is None for o in outline_c) else outline_c,
        **kwargs,
    )


def _validate_polygons(shapes: GeoDataFrame) -> GeoDataFrame:
    """
    Convert Polygons with holes to MultiPolygons to keep interior rings during rendering.

    Parameters
    ----------
    shapes
        GeoDataFrame containing a `geometry` column.

    Returns
    -------
    GeoDataFrame
        ``shapes`` with holed Polygons converted to MultiPolygons.
    """
    if "geometry" not in shapes:
        return shapes

    converted_count = 0
    for idx, geom in shapes["geometry"].items():
        if isinstance(geom, shapely.Polygon) and len(geom.interiors) > 0:
            shapes.at[idx, "geometry"] = shapely.MultiPolygon([geom])
            converted_count += 1

    if converted_count > 0:
        logger.info(
            "Converted %d Polygon(s) with holes to MultiPolygon(s) for correct rendering.",
            converted_count,
        )

    return shapes


def _convert_shapes(
    shapes: GeoDataFrame,
    target_shape: str,
    max_extent: float,
    warn_above_extent_fraction: float = 0.5,
) -> GeoDataFrame:
    """Convert shapes in a GeoDataFrame to the target_shape, using positional indexing."""
    if warn_above_extent_fraction < 0.0 or warn_above_extent_fraction > 1.0:
        warn_above_extent_fraction = 0.5
    warn_shape_size = False

    # work on a copy with a clean positional index
    shapes = shapes.reset_index(drop=True).copy()

    def _circle_to_hexagon(center: shapely.Point, radius: float) -> tuple[shapely.Polygon, None]:
        verts = [
            (
                center.x + radius * math.cos(math.radians(a)),
                center.y + radius * math.sin(math.radians(a)),
            )
            for a in range(30, 390, 60)
        ]
        return shapely.Polygon(verts), None

    def _circle_to_square(center: shapely.Point, radius: float) -> tuple[shapely.Polygon, None]:
        verts = [
            (
                center.x + radius * math.cos(math.radians(a)),
                center.y + radius * math.sin(math.radians(a)),
            )
            for a in range(45, 360, 90)
        ]
        return shapely.Polygon(verts), None

    def _circle_to_circle(center: shapely.Point, radius: float) -> tuple[shapely.Point, float]:
        return center, radius

    def _polygon_to_circle(polygon: shapely.Polygon) -> tuple[shapely.Point, float]:
        coords = np.array(polygon.exterior.coords)
        hull_pts = coords[ConvexHull(coords).vertices]
        center = np.mean(hull_pts, axis=0)
        radius = float(np.max(np.linalg.norm(hull_pts - center, axis=1)))
        nonlocal warn_shape_size
        if 2 * radius > max_extent * warn_above_extent_fraction:
            warn_shape_size = True
        return shapely.Point(center), radius

    def _polygon_to_hexagon(polygon: shapely.Polygon) -> tuple[shapely.Polygon, None]:
        c, r = _polygon_to_circle(polygon)
        return _circle_to_hexagon(c, r)

    def _polygon_to_square(polygon: shapely.Polygon) -> tuple[shapely.Polygon, None]:
        c, r = _polygon_to_circle(polygon)
        return _circle_to_square(c, r)

    def _multipolygon_to_circle(multipolygon: shapely.MultiPolygon) -> tuple[shapely.Point, float]:
        pts = []
        for poly in multipolygon.geoms:
            pts.extend(poly.exterior.coords)
        pts_array = np.array(pts)
        hull_pts = pts_array[ConvexHull(pts_array).vertices]
        center = np.mean(hull_pts, axis=0)
        radius = float(np.max(np.linalg.norm(hull_pts - center, axis=1)))
        nonlocal warn_shape_size
        if 2 * radius > max_extent * warn_above_extent_fraction:
            warn_shape_size = True
        return shapely.Point(center), radius

    def _multipolygon_to_hexagon(multipolygon: shapely.MultiPolygon) -> tuple[shapely.Polygon, None]:
        c, r = _multipolygon_to_circle(multipolygon)
        return _circle_to_hexagon(c, r)

    def _multipolygon_to_square(multipolygon: shapely.MultiPolygon) -> tuple[shapely.Polygon, None]:
        c, r = _multipolygon_to_circle(multipolygon)
        return _circle_to_square(c, r)

    # choose conversion methods
    conversion_methods: dict[str, Any]
    if target_shape == "circle":
        conversion_methods = {
            "Point": _circle_to_circle,
            "Polygon": _polygon_to_circle,
            "MultiPolygon": _multipolygon_to_circle,
        }
    elif target_shape == "hex":
        conversion_methods = {
            "Point": _circle_to_hexagon,
            "Polygon": _polygon_to_hexagon,
            "MultiPolygon": _multipolygon_to_hexagon,
        }
    elif target_shape == "visium_hex":
        # estimate hex radius from point spacing when possible
        point_centers = []
        non_point_count = 0
        for geom in shapes.geometry:
            if geom.geom_type == "Point":
                point_centers.append((geom.x, geom.y))
            else:
                non_point_count += 1
        if non_point_count > 0:
            logger.warning("visium_hex supports Points best. Non-Point geometries will use regular hex conversion.")
        if len(point_centers) >= 2:
            centers = np.array(point_centers, dtype=float)
            # pairwise min distance
            dmin = np.inf
            for i in range(len(centers)):
                diffs = centers[i + 1 :] - centers[i]
                if diffs.size:
                    d = np.min(np.linalg.norm(diffs, axis=1))
                    dmin = min(dmin, d)
            if not np.isfinite(dmin) or dmin <= 0:
                # fallback
                conversion_methods = {
                    "Point": _circle_to_hexagon,
                    "Polygon": _polygon_to_hexagon,
                    "MultiPolygon": _multipolygon_to_hexagon,
                }
            else:
                hex_radius = dmin / math.sqrt(3.0)

                def _circle_to_visium_hex(center: shapely.Point, radius: float) -> tuple[shapely.Polygon, None]:
                    return _circle_to_hexagon(center, hex_radius)

                def _polygon_to_visium_hex(polygon: shapely.Polygon) -> tuple[shapely.Polygon, None]:
                    return _polygon_to_hexagon(polygon)

                def _multipolygon_to_visium_hex(multipolygon: shapely.MultiPolygon) -> tuple[shapely.Polygon, None]:
                    return _multipolygon_to_hexagon(multipolygon)

                conversion_methods = {
                    "Point": _circle_to_visium_hex,
                    "Polygon": _polygon_to_visium_hex,
                    "MultiPolygon": _multipolygon_to_visium_hex,
                }
        else:
            conversion_methods = {
                "Point": _circle_to_hexagon,
                "Polygon": _polygon_to_hexagon,
                "MultiPolygon": _multipolygon_to_hexagon,
            }
    else:
        conversion_methods = {
            "Point": _circle_to_square,
            "Polygon": _polygon_to_square,
            "MultiPolygon": _multipolygon_to_square,
        }

    # ensure radius column exists if needed
    if "radius" not in shapes.columns:
        shapes["radius"] = np.nan

    # convert all geometries using positional indexing
    for i in range(len(shapes)):
        geom = shapes.geometry.iloc[i]
        gtype = geom.geom_type
        if gtype == "Point":
            r = shapes["radius"].iloc[i]
            r = float(r) if np.isfinite(r) else 0.0
            converted, radius = conversion_methods["Point"](geom, r)  # type: ignore[arg-type]
        elif gtype == "Polygon":
            converted, radius = conversion_methods["Polygon"](geom)  # type: ignore[arg-type]
        elif gtype == "MultiPolygon":
            converted, radius = conversion_methods["MultiPolygon"](geom)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Converting shape {gtype} to {target_shape} is not supported.")
        shapes.at[i, "geometry"] = converted
        if radius is not None:
            shapes.at[i, "radius"] = radius

    if warn_shape_size:
        logger.info(
            f"At least one converted shape spans >= {warn_above_extent_fraction * 100:.0f}% of the "
            "original total bound. Results may be suboptimal."
        )

    return shapes
