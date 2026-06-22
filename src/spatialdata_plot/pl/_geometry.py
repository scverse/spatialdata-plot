"""Shape geometry and matplotlib patch construction (extracted from utils.py, see #696)."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.path as mpath
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from matplotlib.collections import PathCollection
from matplotlib.colors import ColorConverter
from scipy.spatial import ConvexHull
from shapely.errors import GEOSException

from spatialdata_plot._logging import logger
from spatialdata_plot.pl.render_params import ShapesRenderParams
from spatialdata_plot.pl.utils import _extract_scalar_value


def _get_centroid_of_path(path: mpath.Path) -> tuple[float, float]:
    vertices = path.vertices
    x = vertices[:, 0]
    y = vertices[:, 1]

    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    # Calculate the centroid coordinates
    centroid_x = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * area)
    centroid_y = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * area)

    return centroid_x, centroid_y


def _scale_path_around_centroid(path: mpath.Path, scale_factor: float) -> None:
    scale_value = _extract_scalar_value(scale_factor, default=1.0)
    centroid = np.asarray(_get_centroid_of_path(path))
    path.vertices = centroid + (path.vertices - centroid) * scale_value


def _scale_geometries(geometries: np.ndarray, scale: float) -> np.ndarray:
    """Scale each geometry about its bounding-box centre (``shapely.affinity.scale``'s default origin).

    Vectorised over all coordinates at once — a per-geometry ``affinity.scale`` loop dominates large renders.
    """
    bbox = shapely.bounds(geometries)  # (n, 4): minx, miny, maxx, maxy
    centre = np.column_stack([(bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2])
    coords, idx = shapely.get_coordinates(geometries, return_index=True)
    return shapely.set_coordinates(geometries.copy(), (coords - centre[idx]) * scale + centre[idx])


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


def _make_paths_from_multipolygon(mp: shapely.MultiPolygon) -> list[mpath.Path]:
    """
    Create matplotlib ``Path``s from a MultiPolygon, preserving holes robustly.

    This follows the same strategy as GeoPandas' internal Polygon plotting:
    each (multi)polygon part becomes a compound Path composed of the exterior
    ring and all interior rings. Orientation is handled by prior geometry
    normalization rather than manual ring reversal.
    """
    paths: list[mpath.Path] = []

    for poly in mp.geoms:
        if poly.is_empty:
            continue

        # Ensure 2D vertices in case geometries carry Z
        exterior = np.asarray(poly.exterior.coords)[..., :2]
        interiors = [np.asarray(ring.coords)[..., :2] for ring in poly.interiors]

        if len(interiors) == 0:
            # Simple polygon without holes
            paths.append(mpath.Path(exterior, closed=True))
            continue

        # Build a compound path: exterior + all interior rings
        paths.append(
            mpath.Path.make_compound_path(
                mpath.Path(exterior, closed=True),
                *[mpath.Path(ring, closed=True) for ring in interiors],
            )
        )

    return paths


def _build_shape_paths(
    shapes: GeoDataFrame,
    scale: float,
) -> tuple[list[mpath.Path], list[int], int]:
    """Build matplotlib ``Path``s from shape geometries, once.

    Built once and shared across the fill and outline ``PathCollection``s in :func:`_render_shapes`.
    Emitting ``Path``s directly avoids constructing one ``matplotlib.patches.*`` object per shape — the
    dominant cost for large shape elements.

    Returns
    -------
    paths
        The matplotlib ``Path``s (a MultiPolygon expands to several paths).
    row_idx
        For each path, the index into the empty-filtered, re-indexed shapes — used to look up the
        per-shape colour.
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

    paths: list[mpath.Path] = []
    row_idx: list[int] = []
    for i, geom in enumerate(geoms):
        geom_type = geom.geom_type
        if geom_type == "Polygon":
            coords = np.asarray(geom.exterior.coords)
            centroid = np.mean(coords, axis=0)
            scaled = centroid + (coords - centroid) * scale_value
            paths.append(mpath.Path(scaled, closed=True))
            row_idx.append(i)
        elif geom_type == "MultiPolygon":
            for p in _make_paths_from_multipolygon(geom):
                _scale_path_around_centroid(p, scale_value)
                paths.append(p)
                row_idx.append(i)
        elif geom_type == "Point":
            radius_value = _extract_scalar_value(radii[i], default=0.0) if radii is not None else 0.0
            paths.append(mpath.Path.circle((geom.x, geom.y), radius_value * scale_value))
            row_idx.append(i)

    return paths, row_idx, len(geoms)


def _get_collection_shape(
    shapes: list[GeoDataFrame],
    c: Any,
    s: float,
    render_params: ShapesRenderParams,
    fill_alpha: None | float = None,
    outline_alpha: None | float = None,
    outline_color: None | str | list[float] | np.ndarray = "white",
    linewidth: float = 0.0,
    prebuilt_paths: tuple[list[mpath.Path], list[int], int] | None = None,
    **kwargs: Any,
) -> PathCollection:
    """Build a PathCollection for shapes.

    ``c`` is the per-row fill: an ``(N, 4)`` RGBA array (from :meth:`ColorSpec.to_rgba`) or a single
    color / list of color specs (broadcast). ``outline_color`` may be an ``(N, 4)`` float RGBA array,
    whose alpha channel is mutated in place to apply ``outline_alpha`` (pass a copy to retain it).
    """
    # Per-row fill is pre-mapped to RGBA by ``ColorSpec.to_rgba`` (an (N, 4) float array, used
    # as-is); otherwise ``c`` is a single color / list of color specs (e.g. the white outline
    # placeholder), which ``to_rgba_array`` broadcasts.
    c_arr = np.asarray(c)
    if c_arr.shape == (len(shapes), 4) and np.issubdtype(c_arr.dtype, np.floating):
        fill_c = c_arr
    else:
        fill_c = np.asarray(ColorConverter().to_rgba_array(c))

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
            outline_c_array = np.asarray(ColorConverter().to_rgba_array(outline_color))
        outline_c_array[..., -1] = outline_alpha
        outline_c = outline_c_array.tolist()
    else:
        outline_c = [None] * fill_c.shape[0]

    # Reuse the shared paths when provided (see _build_shape_paths), else build them.
    paths, row_idx, n_shapes = prebuilt_paths if prebuilt_paths is not None else _build_shape_paths(shapes, s)

    if not paths:
        return PathCollection([])

    # Expand the per-shape fill colours to per-path (a MultiPolygon owns several
    # paths). Preserve the single-colour broadcast used for multi-shape elements.
    broadcast_single = n_shapes > 1 and len(fill_c) == 1
    path_fill = np.repeat(fill_c, len(paths), axis=0) if broadcast_single else fill_c[row_idx]

    return PathCollection(
        paths,
        snap=False,
        lw=linewidth,
        facecolor=path_fill,
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


def _enclosing_circle(coords: np.ndarray) -> tuple[shapely.Point, float]:
    """Enclosing circle from a point cloud: centroid of the convex hull and the max vertex distance."""
    hull_pts = coords[ConvexHull(coords).vertices]
    center = np.mean(hull_pts, axis=0)
    radius = float(np.max(np.linalg.norm(hull_pts - center, axis=1)))
    return shapely.Point(center), radius


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

    def _polygon_to_circle(polygon: shapely.Polygon) -> tuple[shapely.Point, float]:
        center, radius = _enclosing_circle(np.array(polygon.exterior.coords))
        nonlocal warn_shape_size
        if 2 * radius > max_extent * warn_above_extent_fraction:
            warn_shape_size = True
        return center, radius

    def _polygon_to_hexagon(polygon: shapely.Polygon) -> tuple[shapely.Polygon, None]:
        return _circle_to_hexagon(*_polygon_to_circle(polygon))

    def _polygon_to_square(polygon: shapely.Polygon) -> tuple[shapely.Polygon, None]:
        return _circle_to_square(*_polygon_to_circle(polygon))

    def _multipolygon_to_circle(multipolygon: shapely.MultiPolygon) -> tuple[shapely.Point, float]:
        coords = np.array([pt for poly in multipolygon.geoms for pt in poly.exterior.coords])
        center, radius = _enclosing_circle(coords)
        nonlocal warn_shape_size
        if 2 * radius > max_extent * warn_above_extent_fraction:
            warn_shape_size = True
        return center, radius

    def _multipolygon_to_hexagon(multipolygon: shapely.MultiPolygon) -> tuple[shapely.Polygon, None]:
        return _circle_to_hexagon(*_multipolygon_to_circle(multipolygon))

    def _multipolygon_to_square(multipolygon: shapely.MultiPolygon) -> tuple[shapely.Polygon, None]:
        return _circle_to_square(*_multipolygon_to_circle(multipolygon))

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
