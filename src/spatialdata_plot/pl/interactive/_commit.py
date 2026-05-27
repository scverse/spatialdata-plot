"""Convert anybioimage canvas shapes into CS-coord shapely geometries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from shapely.geometry import Point, Polygon, box

PxToCs = Callable[[float, float], tuple[float, float]]


def _make_px_to_cs(xmin: float, xmax: float, y_lo: float, y_hi: float, image_w: int, image_h: int) -> PxToCs:
    """Build an affine mapping (px_x, px_y) → (cs_x, cs_y).

    The y_lo/y_hi are the sorted ylim values; image_h pixels map linearly
    between them. matplotlib image axes with ``origin='upper'`` return
    reversed ylim — sorting normalises that.
    """
    dx = xmax - xmin
    dy = y_hi - y_lo

    def px_to_cs(x_px: float, y_px: float) -> tuple[float, float]:
        return (xmin + (x_px / image_w) * dx, y_lo + (y_px / image_h) * dy)

    return px_to_cs


def _roi_to_polygon(roi: dict[str, Any], px_to_cs: PxToCs) -> Polygon | None:
    """ROI dict ``{x, y, width, height}`` → axis-aligned rectangle Polygon."""
    try:
        x0, y0 = px_to_cs(float(roi["x"]), float(roi["y"]))
        x1, y1 = px_to_cs(float(roi["x"]) + float(roi["width"]), float(roi["y"]) + float(roi["height"]))
    except (KeyError, TypeError, ValueError):
        return None
    poly = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    return poly if not poly.is_empty else None


def _polygon_to_polygon(poly: dict[str, Any], px_to_cs: PxToCs) -> Polygon | None:
    """Polygon dict ``{id, points: [{x, y}, ...]}`` → shapely Polygon (≥3 verts)."""
    pts = poly.get("points") or []
    try:
        cs_verts = [px_to_cs(float(p["x"]), float(p["y"])) for p in pts]
    except (KeyError, TypeError, ValueError):
        return None
    if len(cs_verts) < 3:
        return None
    geom = Polygon(cs_verts)
    return geom if not geom.is_empty else None


def _point_to_circle(pt: dict[str, Any], px_to_cs: PxToCs, radius: float) -> Polygon | None:
    """Point dict ``{x, y}`` → circle Polygon of the given CS-units radius.

    Stored as a polygon so the resulting ShapesModel is uniform-type and
    doesn't need a ``radius`` column.
    """
    try:
        cx, cy = px_to_cs(float(pt["x"]), float(pt["y"]))
    except (KeyError, TypeError, ValueError):
        return None
    geom = Point(cx, cy).buffer(radius)
    return geom if not geom.is_empty else None


def collect_geoms_from_viewer(
    viewer: Any,
    *,
    xmin: float,
    xmax: float,
    y_lo: float,
    y_hi: float,
    image_w: int,
    image_h: int,
    point_radius: float,
) -> list[Polygon]:
    """Read the viewer's three shape stores and convert each entry to a CS-coord Polygon.

    Order of returned geometries: ROIs first, then polygons, then points. Invalid
    entries (missing keys, degenerate geometry) are silently skipped.
    """
    px_to_cs = _make_px_to_cs(xmin, xmax, y_lo, y_hi, image_w, image_h)
    geoms: list[Polygon] = []
    for roi in viewer._rois_data or []:
        g = _roi_to_polygon(roi, px_to_cs)
        if g is not None:
            geoms.append(g)
    for poly in viewer._polygons_data or []:
        g = _polygon_to_polygon(poly, px_to_cs)
        if g is not None:
            geoms.append(g)
    for pt in viewer._points_data or []:
        g = _point_to_circle(pt, px_to_cs, point_radius)
        if g is not None:
            geoms.append(g)
    return geoms
