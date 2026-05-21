"""Convert canvas pixel-coord shapes into a CS-coord ShapesModel."""
from __future__ import annotations

from typing import Any

import geopandas as gpd
import spatialdata as sd
from shapely.geometry import Polygon
from spatialdata.models import ShapesModel

# Tolerance for lasso simplification, in CS units. The lasso path is sampled
# at every mouse-move so a freehand loop easily exceeds 1000 vertices;
# 0.5 keeps shape fidelity while collapsing co-linear points.
_LASSO_SIMPLIFY_TOL = 0.5


def pixel_shape_to_polygon(
    shape: dict[str, Any],
    image_w: int,
    image_h: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> Polygon | None:
    """Convert a single ``DrawCanvas`` shape entry to a CS-coord shapely Polygon.

    Returns ``None`` if the shape is invalid (no verts, <3 verts after
    construction, or empty).

    The matplotlib image axes use ``origin='upper'`` — the *smaller* y-value
    of ``ylim`` corresponds to the top of the rendered image (PNG row 0).
    """
    verts = shape.get("verts") if isinstance(shape, dict) else None
    if not verts:
        return None

    xmin, xmax = float(xlim[0]), float(xlim[1])
    y_lo, y_hi = sorted((float(ylim[0]), float(ylim[1])))

    def px_to_cs(px: float, py: float) -> tuple[float, float]:
        return (
            xmin + (px / image_w) * (xmax - xmin),
            y_lo + (py / image_h) * (y_hi - y_lo),
        )

    cs_verts = [px_to_cs(v[0], v[1]) for v in verts]
    if len(cs_verts) < 3:
        return None
    poly = Polygon(cs_verts)
    if poly.is_empty:
        return None
    if shape.get("type") == "polygon" and len(cs_verts) > 50:
        # Lasso-like (high vertex count) polygons benefit from simplification.
        poly = poly.simplify(_LASSO_SIMPLIFY_TOL, preserve_topology=True)
    return poly


def build_shapes_model(
    polygons: list[Polygon],
    coordinate_system: str,
) -> Any:
    """Wrap shapely polygons in a ShapesModel registered with Identity in ``coordinate_system``."""
    gdf = gpd.GeoDataFrame({"geometry": polygons})
    return ShapesModel.parse(
        gdf,
        transformations={coordinate_system: sd.transformations.Identity()},
    )
