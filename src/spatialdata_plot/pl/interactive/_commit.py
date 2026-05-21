"""Convert canvas pixel-coord shapes into a CS-coord ShapesModel."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from shapely.geometry import Polygon
from spatialdata.models import ShapesModel
from spatialdata.transformations.transformations import Identity

from ._render import RenderExtent

_LASSO_SIMPLIFY_TOL = 0.5
_DENSE_POLYGON_VERTEX_THRESHOLD = 50


def pixel_shape_to_polygon(shape: dict[str, Any], extent: RenderExtent) -> Polygon | None:
    """Convert a single ``DrawCanvas`` shape entry to a CS-coord shapely Polygon.

    Returns ``None`` if the shape is invalid (no verts, <3 verts after
    construction, or empty).
    """
    verts = shape.get("verts") if isinstance(shape, dict) else None
    if not verts:
        return None

    xmin, xmax = float(extent.xlim[0]), float(extent.xlim[1])
    y_lo, y_hi = sorted((float(extent.ylim[0]), float(extent.ylim[1])))
    w, h = extent.image_w, extent.image_h

    cs_verts = [(xmin + (v[0] / w) * (xmax - xmin), y_lo + (v[1] / h) * (y_hi - y_lo)) for v in verts]
    if len(cs_verts) < 3:
        return None
    poly = Polygon(cs_verts)
    if poly.is_empty:
        return None
    if shape.get("type") == "polygon" and len(cs_verts) > _DENSE_POLYGON_VERTEX_THRESHOLD:
        poly = poly.simplify(_LASSO_SIMPLIFY_TOL, preserve_topology=True)
    return poly


def build_shapes_model(polygons: list[Polygon], coordinate_system: str) -> Any:
    """Wrap shapely polygons in a ShapesModel registered with Identity in ``coordinate_system``."""
    gdf = gpd.GeoDataFrame({"geometry": polygons})
    return ShapesModel.parse(gdf, transformations={coordinate_system: Identity()})
