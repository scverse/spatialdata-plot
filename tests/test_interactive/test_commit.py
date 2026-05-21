"""Tests for pixel-coord → CS-coord conversion and ShapesModel construction."""

from __future__ import annotations

from shapely.geometry import Polygon
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import Identity

from spatialdata_plot.pl.interactive._commit import (
    build_shapes_model,
    pixel_shape_to_polygon,
)
from spatialdata_plot.pl.interactive._render import RenderExtent


def _extent(w=100, h=100, xlim=(0.0, 100.0), ylim=(100.0, 0.0)) -> RenderExtent:
    return RenderExtent(w, h, xlim, ylim)


def test_rect_maps_to_full_cs_extent():
    shape = {"type": "rect", "verts": [[0, 0], [100, 0], [100, 100], [0, 100]]}
    poly = pixel_shape_to_polygon(shape, _extent(xlim=(0.0, 50.0), ylim=(50.0, 0.0)))
    assert poly.bounds == (0.0, 0.0, 50.0, 50.0)


def test_rect_subregion():
    shape = {"type": "rect", "verts": [[25, 25], [75, 25], [75, 75], [25, 75]]}
    poly = pixel_shape_to_polygon(shape, _extent())
    assert poly.bounds == (25.0, 25.0, 75.0, 75.0)


def test_y_axis_orientation_matplotlib_image():
    shape = {"type": "polygon", "verts": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    poly = pixel_shape_to_polygon(shape, _extent())
    assert poly.bounds == (0.0, 0.0, 10.0, 10.0)


def test_y_axis_non_reversed_ylim():
    shape = {"type": "polygon", "verts": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    poly = pixel_shape_to_polygon(shape, _extent(ylim=(0.0, 100.0)))
    assert poly.bounds == (0.0, 0.0, 10.0, 10.0)


def test_invalid_shapes_return_none():
    ext = _extent(xlim=(0, 1), ylim=(0, 1))
    assert pixel_shape_to_polygon({"type": "polygon", "verts": []}, ext) is None
    assert pixel_shape_to_polygon({"type": "polygon", "verts": [[1, 1], [2, 2]]}, ext) is None
    assert pixel_shape_to_polygon({}, ext) is None


def test_lasso_simplification_for_high_vertex_count():
    n = 200
    verts = (
        [[i, 0] for i in range(n)]
        + [[n, j] for j in range(n)]
        + [[n - i, n] for i in range(n)]
        + [[0, n - j] for j in range(n)]
    )
    shape = {"type": "polygon", "verts": verts}
    poly = pixel_shape_to_polygon(shape, _extent(w=n, h=n, xlim=(0.0, float(n)), ylim=(float(n), 0.0)))
    assert len(poly.exterior.coords) < 4 * n
    assert poly.bounds == (0.0, 0.0, float(n), float(n))


def test_build_shapes_model_registers_identity_transform():
    polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    sm = build_shapes_model(polys, "my_cs")
    transforms = get_transformation(sm, get_all=True)
    assert "my_cs" in transforms
    assert isinstance(transforms["my_cs"], Identity)
