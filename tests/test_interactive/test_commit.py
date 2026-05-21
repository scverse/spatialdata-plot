"""Tests for pixel-coord → CS-coord conversion and ShapesModel construction."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from spatialdata_plot.pl.interactive._commit import (
    build_shapes_model,
    pixel_shape_to_polygon,
)


def test_rect_maps_to_full_cs_extent():
    """A rect spanning the full PNG maps to the full CS extent."""
    shape = {"type": "rect", "verts": [[0, 0], [100, 0], [100, 100], [0, 100]]}
    poly = pixel_shape_to_polygon(shape, 100, 100, (0.0, 50.0), (50.0, 0.0))
    assert poly.bounds == (0.0, 0.0, 50.0, 50.0)


def test_rect_subregion():
    """A 50% rect maps to the central CS quadrant correctly."""
    shape = {"type": "rect", "verts": [[25, 25], [75, 25], [75, 75], [25, 75]]}
    poly = pixel_shape_to_polygon(shape, 100, 100, (0.0, 100.0), (100.0, 0.0))
    assert poly.bounds == (25.0, 25.0, 75.0, 75.0)


def test_y_axis_orientation_matplotlib_image():
    """matplotlib image axes have origin='upper'; py=0 maps to the smaller y."""
    # PNG pixel (0, 0) = top-left. With ylim=(100, 0) (matplotlib image style),
    # the smaller y (0) is at the top, larger y (100) at the bottom.
    shape = {"type": "polygon", "verts": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    poly = pixel_shape_to_polygon(shape, 100, 100, (0.0, 100.0), (100.0, 0.0))
    # Pixel (0, 0) (top) → CS y = 0; pixel (0, 10) (10px down) → CS y = 10.
    assert poly.bounds == (0.0, 0.0, 10.0, 10.0)


def test_y_axis_non_reversed_ylim():
    """sorted() handles ylim either way round — no orientation flip."""
    shape = {"type": "polygon", "verts": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    # Hand it ylim in non-reversed order; result should be the same.
    poly = pixel_shape_to_polygon(shape, 100, 100, (0.0, 100.0), (0.0, 100.0))
    assert poly.bounds == (0.0, 0.0, 10.0, 10.0)


def test_invalid_shapes_return_none():
    """Empty verts, <3 verts, or empty geometry all yield None."""
    assert pixel_shape_to_polygon({"type": "polygon", "verts": []}, 100, 100, (0, 1), (0, 1)) is None
    assert pixel_shape_to_polygon({"type": "polygon", "verts": [[1, 1], [2, 2]]}, 100, 100, (0, 1), (0, 1)) is None
    assert pixel_shape_to_polygon({}, 100, 100, (0, 1), (0, 1)) is None


def test_lasso_simplification_for_high_vertex_count():
    """Polygons with > 50 verts get .simplify() applied; rectangle-shaped ones stay rectangles."""
    n = 200
    # A near-rectangular path with many co-linear noise points along each edge.
    verts = (
        [[i, 0] for i in range(n)]
        + [[n, j] for j in range(n)]
        + [[n - i, n] for i in range(n)]
        + [[0, n - j] for j in range(n)]
    )
    shape = {"type": "polygon", "verts": verts}
    poly = pixel_shape_to_polygon(shape, n, n, (0.0, n), (float(n), 0.0))
    # After simplification, the rectangle should have far fewer than 4*n vertices.
    assert len(poly.exterior.coords) < 4 * n
    # Bounds preserved.
    assert poly.bounds == (0.0, 0.0, float(n), float(n))


def test_build_shapes_model_registers_identity_transform():
    """build_shapes_model wraps polygons with an Identity transform in the given CS."""
    polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    sm = build_shapes_model(polys, "my_cs")
    # ShapesModel inherits GeoDataFrame; transformations live in .attrs.
    import spatialdata as sd
    transforms = sd.transformations.get_transformation(sm, get_all=True)
    assert "my_cs" in transforms
    assert isinstance(transforms["my_cs"], sd.transformations.Identity)
