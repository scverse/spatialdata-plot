"""Tests for the pixel→CS affine and the anybioimage shape-format adapters."""

from __future__ import annotations

from types import SimpleNamespace

from shapely.geometry import Polygon

from spatialdata_plot.pl.interactive._commit import (
    _make_px_to_cs,
    _point_to_circle,
    _polygon_to_polygon,
    _roi_to_polygon,
    collect_geoms_from_viewer,
)


def _px_to_cs(xmin=0.0, xmax=100.0, y_lo=0.0, y_hi=100.0, w=100, h=100):
    return _make_px_to_cs(xmin, xmax, y_lo, y_hi, w, h)


def test_px_to_cs_identity_when_extent_matches_image():
    f = _px_to_cs()
    assert f(0, 0) == (0.0, 0.0)
    assert f(50, 50) == (50.0, 50.0)
    assert f(100, 100) == (100.0, 100.0)


def test_px_to_cs_scales_to_cs_extent():
    f = _px_to_cs(xmin=10.0, xmax=510.0, y_lo=0.0, y_hi=200.0, w=100, h=100)
    assert f(0, 0) == (10.0, 0.0)
    assert f(100, 100) == (510.0, 200.0)
    assert f(50, 50) == (260.0, 100.0)


def test_roi_to_polygon_axis_aligned():
    f = _px_to_cs()
    poly = _roi_to_polygon({"x": 10, "y": 20, "width": 30, "height": 40}, f)
    assert poly is not None
    assert poly.bounds == (10.0, 20.0, 40.0, 60.0)


def test_roi_with_missing_keys_returns_none():
    f = _px_to_cs()
    assert _roi_to_polygon({"x": 10, "y": 20}, f) is None
    assert _roi_to_polygon({}, f) is None


def test_polygon_dict_to_polygon():
    f = _px_to_cs()
    pts = [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}, {"x": 0, "y": 10}]
    poly = _polygon_to_polygon({"id": "p1", "points": pts}, f)
    assert poly is not None
    assert poly.bounds == (0.0, 0.0, 10.0, 10.0)


def test_polygon_with_too_few_vertices_returns_none():
    f = _px_to_cs()
    assert _polygon_to_polygon({"points": []}, f) is None
    assert _polygon_to_polygon({"points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]}, f) is None


def test_point_to_circle_has_expected_centroid_and_radius():
    f = _px_to_cs()
    poly = _point_to_circle({"x": 50, "y": 50}, f, radius=2.0)
    assert poly is not None
    cx, cy = poly.centroid.x, poly.centroid.y
    assert abs(cx - 50.0) < 1e-9
    assert abs(cy - 50.0) < 1e-9
    # buffered Point: distance from centre to furthest exterior coord ≈ radius
    bx, by = poly.exterior.coords[0]
    assert abs(((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 - 2.0) < 0.05


def test_collect_geoms_orders_rois_polygons_points():
    f_args = dict(xmin=0.0, xmax=100.0, y_lo=0.0, y_hi=100.0, image_w=100, image_h=100, point_radius=1.0)
    viewer = SimpleNamespace(
        _rois_data=[{"x": 0, "y": 0, "width": 10, "height": 10}],
        _polygons_data=[{"points": [{"x": 20, "y": 20}, {"x": 30, "y": 20}, {"x": 30, "y": 30}]}],
        _points_data=[{"x": 80, "y": 80}],
    )
    geoms = collect_geoms_from_viewer(viewer, **f_args)
    assert len(geoms) == 3
    # roi is the axis-aligned rect at (0,0)-(10,10)
    assert geoms[0].bounds == (0.0, 0.0, 10.0, 10.0)
    # polygon centroid roughly at (26.67, 23.33)
    assert geoms[1].geom_type == "Polygon"
    # point became a circle near (80, 80)
    assert abs(geoms[2].centroid.x - 80.0) < 1e-9
    assert abs(geoms[2].centroid.y - 80.0) < 1e-9


def test_collect_geoms_silently_skips_invalid_entries():
    f_args = dict(xmin=0.0, xmax=10.0, y_lo=0.0, y_hi=10.0, image_w=10, image_h=10, point_radius=0.5)
    viewer = SimpleNamespace(
        _rois_data=[{"x": 0, "y": 0}],  # missing width/height
        _polygons_data=[{"points": [{"x": 0, "y": 0}]}],  # only 1 vertex
        _points_data=[{"x": 5}],  # missing y
    )
    assert collect_geoms_from_viewer(viewer, **f_args) == []


def test_collect_geoms_handles_none_buffers():
    f_args = dict(xmin=0.0, xmax=10.0, y_lo=0.0, y_hi=10.0, image_w=10, image_h=10, point_radius=0.5)
    viewer = SimpleNamespace(_rois_data=None, _polygons_data=None, _points_data=None)
    assert collect_geoms_from_viewer(viewer, **f_args) == []


def test_collect_geoms_returns_polygon_typed_geometries():
    """All three input categories should produce Polygons (points → buffered circles)."""
    f_args = dict(xmin=0.0, xmax=10.0, y_lo=0.0, y_hi=10.0, image_w=10, image_h=10, point_radius=0.5)
    viewer = SimpleNamespace(
        _rois_data=[{"x": 1, "y": 1, "width": 1, "height": 1}],
        _polygons_data=[{"points": [{"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 5}]}],
        _points_data=[{"x": 8, "y": 8}],
    )
    geoms = collect_geoms_from_viewer(viewer, **f_args)
    assert all(isinstance(g, Polygon) for g in geoms)
