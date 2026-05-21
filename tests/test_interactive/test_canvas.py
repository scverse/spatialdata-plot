"""Smoke tests for the DrawCanvas anywidget class."""
from __future__ import annotations

import pytest

pytest.importorskip("anywidget")
pytest.importorskip("ipywidgets")


def test_draw_canvas_imports():
    from spatialdata_plot.pl.interactive._canvas import DrawCanvas

    assert DrawCanvas is not None


def test_draw_canvas_default_traitlets():
    from spatialdata_plot.pl.interactive._canvas import DrawCanvas

    c = DrawCanvas()
    assert c.tool == "rectangle"
    assert c.shapes == []
    assert c.image_width == 720
    assert c.image_height == 720
    assert c.clear_trigger == 0
    assert c.close_poly_trigger == 0
    assert c.undo_trigger == 0
    assert c.fit_trigger == 0


def test_draw_canvas_esm_file_is_bundled():
    """The ESM module file must ship with the package."""
    from spatialdata_plot.pl.interactive import _canvas

    assert _canvas._ESM_PATH.exists(), f"{_canvas._ESM_PATH} not bundled"
    assert _canvas._ESM_PATH.suffix == ".js"
    assert _canvas._ESM_PATH.stat().st_size > 0


def test_draw_canvas_traitlet_assignment():
    """Setting traitlets from Python should work (Python → JS sync)."""
    from spatialdata_plot.pl.interactive._canvas import DrawCanvas

    c = DrawCanvas()
    c.tool = "polygon"
    assert c.tool == "polygon"
    c.clear_trigger += 1
    assert c.clear_trigger == 1
