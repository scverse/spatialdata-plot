"""Tests for the chain-aware sdata.pl.annotate() entry point and session save loop."""

from __future__ import annotations

import numpy as np
import pytest
import spatialdata as sd

pytest.importorskip("anywidget")
pytest.importorskip("anybioimage")
pytest.importorskip("ipywidgets")

from spatialdata.models import Image2DModel
from spatialdata.transformations.transformations import Identity

import spatialdata_plot  # noqa: F401  registers .pl


@pytest.fixture
def no_display(monkeypatch):
    """Silence the widget display so tests don't need a notebook frontend."""
    monkeypatch.setattr(
        "spatialdata_plot.pl.interactive._session._InteractiveSession.show",
        lambda self: None,
    )


def _make_sdata(cs_names=("global",)) -> sd.SpatialData:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(3, 32, 32), dtype=np.uint8)
    img = Image2DModel.parse(
        arr,
        dims=("c", "y", "x"),
        transformations={cs: Identity() for cs in cs_names},
    )
    return sd.SpatialData(images={"img": img})


def test_annotate_requires_render_chain_or_explicit_cs(no_display):
    """Annotate with multiple CS and no coordinate_systems= must raise."""
    sdata = _make_sdata(cs_names=("global", "second"))
    with pytest.raises(ValueError, match="exactly one coordinate system"):
        sdata.pl.render_images(element="img").pl.annotate()


def test_annotate_unknown_cs_raises(no_display):
    sdata = _make_sdata()
    with pytest.raises(ValueError, match="Unknown coordinate system"):
        sdata.pl.render_images(element="img").pl.annotate(coordinate_systems="does_not_exist")


def test_annotate_list_of_one_cs_is_accepted(no_display):
    sdata = _make_sdata()
    session = sdata.pl.render_images(element="img").pl.annotate(coordinate_systems=["global"])
    assert session is not None


def test_annotate_list_of_multiple_cs_raises(no_display):
    sdata = _make_sdata(cs_names=("global", "second"))
    with pytest.raises(ValueError, match="single coordinate system"):
        sdata.pl.render_images(element="img").pl.annotate(coordinate_systems=["global", "second"])


def test_annotate_single_cs_inferred_when_only_one_exists(no_display):
    sdata = _make_sdata(cs_names=("global",))
    session = sdata.pl.render_images(element="img").pl.annotate()
    assert session is not None
    # the inferred CS is exposed for the save handler to use
    assert session._cs == "global"


def test_save_writes_shapes_into_sdata(no_display):
    """Drive the save callback directly: ROI on the canvas → sdata.shapes['roi1']."""
    sdata = _make_sdata()
    session = sdata.pl.render_images(element="img").pl.annotate()

    # Inject a single ROI directly into the viewer's sync traitlet — same path
    # the JS side uses when the user drags out a rectangle.
    session.viewer._rois_data = [{"id": "r1", "x": 0, "y": 0, "width": 16, "height": 16}]
    session.name_tx.value = "roi1"
    session._on_save(None)

    assert "roi1" in sdata.shapes
    gdf = sdata.shapes["roi1"]
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].geom_type == "Polygon"


def test_save_without_name_does_not_write(no_display):
    sdata = _make_sdata()
    session = sdata.pl.render_images(element="img").pl.annotate()
    session.viewer._rois_data = [{"id": "r1", "x": 0, "y": 0, "width": 10, "height": 10}]
    session.name_tx.value = "   "  # whitespace-only
    session._on_save(None)
    assert "   " not in sdata.shapes
    assert "" not in sdata.shapes


def test_save_without_shapes_does_not_write(no_display):
    sdata = _make_sdata()
    session = sdata.pl.render_images(element="img").pl.annotate()
    session.name_tx.value = "empty"
    session._on_save(None)
    assert "empty" not in sdata.shapes


def test_clear_wipes_all_three_shape_stores(no_display):
    sdata = _make_sdata()
    session = sdata.pl.render_images(element="img").pl.annotate()
    session.viewer._rois_data = [{"id": "r", "x": 0, "y": 0, "width": 1, "height": 1}]
    session.viewer._polygons_data = [{"id": "p", "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]}]
    session.viewer._points_data = [{"id": "pt", "x": 5, "y": 5}]
    session._on_clear(None)
    assert session.viewer._rois_data == []
    assert session.viewer._polygons_data == []
    assert session.viewer._points_data == []


def test_saved_shapes_carry_identity_in_chosen_cs(no_display):
    from spatialdata.transformations.operations import get_transformation

    sdata = _make_sdata(cs_names=("ortho",))
    session = sdata.pl.render_images(element="img").pl.annotate(coordinate_systems="ortho")
    session.viewer._points_data = [{"id": "p", "x": 16, "y": 16}]
    session.name_tx.value = "dot"
    session._on_save(None)

    transforms = get_transformation(sdata.shapes["dot"], get_all=True)
    assert "ortho" in transforms
    assert isinstance(transforms["ortho"], Identity)
