"""Tests for the user-facing sdata.pl.annotate() validation paths."""

from __future__ import annotations

import numpy as np
import pytest
import spatialdata as sd

pytest.importorskip("anywidget")
pytest.importorskip("ipywidgets")

from spatialdata.models import Image2DModel
from spatialdata.transformations.transformations import Identity

import spatialdata_plot  # noqa: F401  registers .pl


@pytest.fixture
def no_display(monkeypatch):
    monkeypatch.setattr(
        "spatialdata_plot.pl.interactive._session._InteractiveSession.show",
        lambda self: None,
    )


def _make_sdata_with_image() -> sd.SpatialData:
    arr = np.random.default_rng(0).integers(0, 255, size=(3, 32, 32), dtype=np.uint8)
    img = Image2DModel.parse(arr, dims=("c", "y", "x"))
    return sd.SpatialData(images={"img": img})


def test_annotate_unknown_coordinate_system_raises(no_display):
    sdata = _make_sdata_with_image()
    with pytest.raises(ValueError, match="Unknown coordinate system"):
        sdata.pl.annotate("does_not_exist", "img")


def test_annotate_unknown_element_raises(no_display):
    sdata = _make_sdata_with_image()
    with pytest.raises(ValueError, match="Unknown image element"):
        sdata.pl.annotate("global", "no_such_image")


def test_annotate_element_not_in_cs_raises(no_display):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(3, 32, 32), dtype=np.uint8)
    img = Image2DModel.parse(
        arr,
        dims=("c", "y", "x"),
        transformations={"other_cs": Identity()},
    )
    anchor = Image2DModel.parse(
        rng.integers(0, 255, size=(3, 32, 32), dtype=np.uint8),
        dims=("c", "y", "x"),
        transformations={"global": Identity()},
    )
    sdata = sd.SpatialData(images={"img": img, "anchor": anchor})
    with pytest.raises(ValueError, match="not registered in coordinate system"):
        sdata.pl.annotate("global", "img")
