"""Smoke test for the matplotlib → PNG render path."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import spatialdata as sd
from PIL import Image

from spatialdata_plot.pl.interactive._render import _IMAGE_H, _IMAGE_W, render_to_png


def _make_sdata_with_image() -> sd.SpatialData:
    from spatialdata.models import Image2DModel

    arr = np.random.default_rng(0).integers(0, 255, size=(3, 64, 64), dtype=np.uint8)
    img = Image2DModel.parse(arr, dims=("c", "y", "x"))
    return sd.SpatialData(images={"img": img})


def test_render_to_png_returns_valid_png():
    sdata = _make_sdata_with_image()
    png_bytes, extent = render_to_png(sdata, "img", "global")
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    decoded = Image.open(BytesIO(png_bytes))
    assert decoded.size == (extent.image_w, extent.image_h) == (_IMAGE_W, _IMAGE_H)


def test_render_to_png_returns_extent_matching_image():
    sdata = _make_sdata_with_image()
    _, extent = render_to_png(sdata, "img", "global")
    # Image2DModel(c=3, y=64, x=64) with no transformations: xlim covers
    # roughly [0, 64], ylim is reversed under origin='upper'.
    assert extent.xlim[0] <= 0 and extent.xlim[1] >= 63
    assert extent.ylim[0] >= 63 and extent.ylim[1] <= 0
