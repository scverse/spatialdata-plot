"""Smoke test for the matplotlib → PNG render path."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import spatialdata as sd
from PIL import Image

from spatialdata_plot.pl.interactive._render import _IMAGE_H, _IMAGE_W, render_to_png


def _make_sdata_with_image() -> sd.SpatialData:
    """Build a tiny SpatialData with a single 2D image in CS 'global'."""
    from spatialdata.models import Image2DModel

    arr = np.random.default_rng(0).integers(0, 255, size=(3, 64, 64), dtype=np.uint8)
    img = Image2DModel.parse(arr, dims=("c", "y", "x"))
    return sd.SpatialData(images={"img": img})


def test_render_to_png_returns_valid_png():
    sdata = _make_sdata_with_image()
    png_bytes, w, h, xlim, ylim = render_to_png(sdata, "img", "global")
    # PNG signature
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    # Decode and check dimensions roughly match the configured render size.
    decoded = Image.open(BytesIO(png_bytes))
    assert decoded.size == (w, h) == (_IMAGE_W, _IMAGE_H)


def test_render_to_png_returns_extent_matching_image():
    sdata = _make_sdata_with_image()
    _, _, _, xlim, ylim = render_to_png(sdata, "img", "global")
    # For an Image2DModel with shape (c=3, y=64, x=64) and no transformations,
    # xlim should cover [0, 64] and ylim [64, 0] (origin='upper'). Allow ±1
    # for matplotlib edge padding.
    assert xlim[0] <= 0 and xlim[1] >= 63
    assert ylim[0] >= 63 and ylim[1] <= 0
