from __future__ import annotations

from pathlib import Path

import anywidget
import traitlets

_ESM_PATH = Path(__file__).parent / "static" / "draw_canvas.js"

TOOLS = ("rectangle", "polygon", "lasso")


class DrawCanvas(anywidget.AnyWidget):
    """Client-side SVG drawing surface for interactive region selection.

    The image (PNG data URL) is shown as a CSS-transformed background; an
    overlay SVG catches mouse events and emits committed shapes in image-
    pixel coordinates via the ``shapes`` traitlet.

    Convert the pixel-coord shapes to data/CS coordinates with
    :func:`spatialdata_plot.pl.interactive._commit.pixel_shape_to_polygon`.

    Traitlets
    ---------
    image_url
        ``data:image/png;base64,...`` for the rendered image.
    image_width, image_height
        Pixel dimensions of the PNG (used to set the SVG ``viewBox``).
    tool
        One of ``TOOLS``.
    shapes
        List of ``{"type": "rect"|"polygon", "verts": [[x, y], ...]}`` in
        image-pixel coordinates. JS pushes to this on commit.
    clear_trigger, close_poly_trigger, undo_trigger, fit_trigger
        Integer counters. Increment from Python to invoke the corresponding
        JS action; JS observers are stateless w.r.t. the value, only the
        change event matters.
    """

    _esm = _ESM_PATH

    image_url = traitlets.Unicode("").tag(sync=True)
    image_width = traitlets.Int(720).tag(sync=True)
    image_height = traitlets.Int(720).tag(sync=True)
    tool = traitlets.Enum(TOOLS, default_value="rectangle").tag(sync=True)
    shapes = traitlets.List([]).tag(sync=True)
    clear_trigger = traitlets.Int(0).tag(sync=True)
    close_poly_trigger = traitlets.Int(0).tag(sync=True)
    undo_trigger = traitlets.Int(0).tag(sync=True)
    fit_trigger = traitlets.Int(0).tag(sync=True)
