"""Render an image element to a PNG suitable for the DrawCanvas background."""
from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import spatialdata as sd

# Render size: 7 in × 120 dpi = 840 px square. Fixed for v0; pyramid-aware
# downsampling is a v1 feature.
_FIGSIZE = (7, 7)
_DPI = 120
_IMAGE_W = _FIGSIZE[0] * _DPI  # 840
_IMAGE_H = _FIGSIZE[1] * _DPI  # 840


def render_to_png(
    sdata: sd.SpatialData,
    element: str,
    coordinate_system: str,
) -> tuple[bytes, int, int, tuple[float, float], tuple[float, float]]:
    """Render ``element`` in ``coordinate_system`` to PNG bytes.

    The matplotlib axes fills the figure (``[0, 0, 1, 1]`` with axis off) so
    the PNG-pixel ↔ data-coord mapping is exactly ``xlim`` × ``ylim``.

    Returns
    -------
    png_bytes
        PNG-encoded image.
    image_w, image_h
        Pixel dimensions of the PNG.
    xlim, ylim
        ``ax.get_xlim()`` / ``ax.get_ylim()`` at render time. For image axes
        (``origin='upper'``) ``ylim`` is reversed — see
        :func:`._commit.pixel_shape_to_polygon` for the conversion.
    """
    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    sdata.pl.render_images(element=element).pl.show(coordinate_systems=coordinate_system, ax=ax)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, pad_inches=0)
    plt.close(fig)
    return buf.getvalue(), _IMAGE_W, _IMAGE_H, tuple(xlim), tuple(ylim)
