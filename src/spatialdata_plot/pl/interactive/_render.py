"""Render an image element to a PNG suitable for the DrawCanvas background."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import matplotlib.pyplot as plt
import spatialdata as sd

_FIGSIZE = (7, 7)
_DPI = 120
_IMAGE_W = _FIGSIZE[0] * _DPI
_IMAGE_H = _FIGSIZE[1] * _DPI


@dataclass(frozen=True)
class RenderExtent:
    """Geometry of a render — PNG pixel dims + CS-coord limits at render time.

    For matplotlib image axes (``origin='upper'``) ``ylim`` is reversed:
    the smaller y maps to PNG row 0. ``pixel_shape_to_polygon`` accepts
    either orientation.
    """

    image_w: int
    image_h: int
    xlim: tuple[float, float]
    ylim: tuple[float, float]


def render_to_png(
    sdata: sd.SpatialData,
    element: str,
    coordinate_system: str,
) -> tuple[bytes, RenderExtent]:
    """Render ``element`` in ``coordinate_system`` to PNG + its extent.

    The matplotlib axes fills the figure (``[0, 0, 1, 1]`` with axis off) so
    the PNG-pixel ↔ data-coord mapping is exactly ``xlim`` × ``ylim``.
    """
    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI)
    try:
        ax = fig.add_axes([0, 0, 1, 1])
        sdata.pl.render_images(element=element).pl.show(coordinate_systems=coordinate_system, ax=ax)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_axis_off()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=_DPI, pad_inches=0)
    finally:
        plt.close(fig)
    extent = RenderExtent(_IMAGE_W, _IMAGE_H, tuple(xlim), tuple(ylim))
    return buf.getvalue(), extent
