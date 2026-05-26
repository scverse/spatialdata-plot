"""ipywidgets composite around an anybioimage BioImageViewer. Internal."""

from __future__ import annotations

from typing import Any, Literal

import geopandas as gpd
import ipywidgets as W
import numpy as np
import spatialdata as sd
from anybioimage import BioImageViewer
from IPython.display import display
from spatialdata.models import ShapesModel
from spatialdata.transformations.transformations import Identity

from ._commit import collect_geoms_from_viewer

BannerKind = Literal["info", "success", "error", "hint"]

_BANNER_CLASS = {
    "info": "sdp-banner sdp-banner-info",
    "success": "sdp-banner sdp-banner-success",
    "error": "sdp-banner sdp-banner-error",
    "hint": "sdp-banner sdp-banner-hint",
}

_CSS = """
<style>
.sdp-card {
    background: #fafafa;
    border: 1px solid #e4e4e7;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin: 6px 0;
}
.sdp-title { font-weight: 600; color: #18181b; margin-bottom: 4px; font-size: 14px; }
.sdp-context { font-size: 12px; color: #71717a; margin-bottom: 4px; }
.sdp-banner { border-radius: 6px; padding: 8px 12px; font-size: 13px; line-height: 1.3; }
.sdp-banner-info    { background: #f4f4f5; color: #3f3f46; border: 1px solid #e4e4e7; }
.sdp-banner-success { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
.sdp-banner-error   { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
.sdp-banner-hint    { background: #eff6ff; color: #1e3a8a; border: 1px solid #dbeafe; }
</style>
"""


def _fmt_banner(msg: str, kind: BannerKind = "info") -> str:
    return f"<div class='{_BANNER_CLASS[kind]}'>{msg}</div>"


def _to_viewer_rgb(rgb: np.ndarray) -> np.ndarray:
    """Reorder (..., [R, G, B]) → (..., [G, R, B]) so BioImageViewer's defaults display true RGB.

    anybioimage's CHANNEL_COLORS = [green, red, blue, ...] (GFP/RFP/DAPI convention).
    Pre-swapping R↔G in the array makes each channel land in its true colour without
    having to mutate ``_channel_settings`` post-hoc — which triggers a JS-side
    canvas re-render that interferes with the ipywidgets text field's focus.
    """
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        return rgb[..., [1, 0, 2]]
    return rgb


class _InteractiveSession:
    """Drives the BioImageViewer widget. Constructed by :meth:`PlotAccessor.annotate`."""

    def __init__(
        self,
        sdata: sd.SpatialData,
        coordinate_system: str,
        rgb: np.ndarray,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        *,
        point_radius_frac: float = 0.005,
    ) -> None:
        self._sdata = sdata
        self._cs = coordinate_system
        self._rgb = rgb
        self._xlim = xlim
        self._ylim = ylim
        img_h, img_w = rgb.shape[:2]
        xmin, xmax = float(xlim[0]), float(xlim[1])
        y_lo, y_hi = sorted((float(ylim[0]), float(ylim[1])))
        self._H, self._W = img_h, img_w
        self._xmin, self._xmax = xmin, xmax
        self._y_lo, self._y_hi = y_lo, y_hi
        self._point_radius = max(abs(xmax - xmin), abs(y_hi - y_lo)) * point_radius_frac

        self.viewer = BioImageViewer()
        self.viewer.set_image(_to_viewer_rgb(rgb))
        self.viewer.tool_mode = "polygon"

        self._style = W.HTML(value=_CSS)
        self.name_tx = W.Text(value="", placeholder="shape name…", layout=W.Layout(flex="1 1 240px"))
        self.save_btn = W.Button(description="Save", button_style="success", icon="save")
        self.clear_btn = W.Button(description="Clear canvas", icon="trash", button_style="warning")
        self.banner = W.HTML(
            value=_fmt_banner(
                f"Annotating in <b>{coordinate_system!r}</b>. "
                "Draw shapes, type a name, click Save. Polygons close with double-click "
                "or by clicking the first vertex. Tool shortcuts in the canvas toolbar.",
                "hint",
            )
        )

        self.save_btn.on_click(self._on_save)
        self.clear_btn.on_click(self._on_clear)

        controls_row = W.HBox([self.name_tx, self.save_btn, self.clear_btn], layout=W.Layout(gap="6px"))
        card = W.VBox(
            [
                W.HTML(
                    value=(
                        f"<div class='sdp-title'>Annotate</div>"
                        f"<div class='sdp-context'>coordinate system: {coordinate_system!r}</div>"
                    )
                ),
                controls_row,
                self.banner,
                self.viewer,
            ],
            layout=W.Layout(max_width="100%"),
        )
        card.add_class("sdp-card")

        self.controls = W.VBox([self._style, card])

    def show(self) -> None:
        display(self.controls)

    def _set_banner(self, msg: str, kind: BannerKind = "info") -> None:
        self.banner.value = _fmt_banner(msg, kind)

    def _on_save(self, _btn: W.Button) -> None:
        name = self.name_tx.value.strip()
        if not name:
            self._set_banner("Name is required.", "error")
            return
        geoms = collect_geoms_from_viewer(
            self.viewer,
            xmin=self._xmin,
            xmax=self._xmax,
            y_lo=self._y_lo,
            y_hi=self._y_hi,
            image_w=self._W,
            image_h=self._H,
            point_radius=self._point_radius,
        )
        if not geoms:
            self._set_banner(
                "No shapes drawn yet. "
                "If you drew a polygon, did you close it (double-click or click the first vertex)?",
                "error",
            )
            return
        gdf = gpd.GeoDataFrame({"geometry": geoms})
        self._sdata.shapes[name] = ShapesModel.parse(gdf, transformations={self._cs: Identity()})
        self._set_banner(
            f"Saved <b>{name!r}</b> with {len(geoms)} geometry(ies) into "
            "<code>sdata.shapes</code>. Canvas keeps the shapes — Clear to start fresh.",
            "success",
        )

    def _on_clear(self, _btn: W.Button) -> None:
        self.viewer.clear_rois()
        self.viewer.clear_polygons()
        self.viewer.clear_points()
        self._set_banner("Canvas cleared.", "info")
