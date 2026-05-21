"""ipywidgets-based session orchestrating the DrawCanvas. Internal."""
from __future__ import annotations

import base64
from typing import Any, Literal

import ipywidgets as W
import spatialdata as sd
from IPython.display import display
from shapely.geometry import Polygon
from spatialdata.transformations.operations import get_transformation

from ._canvas import DrawCanvas
from ._commit import build_shapes_model, pixel_shape_to_polygon
from ._persist import commit_to_memory
from ._render import RenderExtent, render_to_png

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
.sdp-section-title {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    color: #71717a; margin: 8px 0 4px 0;
}
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


def _validate(sdata: sd.SpatialData, coordinate_system: str, element: str) -> None:
    if coordinate_system not in sdata.coordinate_systems:
        raise ValueError(
            f"Unknown coordinate system {coordinate_system!r}. "
            f"Available: {list(sdata.coordinate_systems)}"
        )
    if element not in sdata.images:
        raise ValueError(f"Unknown image element {element!r}. Available: {list(sdata.images)}")
    transforms = get_transformation(sdata[element], get_all=True)
    if coordinate_system not in transforms:
        raise ValueError(
            f"Image {element!r} is not registered in coordinate system "
            f"{coordinate_system!r}. Registered in: {list(transforms)}"
        )


class _InteractiveSession:
    """Drives the DrawCanvas widget. Constructed by :meth:`PlotAccessor.annotate`."""

    def __init__(
        self,
        sdata: sd.SpatialData,
        coordinate_system: str,
        element: str,
        *,
        persist: bool = True,
    ) -> None:
        _validate(sdata, coordinate_system, element)

        self._sdata = sdata
        self._cs = coordinate_system
        self._element = element
        self._persist_enabled = persist
        self.canvas: DrawCanvas | None = None
        self._extent: RenderExtent | None = None
        self._commits: list[str] = []

        self._style = W.HTML(value=_CSS)

        self.tool_tb = W.ToggleButtons(
            options=[("Rect", "rectangle"), ("Polygon", "polygon"), ("Lasso", "lasso")],
            value="rectangle",
            description="Tool:",
        )
        self.tool_tb.observe(self._on_tool_change, names="value")
        self.close_poly_btn = self._trigger_btn(
            "Close polygon", "check", "close_poly_trigger", tooltip="Enter", disabled=True,
        )
        self.undo_btn = self._trigger_btn(
            "Undo", "rotate-left", "undo_trigger", tooltip="Ctrl+Z", disabled=True,
        )
        self.clear_btn = self._trigger_btn(
            "Clear", "trash", "clear_trigger", tooltip="Delete",
            after=lambda: self._set_banner("Canvas cleared.", "info"),
        )
        self.fit_btn = self._trigger_btn("Fit view", "compress", "fit_trigger", tooltip="F")
        self.shape_count_lbl = W.Label(value="0 shape(s) on canvas")

        self.name_tx = W.Text(value="", placeholder="name…", description="Name:")
        self.save_btn = W.Button(description="Save", button_style="success", icon="save")
        self.save_btn.on_click(self._on_save)

        save_row_widgets: list[W.Widget] = [self.name_tx, self.save_btn]
        self.persist_btn: W.Button | None = None
        if persist:
            self.persist_btn = W.Button(
                description="Write to disk", button_style="warning", icon="hdd-o",
            )
            self.persist_btn.on_click(self._on_persist)
            self.persist_btn.disabled = True
            save_row_widgets.append(self.persist_btn)

        self.banner = W.HTML(value=_fmt_banner(
            f"Annotating <b>{element!r}</b> in coordinate system <b>{coordinate_system!r}</b>. "
            "Pick a tool and draw. Click canvas first so keyboard shortcuts work. "
            "<b>R/P/L</b> tools · <b>Wheel</b> zoom · <b>Shift+drag</b> pan · "
            "<b>Alt+click</b> shape to delete · <b>Ctrl+Z</b> undo · <b>F</b> fit",
            "hint",
        ))
        self.plot_box = W.VBox([])

        def section(label: str) -> W.HTML:
            return W.HTML(value=f"<div class='sdp-section-title'>{label}</div>")

        controls_card = W.VBox([
            W.HTML(value=(
                f"<div class='sdp-title'>Annotate</div>"
                f"<div class='sdp-context'>{element!r} · {coordinate_system!r}</div>"
            )),
            section("Draw"),
            W.HBox([self.tool_tb, self.close_poly_btn, self.undo_btn, self.clear_btn, self.fit_btn]),
            W.HBox([self.shape_count_lbl]),
            section("Save"),
            W.HBox(save_row_widgets),
            self.banner,
        ])
        controls_card.add_class("sdp-card")

        canvas_card = W.VBox([self.plot_box])
        canvas_card.add_class("sdp-card")

        self.controls = W.VBox([self._style, controls_card, canvas_card])

    def show(self) -> None:
        self._render()
        display(self.controls)

    def _set_banner(self, msg: str, kind: BannerKind = "info") -> None:
        self.banner.value = _fmt_banner(msg, kind)

    def _trigger_btn(
        self,
        description: str,
        icon: str,
        trait_name: str,
        *,
        tooltip: str = "",
        disabled: bool = False,
        after: Any = None,
    ) -> W.Button:
        btn = W.Button(description=description, icon=icon, tooltip=tooltip)
        btn.disabled = disabled

        def _on_click(_b: W.Button) -> None:
            if self.canvas is None:
                return
            setattr(self.canvas, trait_name, getattr(self.canvas, trait_name) + 1)
            if after is not None:
                after()

        btn.on_click(_on_click)
        return btn

    def _render(self) -> None:
        png_bytes, extent = render_to_png(self._sdata, self._element, self._cs)
        data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        self._extent = extent

        self.canvas = DrawCanvas(
            image_url=data_url,
            image_width=extent.image_w,
            image_height=extent.image_h,
            tool=self.tool_tb.value,
        )
        self.canvas.observe(self._on_shapes_change, names="shapes")
        self.plot_box.children = (self.canvas,)
        self.shape_count_lbl.value = "0 shape(s) on canvas"

    def _on_shapes_change(self, change: dict[str, Any]) -> None:
        shapes = change["new"] or []
        self.shape_count_lbl.value = f"{len(shapes)} shape(s) on canvas"
        self.undo_btn.disabled = len(shapes) == 0

    def _on_tool_change(self, change: dict[str, Any]) -> None:
        if self.canvas is None:
            return
        self.canvas.tool = change["new"]
        self.close_poly_btn.disabled = change["new"] != "polygon"
        self._set_banner(f"Tool: <b>{change['new']}</b>", "info")

    def _collect_polygons(self) -> list[Polygon]:
        assert self.canvas is not None and self._extent is not None
        polys: list[Polygon] = []
        for sh in self.canvas.shapes:
            p = pixel_shape_to_polygon(sh, self._extent)
            if p is not None:
                polys.append(p)
        return polys

    def _commit_polygons(self, polys: list[Polygon], name: str) -> tuple[str, bool]:
        shapes_model = build_shapes_model(polys, self._cs)
        target = commit_to_memory(self._sdata, shapes_model, name)
        self._commits.append(target)
        return target, target != name

    def _reset_canvas_state(self) -> None:
        assert self.canvas is not None
        self.canvas.clear_trigger += 1
        self.shape_count_lbl.value = "0 shape(s) on canvas"

    def _on_save(self, _btn: W.Button) -> None:
        name = self.name_tx.value.strip()
        if not name:
            self._set_banner("Name is required.", "error")
            return
        if self.canvas is None or not self.canvas.shapes:
            self._set_banner("No shapes drawn yet.", "error")
            return

        polys = self._collect_polygons()
        if not polys:
            self._set_banner(
                f"{len(self.canvas.shapes)} shape(s) on canvas but none parsed as valid polygons.",
                "error",
            )
            return

        target, renamed = self._commit_polygons(polys, name)
        self._reset_canvas_state()

        msg = f"Saved <b>{target!r}</b> with {len(polys)} polygon(s)."
        if renamed:
            msg += " (name collided; renamed)"
        self._set_banner(msg, "success")
        if self.persist_btn is not None:
            self.persist_btn.disabled = self._sdata.path is None

    def _on_persist(self, _btn: W.Button) -> None:
        if not self._persist_enabled:
            return
        if not self._commits:
            self._set_banner("Nothing saved this session yet.", "error")
            return
        target = self._commits[-1]
        try:
            self._sdata.write_element(target, overwrite=True)
        except (ValueError, OSError) as exc:
            self._set_banner(str(exc), "error")
            return
        self._set_banner(f"Persisted <b>{target!r}</b> → {self._sdata.path}", "success")
