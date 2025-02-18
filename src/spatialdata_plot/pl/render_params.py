from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure

_FontWeight = Literal["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
_FontSize = Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]

# replace with
# from spatialdata._types import ColorLike
# once https://github.com/scverse/spatialdata/pull/689/ is in a release
ColorLike = tuple[float, ...] | str


@dataclass
class CmapParams:
    """Cmap params."""

    cmap: Colormap
    norm: Normalize
    na_color: ColorLike
    na_color_modified_by_user: bool = False
    cmap_is_default: bool = True


@dataclass
class FigParams:
    """Figure params."""

    fig: Figure
    ax: Axes
    num_panels: int
    axs: Sequence[Axes] | None = None
    title: str | Sequence[str] | None = None
    ax_labels: Sequence[str] | None = None
    frameon: bool | None = None


@dataclass
class OutlineParams:
    """Cmap params."""

    outline: bool
    outline_color: str | list[float]
    linewidth: float


@dataclass
class LegendParams:
    """Legend params."""

    legend_fontsize: int | float | _FontSize | None = None
    legend_fontweight: int | _FontWeight = "bold"
    legend_loc: str | None = "right margin"
    legend_fontoutline: int | None = None
    na_in_legend: bool = True
    colorbar: bool = True


@dataclass
class ScalebarParams:
    """Scalebar params."""

    scalebar_dx: Sequence[float] | None = None
    scalebar_units: Sequence[str] | None = None


@dataclass
class ShapesRenderParams:
    """Labels render parameters.."""

    cmap_params: CmapParams
    outline_params: OutlineParams
    element: str
    color: str | None = None
    col_for_color: str | None = None
    groups: str | list[str] | None = None
    contour_px: int | None = None
    palette: ListedColormap | list[str] | None = None
    outline_alpha: float = 1.0
    fill_alpha: float = 0.3
    scale: float = 1.0
    transfunc: Callable[[float], float] | None = None
    method: str | None = None
    zorder: int = 0
    table_name: str | None = None
    table_layer: str | None = None
    ds_reduction: Literal["sum", "mean", "any", "count", "std", "var", "max", "min"] | None = None


@dataclass
class PointsRenderParams:
    """Points render parameters.."""

    cmap_params: CmapParams
    element: str
    color: str | None = None
    col_for_color: str | None = None
    groups: str | list[str] | None = None
    palette: ListedColormap | list[str] | None = None
    alpha: float = 1.0
    size: float = 1.0
    transfunc: Callable[[float], float] | None = None
    method: str | None = None
    zorder: int = 0
    table_name: str | None = None
    table_layer: str | None = None
    ds_reduction: Literal["sum", "mean", "any", "count", "std", "var", "max", "min"] | None = None


@dataclass
class ImageRenderParams:
    """Labels render parameters.."""

    cmap_params: list[CmapParams] | CmapParams
    element: str
    channel: list[str] | list[int] | int | str | None = None
    palette: ListedColormap | list[str] | None = None
    alpha: float = 1.0
    percentiles_for_norm: tuple[float | None, float | None] = (None, None)
    scale: str | None = None
    zorder: int = 0


@dataclass
class LabelsRenderParams:
    """Labels render parameters.."""

    cmap_params: CmapParams
    element: str
    color: str | None = None
    groups: str | list[str] | None = None
    contour_px: int | None = None
    outline: bool = False
    palette: ListedColormap | list[str] | None = None
    outline_alpha: float = 1.0
    fill_alpha: float = 0.4
    transfunc: Callable[[float], float] | None = None
    scale: str | None = None
    table_name: str | None = None
    table_layer: str | None = None
    zorder: int = 0
