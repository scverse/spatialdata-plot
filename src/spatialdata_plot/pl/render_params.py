from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, ListedColormap, Normalize, rgb2hex, to_hex, to_rgba
from matplotlib.figure import Figure

_FontWeight = Literal["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
_FontSize = Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]
_DsReduction = Literal["sum", "mean", "any", "count", "std", "var", "max", "min"]
_ImageDsReduction = Literal["max", "min", "mean", "mode", "first", "last", "var", "std"]

# Canonical definition for the package; imported by basic.py and utils.py.
# replace with
# from spatialdata._types import ColorLike
# once https://github.com/scverse/spatialdata/pull/689/ is in a release
ColorLike = tuple[float, ...] | list[float] | str


# NOTE: defined here instead of utils to avoid circular import
@dataclass(kw_only=True)
class Color:
    """Validate, parse and store a single color.

    Accepts a color and an alpha value.
    If no color or "default" is given, the default color "lightgray" is used.
    If no alpha is given, the default of completely opaque is used ("ff").
    At all times, if color indicates an alpha value, for instance as part of a hex string, the alpha parameter takes
    precedence if given.
    """

    color: str
    alpha: str
    default_color_set: bool = False
    user_defined_alpha: bool = False

    def __init__(
        self,
        color: None | str | list[float] | tuple[float, ...] = "default",
        alpha: float | int | None = None,
    ) -> None:
        # 1) Validate alpha value
        if alpha is None:
            self.alpha = "ff"  # default: completely opaque
        elif isinstance(alpha, float | int):
            if alpha <= 1.0 and alpha >= 0.0:
                # Convert float alpha to hex representation
                self.alpha = hex(int(np.round(alpha * 255)))[2:].lower()
                if len(self.alpha) == 1:
                    self.alpha = "0" + self.alpha
                self.user_defined_alpha = True
            else:
                raise ValueError(f"Invalid alpha value `{alpha}`, must lie within [0.0, 1.0].")
        else:
            raise ValueError(f"Invalid alpha value `{alpha}`, must be None or a float | int within [0.0, 1.0].")

        # 2) Validate color value
        if color is None:
            self.color = to_hex("lightgray", keep_alpha=False)
            # setting color to None should lead to full transparency (except alpha is set manually)
            if alpha is None:
                self.alpha = "00"
        elif color == "default":
            self.default_color_set = True
            self.color = to_hex("lightgray", keep_alpha=False)
        elif isinstance(color, str):
            # already hex
            if color.startswith("#"):
                if len(color) not in [7, 9]:
                    raise ValueError("Invalid hex color length: only formats '#RRGGBB' and '#RRGGBBAA' are supported.")
                self.color = color.lower()
                if not all(c in "0123456789abcdef" for c in self.color[1:]):
                    raise ValueError("Invalid hex color: contains non-hex characters")
                if len(self.color) == 9:
                    if alpha is None:
                        self.alpha = self.color[7:]
                        self.user_defined_alpha = True
                    self.color = self.color[:7]
            else:
                try:
                    float(color)
                except ValueError:
                    # we're not dealing with what matplotlib considers greyscale
                    pass
                else:
                    raise TypeError(
                        f"Invalid type `{type(color)}` for a color, expecting str | None | tuple[float, ...] | "
                        "list[float]. Note that unlike in matplotlib, giving a string of a number within [0, 1] as a "
                        "greyscale value is not supported here!"
                    )
                # matplotlib raises ValueError in case of invalid color name
                self.color = to_hex(color, keep_alpha=False)
        elif isinstance(color, list | tuple):
            if len(color) < 3:
                raise ValueError(f"Color `{color}` can't be interpreted as RGB(A) array, needs 3 or 4 values!")
            if len(color) > 4:
                raise ValueError(f"Color `{color}` can't be interpreted as RGB(A) array, needs 3 or 4 values!")
            # get first 3-4 values
            r, g, b = color[0], color[1], color[2]
            a = 1.0
            if len(color) == 4:
                a = color[3]
                self.user_defined_alpha = True
            if (
                not isinstance(r, int | float)
                or not isinstance(g, int | float)
                or not isinstance(b, int | float)
                or not isinstance(a, int | float)
            ):
                raise ValueError(f"Invalid color `{color}`, all values in RGB(A) array must be int or float.")
            if any(np.array([r, g, b, a]) > 1) or any(np.array([r, g, b, a]) < 0):
                raise ValueError(f"Invalid color `{color}`, all values in RGB(A) array must be within [0.0, 1.0].")
            self.color = rgb2hex((r, g, b, a), keep_alpha=False)
            if alpha is None:
                self.alpha = rgb2hex((r, g, b, a), keep_alpha=True)[7:]
        else:
            raise TypeError(
                f"Invalid type `{type(color)}` for color, expecting str | None | tuple[float, ...] | list[float]."
            )

    def get_hex_with_alpha(self) -> str:
        """Get color value as '#RRGGBBAA'."""
        return self.color + self.alpha

    def get_hex(self) -> str:
        """Get color value as '#RRGGBB'."""
        return self.color

    def get_alpha_as_float(self) -> float:
        """Return alpha as value within [0.0, 1.0]."""
        return int(self.alpha, 16) / 255

    def color_modified_by_user(self) -> bool:
        """Get whether a color was passed when the object was created."""
        return not self.default_color_set

    def alpha_is_user_defined(self) -> bool:
        """Get whether an alpha was set during object creation."""
        return self.user_defined_alpha

    def is_fully_transparent(self) -> bool:
        """Check whether this color is fully transparent (alpha == 0)."""
        return self.alpha == "00"


def colormap_with_alpha(cmap: Colormap, alpha: float, na_color: str) -> Colormap:
    """Return ``cmap`` rebuilt with a uniform ``alpha`` and ``na_color`` as the bad/NaN color.

    Resampling at ``linspace(0, 1, N)`` is lossless (matplotlib quantizes ``__call__`` into ``N`` bins).
    """
    lut = cmap(np.linspace(0, 1, cmap.N))
    lut[:, -1] = alpha
    new = ListedColormap(lut, name=cmap.name)
    # Apply alpha to under/over too, matching the old ``_lut[:, -1] = alpha`` (which hit every row).
    new.set_extremes(
        bad=[*to_rgba(na_color)[:3], alpha],
        under=[*cmap.get_under()[:3], alpha],
        over=[*cmap.get_over()[:3], alpha],
    )
    return new


@dataclass
class CmapParams:
    """Cmap params."""

    cmap: Colormap
    norm: Normalize
    na_color: Color
    cmap_is_default: bool = True

    def fresh_norm(self) -> Normalize:
        """Return a copy of ``norm`` safe to apply/autoscale without mutating the shared one.

        ``Normalize.__call__`` autoscales ``vmin``/``vmax`` in place when unset, which would leak one
        element's data range into later elements that reuse the same ``CmapParams``.
        """
        return copy(self.norm)


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
    """Outline params."""

    outer_outline_color: Color | None = None
    outer_outline_linewidth: float = 1.5
    inner_outline_color: Color | None = None
    inner_outline_linewidth: float = 0.5


@dataclass
class LegendParams:
    """Legend params."""

    legend_fontsize: int | float | _FontSize | None = None
    legend_fontweight: int | _FontWeight = "bold"
    legend_loc: str | None = "right margin"
    legend_fontoutline: int | None = None
    na_in_legend: bool = True
    colorbar: bool = True
    # Optional explicit titles for the fill / outline categorical legends. When unset,
    # both legends are untitled unless both fill and outline are colored by an obs
    # column, in which case they default to "fill" / "outline" to disambiguate.
    legend_title: str | None = None
    outline_legend_title: str | None = None


@dataclass
class ColorbarSpec:
    """Data required to create a colorbar."""

    ax: Axes
    mappable: ScalarMappable
    params: dict[str, object] | None = None
    label: str | None = None
    alpha: float | None = None


@dataclass
class ChannelLegendEntry:
    """A single channel-to-color mapping for the categorical channel legend."""

    channel_name: str
    color_hex: str


CBAR_DEFAULT_LOCATION = "right"
CBAR_DEFAULT_FRACTION = 0.075
CBAR_DEFAULT_PAD = 0.015


@dataclass
class ScalebarParams:
    """Scalebar params."""

    scalebar_dx: Sequence[float] | None = None
    scalebar_units: Sequence[str] | None = None
    scalebar_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class RenderParams:
    """Fields shared by every ``*RenderParams``.

    These four are the only fields with identical type and default across all five renderers.
    ``kw_only=True`` keeps field order irrelevant across inheritance (subclasses add required fields
    such as ``cmap_params``), and all call sites construct these dataclasses by keyword. Subclasses
    carry their renderer-specific fields (including ``cmap_params``, whose type varies per renderer).
    """

    element: str
    zorder: int = 0
    colorbar: bool | str | None = "auto"
    colorbar_params: dict[str, object] | None = None


@dataclass(kw_only=True)
class ShapesRenderParams(RenderParams):
    """Shapes render parameters."""

    cmap_params: CmapParams
    outline_params: OutlineParams
    color: Color | None = None
    col_for_color: str | None = None
    col_for_outline_color: str | None = None
    outline_table_name: str | None = None
    groups: str | list[str] | None = None
    palette: ListedColormap | dict[str, str] | list[str] | None = None
    outline_alpha: tuple[float, float] = (1.0, 1.0)
    fill_alpha: float = 0.3
    scale: float = 1.0
    transfunc: Callable[[float], float] | None = None
    method: str | None = None
    table_name: str | None = None
    table_layer: str | None = None
    shape: Literal["circle", "hex", "visium_hex", "square"] | None = None
    # Fast mode: render each shape as a single dot at its centroid instead of its geometry.
    as_points: bool = False
    size: float = 1.0  # marker size for as_points (matplotlib scatter ``s``)
    ds_reduction: _DsReduction | None = None
    # Multi-panel color: when set, this render entry belongs to the panel identified by this
    # color key. ``None`` means the entry is shared across every panel (e.g. a background layer).
    panel_key: str | None = None


@dataclass(kw_only=True)
class PointsRenderParams(RenderParams):
    """Points render parameters."""

    cmap_params: CmapParams
    color: Color | None = None
    col_for_color: str | None = None
    groups: str | list[str] | None = None
    palette: ListedColormap | dict[str, str] | list[str] | None = None
    alpha: float = 1.0
    size: float = 1.0
    transfunc: Callable[[float], float] | None = None
    method: str | None = None
    table_name: str | None = None
    table_layer: str | None = None
    ds_reduction: _DsReduction | None = None
    density: bool = False
    density_how: Literal["linear", "log", "cbrt", "eq_hist"] = "linear"


@dataclass(kw_only=True)
class ImageRenderParams(RenderParams):
    """Image render parameters."""

    cmap_params: list[CmapParams] | CmapParams
    channel: list[str] | list[int] | int | str | None = None
    palette: ListedColormap | list[str] | None = None
    alpha: float = 1.0
    scale: str | None = None
    transfunc: Callable[[np.ndarray], np.ndarray] | list[Callable[[np.ndarray], np.ndarray]] | None = None
    grayscale: bool = False
    channels_as_legend: bool = False
    method: Literal["matplotlib", "datashader"] | None = None
    ds_reduction: _ImageDsReduction | None = None


@dataclass(kw_only=True)
class LabelsRenderParams(RenderParams):
    """Labels render parameters."""

    cmap_params: CmapParams
    color: Color | None = None
    col_for_color: str | None = None
    col_for_outline_color: str | None = None
    outline_table_name: str | None = None
    groups: str | list[str] | None = None
    contour_px: int | None = None
    palette: ListedColormap | dict[str, str] | list[str] | None = None
    outline_alpha: float = 1.0
    outline_color: Color | None = None
    fill_alpha: float = 0.4
    scale: str | None = None
    table_name: str | None = None
    table_layer: str | None = None
    transfunc: Callable[[float], float] | None = None
    # Fast mode: render each label as a single dot at its centroid instead of the mask.
    as_points: bool = False
    size: float = 1.0  # marker size for as_points (matplotlib scatter ``s``)
    # Backend for the as_points centroids: None auto-selects (datashader above ~50k dots).
    method: str | None = None
    # Multi-panel color: when set, this render entry belongs to the panel identified by this
    # color key. ``None`` means the entry is shared across every panel (e.g. a background layer).
    panel_key: str | None = None


@dataclass(kw_only=True)
class GraphRenderParams(RenderParams):
    """Graph render parameters."""

    connectivity_obsp_key: str = "spatial_connectivities"
    table_name: str | None = None
    color: Color | None = None
    obs_col: str | None = None
    obsp_key: str | None = None
    cmap_params: CmapParams | None = None
    palette_map: dict[str, str] | None = None
    na_color: Color | None = None
    color_source: Literal["scalar", "obsp", "obs_categorical", "obs_continuous"] = "scalar"
    groups: list[str] | str | None = None
    group_key: str | None = None
    edge_width: float | Literal["weight"] = 1.0
    edge_alpha: float | Literal["weight"] = 1.0
    weight_key: str | None = None
    linestyle: str | Sequence[str] = "solid"
    rasterize: bool = True
    include_self_loops: bool = False
