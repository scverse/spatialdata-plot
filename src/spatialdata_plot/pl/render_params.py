from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap, Normalize, rgb2hex, to_hex
from matplotlib.figure import Figure

_FontWeight = Literal["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
_FontSize = Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]

# replace with
# from spatialdata._types import ColorLike
# once https://github.com/scverse/spatialdata/pull/689/ is in a release
ColorLike = tuple[float, ...] | str


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
        self, color: None | str | list[float] | tuple[float, ...] = "default", alpha: float | int | None = None
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


@dataclass
class CmapParams:
    """Cmap params."""

    cmap: Colormap
    norm: Normalize
    na_color: Color
    # na_color_modified_by_user: bool = False # NOTE: na_color stores that info already
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

    # outer_outline: bool
    outer_outline_color: Color | None = None
    outer_outline_linewidth: float = 1.5
    # inner_outline: bool = False
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


@dataclass
class ScalebarParams:
    """Scalebar params."""

    scalebar_dx: Sequence[float] | None = None
    scalebar_units: Sequence[str] | None = None


@dataclass
class ShapesRenderParams:
    """Shapes render parameters.."""

    cmap_params: CmapParams
    outline_params: OutlineParams
    element: str
    color: Color | None = None
    col_for_color: str | None = None
    groups: str | list[str] | None = None
    contour_px: int | None = None
    palette: ListedColormap | list[str] | None = None
    outline_alpha: tuple[float, float] = (1.0, 1.0)
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
    color: Color | None = None
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
    """Image render parameters.."""

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
