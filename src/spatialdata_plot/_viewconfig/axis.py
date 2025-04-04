from typing import Any

import matplotlib.colors as mcolors
from matplotlib.axes import Axes

from spatialdata_plot.pl.utils import to_hex_alpha


def create_axis_block(ax: Axes, axis_scales_block: list[dict[str, Any]], dpi: float) -> list[dict[str, Any]]:
    axis_array = []
    for scale in axis_scales_block:
        axis_config = {"scale": scale["name"]}
        if scale["name"] == "X_scale":
            axis = ax.xaxis
        elif scale["name"] == "Y_scale":
            axis = ax.yaxis

        axis_props = axis.properties()

        axis_config["orient"] = axis.get_label_position()

        axis_line_props = ax.spines[axis_config["orient"]].properties()
        axis_config["domain"] = axis_line_props["visible"]  # domain is whether axis line should be visible.
        axis_config["domainOpacity"] = axis_line_props["alpha"] if axis_line_props["alpha"] else 1
        axis_config["domainColor"] = mcolors.to_hex(axis_line_props["edgecolor"])
        axis_config["domainWidth"] = (axis_line_props["linewidth"] * dpi) / 72
        axis_config["grid"] = axis_props["tick_params"]["gridOn"]

        # making the assumption here that all gridlines look the same
        if axis_config["grid"]:
            axis_config["gridOpacity"] = axis_props["gridlines"][0].properties()["alpha"]
            axis_config["gridCap"] = axis_props["gridlines"][0].properties()["dash_capstyle"]
            grid_color = float(axis_props["gridlines"][0].properties()["markeredgecolor"])
            axis_config["gridColor"] = to_hex_alpha([grid_color] * 3)
            axis_config["gridWidth"] = (axis_props["gridlines"][0].properties()["markeredgewidth"] * dpi) / 72
        axis_config["labelFont"] = axis_props["majorticklabels"][0].get_fontname()
        axis_config["labelFontSize"] = (axis_props["majorticklabels"][0].get_size() * dpi) / 72
        axis_config["labelFontStyle"] = axis_props["majorticklabels"][0].get_fontstyle()
        axis_config["labelFontWeight"] = axis_props["majorticklabels"][0].get_fontweight()
        axis_config["tickCount"] = len(axis_props["ticklocs"])
        if axis_config["tickCount"] != 0:
            tick_props = axis_props["ticklines"][0].properties()
            axis_config["ticks"] = tick_props["visible"]
            axis_config["tickOpacity"] = tick_props["alpha"] if tick_props["alpha"] else 1
            if axis_config["ticks"] and axis_config["tickOpacity"] != 0:
                axis_config["tickColor"] = mcolors.to_hex(tick_props["color"])
                axis_config["tickCap"] = tick_props["dash_capstyle"]
                axis_config["tickWidth"] = (tick_props["linewidth"] * dpi) / 72
                axis_config["tickSize"] = (
                    tick_props["markersize"] * dpi
                ) / 72  # also marker edge width, but vega doesn't have an equivalent for that.

        label = axis_props["label_text"]
        if label != "":
            axis_config["title"] = label
            label_props = axis_props["label"].properties()

            axis_config["titleAlign"] = label_props["horizontalalignment"]
            axis_config["titleBaseline"] = label_props["verticalalignment"]
            axis_config["titleColor"] = mcolors.to_hex(label_props["color"])
            axis_config["titleFont"] = label_props["fontname"]
            axis_config["titleFontSize"] = (label_props["fontsize"] * dpi) / 72
            axis_config["titleFontWeight"] = label_props["fontweight"]
            axis_config["titleOpacity"] = label_props["alpha"] if label_props["alpha"] else 1
        axis_config["zindex"] = axis_props["zorder"]

        axis_array.append(axis_config)
    return axis_array
