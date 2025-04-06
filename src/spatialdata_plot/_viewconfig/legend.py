from typing import Any

import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

from spatialdata_plot._viewconfig.misc import enforce_common_decimal_format
from spatialdata_plot.pl.utils import to_hex_alpha


def _create_legend_title_config(title_obj: Text, dpi: float) -> dict[str, Any]:
    """Create the vega like legend title object.

    This creates the object containing information pertaining to the legend title. This will be added to the legend
    object.

    Parameters
    ----------
    title_obj : Text
        The legend title object in matplotlib.
    dpi: float
        dots per inch used to convert fontsizes to from standard unit to size in pixels.

    Returns
    -------
    The legend title object.
    """
    title_props = title_obj.properties()
    return {
        "title": title_props["text"],
        "titleOrient": "top",
        "titleAlign": title_props["horizontalalignment"],
        "titleBaseline": title_props["verticalalignment"],
        "titleColor": title_props["color"],
        "titleFont": title_props["fontname"],
        "titleFontSize": (title_props["fontsize"] * dpi) / 72,
        "titleFontStyle": title_props["fontstyle"],
        "titleFontWeight": title_props["fontweight"],
    }


def _extract_legend_label_properties(label_texts: list[Text], dpi: float) -> dict[str, Any]:
    """Extract common legend properties for reuse."""
    text_props = label_texts[0].properties()

    return {
        "labelAlign": text_props["horizontalalignment"],
        "labelColor": to_hex_alpha(text_props["color"]),
        "labelFont": text_props["fontname"],
        "labelFontSize": (text_props["fontsize"] * dpi) / 72,
        "labelFontStyle": text_props["fontstyle"],
        "labelFontWeight": text_props["fontweight"],
    }


def create_categorical_legend(fig: Figure, color_scale_array: list[dict[str, Any]], ax: Axes) -> list[dict[str, Any]]:
    """Create vega like categorical legend array.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure.
    color_scale_array : list[dict[str, Any]]
        The vega like color scale array for which the vega like legend array will be created.
    ax : Axes
        A matplotlib Axes object.

    Returns
    -------
    The vega like categorical legend array.
    """
    legend_array: list[dict[str, Any]] = []
    legend = ax.legend()
    label_props = _extract_legend_label_properties(legend.get_texts(), fig.dpi)
    frame = legend.get_frame()

    for color_object in color_scale_array:
        legend_object = {
            "type": "discrete",
            "direction": "horizontal" if legend._ncols == 0 else "vertical",
            "fill": color_object["name"],
            "orient": "none",  # required by vega usually to explicitly use legend position X and Y
            "columns": legend._ncols,
            "columnPadding": (legend.columnspacing * fig.dpi) / 72,
            "rowPadding": (legend.labelspacing * fig.dpi) / 72,
            "padding": (legend.borderpad * fig.dpi) / 72,
            "fillColor": to_hex_alpha(frame.get_facecolor()),
            "strokeColor": to_hex_alpha(frame.get_edgecolor()),
            "strokeWidth": (frame.get_linewidth() * fig.dpi) / 72,
            "labelOffset": (legend.handletextpad * fig.dpi) / 72,
            **label_props,
            "legendX": legend.get_tightbbox().bounds[0],
            "legendY": fig.bbox.height - legend.get_tightbbox().bounds[1] - legend.get_tightbbox().bounds[3],
        }

        if legend.get_title().get_text() != "":
            legend_title_object = _create_legend_title_config(legend.get_title(), fig.dpi)
            legend_object |= legend_title_object

        legend_array.append(legend_object)
    return legend_array


def create_colorbar_legend(
    fig: Figure, color_scale_array: list[dict[str, Any]], legend_count: int
) -> list[dict[str, Any]]:
    """Create the vega like legend array containing the colorbar information.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure.
    color_scale_array : list[dict[str, Any]]
        The vega like color scale array for which the vega like legend array will be created.
    legend_count : int
        The number of already created legend objects.

    Returns
    -------
    The vega like colorbar legend array.
    """
    legend_array: list[dict[str, Any]] = []
    cbars = []
    for ax in fig.axes:
        cbar = getattr(ax.properties()["axes_locator"], "_cbar", None) if ax.properties()["axes_locator"] else None
        if cbar:
            cbars.append(cbar)

    if cbars:
        for col_config in color_scale_array:
            cbar = cbars[legend_count]

            axis_props = cbar.ax.properties()
            if cbar.orientation == "vertical":
                gradient_length = cbar.ax.get_position().bounds[-1] * fig.get_figheight() * fig.dpi
                labels = axis_props["yticklabels"]
            else:
                gradient_length = cbar.ax.get_position().bounds[-2] * fig.get_figwidth() * fig.dpi
                labels = axis_props["xticklabels"]
            if col_config["type"] == "linear":
                legend_type = "gradient"

            common_props = _extract_legend_label_properties(labels, fig.dpi)
            spine_outline = cbar.outline.properties()  # outline of the colorbar lining

            stroke_color = mcolors.to_hex(spine_outline["facecolor"]) if spine_outline["facecolor"][-1] > 0 else None
            legend_title_object = _create_legend_title_config(cbar.ax.title, fig.dpi)
            # TODO: do we require padding? it is not obvious to get from matplotlib
            legend_object = {
                "type": legend_type,
                "direction": cbar.orientation,
                "orient": "none",  # Required in vega in order to use the x and y position
                "fill": color_scale_array[0]["name"],
                "fillColor": mcolors.to_hex(cbar.ax.get_facecolor()),
                "gradientLength": gradient_length,  # alpha if alpha := getattr(cbar.cmap, "_lut", None)[0][-1] else
                "gradientOpacity": cbar.mappable.get_alpha(),
                "gradientThickness": (cbar.ax.get_position().bounds[2] * fig.dpi) / 72,
                "gradientStrokeColor": stroke_color,
                "gradientStrokeWidth": (spine_outline["linewidth"] * fig.dpi) / 72 if stroke_color else None,
                "values": enforce_common_decimal_format(list(cbar.ax.get_yticks())),
                **common_props,
                "legendX": cbar.ax.get_tightbbox().bounds[0],
                "legendY": fig.bbox.height - cbar.ax.get_tightbbox().bounds[1] - cbar.ax.get_tightbbox().bounds[3],
                "zindex": axis_props["zorder"],
            }
            if legend_title_object["title"] != "":
                legend_object |= legend_title_object
            legend_array.append(legend_object)
    return legend_array
