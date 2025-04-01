from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from spatialdata_plot._viewconfig.data import (
    create_base_level_sdata_object,
    create_derived_data_object,
    create_table_data_object,
)
from spatialdata_plot._viewconfig.layout import create_padding_object
from spatialdata_plot._viewconfig.misc import VegaAlignment
from spatialdata_plot._viewconfig.scales import (
    create_axis_scale_array,
    create_colorscale_array_image,
    create_colorscale_array_points_shapes_labels,
)
from spatialdata_plot.pl.render_params import (
    FigParams,
    ImageRenderParams,
    LabelsRenderParams,
    PointsRenderParams,
    ShapesRenderParams,
)

Params = ImageRenderParams | LabelsRenderParams | PointsRenderParams | ShapesRenderParams

if TYPE_CHECKING:
    from matplotlib.text import Text
    from spatialdata import SpatialData


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


def _create_categorical_legend(fig: Figure, color_scale_array: list[dict[str, Any]], ax: Axes) -> list[dict[str, Any]]:
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
    legend_bbox_props = legend.get_frame().properties()
    legend_bbox = legend.get_tightbbox()

    for color_object in color_scale_array:
        fill_color = legend.get_frame().get_facecolor()
        legend_object = {
            "type": "discrete",
            "direction": "horizontal" if legend._ncols == 0 else "vertical",
            "fill": color_object["name"],
            "orient": "none",
            "columns": legend._ncols,
            "columnPadding": (legend.columnspacing * fig.dpi) / 72,
            "rowPadding": (legend.labelspacing * fig.dpi) / 72,
            "fillColor": mcolors.to_hex(fill_color),
            "padding": (legend.borderpad * fig.dpi) / 72,
            "strokeColor": mcolors.to_hex(legend_bbox_props["edgecolor"]),
            "strokeWidth": (legend_bbox_props["linewidth"] * fig.dpi)
            / 72,  # Different from Vega as vega expects a vega scale here!
            "labelAlign": legend.get_texts()[0].get_ha(),
            "labelColor": mcolors.to_hex(legend.get_texts()[0].get_color()),
            "labelFont": legend.get_texts()[0].get_fontname(),
            "labelFontSize": (legend.get_texts()[0].get_fontsize() * fig.dpi) / 72,
            "labelFontStyle": legend.get_texts()[0].get_fontstyle(),
            "labelFontWeight": legend.get_texts()[0].get_fontweight(),
            "labelOffset": (legend.handletextpad * fig.dpi) / 72,
            "legendX": legend_bbox.bounds[0],
            "legendY": fig.bbox.height - legend_bbox.bounds[1] - legend_bbox.bounds[3],
        }

        if legend.get_title().get_text() != "":
            legend_title_object = _create_legend_title_config(legend.get_title(), fig.dpi)
            legend_object.update(legend_title_object)

        legend_array.append(legend_object)
    return legend_array


def _create_colorbar_legend(
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

    if len(cbars) != 0:
        for col_config in color_scale_array:
            cbar = cbars[legend_count]

            axis_props = cbar.ax.properties()
            if cbar.orientation == "vertical":
                gradient_length = cbar.ax.get_position().bounds[-1] * fig.get_figheight() * fig.dpi
                label = axis_props["yticklabels"][0].properties()
            else:
                gradient_length = cbar.ax.get_position().bounds[-2] * fig.get_figwidth() * fig.dpi
                label = axis_props["xticklabels"][0].properties()
            if col_config["type"] == "linear":
                legend_type = "gradient"
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
                "values": list(cbar.ax.get_yticks()),
                "labelAlign": label["horizontalalignment"],
                "labelColor": mcolors.to_hex(label["color"]),
                "labelFont": label["fontname"],
                "labelFontSize": (label["fontsize"] * fig.dpi) / 72,
                "labelFontStyle": label["fontstyle"],
                "labelFontWeight": label["fontweight"],
                "legendX": cbar.ax.get_tightbbox().bounds[0],
                "legendY": fig.bbox.height - cbar.ax.get_tightbbox().bounds[1] - cbar.ax.get_tightbbox().bounds[3],
                "zindex": axis_props["zorder"],
            }
            if legend_title_object["title"] != "":
                legend_object.update(legend_title_object)
            legend_array.append(legend_object)
    return legend_array


def _create_scales_legends_marks(
    fig: Figure,
    ax: Axes,
    data_object: dict[str, Any],
    params: Params,
    call_count: int,
    legend_count: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create vega like data object for SpatialData elements.

    Each object for a SpatialData element contains an additional transform that
    is not entirely corresponding to the vega spec but aims to allow for retrieving
    the specific element and transforming it to a particular coordinate space.

    Parameters
    ----------
    params: Params
        The render parameters used in spatialdata-plot for the particular type of SpatialData
        element.
    """
    marks_object: dict[str, Any] = {}
    color_scale_array: list[dict[str, Any]] = []
    legend_array: list[dict[str, Any]] = []

    if isinstance(params, ImageRenderParams):  # second part to shut up mypy
        color_scale_array = create_colorscale_array_image(params.cmap_params, data_object["name"], params.channel)
        legend_array = _create_colorbar_legend(fig, color_scale_array, legend_count)
        marks_object = _create_raster_image_marks_object(ax, params, data_object, call_count, color_scale_array)
    if isinstance(params, LabelsRenderParams):
        if params.colortype is not None:
            color_scale_array = create_colorscale_array_points_shapes_labels(params.colortype, params, data_object)
        if not mcolors.is_color_like(params.colortype):
            legend_array = _create_colorbar_legend(fig, color_scale_array, legend_count)
        if isinstance(params.colortype, dict):
            legend_array = _create_categorical_legend(fig, color_scale_array, ax)
        marks_object = _create_raster_label_marks_object(ax, params, data_object, call_count, color_scale_array)
    if isinstance(params, PointsRenderParams | ShapesRenderParams):
        if params.colortype is not None:
            color_scale_array = create_colorscale_array_points_shapes_labels(params.colortype, params, data_object)
            if params.colortype == "continuous":
                legend_array = _create_colorbar_legend(fig, color_scale_array, legend_count)
            if isinstance(params.colortype, dict):
                legend_array = _create_categorical_legend(fig, color_scale_array, ax)

        if isinstance(params, PointsRenderParams):
            marks_object = _create_points_symbol_marks_object(ax, params, data_object, call_count, color_scale_array)
        else:
            marks_object = _create_shapes_marks_object(ax, params, data_object, call_count, color_scale_array)

    return marks_object, color_scale_array, legend_array


def _create_raster_image_marks_object(
    ax: Axes,
    params: ImageRenderParams,
    data_object: dict[str, Any],
    call_count: int,
    color_scale_array: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(color_scale_array) == 1:
        fill_color = [{"scale": color_scale_array[0]["name"], "value": "value"}]
    else:
        fill_color = [
            {"scale": color_scale["name"], "field": f"channel_{index}"}
            for index, color_scale in enumerate(color_scale_array)
        ]

    return {
        "type": "raster_image",
        "from": {"data": data_object["name"]},
        "zindex": ax.properties()["images"][call_count].zorder,
        "encode": {
            "enter": {
                "opacity": {"value": params.alpha},
                "fill": fill_color,
            }
        },
    }


def _create_shapes_marks_object(
    ax: Axes,
    params: ShapesRenderParams,
    data_object: dict[str, Any],
    call_count: int,
    color_scale_array: list[dict[str, Any]],
) -> dict[str, Any]:
    encode_update = None
    if not color_scale_array and not params.color:
        fill_color = {"value": mcolors.to_hex(params.cmap_params.na_color, keep_alpha=False)}
    elif not color_scale_array and params.color:
        fill_color = {"value": mcolors.to_hex(params.color, keep_alpha=False)}
    elif color_scale_array and (params.color or params.col_for_color):
        if isinstance(params.colortype, dict):
            encode_update = {
                "fill": [
                    {
                        "test": f"!isValid(datum.{params.col_for_color})",
                        "value": mcolors.to_hex(params.cmap_params.na_color, keep_alpha=False),
                    }
                ]
            }
            fill_color = {"scale": color_scale_array[0]["name"], "field": params.col_for_color}
        else:
            value = val[0] if isinstance(val := color_scale_array[0]["domain"]["field"], list) else val
            fill_color = {"scale": color_scale_array[0]["name"], "value": value}
            encode_update = {"fill": []}
            encode_update["fill"].append(
                {
                    "test": f"!isValid(datum.{params.col_for_color})",
                    "value": mcolors.to_hex(params.cmap_params.na_color, keep_alpha=False),
                }
            )
            if (params.cmap_params.norm.vmin is not None or params.cmap_params.norm.vmax is not None) and (
                params.cmap_params.cmap.get_under() is not None or params.cmap_params.cmap.get_over() is not None
            ):
                # or condition doesn't reach second condition if first condition is met
                under_col = params.cmap_params.cmap.get_under()
                over_col = params.cmap_params.cmap.get_over()
                if under_col is not None:
                    encode_update["fill"].append(
                        {
                            "test": f"datum.{value}) < {params.cmap_params.norm.vmin}",
                            "value": mcolors.to_hex(under_col, keep_alpha=False),
                        }
                    )
                if over_col is not None:
                    encode_update["fill"].append(
                        {
                            "test": f"datum.{value}) > {params.cmap_params.norm.vmax}",
                            "value": mcolors.to_hex(over_col, keep_alpha=False),
                        }
                    )

    shapes_object = {
        "type": "path",
        "from": {"data": data_object["name"]},
        "zindex": params.zorder,
        "encode": {
            "enter": {
                "x": {"scale": "X_scale", "field": "x"},
                "y": {"scale": "Y_scale", "field": "y"},
                "scaleX": params.scale,
                "scaleY": params.scale,
                "fill": fill_color,
                "fillOpacity": {"value": params.fill_alpha},
            }
        },
    }

    if params.outline_params.outline and params.outline_alpha != 0:
        outline_par = params.outline_params
        stroke_color = {"value": mcolors.to_hex(outline_par.outline_color, keep_alpha=False)}

        shapes_object["encode"]["enter"].update(  # type: ignore[index]
            {
                "stroke": stroke_color,
                "strokeWidth": {"value": outline_par.linewidth},
                "strokeOpacity": {"value": params.outline_alpha},
            }
        )

    return shapes_object


def _create_points_symbol_marks_object(
    ax: Axes,
    params: PointsRenderParams,
    data_object: dict[str, Any],
    call_count: int,
    color_scale_array: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    encode_update = {}
    if not color_scale_array and not params.color:
        fill_color = {"value": mcolors.to_hex(params.cmap_params.na_color, keep_alpha=False)}
    elif not color_scale_array and params.color:
        fill_color = {"value": mcolors.to_hex(params.color, keep_alpha=False)}
    elif color_scale_array and (params.color or params.col_for_color):
        if isinstance(params.colortype, dict):
            encode_update["fill"] = [
                {
                    "test": f"!isValid(datum.{params.col_for_color})",
                    "value": mcolors.to_hex(params.cmap_params.na_color, keep_alpha=False),
                }
            ]
            fill_color = {"scale": color_scale_array[0]["name"], "field": params.col_for_color}
        else:
            value = val[0] if isinstance(val := color_scale_array[0]["domain"]["field"], list) else val
            fill_color = {"scale": color_scale_array[0]["name"], "value": value}
            # encode_update = {"fill": []}
            encode_update["fill"] = [
                {
                    "test": f"!isValid(datum.{params.col_for_color})",
                    "value": mcolors.to_hex(params.cmap_params.na_color, keep_alpha=False),
                }
            ]

            if (params.cmap_params.norm.vmin is not None or params.cmap_params.norm.vmax is not None) and (
                params.cmap_params.cmap.get_under() is not None or params.cmap_params.cmap.get_over() is not None
            ):
                # or condition doesn't reach second condition if first condition is met
                under_col = params.cmap_params.cmap.get_under()
                over_col = params.cmap_params.cmap.get_over()
                if under_col is not None:
                    encode_update["fill"].append(
                        {
                            "test": f"datum.{value}) < {params.cmap_params.norm.vmin}",
                            "value": mcolors.to_hex(under_col, keep_alpha=False),
                        }
                    )
                if over_col is not None:
                    encode_update["fill"].append(
                        {
                            "test": f"datum.{value}) > {params.cmap_params.norm.vmax}",
                            "value": mcolors.to_hex(over_col, keep_alpha=False),
                        }
                    )

    points_object = {
        "type": "symbol",
        "from": {"data": data_object["name"]},
        "zindex": params.zorder,
        "encode": {
            "enter": {
                "x": {"scale": "X_scale", "field": "x"},
                "y": {"scale": "Y_scale", "field": "y"},
                "stroke": fill_color,
                "fill": fill_color,
                "fillOpacity": {"value": params.alpha},
                "size": {"value": params.size},
                "shape": {"value": "circle"},
            }
        },
    }
    if encode_update:
        # TODO: check if we can give info that na-color is used prior to adding this. If so then add if conditional.
        points_object["encode"]["update"] = encode_update  # type: ignore[index]

    return points_object


def _create_raster_label_marks_object(
    ax: Axes,
    params: LabelsRenderParams,
    data_object: dict[str, Any],
    call_count: int,
    color_scale_array: list[dict[str, Any]],
) -> dict[str, Any]:

    if params.colortype == "continuous":
        color_col = color_scale_array[0]["domain"]["field"][0]
        fill_color = [{"scale": color_scale_array[0]["name"], "value": color_col}]
        encode_update = {
            "fill": [
                {"test": "isValid(datum.value)", "scale": color_scale_array[0]["name"], "field": color_col},
                {"value": params.cmap_params.na_color},
            ]
        }
    if isinstance(params.colortype, dict):
        color_col = params.color
        fill_color = [{"scale": color_scale_array[0]["name"], "value": color_col}]
        encode_update = {
            "fill": [
                {"test": "isValid(datum.value)", "scale": color_scale_array[0]["name"], "field": color_col},
                {"value": params.cmap_params.na_color},
            ]
        }
    if params.colortype == "random":
        fill_color = [{"scale": color_scale_array[0]["name"], "value": "value"}]
    if mcolors.is_color_like(params.colortype):
        fill_color = [{"value": params.colortype}]

    labels_object = {
        "type": "raster_label",
        "from": {"data": data_object["name"]},
        "zindex": ax.properties()["images"][call_count].zorder,
        "encode": {
            "enter": {
                "stroke": fill_color,
                "fill": fill_color,
                "fillOpacity": {"value": params.fill_alpha},
                "strokeOpacity": {"value": params.outline_alpha},
                "strokeWidth": {"value": params.contour_px},
            }
        },
    }

    if params.colortype == "continuous":
        labels_object["encode"]["update"] = encode_update
    return labels_object


def strip_call(s: str) -> str:
    """Strip leading digit and underscore from call."""
    return re.sub(r"^\d+_", "", s)


def _create_data_configs(
    sdata: SpatialData, fig: Figure, ax: Axes, cs: str, sdata_path: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create the vega json array value to the data key.

    The data array in the SpatialData vegalike viewconfig consists out of
    an object for the base level of the SpatialData zarr store and subsequently
    derived individual SpatialData elements.

    Parameters
    ----------
    plotting_tree: OrderedDict[str, Params]
        Dictionary with as keys the render calls prefixed with the index of the render call. Render calls are either
        render_images, render_labels, render_points, or render_shapes. The values in the dict are the parameters
        corresponding to the render call.
    cs: str
        The name of the coordinate system in which the SpatialData elements were plotted.
    sdata_path: str
        The location of the SpatialData zarr store.
    """
    data_array = []
    marks_array = []
    color_scale_array_full: list[dict[str, Any]] = []
    legend_array_full = []
    url = str(Path("sdata.zarr"))

    if sdata_path:
        url = sdata_path

    base_block = create_base_level_sdata_object(url)
    data_array.append(base_block)

    counters = {"render_images": 0, "render_labels": 0, "render_points": 0, "render_shapes": 0}
    for call, params in sdata.plotting_tree.items():
        call = strip_call(call)
        table_id = None
        if table := getattr(params, "table_name", None):
            data_array.append(create_table_data_object(table, base_block["name"], params.table_layer))
            table_id = data_array[-1]["name"]
        data_array.append(create_derived_data_object(sdata, call, params, base_block["name"], cs, table_id))
        marks_object, color_scale_array, legend_array = _create_scales_legends_marks(
            fig, ax, data_array[-1], params, counters[call], len(color_scale_array_full)
        )

        marks_array.append(marks_object)
        if color_scale_array:
            color_scale_array_full += color_scale_array
        legend_array_full += legend_array
        counters[call] += 1

    return data_array, marks_array, color_scale_array_full, legend_array_full


def _create_title_config(ax: Axes, fig: Figure) -> dict[str, Any]:
    """Create a vega title object for a spatialdata view configuration.

    Note that not all field values as obtained from matplotlib are supported by the official
    vega specification.

    Parameters
    ----------
    ax : Axes
        A matplotlib Axes instance which represents one (sub)plot in a matplotlib figure.
    fig : Figure
        The matplotlib figure. The top level container for all the plot elements.
    """
    title_text = ax.get_title()
    title_obj = ax.title
    title_font = title_obj.get_fontproperties()

    return {
        "text": title_text,
        "orient": "top",  # there is not really a nice conversion here of matplotlib to vega
        "anchor": VegaAlignment.from_matplotlib(title_obj.get_horizontalalignment()),
        "baseline": title_obj.get_va(),
        "color": title_obj.get_color(),
        "font": title_obj.get_fontname(),
        "fontSize": (title_font.get_size() * fig.dpi) / 72,
        "fontStyle": title_obj.get_fontstyle(),
        "fontWeight": title_font.get_weight(),
    }


def _create_axis_block(ax: Axes, axis_scales_block: list[dict[str, Any]], dpi: float) -> list[dict[str, Any]]:
    axis_array = []
    for scale in axis_scales_block:
        axis_config = {}
        axis_config["scale"] = scale["name"]
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
            axis_config["gridColor"] = mcolors.to_hex([grid_color] * 3)
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
        if label == "":
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


def create_viewconfig(sdata: SpatialData, fig_params: FigParams, legend_params: Any, cs: str) -> dict[str, Any]:
    fig = fig_params.fig
    ax = fig_params.ax
    data_array, marks_array, color_scale_array, legend_array = _create_data_configs(sdata, fig, ax, cs, sdata._path)

    scales_array = create_axis_scale_array(ax)
    axis_array = _create_axis_block(ax, scales_array, fig.dpi)

    scales = scales_array + color_scale_array if len(color_scale_array) > 0 else scales_array
    # TODO: check why attrs does not respect ordereddict when writing sdata
    viewconfig = {
        "$schema": "https://spatialdata-plot.github.io/schema/viewconfig/v1.json",
        "height": fig.bbox.height,  # matplotlib uses inches, but vega uses absolute pixels
        "width": fig.bbox.width,
        "padding": create_padding_object(fig),
        "title": _create_title_config(ax, fig),
        "data": data_array,
        "scales": scales,
    }

    viewconfig["axes"] = axis_array
    if len(legend_array) > 0:
        viewconfig["legend"] = legend_array
    viewconfig["marks"] = marks_array

    return viewconfig
