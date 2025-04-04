from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from spatialdata_plot._viewconfig.axis import create_axis_block
from spatialdata_plot._viewconfig.data import (
    create_base_level_sdata_object,
    create_derived_data_object,
    create_table_data_object,
)
from spatialdata_plot._viewconfig.layout import create_padding_object, create_title_config
from spatialdata_plot._viewconfig.legend import create_categorical_legend, create_colorbar_legend
from spatialdata_plot._viewconfig.misc import strip_call
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
    from spatialdata import SpatialData


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
        legend_array = create_colorbar_legend(fig, color_scale_array, legend_count)
        marks_object = _create_raster_image_marks_object(ax, params, data_object, call_count, color_scale_array)
    if isinstance(params, LabelsRenderParams):
        if params.colortype is not None:
            color_scale_array = create_colorscale_array_points_shapes_labels(params.colortype, params, data_object)
        if params.colortype == "continuous":
            legend_array = create_colorbar_legend(fig, color_scale_array, legend_count)
        if isinstance(params.colortype, dict):
            legend_array = create_categorical_legend(fig, color_scale_array, ax)
        marks_object = _create_raster_label_marks_object(ax, params, data_object, call_count, color_scale_array)
    if isinstance(params, PointsRenderParams | ShapesRenderParams):
        if params.colortype is not None:
            color_scale_array = create_colorscale_array_points_shapes_labels(params.colortype, params, data_object)
            if params.colortype == "continuous":
                legend_array = create_colorbar_legend(fig, color_scale_array, legend_count)
            if isinstance(params.colortype, dict):
                legend_array = create_categorical_legend(fig, color_scale_array, ax)

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
    marks_array = []
    color_scale_array_full: list[dict[str, Any]] = []
    legend_array_full = []
    url = str(Path("sdata.zarr"))

    if sdata_path:
        url = sdata_path

    base_block = create_base_level_sdata_object(url)
    data_array = [base_block]

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


def create_viewconfig(sdata: SpatialData, fig_params: FigParams, legend_params: Any, cs: str) -> dict[str, Any]:
    fig = fig_params.fig
    ax = fig_params.ax
    data_array, marks_array, color_scale_array, legend_array = _create_data_configs(sdata, fig, ax, cs, sdata._path)

    scales_array = create_axis_scale_array(ax)
    axis_array = create_axis_block(ax, scales_array, fig.dpi)

    scales = scales_array + color_scale_array if len(color_scale_array) > 0 else scales_array
    # TODO: check why attrs does not respect ordereddict when writing sdata
    viewconfig = {
        "$schema": "https://spatialdata-plot.github.io/schema/viewconfig/v1.json",
        "height": fig.bbox.height,
        "width": fig.bbox.width,
        "padding": create_padding_object(fig),
        "title": create_title_config(ax, fig),
        "data": data_array,
        "scales": scales,
        "axes": axis_array,
    }

    if len(legend_array) > 0:
        viewconfig["legend"] = legend_array
    viewconfig["marks"] = marks_array

    return viewconfig
