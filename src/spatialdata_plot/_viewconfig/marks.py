from typing import Any

import matplotlib.colors as mcolors
from matplotlib.axes import Axes

from spatialdata_plot.pl.render_params import (
    ImageRenderParams,
    LabelsRenderParams,
    PointsRenderParams,
    ShapesRenderParams,
)


def _create_marks_fill_color_from_params(
    params: PointsRenderParams | ShapesRenderParams, color_scale_array: list[dict[str, Any]] | None
) -> dict[str, Any] | None:
    """
    Create the fill color object for a vega like mark for a points or shapes element.

    Parameters
    ----------
    params : PointsRenderParams | ShapesRenderParams
        The render parameters used for visualizing the SpatialData points or shapes element.
    color_scale_array : list[dict[str, Any]]
        The vega like color scale array containing the color scale used in the vega like mark object.

    Returns
    -------
    list[dict[str, Any]] | None
        The fill color object for the points or shapes element.
    """
    if not color_scale_array and not params.color:
        return {"value": mcolors.to_hex(params.cmap_params.na_color, keep_alpha=False)}
    if color_scale_array is not None and params.color:
        return {"value": mcolors.to_hex(params.color, keep_alpha=False)}

    if color_scale_array and (params.color or params.col_for_color):
        if isinstance(params.colortype, dict):
            return {"scale": color_scale_array[0]["name"], "field": params.col_for_color}
        value = color_scale_array[0]["domain"]["field"]
        if isinstance(value, list):
            value = value[0]
        return {"scale": color_scale_array[0]["name"], "value": value}
    return None


def _create_encode_update(params: PointsRenderParams | ShapesRenderParams, field_name: str) -> list[dict[str, Any]]:
    """Create the encode update object for a vega like mark for a points or shapes element.

    This object is only created when either a column used to color the mark has a value of NaN or when the
    provided colormap does not perform clipping and values of the column fall beyond the range of vmin and vmax.

    Parameters
    ----------
    params : PointsRenderParams | ShapesRenderParams
        The render parameters used for visualizing the SpatialData points or shapes element.
    field_name : str
        The name of the column used to color the data.

    Returns
    -------
    The encode update object for the points or shapes element.
    """
    hex_na = mcolors.to_hex(params.cmap_params.na_color, keep_alpha=False)
    update = [
        {
            "test": f"!isValid(datum.{params.col_for_color})",
            "value": hex_na,
        }
    ]

    norm = params.cmap_params.norm
    cmap = params.cmap_params.cmap
    if (norm.vmin is not None or norm.vmax is not None) and (
        cmap.get_under() is not None or cmap.get_over() is not None
    ):
        if cmap.get_under() is not None:
            update.append(
                {
                    "test": f"datum.{field_name}) < {norm.vmin}",
                    "value": mcolors.to_hex(cmap.get_under(), keep_alpha=False),
                }
            )
        if cmap.get_over() is not None:
            update.append(
                {
                    "test": f"datum.{field_name}) > {norm.vmax}",
                    "value": mcolors.to_hex(cmap.get_over(), keep_alpha=False),
                }
            )
    return update


def create_raster_image_marks_object(
    ax: Axes,
    params: ImageRenderParams,
    data_object: dict[str, Any],
    call_count: int,
    color_scale_array: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a vega like marks object for visualizing a SpatialData image element.

    Note that there is no equivalent raster image marks object specification in vega. This is because
    vega has no support for visualization of images.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes object representing the (sub-) plot in which the SpatialData image element is visualized.
    params : ImageRenderParams
        The render parameters used for visualizing the SpatialData image element.
    data_object : dict[str, Any]
        A vega like data object which correspond to the SpatialData image element which is visualized.
    call_count : int
        The number indicating the index of the render call that visualized the SpatialData image element.
    color_scale_array : list[dict[str, Any]]
        A vega like array containing the color scale objects containing the information which colors are used
        for visualizing the SpatialData image element.

    Returns
    -------
    dict[str, Any]
        The vega like marks object pertaining to the SpatialData image element that is visualized.
    """
    fill_color = (
        [{"scale": color_scale_array[0]["name"], "value": "value"}]
        if len(color_scale_array) == 1
        else [{"scale": cs["name"], "field": f"channel_{i}"} for i, cs in enumerate(color_scale_array)]
    )

    data_id = (
        data_object["name"] if data_object["transform"][-1]["type"] != "formula" else data_object["transform"][-1]["as"]
    )
    return {
        "type": "raster_image",
        "from": {"data": data_id},
        "zindex": ax.properties()["images"][call_count].zorder,
        "encode": {"enter": {"opacity": {"value": params.alpha}, "fill": fill_color}},
    }


def create_shapes_marks_object(
    params: ShapesRenderParams,
    data_object: dict[str, Any],
    color_scale_array: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a vega like marks object for visualizing a SpatialData shapes element.

    Note that the mark object differs from a vega like mark object in the way that the data is defined.

    Parameters
    ----------
    params : ShapesRenderParams
        The render parameters used for visualizing the SpatialData shapes element.
    data_object : dict[str, Any]
        A vega like data object which correspond to the SpatialData shapes element which is visualized.
    color_scale_array : list[dict[str, Any]]
        A vega like array containing the color scale objects containing the information which colors are used
        for visualizing the SpatialData shapes element.

    Returns
    -------
    dict[str, Any]
        The vega like marks object pertaining to the SpatialData shapes element that is visualized.
    """
    encode_update = {}
    fill_color = _create_marks_fill_color_from_params(params, color_scale_array)

    if color_scale_array and isinstance(params.colortype, dict | str):
        field = params.col_for_color or color_scale_array[0]["domain"]["field"]
        if isinstance(field, list):
            field = field[0]
        encode_update["fill"] = _create_encode_update(params, field)

    mark = {
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

    if encode_update:
        mark["encode"]["update"] = encode_update  # type: ignore[index]

    if params.outline_params.outline and params.outline_alpha != 0:
        outline_par = params.outline_params
        stroke_color = {"value": mcolors.to_hex(outline_par.outline_color, keep_alpha=False)}

        mark["encode"]["enter"].update(  # type: ignore[index]
            {
                "stroke": stroke_color,
                "strokeWidth": {"value": outline_par.linewidth},
                "strokeOpacity": {"value": params.outline_alpha},
            }
        )

    return mark


def create_points_symbol_marks_object(
    params: PointsRenderParams,
    data_object: dict[str, Any],
    color_scale_array: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Create a vega like marks object for visualizing a SpatialData points element.

    Note that the mark object differs from a vega like mark object in the way that the data is defined.

    Parameters
    ----------
    params : PointsRenderParams
        The render parameters used for visualizing the SpatialData points element.
    data_object : dict[str, Any]
        A vega like data object which correspond to the SpatialData points element which is visualized.
    color_scale_array : list[dict[str, Any]]
        A vega like array containing the color scale objects containing the information which colors are used
        for visualizing the SpatialData points element.

    Returns
    -------
    dict[str, Any]
        The vega like marks object pertaining to the SpatialData points element that is visualized.
    """
    fill_color = _create_marks_fill_color_from_params(params, color_scale_array)
    encode_update = {}

    if color_scale_array and isinstance(params.colortype, dict | str):
        field = params.col_for_color or color_scale_array[0]["domain"]["field"]
        if isinstance(field, list):
            field = field[0]
        encode_update["fill"] = _create_encode_update(params, field)

    mark = {
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
        mark["encode"]["update"] = encode_update  # type: ignore[index]

    return mark


def create_raster_label_marks_object(
    ax: Axes,
    params: LabelsRenderParams,
    data_object: dict[str, Any],
    call_count: int,
    color_scale_array: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a vega like marks object for visualizing a SpatialData image element.

    Note that there is no equivalent raster image marks object specification in vega. This is because
    vega has no support for visualization of labels.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes object representing the (sub-) plot in which the SpatialData labels element is visualized.
    params : ImageRenderParams
        The render parameters used for visualizing the SpatialData labels element.
    data_object : dict[str, Any]
        A vega like data object which correspond to the SpatialData labels element which is visualized.
    call_count : int
        The number indicating the index of the render call that visualized the SpatialData labels element.
    color_scale_array : list[dict[str, Any]]
        A vega like array containing the color scale objects containing the information which colors are used
        for visualizing the SpatialData labels element.

    Returns
    -------
    dict[str, Any]
        The vega like marks object pertaining to the SpatialData labels element that is visualized.
    """
    fill_color = [{"value": params.colortype}]
    encode_update = None

    if params.colortype == "continuous":
        if isinstance(field := color_scale_array[0]["domain"]["field"], list):
            field = field[0]

        fill_color = [{"scale": color_scale_array[0]["name"], "value": field}]
        encode_update = {
            "fill": [
                {"test": "isValid(datum.value)", "scale": color_scale_array[0]["name"], "field": field},
                {"value": params.cmap_params.na_color},
            ]
        }
    elif isinstance(params.colortype, dict) and color_scale_array is not None:
        fill_color = [{"scale": color_scale_array[0]["name"], "value": params.color}]
        encode_update = {
            "fill": [
                {"test": "isValid(datum.value)", "scale": color_scale_array[0]["name"], "field": params.color},
                {"value": params.cmap_params.na_color},
            ]
        }
    elif params.colortype == "random":
        fill_color = [{"scale": color_scale_array[0]["name"], "value": "value"}]

    mark = {
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

    if encode_update:
        mark["encode"]["update"] = encode_update

    return mark
