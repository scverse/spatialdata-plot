from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from spatialdata_plot._viewconfig.marks import (
    create_points_symbol_marks_object,
    create_raster_image_marks_object,
    create_raster_label_marks_object,
    create_shapes_marks_object,
)
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
        marks_object = create_raster_image_marks_object(ax, params, data_object, call_count, color_scale_array)
    if isinstance(params, LabelsRenderParams):
        if params.colortype is not None:
            color_scale_array = create_colorscale_array_points_shapes_labels(params.colortype, params, data_object)
        if params.colortype == "continuous":
            legend_array = create_colorbar_legend(fig, color_scale_array, legend_count)
        if isinstance(params.colortype, dict):
            legend_array = create_categorical_legend(fig, color_scale_array, ax)
        marks_object = create_raster_label_marks_object(ax, params, data_object, call_count, color_scale_array)
    if isinstance(params, PointsRenderParams | ShapesRenderParams):
        if params.colortype is not None:
            color_scale_array = create_colorscale_array_points_shapes_labels(params.colortype, params, data_object)
            if params.colortype == "continuous":
                legend_array = create_colorbar_legend(fig, color_scale_array, legend_count)
            if isinstance(params.colortype, dict):
                legend_array = create_categorical_legend(fig, color_scale_array, ax)

        if isinstance(params, PointsRenderParams):
            marks_object = create_points_symbol_marks_object(params, data_object, color_scale_array)
        else:
            marks_object = create_shapes_marks_object(params, data_object, color_scale_array)

    return marks_object, color_scale_array, legend_array


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
