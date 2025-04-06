from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from spatialdata import SpatialData

from spatialdata_plot._viewconfig.axis import create_axis_block
from spatialdata_plot._viewconfig.data import (
    create_base_level_sdata_object,
    create_derived_data_object,
    create_table_data_object,
)
from spatialdata_plot._viewconfig.legend import create_categorical_legend, create_colorbar_legend
from spatialdata_plot._viewconfig.marks import (
    create_points_symbol_marks_object,
    create_raster_image_marks_object,
    create_raster_label_marks_object,
    create_shapes_marks_object,
)
from spatialdata_plot._viewconfig.misc import VegaAlignment, strip_call
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


def _colortype_to_scale_legend(
    fig: Figure,
    ax: Axes,
    params: LabelsRenderParams | PointsRenderParams | ShapesRenderParams,
    data_object: dict[str, Any],
    legend_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    color_scale_array = []
    legend_array = []

    if params.colortype is not None:
        color_scale_array = create_colorscale_array_points_shapes_labels(params.colortype, params, data_object)
        if params.colortype == "continuous":
            legend_array = create_colorbar_legend(fig, color_scale_array, legend_count)
        elif isinstance(params.colortype, dict):
            legend_array = create_categorical_legend(fig, color_scale_array, ax)

    return color_scale_array, legend_array


def create_padding_object(fig: Figure) -> dict[str, float]:
    """Get the padding parameters for a vega viewconfiguration.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure. The top level container for all the plot elements.
    """
    # contains also wspace and hspace but does not seem to be used by vega here.
    padding_obj = fig.subplotpars
    return {
        "left": (padding_obj.left * fig.bbox.width),
        "top": ((1 - padding_obj.top) * fig.bbox.height),
        "right": ((1 - padding_obj.right) * fig.bbox.width),
        "bottom": (padding_obj.bottom * fig.bbox.height),
    }


def create_title_config(ax: Axes, fig: Figure) -> dict[str, Any]:
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


def _create_scales_legends_marks(
    fig: Figure,
    ax: Axes,
    data_object: dict[str, Any],
    params: Params,
    call_count: int,
    legend_count: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create vega like scales, legend and mark arrays for the viewconfiguration.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure.
    ax : Axes
        Matplotlib Axes object representing the (sub-) plot in which the SpatialData labels element is visualized.
    data_object: dict[str, Any]
        A vega like data object pertaining to a Spatialdata element that is visualized.
    params: Params
        The render parameters used in spatialdata-plot for the particular type of SpatialData
        element.
    call_count : int
        The number indicating the index of the render call that visualized the SpatialData element.
    legend_count : int
        The number of already created legend objects.

    Returns
    -------
    marks_object : dict[str, Any]
        The vega like marks object for a given spatialdata element.
    color_scale_array : list[dict[str, Any]]
        An array of vega like color scale object containing the information for applying colors to a mark
    legend_array : list[dict[str, Any]]
        An array of vega like legend objects for a given spatialdata element colored based on a given
        color object.
    """
    marks_object: dict[str, Any] = {}
    color_scale_array: list[dict[str, Any]] = []
    legend_array: list[dict[str, Any]] = []

    match params:
        case ImageRenderParams():
            color_scale_array = create_colorscale_array_image(params.cmap_params, data_object["name"], params.channel)
            legend_array = create_colorbar_legend(fig, color_scale_array, legend_count)
            marks_object = create_raster_image_marks_object(ax, params, data_object, call_count, color_scale_array)
        case LabelsRenderParams() | PointsRenderParams() | ShapesRenderParams():
            color_scale_array, legend_array = _colortype_to_scale_legend(fig, ax, params, data_object, legend_count)

    match params:
        case LabelsRenderParams():
            marks_object = create_raster_label_marks_object(ax, params, data_object, call_count, color_scale_array)
        case PointsRenderParams():
            marks_object = create_points_symbol_marks_object(params, data_object, color_scale_array)
        case ShapesRenderParams():
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
    sdata : SpatialData
        The SpatialData object from which elements are visualized.
    fig : Figure
        The matplotlib figure.
    ax : Axes
        Matplotlib Axes object representing the (sub-) plot in which the SpatialData labels element is visualized.
    cs: str
        The name of the coordinate system in which the SpatialData elements were plotted.
    sdata_path: str
        The location of the SpatialData zarr store.

    Returns
    -------
    data_array : list[dict[str, Any]]
        An array of vega like data objects pertaining to visualized SpatialData elements.
    marks_array: list[dict[str, Any]]
        An array of vega like marks objects, each pertaining to one render call.
    color_scale_array_full : list[dict[str, Any]]
        An array of vega like color scale objects, each containing information regarding the coloring
        used to visualize a particular SpatialData element.
    legend_array_full: list[dict[str, Any]]
        An array of vega like legend objects, each pertaining to one legend.
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


def create_viewconfig(sdata: SpatialData, fig_params: FigParams, cs: str) -> dict[str, Any]:
    """Create a vega like view configuration based on the spatialdata-plot visualization.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object from which elements are visualized.
    fig_params : FigParams
        The figure parameters containing for example the matplotlib figure and axes.
    cs: str
        The name of the coordinate system in which the SpatialData elements were plotted.
    """
    fig = fig_params.fig
    ax = fig_params.ax
    data_array, marks_array, color_scale_array, legend_array = _create_data_configs(sdata, fig, ax, cs, sdata._path)

    scales_array = create_axis_scale_array(ax)
    axis_array = create_axis_block(ax, scales_array, fig.dpi)

    scales = scales_array + color_scale_array if len(color_scale_array) > 0 else scales_array

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
