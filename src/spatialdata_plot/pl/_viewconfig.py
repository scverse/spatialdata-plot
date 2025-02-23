from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import matplotlib.colors as mcolors
import spatialdata
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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


class VegaAlignment(Enum):
    LEFT = "start"
    CENTER = "middle"
    RIGHT = "end"

    @classmethod
    def from_matplotlib(cls, alignment: str) -> str:
        """Convert Matplotlib horizontal alignment to Vega alignment."""
        mapping = {"left": cls.LEFT, "center": cls.CENTER, "right": cls.RIGHT}
        return mapping.get(alignment, cls.CENTER).value


def _create_axis_scale_block(ax: Axes) -> list[dict[str, Any]]:
    """Create vega scales object pertaining to both the x and the y axis.

    Parameters
    ----------
    ax : Axes
        A matplotlib Axes instance which represents one (sub)plot in a matplotlib figure.
    """
    scales = []
    scales.append(_get_axis_scale_config(ax, "x"))
    scales.append(_get_axis_scale_config(ax, "y"))
    return scales


def _get_axis_scale_config(ax: Axes, axis_name: str) -> dict[str, Any]:
    """Provide a vega like scales object particular for one of the plotting axes.

    Note that in vega, this config also contains the fields reverse and zero.
    However, given that we specify the domain explicitly, these are not required here.

    Parameters
    ----------
    ax : Axes
        A matplotlib Axes instance which represents one (sub)plot in a matplotlib figure.
    axis_name: str
        Which axis the config should be made for, either "x" or "y".
    """
    scale: dict[str, Any] = {}
    scale["name"] = f"{axis_name.upper()}_scale"
    if axis_name == "x":
        scale["type"] = ax.get_xaxis().get_scale()
        scale["domain"] = [ax.get_xlim()[0].item(), ax.get_xlim()[1].item()]
        scale["range"] = "width"
    if axis_name == "y":
        scale["type"] = ax.get_yaxis().get_scale()
        scale["domain"] = [ax.get_ylim()[0].item(), ax.get_ylim()[1].item()]
        scale["range"] = "height"
    return scale


def _create_padding_object(fig: Figure) -> dict[str, float]:
    """Get the padding parameters for a vega viewconfiguration.

    Given that matplotlib gives the padding parameters as a fraction of the the figure width or height and
    vega gives it as absolute number of pixels we need to convert from the fraction to the number of pixels.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure. The top level container for all the plot elements.
    """
    fig_width_pixels, fig_height_pixels = fig.get_size_inches() * fig.dpi
    # contains also wspace and hspace but does not seem to be used by vega here.
    padding_obj = fig.subplotpars
    return {
        "left": (padding_obj.left * fig_width_pixels).item(),
        "top": ((1 - padding_obj.top) * fig_height_pixels).item(),
        "right": ((1 - padding_obj.right) * fig_width_pixels).item(),
        "bottom": (padding_obj.bottom * fig_height_pixels).item(),
    }


def _create_base_level_sdata_block(url: str) -> dict[str, Any]:
    """Create the vega json object for the SpatialData zarr store.

    Parameters
    ----------
    url : Path
        The location of the SpatialData zarr store.

    This config is to be added to the vega data field block.
    """
    return {
        "name": str(uuid4()),
        "url": url,
        "format": {"type": "SpatialData", "version": spatialdata.__version__},
    }


def _create_derived_data_block(
    ax: Axes, call: str, params: Params, base_uuid: str, cs: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create vega like data object for SpatialData elements.

    Each object for a SpatialData element contains an additional transform that
    is not entirely corresponding to the vega spec but aims to allow for retrieving
    the specific element and transforming it to a particular coordinate space.

    Parameters
    ----------
    call: str
        The render call from spatialdata plot, either render_images, render_labels, render_points
        or render_shapes, prefixed by n_ where n is the index of the render call starting from 0.
    params: Params
        The render parameters used in spatialdata-plot for the particular type of SpatialData
        element.
    base_uuid: str
        Unique identifier used to refer to the base level SpatialData zarr store in the vega
        like view configuration.
    cs: str
        The name of the coordinate system in which the SpatialData element was plotted.
    """
    data_object: dict[str, Any] = {}
    img_counter = 0

    data_object["name"] = params.element + "_" + str(uuid4())

    # TODO: think about versioning of individual spatialdata elements
    if "render_images" in call and isinstance(params, ImageRenderParams):
        data_object["format"] = {"type": "spatialdata_image", "version": 0.1}
        marks_object = _create_raster_image_marks_object(ax, call, params, data_object["name"], img_counter)
        img_counter += 1
    elif "render_labels" in call:
        data_object["format"] = {"type": "spatialdata_label", "version": 0.1}
        marks_object = {"a": 5}
    elif "render_points" in call:
        data_object["format"] = {"type": "spatialdata_point", "version": 0.1}
        marks_object = {"a": 5}
    elif "render_shapes" in call:
        data_object["format"] = {"type": "spatialdata_shape", "version": 0.1}
        marks_object = {"a": 5}
    else:
        raise ValueError(f"Unknown call: {call}")

    data_object["source"] = base_uuid
    data_object["transform"] = [{"type": "filter_element", "expr": params.element}, {"type": "filter_cs", "expr": cs}]

    # TODO: complete this part
    if "render_images" in call and isinstance(params, ImageRenderParams):  # second part to shut up mypy
        multiscale = "full" if not params.scale else params.scale
        data_object["transform"].append({"type": "filter_scale", "expr": multiscale})
        data_object["transform"].append({"type": "filter_channel", "expr": params.channel})
    return data_object, marks_object


def _create_raster_image_marks_object(
    ax: Axes, call: str, params: ImageRenderParams, element_uuid: str, counter: int
) -> dict[str, Any]:

    return {
        "type": "raster_image",
        "from": {"data": element_uuid},
        "zindex": ax.properties()["images"][counter].zorder,
        "encode": {"enter": {"opacity": {"value": ax.properties()["images"][counter].properties()["alpha"]}}},
    }


# def plotting_tree_dict_to_marks(plotting_tree_dict):
#     out = [] # caller will set { ..., "marks": out }
#     for pl_call_id, pl_call_params in plotting_tree_dict.items():
#         if pl_call_id.endswith("_render_images"):
#             for channel_index in pl_call_params["channel"]:
#                 out.append({
#                   "type": "raster_image",
#                   "from": {"data": sdata_element_to_uuid(pl_call_params["element"])},
#                   "zindex": pl_call_params["zorder"],
#                   "encode": {
#                       "opacity": { "value": pl_call_params.get("alpha") },
#                       "color": {"scale": get_scale_name(pl_call_params), "field": channel_index }
#                   }
#                 })
#         if pl_call_id.endswith("_render_shapes"):
#             out.append({
#                 "type": "shape",
#                 "from": {"data": sdata_element_to_uuid(pl_call_params["element"])},
#                 "zindex": pl_call_params["zorder"],
#                 "encode": {
#                     "fillOpacity": {"value": pl_call_params.get("fill_alpha")},
#                     "fillColor": get_shapes_color_encoding(pl_call_params),
#                     "strokeWidth": {"value": pl_call_params.get("outline_width")},
#                   # TODO: check whether this is the key used in the spatial plotting tree # TODO: what are the units?
#                     "strokeColor": {"value": pl_call_params.get("outline_color")},
#                     "strokeOpacity": {"value": pl_call_params.get("outline_alpha")},
#                 }
#             })
#         if pl_call_id.endswith("_render_points"):
#             out.append({
#                 "type": "point",
#                 "from": {"data": sdata_element_to_uuid(pl_call_params["element"])},
#                 "zindex": pl_call_params["zorder"],
#                 "encode": {
#                     "opacity": {"value": pl_call_params.get("alpha")},
#                     "color": get_shapes_color_encoding(pl_call_params),
#                     "size": {"value": pl_call_params.get("size")},
#                 }
#             })
#         if pl_call_id.endswith("_render_labels"):
#             out.append({
#                 "type": "raster_labels",
#                 "from": {"data": sdata_element_to_uuid(pl_call_params["element"])},
#                 "zindex": pl_call_params["zorder"],
#                 "encode": {
#                     "opacity": {"value": pl_call_params.get("alpha")},
#                     "fillColor": get_shapes_color_encoding(pl_call_params),
#                     "strokeColor": get_shapes_color_encoding(pl_call_params),
#                     "strokeWidth": {"value": pl_call_params.get("contour_px")},
#                     # TODO: check whether this is the key used in the spatial plotting tree
#                     "strokeOpacity": {"value": pl_call_params.get("outline_alpha")},
#                     # TODO: check whether this is the key used in the spatial plotting tree
#                     "fillOpacity": {"value": pl_call_params.get("fill_alpha")},
#                     # TODO: check whether this is the key used in the spatial plotting tree
#                 }
#             })


def _create_data_configs(
    plotting_tree: OrderedDict[str, Params], ax: Axes, cs: str, sdata_path: str
) -> list[dict[str, Any]]:
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
    data = []
    url = str(Path("sdata.zarr"))

    if sdata_path:
        url = sdata_path

    base_block = _create_base_level_sdata_block(url)
    data.append(base_block)
    for call, params in plotting_tree.items():
        data_object, marks_object = _create_derived_data_block(ax, call, params, base_block["name"], cs)
        data.append(data_object)

    return data


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
        axis_config["domainColor"] = mcolors.to_hex(axis_line_props["edgecolor"])[:-2]
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
    data_block = _create_data_configs(sdata.plotting_tree, ax, cs, sdata._path)

    axis_scales_block = _create_axis_scale_block(ax)
    axis_array = _create_axis_block(ax, axis_scales_block, fig.dpi)

    # TODO: check why attrs does not respect ordereddict when writing sdata
    return {
        "$schema": "https://spatialdata-plot.github.io/schema/viewconfig/v1.json",
        "height": fig.get_figheight() * fig.dpi,  # matplotlib uses inches, but vega uses absolute pixels
        "width": fig.get_figwidth() * fig.dpi,
        "padding": _create_padding_object(fig),
        "title": _create_title_config(ax, fig),
        "data": data_block,
        "scales": axis_scales_block,
        "axes": axis_array,
    }
