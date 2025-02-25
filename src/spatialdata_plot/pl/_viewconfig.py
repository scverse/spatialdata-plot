from __future__ import annotations

import re
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
    CmapParams,
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


def _get_colorbar_orient(ax, cbar):
    """
    Determine the Vega legend orientation based on the position of a Matplotlib colorbar relative to the main plot.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The main plot axes containing the actual plotted data.
    cbar : Colorbar
        The matplotlib colorbar object.

    Returns
    -------
    str
        Vega `legend.orient` value ('top', 'bottom', 'left', 'right', or 'overlapping').
    """
    main_bbox = ax.get_position()  # Main plot bounding box
    cbar_bbox = cbar.ax.get_position()  # Colorbar bounding box

    if cbar_bbox.y1 <= main_bbox.y0:
        return "bottom"  # Colorbar is below
    if cbar_bbox.y0 >= main_bbox.y1:
        return "top"  # Colorbar is above
    if cbar_bbox.x1 <= main_bbox.x0:
        return "left"  # Colorbar is to the left
    if cbar_bbox.x0 >= main_bbox.x1:
        return "right"  # Colorbar is to the right

    return "overlapping"  # Unusual case


def _create_colorscale_image(cmap_params: CmapParams) -> tuple[list[dict[str, Any]], float]:
    cmaps = [cmap_params.cmap] if not isinstance(cmap_params, list) else [param.cmap for param in cmap_params]
    cmaps = cmaps[0] if isinstance(cmaps[0], list) else cmaps  # Happens if palette is specified as list of strings
    color_scale_array: list[dict[str, Any]] = []
    for cmap in cmaps:
        # TODO: check why listedcolormap is only passed on when we specify channel.
        if isinstance(cmap, mcolors.ListedColormap):
            type_scale = "linear"
            if cmap.name == "from_list":  # default name when cmap is custom.
                # TODO: complete this for all types of cmaps
                pass
            else:
                color_range = {"scheme": cmap.name, "count": cmap.N}
        elif isinstance(cmap, mcolors.LinearSegmentedColormap):
            type_scale = "linear"
            if cmap.name == "custom_colormap":
                pass
            else:
                color_range = {"scheme": cmap.name, "count": cmap.N}
        color_scale_object = {
            "name": f"color_{str(uuid4())}",
            "type": type_scale,
            # The domain in matplotlib seems to be 0-1 always for images, but for example for napari should be
            # interpreted as relative
            "domain": [0, 1],
            "range": color_range,
        }

        color_scale_array.append(color_scale_object)

    return color_scale_array


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


def _create_legend_title_config(title_obj, dpi):
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


def _create_colorbar_legend(fig, color_scale_array):
    legend_array: list[dict[str, Any]] = []
    axes = fig.get_axes()
    for col_config in color_scale_array:
        cbar = None
        for ax in axes:
            cbar = getattr(ax.properties()["axes_locator"], "_cbar") if ax.properties()["axes_locator"] else None
            if not cbar:
                continue
            if cbar.cmap.name != col_config["range"]["scheme"]:
                cbar = None
                continue
            break

        if not cbar:
            continue

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
            "gradientLength": gradient_length,
            "gradientOpacity": cbar.cmap._lut[-0][-1],
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
            "legendX": cbar.ax.get_position().bounds[0] * fig.get_figwidth() * fig.dpi,
            "legendY": (1 - (cbar.ax.get_position().bounds[1] + cbar.ax.get_position().bounds[-1]))
            * fig.get_figheight()
            * fig.dpi,
            "zindex": axis_props["zorder"],
        }
        if legend_title_object["title"] != "":
            legend_object.update(legend_title_object)
        legend_array.append(legend_object)
    return legend_array


def _create_derived_data_block(
    fig, ax: Axes, call: str, params: Params, base_uuid: str, cs: str, call_count: int
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
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
    marks_object: dict[str, Any] = {}
    color_scale_array: list[dict[str, Any]] = []
    legend_array: list[dict[str, Any]] = []

    data_object["name"] = params.element + "_" + str(uuid4())

    # TODO: think about versioning of individual spatialdata elements
    if "render_images" in call and isinstance(params, ImageRenderParams):
        data_object["format"] = {"type": "spatialdata_image", "version": 0.1}
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

    if "render_images" in call and isinstance(params, ImageRenderParams):  # second part to shut up mypy
        multiscale = "full" if not params.scale else params.scale
        data_object["transform"].append({"type": "filter_scale", "expr": multiscale})
        data_object["transform"].append({"type": "filter_channel", "expr": params.channel})
        # Use isinstance because of possible 0 value
        norm = params.cmap_params.norm if not isinstance(params.cmap_params, list) else params.cmap_params[0].norm
        if isinstance(vmin := norm.vmin, float) and isinstance(vmax := norm.vmax, float):

            if norm.clip:
                formula = f"clamp((datum.value - {vmin}) / ({vmax} - {vmin}), 0, 1)"
            else:
                formula = f"(datum.value - {vmin}) / ({vmax} - {vmin})"
            data_object["transform"].append({"type": "formula", "expr": formula, "as": str(uuid4())})

        color_scale_array = _create_colorscale_image(params.cmap_params)
        legend_array = _create_colorbar_legend(fig, color_scale_array)
        marks_object = _create_raster_image_marks_object(ax, call, params, data_object, call_count, color_scale_array)
    return data_object, marks_object, color_scale_array, legend_array


def _create_raster_image_marks_object(
    ax: Axes,
    call: str,
    params: ImageRenderParams,
    data_object: dict[str, Any],
    call_count: int,
    color_scale_array: list[dict[str, Any]],
    # image_alpha : float
) -> dict[str, Any]:
    if len(color_scale_array) == 1:
        fill_color = [{"scale": color_scale_array[0]["name"], "value": "intensity_value"}]
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


def strip_call(s: str) -> str:
    """Strip leading digit and underscore from call."""
    return re.sub(r"^\d+_", "", s)


def _create_data_configs(
    plotting_tree: OrderedDict[str, Params], fig, ax: Axes, cs: str, sdata_path: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
    color_scale_array_full = []
    legend_array_full = []
    url = str(Path("sdata.zarr"))

    if sdata_path:
        url = sdata_path

    base_block = _create_base_level_sdata_block(url)
    data_array.append(base_block)

    counters = {"render_images": 0, "render_labels": 0, "render_points": 0, "render_shapes": 0}
    for call, params in plotting_tree.items():
        call = strip_call(call)
        data_object, marks_object, color_scale_array, legend_array = _create_derived_data_block(
            fig, ax, call, params, base_block["name"], cs, counters[call]
        )
        data_array.append(data_object)
        marks_array.append(marks_object)
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
    data_array, marks_array, color_scale_array, legend_array = _create_data_configs(
        sdata.plotting_tree, fig, ax, cs, sdata._path
    )

    scales_array = _create_axis_scale_block(ax)
    axis_array = _create_axis_block(ax, scales_array, fig.dpi)

    scales = axis_array + color_scale_array if len(color_scale_array) > 0 else axis_array
    # TODO: check why attrs does not respect ordereddict when writing sdata
    viewconfig = {
        "$schema": "https://spatialdata-plot.github.io/schema/viewconfig/v1.json",
        "height": fig.get_figheight() * fig.dpi,  # matplotlib uses inches, but vega uses absolute pixels
        "width": fig.get_figwidth() * fig.dpi,
        "padding": _create_padding_object(fig),
        "title": _create_title_config(ax, fig),
        "data": data_array,
        "scales": scales,
    }

    viewconfig["axes"] = axis_array
    if len(legend_array) > 0:
        viewconfig["legend"] = legend_array
    viewconfig["marks"] = marks_array

    return viewconfig


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
