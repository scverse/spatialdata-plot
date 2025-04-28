from typing import Any, Literal
from uuid import uuid4

import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from pydantic import BaseModel

from spatialdata_plot.pl.render_params import CmapParams, LabelsRenderParams, PointsRenderParams, ShapesRenderParams


def create_axis_scale_array(ax: Axes) -> list[dict[str, Any]]:
    """Create vega scales object pertaining to both the x and the y axis.

    Parameters
    ----------
    ax : Axes
        A matplotlib Axes instance which represents one (sub)plot in a matplotlib figure.

    Returns
    -------
    scales: list[dict[str, Any]]
        An array containing individual scales objects with the parameters for the x and y axis of a plot.
    """
    scales = []
    scales.append(get_axis_scale_object(ax, "x"))
    scales.append(get_axis_scale_object(ax, "y"))
    return scales


def get_axis_scale_object(ax: Axes, axis_name: str) -> dict[str, Any]:
    """Provide a vega like scales object particular for one of the plotting axes.

    Note that in vega, this config also contains the fields reverse and zero.
    However, given that we specify the domain explicitly, these are not required here.

    Parameters
    ----------
    ax : Axes
        A matplotlib Axes instance which represents one (sub)plot in a matplotlib figure.
    axis_name: str
        Which axis the config should be made for, either "x" or "y".

    Returns
    -------
    scale: dict[str, Any]
        A vega like scale object containing the type of scale, the domain (xlim or ylim) and the range (`width` for
        x axis and `height` for y axis).
    """
    scale_type = ax.get_xaxis().get_scale() if axis_name == "x" else ax.get_yaxis().get_scale()
    domain = (
        [ax.get_xlim()[0].item(), ax.get_xlim()[1].item()]
        if axis_name == "x"
        else [ax.get_ylim()[0].item(), ax.get_ylim()[1].item()]
    )

    return {
        "name": f"{axis_name.upper()}_scale",
        "type": scale_type,
        "domain": domain,
        "range": "width" if axis_name == "x" else "height",
    }


def _generate_color_scale_object(
    name: str,
    type_scale: str,
    domain: list[str] | dict[str, Any],
    color_range: list[str] | dict[str, str | int],
) -> dict[str, Any]:
    """Create vega like color scale object.

    This function is a helper function to generate any kind of color scale object, whether linear or categorical /
    ordinal.

    Parameters
    ----------
    name : str
        The name by which others parts of the view configuration can refer to the color scale object.
    type_scale : str
        The type of color scale. Usually either `linear` or `ordinal`.
    domain : list[str] | dict[str, str]
        The domain of the color scale, meaning the actual values to which a color must be mapped. Can be either
        an array of strings or if the `type` is `linear` it can be an object containing  `data` and `field` keys, which
        are derived from a data object.
    color_range : list[str] | dict[str, str | int]
        The range of the color scale, meaning the colors which are mapped to a particular value given by the `domain`.
        Either an object containing the `scheme` with as value the colormap name and the `count` stating the number
        of colors in the colormap. Otherwise, it is an array of hex colors represented by strings.

    Returns
    -------
    A vega-like color scale object.
    """
    return {
        "name": name,
        "type": type_scale,
        "domain": domain,
        "range": color_range,
    }


def _create_random_colorscale(data_id: str) -> dict[str, Any]:
    """Create a vega-like colorscale for random colors.

    There is no way currently to create a vega color scale object with random colors without serializing those. Need to
    find a way to agree on this.

    Parameters
    ----------
    data_id: str
        The name of the derived data object that pertains to a spatialdata element for which a color scale array object
        is created.

    Returns
    -------
    A vega like color scale object for random color assignment.
    """
    return _generate_color_scale_object(f"color_{uuid4()}", "ordinal", {"data": data_id, "field": "value"}, ["random"])


def create_categorical_colorscale(color_mapping: dict[str, str]) -> list[dict[str, Any]]:
    """Create a categorical Vega-like color scale array.

    Parameters
    ----------
    color_mapping: dict[str, str]
        A mapping of individual values in usually a table column to their corresponding hex color in the visualization.

    Returns
    -------
    A vega like color scale array containing in this case one color scale object.
    """
    return [
        _generate_color_scale_object(
            f"color_{uuid4()}", "ordinal", list(color_mapping.keys()), list(color_mapping.values())
        )
    ]


def _process_colormap(cmap: CmapParams) -> dict[str, Any]:
    """Process colormap to return a Vega color range dictionary.

    cmap: CmapParams
        An instance of CmapParams containing information pertaining colormap used and normalization applied.

    Returns
    -------
    The `range` value of a vega like color scale object.
    """
    if isinstance(cmap, mcolors.ListedColormap | mcolors.LinearSegmentedColormap):
        if cmap.name in {"from_list", "custom_colormap"}:
            # TODO: Handle custom colormap logic
            return {}
        if cmap.name.startswith("#"):
            color = mcolors.to_hex(cmap.name)
            for name, hex_val in mcolors.CSS4_COLORS.items():
                if color.lower() == hex_val.lower():
                    cmap.name = name

        return {"scheme": cmap.name, "count": cmap.N}
    return {}


def create_colorscale_array_points_shapes_labels(
    coloring: dict[str, str] | Literal["continuous"] | str,
    params: PointsRenderParams | ShapesRenderParams | LabelsRenderParams,
    data_object: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create a vega like colorscale array based on the colormap parameters for points or shapes.

    Parameters
    ----------
    coloring: dict[str, str] | str | None
        Either a categorical mapping of values to hex color strings or the literal `continuous` in case of a
        numerical column being used to determine the color of SpatialData shapes or points.
    params: PointsRenderParams | ShapesRenderParams
        The render parameters for a given SpatialData points or shapes element.
    data_object: dict[str, Any]
        A vega like data object pertaining to a Spatialdata points or shapes element
        to which the color scale is applied.

    Returns
    -------
    A vega like colorscale array containing in this case one color scale object pertaining to a SpatialData points
    or shapes element.
    """
    color_scale_object = {"name": f"color_{uuid4()}"}

    if isinstance(coloring, dict):
        color_scale_object.update(
            _generate_color_scale_object(
                color_scale_object["name"],
                "ordinal",
                list(coloring.keys()),
                [mcolors.to_hex(col, keep_alpha=False) for col in coloring.values()],
            )
        )
    elif coloring == "continuous":
        if isinstance(params, LabelsRenderParams):
            field = data_object["transform"][-1].get("as") or params.color
        else:
            field = data_object["transform"][-1].get("as") or params.col_for_color
        color_scale_object.update(
            _generate_color_scale_object(
                color_scale_object["name"],
                "linear",
                {"data": data_object["name"], "field": field},
                _process_colormap(params.cmap_params.cmap),
            )
        )
    elif coloring == "random":
        color_scale_object.update(
            _create_random_colorscale(
                data_object["name"],
            )
        )

    return [color_scale_object]


def create_colorscale_array_image(
    cmap_params: list[CmapParams] | CmapParams, data_id: str, field: list[str] | list[int] | int | str | None
) -> list[dict[str, Any]]:
    """Create a Vega-like color scale array for a SpatialData image element.

    cmap_params: list[CmapParams] | CmapParams
        The colormap parameters used to color a particular SpatialData image element.
    data_id: str
        The `name` in the vega like derived data object pertaining to a SpatialData image element.
    field: list[str] | list[int] | int | str | None
        The part of the SpatialData image or labels element to which apply the color. If `string` or
        `list` of `strings` it pertains to individual channel names, if `int` or `list` of `int`
        it pertains to the index or indices of image channel(s).

    Returns
    -------
    A color scale array containing vega-like color scale objects pertaining to a SpatialData image element.
    """
    color_scale_array = []
    cmaps = [param.cmap for param in cmap_params] if isinstance(cmap_params, list) else [cmap_params.cmap]
    cmaps = cmaps[0] if isinstance(cmaps[0], list) else cmaps

    for index, cmap in enumerate(cmaps):

        color_range = _process_colormap(cmap)
        field_val = field
        if isinstance(field, int | list) or (field is None and len(cmaps) != 1):
            field_val = f"channel_{index}"
        if isinstance(field, list) and len(field) == 1:
            field_val = "value"

        field_val = field_val or "value"

        color_scale_array.append(
            _generate_color_scale_object(
                f"color_{uuid4()}", "linear", {"data": data_id, "field": field_val}, color_range
            )
        )

    return color_scale_array


class AxisScaleObject(BaseModel):
    """Represents a scale configuration for a single axis in a vega-like format.

    Attributes
    ----------
    name : str
        The name of the scale, typically formatted as "X_scale" or "Y_scale".
    type : Literal["linear", "log", "symlog", "logit"]
        The type of scale used for the axis, matching common matplotlib scale types.
    domain : list[float]
        The domain of the axis, defined by the minimum and maximum values.
    range : Literal["width", "height"]
        The mapping of the axis to the corresponding plot dimension.
    """

    name: str
    type: Literal[
        "asinh", "function", "functionlog", "linear", "log", "logit", "symlog"
    ]  # Common matplotlib scale types
    domain: list[float]
    range: Literal["width", "height"]

    @classmethod
    def get_axis_scale_object_from_mpl(cls, ax: Axes, axis_name: str) -> "AxisScaleObject":
        """Generate a scale object for a given axis in a vega-like format.

        Parameters
        ----------
        ax : Axes
            A matplotlib Axes instance representing a subplot.
        axis_name : str
            The axis to configure, either "x" or "y".

        Returns
        -------
        AxisScale
            A validated scale configuration for the specified axis.
        """
        scale_type = ax.get_xaxis().get_scale() if axis_name == "x" else ax.get_yaxis().get_scale()
        domain = (
            [ax.get_xlim()[0].item(), ax.get_xlim()[1].item()]
            if axis_name == "x"
            else [ax.get_ylim()[0].item(), ax.get_ylim()[1].item()]
        )

        return cls(
            name=f"{axis_name.upper()}_scale",
            type=scale_type,
            domain=domain,
            range="width" if axis_name == "x" else "height",
        )


class AxisScaleArray(BaseModel):
    """Represents an array of AxisScaleObject instances."""

    scales: list[AxisScaleObject]

    @classmethod
    def create_axis_scale_array_from_mpl(cls, ax: Axes) -> "AxisScaleArray":
        """Create a list of scale objects for both the x and y axes.

        Parameters
        ----------
        ax : Axes
            A matplotlib Axes instance representing a subplot.

        Returns
        -------
        list[AxisScale]
            A list containing scale configurations for both x and y axes.
        """
        return cls(
            scales=[
                AxisScaleObject.get_axis_scale_object_from_mpl(ax, "x"),
                AxisScaleObject.get_axis_scale_object_from_mpl(ax, "y"),
            ]
        )
