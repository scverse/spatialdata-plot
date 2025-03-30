from typing import Any, Literal

from matplotlib.axes import Axes
from pydantic import BaseModel


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
