from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from spatialdata_plot._viewconfig.misc import VegaAlignment


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
