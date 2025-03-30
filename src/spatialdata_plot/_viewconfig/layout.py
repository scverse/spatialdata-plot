from matplotlib.figure import Figure


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
