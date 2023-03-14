from collections import OrderedDict
from collections.abc import Iterable
from typing import Optional, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata as sd
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from matplotlib.colors import ListedColormap, to_rgb
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from skimage.segmentation import find_boundaries
from spatial_image import SpatialImage

from spatialdata_plot.pl._categorical_utils import (
    add_colors_for_categorical_sample_annotation,
)

from ..accessor import register_spatial_data_accessor
from ..pp.utils import _get_region_key, _verify_plotting_tree_exists


@register_spatial_data_accessor("pl")
class PlotAccessor:
    """
    A class to provide plotting functions for `SpatialData` objects.

    Parameters
    ----------
    sdata : sd.SpatialData
        The `SpatialData` object to provide plotting functions for.

    Attributes
    ----------
    sdata : sd.SpatialData
        The `SpatialData` object to provide plotting functions for.

    Notes
    -----
    This class provides a number of methods that can be used to generate
    plots of the data stored in a `SpatialData` object. These methods are
    accessed via the `SpatialData.pl` accessor.

    Examples
    --------
    To plot the images in a `SpatialData` object, use the `plot_images`
    method:

    >>> sdata.pl.plot_images()

    To plot the labels in a `SpatialData` object, use the `plot_labels`
    method:

    >>> sdata.pl.plot_labels()

    To plot the points in a `SpatialData` object, use the `plot_points`
    method:

    >>> sdata.pl.plot_points()

    To plot the shapes in a `SpatialData` object, use the `plot_shapes`
    method:

    >>> sdata.pl.plot_shapes()

    To plot the table in a `SpatialData` object, use the `plot_table`
    method:

    >>> sdata.pl.plot_table()

    To plot the polygons in a `SpatialData` object, use the `plot_polygons`
    method:

    >>> sdata.pl.plot_polygons()

    """

    @property
    def sdata(self) -> sd.SpatialData:
        """The `SpatialData` object to provide plotting functions for."""
        return self._sdata

    @sdata.setter
    def sdata(self, sdata: sd.SpatialData) -> None:
        self._sdata = sdata

    def __init__(self, sdata: sd.SpatialData) -> None:
        self._sdata = sdata

    def _copy(
        self,
        images: Union[None, dict[str, Union[SpatialImage, MultiscaleSpatialImage]]] = None,
        labels: Union[None, dict[str, Union[SpatialImage, MultiscaleSpatialImage]]] = None,
        points: Union[None, dict[str, DaskDataFrame]] = None,
        shapes: Union[None, dict[str, GeoDataFrame]] = None,
        table: Union[None, AnnData] = None,
    ) -> sd.SpatialData:
        """Copy the current `SpatialData` object, optionally modifying some of its attributes.

        Parameters
        ----------
        images : Union[None, dict], optional
            A dictionary containing image data to replace the images in the
            original `SpatialData` object, or `None` to keep the original
            images. Defaults to `None`.
        labels : Union[None, dict], optional
            A dictionary containing label data to replace the labels in the
            original `SpatialData` object, or `None` to keep the original
            labels. Defaults to `None`.
        points : Union[None, dict], optional
            A dictionary containing point data to replace the points in the
            original `SpatialData` object, or `None` to keep the original
            points. Defaults to `None`.
        shapes : Union[None, dict], optional
            A dictionary containing shape data to replace the shapes in the
            original `SpatialData` object, or `None` to keep the original
            shapes. Defaults to `None`.
        table : Union[dict, AnnData], optional
            A dictionary or `AnnData` object containing table data to replace
            the table in the original `SpatialData` object, or `None` to keep
            the original table. Defaults to `None`.

        Returns
        -------
        sd.SpatialData
            A new `SpatialData` object that is a copy of the original
            `SpatialData` object, with any specified modifications.

        Notes
        -----
        This method creates a new `SpatialData` object with the same metadata
        and similar data as the original `SpatialData` object. The new object
        can be modified without affecting the original object.

        """
        return sd.SpatialData(
            images=self._sdata.images if images is None else images,
            labels=self._sdata.labels if labels is None else labels,
            points=self._sdata.points if points is None else points,
            shapes=self._sdata.shapes if shapes is None else shapes,
            table=self._sdata.table if table is None else table,
        )

    def _subplots(
        self, num_images: int, ncols: int = 4, width: int = 4, height: int = 3
    ) -> Union[plt.Figure, plt.Axes]:
        """Helper function to set up axes for plotting.

        Parameters
        ----------
        num_images : int
            Number of images to plot. Must be greater than 1.
        ncols : int, optional
            Number of columns in the subplot grid, by default 4
        width : int, optional
            Width of each subplot, by default 4

        Returns
        -------
        Union[plt.Figure, plt.Axes]
            Matplotlib figure and axes object.
        """
        # if num_images <= 1:
        # raise ValueError("Number of images must be greater than 1.")

        if num_images < ncols:
            nrows = 1
            ncols = num_images
        else:
            nrows, reminder = divmod(num_images, ncols)

            if nrows == 0:
                nrows = 1
            if reminder > 0:
                nrows += 1

        fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))

        if not isinstance(axes, Iterable):
            axes = [axes]

        # get rid of the empty axes
        # _ = [ax.axis("off") for ax in axes.flatten()[num_images:]]
        return fig, axes

    def render_labels(
        self,
        cell_key: str,
        color_key: Optional[Union[str, None]] = None,
        border_alpha: float = 1.0,
        border_color: Optional[Union[str, None]] = None,
        fill_alpha: float = 0.5,
        fill_color: Optional[Union[str, None]] = None,
        mode: str = "thick",
        palette: Optional[list[str]] = None,
        add_legend: bool = True,
    ) -> matplotlib.pyplot.Axes:
        """Plot cell labels for a scanpy object.

        Parameters
        ----------
        self : object
            The scanpy object.
        cell_key : str
            The name of the column in the table to use for labeling cells.
        color_key : str or None, optional (default: None)
            The name of the column in the table to use for coloring cells.
        border_alpha : float, optional (default: 1.0)
            The alpha value of the label border. Must be between 0 and 1.
        border_color : str or None, optional (default: None)
            The color of the border of the labels.
        fill_alpha : float, optional (default: 0.5)
            The alpha value of the fill of the labels. Must be between 0 and 1.
        fill_color : str or None, optional (default: None)
            The color of the fill of the labels.
        mode : str, optional (default: 'thick')
            The rendering mode of the labels. Must be one of 'thick', 'inner',
            'outer', or 'subpixel'.
        palette : list or None, optional (default: None)
            The color palette to use when coloring cells. If None, a default
            palette will be used.
        add_legend : bool, optional (default: True)
            Whether to add a legend to the plot.

        Returns
        -------
        matplotlib.pyplot.Axes
            The resulting plot axes.

        Raises
        ------
        TypeError
            If any of the parameters have an invalid type.
        ValueError
            If any of the parameters have an invalid value.
            If the provided cell_key or color_key is not a valid table column.
            If the provided mode is not one of 'thick', 'inner', 'outer', or
            'subpixel'.

        Notes
        -----
        This function plots cell labels for a spatialdata object. The cell labels are
        based on a column in the table, and can optionally be colored based on another
        column in the table. The resulting plot can be customized by specifying the
        alpha, color, and rendering mode of the labels, as well as whether to add a
        legend to the plot.
        """
        if not isinstance(cell_key, str):
            raise TypeError("Parameter 'cell_key' must be a string.")

        if cell_key not in self._sdata.table.obs.columns:
            raise ValueError(f"The provided cell_key '{cell_key}' is not a valid table column.")

        if color_key is not None:
            if not isinstance(color_key, (str, type(None))):
                raise TypeError("Parameter 'color_key' must be a string.")

            if color_key not in self._sdata.table.obs.columns:
                raise ValueError(f"The provided color_key '{color_key}' is not a valid table column.")

        if not isinstance(border_alpha, (int, float)):
            raise TypeError("Parameter 'border_alpha' must be a float.")

        if not (border_alpha <= 1 and border_alpha >= 0):
            raise ValueError("Parameter 'border_alpha' must be between 0 and 1.")

        if border_color is not None:
            if not isinstance(color_key, (str, type(None))):
                raise TypeError("If specified, parameter 'border_color' must be a string.")

        if not isinstance(fill_alpha, (int, float)):
            raise TypeError("Parameter 'fill_alpha' must be a float.")

        if not (fill_alpha <= 1 and fill_alpha >= 0):
            raise ValueError("Parameter 'fill_alpha' must be between 0 and 1.")

        if fill_color is not None:
            if not isinstance(fill_color, (str, type(None))):
                raise TypeError("If specified, parameter 'fill_color' must be a string.")

        valid_modes = ["thick", "inner", "outer", "subpixel"]
        if not isinstance(mode, str):
            raise TypeError("Parameter 'mode' must be a string.")

        if mode not in valid_modes:
            raise ValueError("Parameter 'mode' must be one of 'thick', 'inner', 'outer', 'subpixel'.")

        if not isinstance(add_legend, bool):
            raise TypeError("Parameter 'add_legend' must be a boolean.")

        # self._sdata = _verify_plotting_tree_exists(self._sdata)

        # get current number of steps to create a unique key
        table = self._sdata.table.copy()
        add_colors_for_categorical_sample_annotation(table, cell_key, table.obs[color_key], palette=palette)

        sdata = self._copy(table=table)
        sdata = _verify_plotting_tree_exists(sdata)
        n_steps = len(sdata.plotting_tree.keys())
        sdata.plotting_tree[f"{n_steps+1}_render_labels"] = {
            "cell_key": cell_key,
            "color_key": color_key,
            "border_alpha": border_alpha,
            "border_color": border_color,
            "fill_alpha": fill_alpha,
            "fill_color": fill_color,
            "mode": mode,
            "palette": palette,
            "add_legend": add_legend,
        }

        return sdata

    def _render_labels(
        self,
        params: dict[str, Union[str, int, float]],
        key: str,
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.SubplotBase,
    ) -> None:
        region_key = _get_region_key(self._sdata)

        # subset table to only the entires specified by 'key'
        table = self._sdata.table.obs
        table = table[table[region_key] == key]

        # If palette is not None, table.uns contains the relevant vector
        if f"{params['cell_key']}_colors" in self._sdata.table.uns.keys():
            colors = [to_rgb(c) for c in self._sdata.table.uns[f"{params['cell_key']}_colors"]]
            colors = [tuple(list(c) + [1]) for c in colors]

        groups = self._sdata.table.obs[params["color_key"]].unique()
        group_to_color = pd.DataFrame({params["color_key"]: groups, "color": colors})

        segmentation = self._sdata.labels[key].values

        for group in groups:
            vaid_cell_ids = table[table[params["color_key"]] == group][params["cell_key"]].values

            # define all out-of-group cells as background
            in_group_mask = segmentation.copy()
            in_group_mask[~np.isin(segmentation, vaid_cell_ids)] = 0

            # get correct color for the group
            group_color = list(group_to_color[group_to_color[params["color_key"]] == group].color.values[0])

            if params["fill_alpha"] != 0:
                infill_mask = in_group_mask > 0

                fill_color = group_color.copy()
                fill_color[-1] = params["fill_alpha"]
                colors = [[0, 0, 0, 0], fill_color]  # add transparent for bg

                ax.imshow(infill_mask, cmap=ListedColormap(colors), interpolation="nearest")

            if params["border_alpha"] != 0:
                border_mask = find_boundaries(in_group_mask, mode=params["mode"])
                border_mask = np.ma.masked_array(in_group_mask, ~border_mask)

                border_color = group_color.copy()
                border_color[-1] = params["border_alpha"]

                ax.imshow(border_mask, cmap=ListedColormap([border_color]), interpolation="nearest")

        if params["add_legend"]:
            patches = []
            for group, color in group_to_color.values:
                patches.append(mpatches.Patch(color=color, label=group))

            fig.legend(handles=patches, bbox_to_anchor=(0.9, 0.9), loc="upper left", frameon=False)

        ax.set_title(key)
        ax.set_xlabel("spatial1")
        ax.set_ylabel("spatial2")
        ax.set_xticks([])
        ax.set_yticks([])

    def show(
        self,
        ax: Union[plt.Axes, None] = None,
        ncols: int = 4,
        width: int = 4,
        height: int = 3,
        bg_color: str = "black",
        **kwargs: str,
    ) -> sd.SpatialData:
        """Plot the images in the SpatialData object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object to plot on. If None, a new figure is created.
            Works only if there is one image in the SpatialData object.
        ncols : int, optional
            Number of columns in the figure. Default is 4.
        width : int, optional
            Width of each subplot. Default is 4.
        height : int, optional
            Height of each subplot. Default is 3.

        Returns
        -------
        sd.SpatialData
            A SpatialData object.
        """
        if not isinstance(ax, (matplotlib.pyplot.Axes, type(None))):
            raise TypeError("If provided, Parameter 'ax' must be of type 'matplotlib.pyplot.Axes.")

        if not isinstance(ncols, int):
            raise TypeError("Parameter 'ncols' must be an integer.")

        if not ncols >= 1:
            raise ValueError("Parameter 'ncols' must be at least 1.")

        if not isinstance(width, int):
            raise ValueError("Parameter 'width' must be an integer.")

        if not width > 0:
            raise ValueError("Parameter 'width' must be greater than 0.")

        if not isinstance(height, int):
            raise TypeError("Parameter 'height' must be an integer.")

        if not height > 0:
            raise ValueError("Parameter 'height' must be greater than 0.")

        if not isinstance(bg_color, str):
            raise TypeError("If specified, parameter 'bg_color' must be a string.")

        image_data = self._sdata.images
        num_images = len(image_data)

        if num_images == 0:
            raise ValueError("No images found in the SpatialData object.")

        # Evaluate execution tree for plotting

        valid_commands = [
            "get_images",
            "get_channels",
            "get_shapes",  # formerly polygons
            "get_bb",
            "render_images",
            "render_channels",
            "render_shapes",
            "render_points",
            "render_labels",
        ]

        if not hasattr(self._sdata, "plotting_tree") or len(self._sdata.plotting_tree.keys()) == 0:
            raise ValueError("No operations have been performed yet.")

        elif len(self._sdata.plotting_tree.keys()) > 0:
            render_cmds = OrderedDict()

            for cmd, params in self._sdata.plotting_tree.items():
                # strip prefix from cmd and verify it's valid
                cmd = "_".join(cmd.split("_")[1:])

                if cmd not in valid_commands:
                    raise ValueError(f"Command {cmd} is not valid.")

                elif "render" in cmd:
                    # verify that rendering commands have been called before
                    render_cmds[cmd] = params

            if len(render_cmds.keys()) == 0:
                raise TypeError("Please specify what to plot using the 'render_*' functions before calling 'imshow().")

            # set up canvas
            num_images = len(self._sdata.images.keys())
            fig, axs = self._subplots(num_images, ncols, width, height)

            # Set background color
            for idx, ax in enumerate(axs):
                key = list(self._sdata.labels.keys())[idx]
                ax.imshow(self._sdata.labels[key].values, cmap=ListedColormap([bg_color]))

            # go through tree
            for cmd, params in render_cmds.items():
                if cmd == "render_images":
                    # self._render_images(params, axs)
                    pass

                elif cmd == "render_channels":
                    # self._render_channels(params, axs)
                    pass

                elif cmd == "render_shapes":
                    # self._render_shapes(params, axs)
                    pass

                elif cmd == "render_points":
                    # for ax in axs:
                    # self._render_points(params, ax)
                    pass

                elif cmd == "render_labels":
                    for idx, ax in enumerate(axs):
                        key = list(self._sdata.labels.keys())[idx]
                        self._render_labels(params=params, key=key, fig=fig, ax=ax)

                else:
                    raise NotImplementedError(f"Command '{cmd}' is not supported.")

        return fig, axs

    def scatter(
        self,
        x: str,
        y: str,
        color: Union[str, None] = None,
        ax: plt.Axes = None,
        ncols: int = 4,
        width: int = 4,
        height: int = 3,
        **kwargs: str,
    ) -> sd.SpatialData:
        """Plots a scatter plot of observations in the table of a SpatialData object.

        Parameters
        ----------
        x : str
            Column name of the x-coordinates.
        y : str
            Column name of the y-coordinates.
        color : str, optional
            Column name of the color, by default None.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object to plot on. If None, a new figure is created.
            Only works if there is one image in the SpatialData object.
        ncols : int, optional
            Number of columns in the figure. Default is 4.
        width : int, optional
            Width of each subplot. Default is 4.
        height : int, optional
            Height of each subplot. Default is 3.
        kwargs : dict
            Additional keyword arguments to pass to the scatter plot.

        Returns
        -------
        sd.SpatialData
            A SpatialData object.
        """
        image_data = self._sdata.images
        region_key = self._sdata.pp.get_region_key()
        regions = self._sdata.table.obs[region_key].unique().tolist()

        region_mapping = {k.split("/")[-1]: k for k in regions}

        for k in image_data.keys():
            if k not in region_mapping.keys():
                del region_mapping[k]

        num_images = len(region_mapping)

        if num_images == 1:
            ax = ax or plt.gca()
            # TODO: support for labels instead of colors (these can be cell types for example) => Automatic coloring.
            if color is not None:
                kwargs["c"] = self._sdata.table.obs[color]
            ax.scatter(self._sdata.table.obs[x], self._sdata.table.obs[y], **kwargs)
            key = list(image_data.keys())[0]
            ax.set_title(key)
            ax.margins(x=0.01)
            ax.margins(y=0.01)

        if num_images > 1:
            fig, axes = self._subplots(num_images, ncols, width, height)

            for i, (ax, (k, v)) in enumerate(zip(np.ravel(axes), region_mapping.items())):
                if i < num_images:
                    sub_table = self._sdata.table.obs[self._sdata.table.obs[region_key] == v]

                    if color is not None:
                        kwargs["c"] = sub_table[color]

                    ax.scatter(sub_table[x], sub_table[y], **kwargs)
                    ax.set_title(k)
                    ax.margins(x=0.01)
                    ax.margins(y=0.01)

        return self._sdata
