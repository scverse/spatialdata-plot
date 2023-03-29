from collections import OrderedDict
from typing import Callable, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata_plot.pl._categorical_utils import (
    add_colors_for_categorical_sample_annotation,
)

from ..accessor import register_spatial_data_accessor
from ..pp.utils import _get_instance_key, _get_region_key, _verify_plotting_tree_exists
from .render import _render_channels, _render_images, _render_labels
from .utils import _get_random_hex_colors, _get_subplots


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
        sdata = sd.SpatialData(
            images=self._sdata.images if images is None else images,
            labels=self._sdata.labels if labels is None else labels,
            points=self._sdata.points if points is None else points,
            shapes=self._sdata.shapes if shapes is None else shapes,
            table=self._sdata.table if table is None else table,
        )
        sdata.plotting_tree = self._sdata.plotting_tree if hasattr(self._sdata, "plotting_tree") else OrderedDict()

        return sdata

    def render_images(
        self,
        palette: Optional[list[str]] = None,
        trans_fun: Optional[Callable[[xr.DataArray], xr.DataArray]] = None,
    ) -> matplotlib.pyplot.Axes:
        """Render images for a scanpy object.

        Parameters
        ----------
        self : object
            The scanpy object.
        palette : list[str], optional (default: None)
            A list of colors to use for rendering the images. If `None`, a
            random palette will be generated.
        fun : callable, optional (default: None)
            A function to apply to the images before rendering. If `None`, no
            function will be applied.

        Returns
        -------
        matplotlib.pyplot.Axes
            The axes object containing the rendered images.

        """
        sdata = self._copy()
        sdata = _verify_plotting_tree_exists(sdata)
        n_steps = len(sdata.plotting_tree.keys())
        sdata.plotting_tree[f"{n_steps+1}_render_images"] = {
            "palette": palette,
            "trans_fun": trans_fun,
        }

        return sdata

    def render_channels(
        self,
        channels: Union[list[str], list[int]],
        colors: list[str],
        clip: bool = True,
        normalize: bool = True,
        background: str = "black",
        pmin: float = 3.0,
        pmax: float = 99.8,
    ) -> sd.SpatialData:
        """Renders selected channels.

        Parameters:
        -----------
        self: object
            The sdata object
        channels: Union[List[str], List[int]]
            The channels to plot
        colors: List[str]
            The colors for the channels

        """
        if not isinstance(channels, list):
            raise TypeError("Parameter 'channels' must be a list.")

        if not isinstance(colors, list):
            raise TypeError("Parameter 'colors' must be a list.")

        if not isinstance(clip, bool):
            raise TypeError("Parameter 'clip' must be a bool.")

        if not isinstance(normalize, bool):
            raise TypeError("Parameter 'normalize' must be a bool.")

        if not isinstance(background, str):
            raise TypeError("Parameter 'background' must be a str.")

        if not isinstance(pmin, float):
            raise TypeError("Parameter 'pmin' must be a str.")

        if not isinstance(pmax, float):
            raise TypeError("Parameter 'pmax' must be a str.")

        sdata = self._copy()
        sdata = _verify_plotting_tree_exists(sdata)
        n_steps = len(sdata.plotting_tree.keys())

        sdata.plotting_tree[f"{n_steps+1}_render_channels"] = {
            "channels": channels,
            "colors": colors,
            "clip": clip,
            "normalize": normalize,
            "background": background,
            "pmin": pmin,
            "pmax": pmax,
        }

        return sdata

    def render_labels(
        self,
        cell_key: Optional[Union[str, None]] = None,
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
        if cell_key is not None:
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

        sdata = self._copy()
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

        if not hasattr(self._sdata, "plotting_tree") or len(self._sdata.plotting_tree.keys()) == 0:
            raise ValueError("No operations have been performed yet.")

        # copy the SpatialData object so we don't modify the original
        plotting_tree = self._sdata.plotting_tree
        sdata = self._copy()
        num_images = len(sdata.images.keys())

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

        if len(plotting_tree.keys()) > 0:
            render_cmds = OrderedDict()

            for cmd, params in plotting_tree.items():
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
            if ax is None:
                num_images = len(sdata.images.keys())
                _, axs = _get_subplots(num_images, ncols, width, height)
            elif isinstance(ax, matplotlib.pyplot.Axes):
                axs = np.array([ax])
            elif isinstance(ax, list):
                axs = ax

            # Set background color
            for _, ax in enumerate(axs.flatten()):
                ax.set_facecolor(bg_color)
                # key = list(sdata.labels.keys())[idx]
                # ax.imshow(sdata.labels[key].values, cmap=ListedColormap([bg_color]))

            # go through tree
            for cmd, params in render_cmds.items():
                keys = list(sdata.images.keys())
                if cmd == "render_images":
                    for key, ax in zip(keys, axs.flatten()):
                        _render_images(sdata=sdata, params=params, key=key, ax=ax)

                elif cmd == "render_channels":
                    # self._render_channels(params, axs)
                    for key, ax in zip(keys, axs.flatten()):
                        _render_channels(sdata=sdata, key=key, ax=ax, **params)

                elif cmd == "render_shapes":
                    # self._render_shapes(params, axs)
                    pass

                elif cmd == "render_points":
                    # for ax in axs:
                    # self._render_points(params, ax)
                    pass

                elif cmd == "render_labels":
                    if (
                        sdata.table is not None
                        and isinstance(params["color_key"], str)
                        and isinstance(params["cell_key"], str)
                    ):
                        # If we have a table and proper keys, generate categolrical
                        # colours which are stored in the 'uns' of the table.

                        add_colors_for_categorical_sample_annotation(
                            adata=sdata.table,
                            key=params["cell_key"],
                            vec=sdata.table.obs[params["color_key"]],
                            palette=params["palette"],
                        )

                    else:
                        # If any of the previous conditions are not met, generate random
                        # colors for each cell id

                        if sdata.table is not None:
                            # annoying case since number of cells in labels can be
                            # different from number of cells in table. So we just use
                            # the index and randomise colours for it

                            # has a table, so it has a region key
                            region_key = _get_region_key(sdata)

                            cell_ids_per_label = {}
                            for key in list(sdata.labels.keys()):
                                cell_ids_per_label[key] = len(sdata.table.obs.query(f"{region_key} == '{key}'"))

                            region_key = _get_region_key(sdata)
                            instance_key = _get_instance_key(sdata)
                            params["cell_key"] = instance_key
                            params["color_key"] = instance_key
                            params["add_legend"] = False
                            # TODO(ttreis) log the decision not to display a legend

                            distinct_cells = len(sdata.table.obs[params["color_key"]].unique())

                        elif sdata.table is None:
                            # No table, create one

                            cell_ids_per_label = {}
                            for key in list(sdata.labels.keys()):
                                cell_ids_per_label[key] = sdata.labels[key].values.max()

                            region_key = "tmp_label_id"
                            instance_key = "tmp_cell_id"
                            params["cell_key"] = instance_key
                            params["color_key"] = instance_key
                            params["add_legend"] = False
                            # TODO(ttreis) log the decision not to display a legend

                            tmp_table = pd.DataFrame(
                                {
                                    region_key: [k for k, v in cell_ids_per_label.items() for _ in range(v)],
                                    instance_key: [i for _, v in cell_ids_per_label.items() for i in range(v)],
                                }
                            )

                            distinct_cells = max(list(cell_ids_per_label.values()))

                        if sdata.table is not None:
                            print("Plotting a lot of cells with random colors, might take a while...")
                            sdata.table.uns[f"{instance_key}_colors"] = _get_random_hex_colors(distinct_cells)

                        elif sdata.table is None:
                            table = AnnData(X=np.zeros((tmp_table.shape[0], 1)), obs=tmp_table)
                            table.uns = {}
                            table.uns["spatialdata_attrs"] = {}
                            table.uns["spatialdata_attrs"]["region"] = list(sdata.labels.keys())
                            table.uns["spatialdata_attrs"]["instance_key"] = instance_key
                            table.uns["spatialdata_attrs"]["region_key"] = region_key
                            table.uns[f"{instance_key}_colors"] = _get_random_hex_colors(distinct_cells)

                            del sdata.table
                            sdata.table = table

                    for idx, ax in enumerate(axs):
                        key = list(sdata.labels.keys())[idx]
                        _render_labels(sdata=sdata, params=params, key=key, ax=ax)

                else:
                    raise NotImplementedError(f"Command '{cmd}' is not supported.")

        return axs

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
            fig, axes = _get_subplots(num_images, ncols, width, height)

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
