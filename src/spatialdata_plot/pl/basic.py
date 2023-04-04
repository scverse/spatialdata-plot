from collections import OrderedDict
from typing import Callable, Optional, Union

import geopandas as gpd
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
from spatialdata import transform
from spatialdata.transformations import get_transformation

from spatialdata_plot.pl._categorical_utils import (
    add_colors_for_categorical_sample_annotation,
)

from ..accessor import register_spatial_data_accessor
from ..pp.utils import _get_instance_key, _get_region_key, _verify_plotting_tree_exists
from .render import (
    _render_channels,
    _render_images,
    _render_labels,
    _render_points,
    _render_shapes,
)
from .utils import (
    _get_color_key_dtype,
    _get_color_key_values,
    _get_hex_colors_for_continous_values,
    _get_random_hex_colors,
    _get_subplots,
)


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

    def render_shapes(
        self,
        palette: Optional[Union[str, list[str], None]] = None,
        instance_key: Optional[str] = None,
        color_key: Optional[str] = None,
        **scatter_kwargs: Optional[str],
    ) -> sd.SpatialData:
        """Render the shapes contained in the given sd.SpatialData object

        Parameters
        ----------
        self : sd.SpatialData
            The sd.SpatialData object.
        palette : list[str], optional (default: None)
            A list of colors to use for rendering the images. If `None`, the
            default colors will be used.
        instance_key : str
            The name of the column in the table that identifies individual shapes
        color_key : str or None, optional (default: None)
            The name of the column in the table to use for coloring shapes.

        Returns
        -------
        sd.SpatialData
            The input sd.SpatialData with a command added to the plotting tree

        """
        if palette is not None:
            if isinstance(palette, str):
                palette = [palette]

            if isinstance(palette, list):
                if not all(isinstance(p, str) for p in palette):
                    raise TypeError("The palette argument must be a list of strings or a single string.")
            else:
                raise TypeError("The palette argument must be a list of strings or a single string.")

        if instance_key is not None or color_key is not None:
            if not hasattr(self._sdata, "table"):
                raise ValueError("SpatialData object does not have a table.")

            if not hasattr(self._sdata.table, "uns"):
                raise ValueError("Table in SpatialData object does not have 'uns'.")

            if not hasattr(self._sdata.table, "obs"):
                raise ValueError("Table in SpatialData object does not have 'obs'.")

            if isinstance(color_key, str) and not isinstance(instance_key, str):
                raise ValueError("When giving a 'color_key', an 'instance_key' must also be given.")

            if color_key is not None and not isinstance(color_key, str):
                raise TypeError("When giving a 'color_key', it must be of type 'str'.")

            if instance_key is not None and not isinstance(instance_key, str):
                raise TypeError("When giving a 'instance_key', it must be of type 'str'.")

            if instance_key not in self._sdata.table.obs.columns:
                raise ValueError(f"Column '{instance_key}' not found in 'obs'.")

            if color_key not in self._sdata.table.obs.columns and color_key not in self._sdata.table.to_df().columns:
                raise ValueError(f"Column '{instance_key}' not found in data.")

        sdata = self._copy()
        sdata = _verify_plotting_tree_exists(sdata)
        n_steps = len(sdata.plotting_tree.keys())
        sdata.plotting_tree[f"{n_steps+1}_render_shapes"] = {
            "palette": palette,
            "instance_key": instance_key,
            "color_key": color_key,
        }

        return sdata

    def render_points(
        self,
        palette: Optional[Union[str, list[str], None]] = None,
        color_key: Optional[str] = None,
        **scatter_kwargs: Optional[str],
    ) -> sd.SpatialData:
        """Render the points contained in the given sd.SpatialData object

        Parameters
        ----------
        self : sd.SpatialData
            The sd.SpatialData object.
        palette : list[str], optional (default: None)
            A list of colors to use for rendering the images. If `None`, the
            default colors will be used.
        instance_key : str
            The name of the column in the table that identifies individual shapes
        color_key : str or None, optional (default: None)
            The name of the column in the table to use for coloring shapes.

        Returns
        -------
        sd.SpatialData
            The input sd.SpatialData with a command added to the plotting tree

        """
        if palette is not None:
            if isinstance(palette, str):
                palette = [palette]

            if isinstance(palette, list):
                if not all(isinstance(p, str) for p in palette):
                    raise TypeError("The palette argument must be a list of strings or a single string.")
            else:
                raise TypeError("The palette argument must be a list of strings or a single string.")

        if color_key is not None and not isinstance(color_key, str):
            raise TypeError("When giving a 'color_key', it must be of type 'str'.")

        sdata = self._copy()
        sdata = _verify_plotting_tree_exists(sdata)
        n_steps = len(sdata.plotting_tree.keys())
        sdata.plotting_tree[f"{n_steps+1}_render_points"] = {
            "palette": palette,
            "color_key": color_key,
        }

        return sdata

    def render_images(
        self,
        palette: Optional[Union[str, list[str]]] = None,
        trans_fun: Optional[Callable[[xr.DataArray], xr.DataArray]] = None,
    ) -> sd.SpatialData:
        """Render the images contained in the given sd.SpatialData object

        Parameters
        ----------
        self : sd.SpatialData
            The sd.SpatialData object.
        palette : list[str], optional (default: None)
            A list of colors to use for rendering the images. If `None`, the
            default colors will be used.
        trans_fun : callable, optional (default: None)
            A function to apply to the images before rendering. If `None`, no
            function will be applied.

        Returns
        -------
        sd.SpatialData
            The input sd.SpatialData with a command added to the plotting tree

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
        normalize: bool = True,
        clip: bool = True,
        background: str = "black",
        pmin: float = 3.0,
        pmax: float = 99.8,
    ) -> sd.SpatialData:
        """Renders selected channels.

        Parameters:
        -----------
        self: object
            The SpatialData object
        channels: Union[List[str], List[int]]
            The channels to plot
        colors: List[str]
            The colors for the channels. Must be at least as long as len(channels).
        normalize: bool
            Perform quantile normalisation (using pmin, pmax)
        clip: bool
            Clips the merged image to the range (0, 1).
        background: str
            Background color (defaults to black).
        pmin: float
            Lower percentile for quantile normalisation (defaults to 3.-).
        pmax: float
            Upper percentile for quantile normalisation (defaults to 99.8).

        Raises
        ------
        TypeError
            If any of the parameters have an invalid type.
        ValueError
            If any of the parameters have an invalid value.

        Returns
        -------
        sd.SpatialData
            A new `SpatialData` object that is a copy of the original
            `SpatialData` object, with an updated plotting tree.
        """
        if not isinstance(channels, list):
            raise TypeError("Parameter 'channels' must be a list.")

        if not isinstance(colors, list):
            raise TypeError("Parameter 'colors' must be a list.")

        if len(channels) > len(colors):
            raise ValueError("Number of colors must have at least the same length as the number of selected channels.")

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

        if (pmin < 0.0) or (pmin > 100.0) or (pmax < 0.0) or (pmax > 100.0):
            raise ValueError("Percentiles must be in the range 0 < pmin/pmax < 100.")

        if pmin > pmax:
            raise ValueError("Percentile parameters must satisfy pmin < pmax.")

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
        instance_key: Optional[Union[str, None]] = None,
        color_key: Optional[Union[str, None]] = None,
        border_alpha: float = 1.0,
        border_color: Optional[Union[str, None]] = None,
        fill_alpha: float = 0.5,
        fill_color: Optional[Union[str, None]] = None,
        mode: str = "thick",
        palette: Optional[Union[str, list[str]]] = None,
        add_legend: bool = True,
    ) -> sd.SpatialData:
        """Render the labels contained in the given sd.SpatialData object

        Parameters
        ----------
        self : object
            sd.SpatialData
        instance_key : str
            The name of the column in the table that identifies individual labels
        color_key : str or None, optional (default: None)
            The name of the column in the table to use for coloring labels.
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
        palette : str, list or None, optional (default: None)
            The color palette to use when coloring cells. If None, a default
            palette will be used.
        add_legend : bool, optional (default: True)
            Whether to add a legend to the plot.

        Returns
        -------
        sd.SpatialData
            The input sd.SpatialData with a command added to the plotting tree

        Raises
        ------
        TypeError
            If any of the parameters have an invalid type.
        ValueError
            If any of the parameters have an invalid value.
            If the provided instance_key or color_key is not a valid table column.
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
        if instance_key is not None:
            if not isinstance(instance_key, str):
                raise TypeError("Parameter 'instance_key' must be a string.")

            if instance_key not in self._sdata.table.obs.columns:
                raise ValueError(f"The provided instance_key '{instance_key}' is not a valid table column.")

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
            "instance_key": instance_key,
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

        # Evaluate execution tree for plotting

        valid_commands = [
            "get_elements",
            "get_bb",
            "render_images",
            "render_shapes",
            "render_labels",
            "render_points",
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

            for cmd, _ in render_cmds.items():
                if cmd == "render_images" and len(sdata.images.keys()) == 0:
                    raise ValueError("No images found in the SpatialData object.")

                elif cmd == "render_shapes" and len(sdata.shapes.keys()) == 0:
                    raise ValueError("No shapes found in the SpatialData object.")

                elif cmd == "render_points" and len(sdata.points.keys()) == 0:
                    raise ValueError("No points found in the SpatialData object.")

                elif cmd == "render_labels" and len(sdata.labels.keys()) == 0:
                    raise ValueError("No labels found in the SpatialData object.")

            # set up canvas
            if ax is None:
                num_images = len(sdata.coordinate_systems)
                fig, axs = _get_subplots(num_images, ncols, width, height)
            elif isinstance(ax, matplotlib.pyplot.Axes):
                axs = np.array([ax])
            elif isinstance(ax, list):
                axs = ax

            # Set background color
            for _, ax in enumerate(axs.flatten()):
                ax.set_facecolor(bg_color)
                # key = list(sdata.labels.keys())[idx]
                # ax.imshow(sdata.labels[key].values, cmap=ListedColormap([bg_color]))

            # transform all elements
            for cmd, _ in render_cmds.items():
                if cmd == "render_images":
                    for key in sdata.images.keys():
                        img_transformation = get_transformation(sdata.images[key], get_all=True)
                        img_transformation = list(img_transformation.values())[0]
                        sdata.images[key] = transform(sdata.images[key], img_transformation)

                elif cmd == "render_shapes":
                    for key in sdata.shapes.keys():
                        shape_transformation = get_transformation(sdata.shapes[key], get_all=True)
                        shape_transformation = list(shape_transformation.values())[0]
                        sdata.shapes[key] = transform(sdata.shapes[key], shape_transformation)

                elif cmd == "render_labels":
                    for key in sdata.labels.keys():
                        label_transformation = get_transformation(sdata.labels[key], get_all=True)
                        label_transformation = list(label_transformation.values())[0]
                        sdata.labels[key] = transform(sdata.labels[key], label_transformation)

            # get biggest image after transformations to set ax size
            x_dims = []
            y_dims = []

            for cmd, _ in render_cmds.items():
                if cmd == "render_images":
                    y_dims += [(0, x.shape[1]) for x in sdata.images.values()]
                    x_dims += [(0, x.shape[2]) for x in sdata.images.values()]

                elif cmd == "render_shapes":
                    for key in sdata.shapes.keys():
                        points = []
                        polygons = []

                        for _, row in sdata.shapes[key].iterrows():
                            if row["geometry"].type == "Point":
                                points.append(row)
                            else:
                                polygons.append(row)

                        if len(points) > 0:
                            points_df = gpd.GeoDataFrame(data=points)
                            x_dims += [(min(points_df.geometry.x), max(points_df.geometry.x))]
                            y_dims += [(min(points_df.geometry.y), max(points_df.geometry.y))]

                        if len(polygons) > 0:
                            for p in polygons:
                                minx, miny, maxx, maxy = p.geometry.bounds
                                x_dims += [(minx, maxx)]
                                y_dims += [(miny, maxy)]

                elif cmd == "render_labels":
                    y_dims += [(0, x.shape[0]) for x in sdata.labels.values()]
                    x_dims += [(0, x.shape[1]) for x in sdata.labels.values()]

            max_x = [max(values) for values in zip(*x_dims)]
            min_x = [min(values) for values in zip(*x_dims)]
            max_y = [max(values) for values in zip(*y_dims)]
            min_y = [min(values) for values in zip(*y_dims)]

            extent = {"x": [min_x[0], max_x[1]], "y": [max_y[1], min_y[0]]}

            # go through tree
            for cmd, params in render_cmds.items():
                keys = list(sdata.images.keys())

                if cmd == "render_images":
                    for key, ax in zip(keys, axs.flatten()):
                        _render_images(sdata=sdata, params=params, key=key, ax=ax, extent=extent)

                elif cmd == "render_channels":
                    for key, ax in zip(keys, axs.flatten()):
                        _render_channels(sdata=sdata, key=key, ax=ax, **params)

                elif cmd == "render_shapes":
                    if (
                        sdata.table is not None
                        and isinstance(params["instance_key"], str)
                        and isinstance(params["color_key"], str)
                    ):
                        color_key_dtype = _get_color_key_dtype(sdata, params["color_key"])

                        if isinstance(color_key_dtype, pd.core.dtypes.dtypes.CategoricalDtype):
                            # If we have a table and proper keys, generate categolrical
                            # colours which are stored in the 'uns' of the table.
                            add_colors_for_categorical_sample_annotation(
                                adata=sdata.table,
                                key=params["instance_key"],
                                vec=_get_color_key_values(sdata, params["color_key"]),
                                palette=params["palette"],
                            )

                        elif isinstance(color_key_dtype, np.dtype):
                            # if it's not categorical, we assume continous values
                            colors = _get_hex_colors_for_continous_values(
                                _get_color_key_values(sdata, params["color_key"])
                            )
                            sdata.table.uns[f"{params['color_key']}_colors"] = colors

                        else:
                            raise ValueError("The dtype of the 'color_key' column must be categorical or numeric.")

                    for idx, ax in enumerate(axs):
                        key = list(sdata.shapes.keys())[idx]
                        _render_shapes(sdata=sdata, params=params, key=key, ax=ax, extent=extent)

                elif cmd == "render_points":
                    for idx, ax in enumerate(axs):
                        key = list(sdata.points.keys())[idx]
                        if params["color_key"] is not None:
                            if params["color_key"] not in sdata.points[key].columns:
                                raise ValueError(
                                    f"The column '{params['color_key']}' is not present in the 'metadata' of the points."
                                )

                        _render_points(sdata=sdata, params=params, key=key, ax=ax, extent=extent)

                elif cmd == "render_labels":
                    if (
                        sdata.table is not None
                        and isinstance(params["instance_key"], str)
                        and isinstance(params["color_key"], str)
                    ):
                        # If we have a table and proper keys, generate categolrical
                        # colours which are stored in the 'uns' of the table.

                        add_colors_for_categorical_sample_annotation(
                            adata=sdata.table,
                            key=params["instance_key"],
                            vec=_get_color_key_values(sdata, params["color_key"]),
                            palette=params["palette"],
                        )

                    else:
                        # If any of the previous conditions are not met, generate random
                        # colors for each cell id

                        N_DISTINCT_FOR_RANDOM = 30

                        if sdata.table is not None:
                            # annoying case since number of cells in labels can be
                            # different from number of cells in table. So we just use
                            # the index and randomise colours for it

                            # add fake column for limiting the amount of different colors
                            sdata.table.obs["fake"] = np.random.randint(
                                0, N_DISTINCT_FOR_RANDOM, sdata.table.obs.shape[0]
                            )

                            # has a table, so it has a region key
                            region_key = _get_region_key(sdata)

                            cell_ids_per_label = {}
                            for key in list(sdata.labels.keys()):
                                cell_ids_per_label[key] = len(sdata.table.obs.query(f"{region_key} == '{key}'"))

                            region_key = _get_region_key(sdata)
                            instance_key = _get_instance_key(sdata)
                            params["instance_key"] = instance_key
                            params["color_key"] = "fake"
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
                            params["instance_key"] = instance_key
                            params["color_key"] = instance_key
                            params["add_legend"] = False
                            # TODO(ttreis) log the decision not to display a legend

                            tmp_table = pd.DataFrame(
                                {
                                    region_key: [k for k, v in cell_ids_per_label.items() for _ in range(v)],
                                    instance_key: [i for _, v in cell_ids_per_label.items() for i in range(v)],
                                }
                            )

                            tmp_table["fake"] = np.random.randint(0, N_DISTINCT_FOR_RANDOM, len(tmp_table))
                            distinct_cells = max(list(cell_ids_per_label.values()))

                        if sdata.table is not None:
                            sdata.table.uns[f"{instance_key}_colors"] = _get_random_hex_colors(distinct_cells)

                        elif sdata.table is None:
                            data = np.zeros((tmp_table.shape[0], 1))
                            table = AnnData(X=data, obs=tmp_table, dtype=data.dtype)
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
                        _render_labels(sdata=sdata, params=params, key=key, ax=ax, extent=extent)

                else:
                    raise NotImplementedError(f"Command '{cmd}' is not supported.")

        return axs
