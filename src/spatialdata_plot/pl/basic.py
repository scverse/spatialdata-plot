from collections import OrderedDict
from collections.abc import Iterable
from typing import List, Optional, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata as sd
from anndata import AnnData
from matplotlib.colors import ListedColormap, to_rgb
from skimage.segmentation import find_boundaries
from spatialdata._core._spatialdata_ops import get_transformation


from spatialdata_plot.pl._categorical_utils import (
    add_colors_for_categorical_sample_annotation,
)

from ..accessor import register_spatial_data_accessor


@register_spatial_data_accessor("pl")
class PlotAccessor:
    def __init__(self, sdata):
        self._sdata = sdata

    def _copy(
        self,
        images: Union[None, dict] = None,
        labels: Union[None, dict] = None,
        points: Union[None, dict] = None,
        shapes: Union[None, dict] = None,
        table: Union[dict, AnnData] = None,
    ) -> sd.SpatialData:
        """
        Helper function to copies the references from the original SpatialData
        object to the subsetted SpatialData object.
        """


        return sd.SpatialData(
            images=self._sdata.images if images is None else images,
            labels=self._sdata.labels if labels is None else labels,
            points=self._sdata.points if points is None else points,
            shapes=self._sdata.shapes if shapes is None else shapes,
            table=self._sdata.table if table is None else table,
        )

    def _get_coordinate_system_mapping(self) -> dict:
        has_images = hasattr(self._sdata, "images")
        has_labels = hasattr(self._sdata, "labels")
        has_polygons = hasattr(self._sdata, "polygons")

        coordsys_keys = self._sdata.coordinate_systems
        image_keys = self._sdata.images.keys() if has_images else []
        label_keys = self._sdata.labels.keys() if has_labels else []
        polygon_keys = self._sdata.images.keys() if has_polygons else []

        mapping = {}

        if len(coordsys_keys) < 1:
            raise ValueError("SpatialData object must have at least one coordinate system to generate a mapping.")

        for key in coordsys_keys:
            mapping[key] = []

            for image_key in image_keys:
                transformations = get_transformation(self._sdata.images[image_key], get_all=True)

                if key in list(transformations.keys()):
                    mapping[key].append(image_key)

            for label_key in label_keys:
                transformations = get_transformation(self._sdata.labels[label_key], get_all=True)

                if key in list(transformations.keys()):
                    mapping[key].append(label_key)

            for polygon_key in polygon_keys:
                transformations = get_transformation(self._sdata.polygons[polygon_key], get_all=True)

                if key in list(transformations.keys()):
                    mapping[key].append(polygon_key)

        return mapping

    def _get_region_key(self) -> str:
        "Quick access to the data's region key."

        if not hasattr(self._sdata, "table"):
            raise ValueError("SpatialData object does not have a table.")

        return self._sdata.table.uns["spatialdata_attrs"]["region_key"]

    def _get_instance_key(self) -> str:
        if not hasattr(self._sdata, "table"):
            raise ValueError("SpatialData object does not have a table.")

        "Quick access to the data's instance key."

        return self._sdata.table.uns["spatialdata_attrs"]["instance_key"]

    def _verify_plotting_tree_exists(self):
        if not hasattr(self._sdata, "plotting_tree"):
            self._sdata.plotting_tree = OrderedDict()


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
        palette: Optional[List[str]] = None,
        add_legend: bool = True,
    ) -> matplotlib.pyplot.Axes:
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

        self._verify_plotting_tree_exists()

        # get current number of steps to create a unique key
        table = self._sdata.table.copy()
        add_colors_for_categorical_sample_annotation(table, cell_key, table.obs[color_key], palette=palette)

        sdata = self._copy(table=table)
        sdata.pp._verify_plotting_tree_exists()
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

    def _render_images(self, params, axs):
        pass

    def _render_channels(self, params, axs):
        pass

    def _render_shapes(self, params, axs):
        pass

    def _render_points(self, params, axs):
        pass

    def _render_labels(self, params, key: str, fig, ax):
        region_key = self._get_region_key()

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
        **kwargs,
    ) -> sd.SpatialData:
        """
        Plot the images in the SpatialData object.

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
                    self._render_images(params, axs)

                elif cmd == "render_channels":
                    self._render_channels(params, axs)

                elif cmd == "render_shapes":
                    self._render_shapes(params, axs)

                elif cmd == "render_points":
                    for ax in axs:
                        self._render_points(params, ax)

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
        **kwargs,
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
            key = [k for k in image_data.keys()][0]
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
