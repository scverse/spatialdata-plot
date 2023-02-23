from typing import Union, Optional
from collections import OrderedDict
from matplotlib.colors import Colormap, BoundaryNorm
import numpy as np
import spatialdata as sd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.cm import get_cmap
from anndata import AnnData
import pandas as pd
from skimage.segmentation import find_boundaries


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

        if not hasattr(self._sdata, "table"):
            raise ValueError("SpatialData object does not have a table.")

        if not hasattr(self._sdata.table, "uns"):
            raise ValueError("Table in SpatialData object does not have a 'uns' attribute.")

        if "plotting_tree" not in self._sdata.table.uns.keys():
            self._sdata.table.uns["plotting_tree"] = OrderedDict()

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

        # get rid of the empty axes
        # _ = [ax.axis("off") for ax in axes.flatten()[num_images:]]
        return fig, axes

    def render_labels(
        self,
        cell_key: str,
        colour_key: Optional[Union[str, None]] = None,
        border_alpha: float = 1.0,
        border_colour: Optional[Union[str, None]] = None,
        fill_alpha: float = 1.0,
        fill_colour: Optional[Union[str, None]] = None,
        mode: str = "outer",
        bg_colour: Optional[Union[str, None]] = None,
        cmap: Optional[Union[str, Colormap]] = matplotlib.pyplot.cm.gist_rainbow,
    ) -> matplotlib.pyplot.Axes:

        if not isinstance(cell_key, str):
            raise TypeError("Parameter 'cell_key' must be a string.")

        if cell_key not in self._sdata.table.obs.columns:
            raise ValueError(f"The provided cell_key '{cell_key}' is not a valid table column.")

        if colour_key is not None:

            if not isinstance(colour_key, (str, type(None))):
                raise TypeError("Parameter 'colour_key' must be a string.")

            if colour_key not in self._sdata.table.obs.columns:
                raise ValueError(f"The provided colour_key '{colour_key}' is not a valid table column.")

        if not isinstance(border_alpha, (int, float)):
            raise TypeError("Parameter 'border_alpha' must be a float.")

        if not (border_alpha <= 1 and border_alpha >= 0):
            raise ValueError("Parameter 'border_alpha' must be between 0 and 1.")

        if border_colour is not None:

            if not isinstance(colour_key, (str, type(None))):
                raise TypeError("If specified, parameter 'border_colour' must be a string.")

        if not isinstance(fill_alpha, (int, float)):
            raise TypeError("Parameter 'fill_alpha' must be a float.")

        if not (fill_alpha <= 1 and fill_alpha >= 0):
            raise ValueError("Parameter 'fill_alpha' must be between 0 and 1.")

        if fill_colour is not None:

            if not isinstance(fill_colour, (str, type(None))):
                raise TypeError("If specified, parameter 'fill_colour' must be a string.")

        valid_modes = ["thick", "inner", "outer", "subpixel"]
        if not isinstance(mode, str):
            raise TypeError("Parameter 'mode' must be a string.")

        if mode not in valid_modes:
            raise ValueError("Parameter 'mode' must be one of 'thick', 'inner', 'outer', 'subpixel'.")

        if bg_colour is not None:

            if not isinstance(bg_colour, (str, type(None))):
                raise TypeError("If specified, parameter 'bg_colour' must be a string.")

        # TODO(ttreis): Steal cmap type checking from squidpy

        self._verify_plotting_tree_exists()

        # get current number of steps to create a unique key
        table = self._sdata.table.copy()
        n_steps = len(table.uns["plotting_tree"].keys())
        table.uns["plotting_tree"][f"{n_steps+1}_render_labels"] = {
            "cell_key": cell_key,
            "colour_key": colour_key,
            "border_alpha": border_alpha,
            "border_colour": border_colour,
            "fill_alpha": fill_alpha,
            "fill_colour": fill_colour,
            "mode": mode,
            "bg_colour": bg_colour,
            "cmap": cmap,
        }

        return self._copy(table=table)

    def _render_images(self, params, axs):

        pass

    def _render_channels(self, params, axs):

        pass

    def _render_shapes(self, params, axs):

        pass

    def _render_points(self, params, axs):

        pass

    def _render_labels(self, params, key: str, ax):

        instance_key = self._get_instance_key()
        region_key = self._get_region_key()
        # Get matching of regions to images
        regions = self._sdata.table.obs[region_key].unique().tolist()
        region_mapping = {k.split("/")[-1]: k for k in regions}

        for k in self._sdata.images.keys():
            if k not in region_mapping.keys():
                del region_mapping[k]

        table = self._sdata.table.obs
        table = table[table[region_key] == region_mapping[key]]
        groups = self._sdata.table.obs[params["colour_key"]].unique()
        seg = self._sdata.labels[key].values

        # Prepare cmap
        if isinstance(params["cmap"], str):
            try:
                cmap = get_cmap(params["cmap"])
            except ValueError:
                raise ValueError(f"Colormap '{params['cmap']}' not found.")
        elif isinstance(params["cmap"], matplotlib.colors.Colormap):
            cmap = params["cmap"]
        else:
            raise TypeError("Parameter 'cmap' must be a string or a matplotlib.colors.Colormap.")

        if len(groups) > 256:
            raise ValueError("Too many colours needed for plotting.")

        colors = ListedColormap(cmap(np.linspace(0, 1, len(groups) + 1)))
        group_to_colour = pd.DataFrame(
            {params["colour_key"]: groups, "color": [color for color in colors(range(len(groups)))]}
        )

        cells_ids_in_seg = np.unique(seg)

        cell_to_group = pd.DataFrame(
            table[[params["cell_key"], params["colour_key"]]].values,
            columns=[params["cell_key"], params["colour_key"]],
        )
        cell_to_colour = cell_to_group.merge(group_to_colour, on=params["colour_key"], how="left").drop(
            params["colour_key"], axis=1
        )

        # apply alpha to background
        cell_to_colour["color_fill"] = [[c[0], c[1], c[2], params["fill_alpha"]] for c in cell_to_colour.color]
        # cell_to_colour["color_border"] = [[c[0], c[1], c[2], params["border_alpha"]] for c in cell_to_colour.color]

        # Add group color if cell_id is in table, else background color
        color_list = pd.DataFrame({"cell_id": cells_ids_in_seg})
        color_list = color_list.merge(cell_to_colour, on="cell_id", how="left").drop("color", axis=1)

        color_list["color_fill"] = color_list["color_fill"].fillna(params["bg_colour"])
        # color_list["color_border"] = color_list["color_border"].fillna(params["bg_colour"])

        bounds = np.arange(len(cells_ids_in_seg) + 1)
        cmap_fill = ListedColormap(color_list["color_fill"].values)
        norm = BoundaryNorm(bounds, cmap_fill.N)

        # First plot the infill, if desired
        if params["fill_alpha"] != 0:

            for group in groups:

                vaid_cell_ids = table[table[params["colour_key"]] == group][params["cell_key"]].values
                seg
                seg_for_border = seg.copy()
                seg_for_border[~np.isin(seg, vaid_cell_ids)] = 0
                seg_for_fill = seg_for_border > 0
                # print(seg_for_fill)
                border_mask = find_boundaries(seg_for_border, mode=params["mode"])
                seg_masked = np.ma.masked_array(seg_for_border, ~border_mask)

                group_color = list(group_to_colour[group_to_colour[params["colour_key"]] == group].color.values[0])
                group_color[-1] = params["fill_alpha"]

                colors = [[0,0,0,0], group_color]

                ax.imshow(seg_for_fill, cmap=ListedColormap(colors), interpolation="nearest")

        # Then plot the borders, if desired
        if params["border_alpha"] != 0:

            # Extract the segmentation for each group
            for group in groups:

                vaid_cell_ids = table[table[params["colour_key"]] == group][params["cell_key"]].values
                
                seg_for_group = seg.copy()
                seg_for_group[~np.isin(seg, vaid_cell_ids)] = 0
                # print(seg_for_group)
                border_mask = find_boundaries(seg_for_group, mode=params["mode"])
                seg_masked = np.ma.masked_array(seg_for_group, ~border_mask)
                #                 #
                colors = [ list(group_to_colour[group_to_colour[params["colour_key"]] == group].color.values[0])]

                ax.imshow(seg_masked, cmap=ListedColormap(colors), interpolation="nearest")

        ax.set_title(key)
        ax.set_xlabel("spatial1")
        ax.set_ylabel("spatial2")
        ax.set_xticks([])
        ax.set_yticks([])

    def imshow(
        self, ax: Union[plt.Axes, None] = None, ncols: int = 4, width: int = 4, height: int = 3, **kwargs
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

        if (
            "plotting_tree" not in self._sdata.table.uns.keys()
            or len(self._sdata.table.uns["plotting_tree"].keys()) == 0
        ):

            raise ValueError("No operations have been performed yet.")

            # if num_images == 1:
            #     ax = ax or plt.gca()
            #     key = [k for k in image_data.keys()][0]
            #     ax.imshow(image_data[key].values.T)
            #     ax.set_title(key)

            # if num_images > 1:
            #     fig, axes = self._subplots(num_images, ncols, width, height)

            #     # iterate over each image and plot it onto the axes
            #     for i, (ax, (k, v)) in enumerate(zip(np.ravel(axes), image_data.items())):
            #         if i < num_images:
            #             ax.imshow(v.values.T)
            #             ax.set_title(k)

        elif len(self._sdata.table.uns["plotting_tree"].keys()) > 0:

            render_cmds = OrderedDict()

            for cmd, params in self._sdata.table.uns["plotting_tree"].items():

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
                        self._render_labels(params=params, key=key, ax=ax)

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
