from typing import Union

import numpy as np
import spatialdata as sd
from matplotlib import pyplot as plt

from ..accessor import register_spatial_data_accessor


@register_spatial_data_accessor("pl")
class PlotAccessor:
    def __init__(self, sdata):
        self._sdata = sdata

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
        if num_images <= 1:
            raise ValueError("Number of images must be greater than 1.")

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
        _ = [ax.axis("off") for ax in axes.flatten()[num_images:]]
        return fig, axes

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

        if num_images == 1:
            ax = ax or plt.gca()
            key = [k for k in image_data.keys()][0]
            ax.imshow(image_data[key].values.T)
            ax.set_title(key)

        if num_images > 1:
            fig, axes = self._subplots(num_images, ncols, width, height)

            # iterate over each image and plot it onto the axes
            for i, (ax, (k, v)) in enumerate(zip(np.ravel(axes), image_data.items())):
                if i < num_images:
                    ax.imshow(v.values.T)
                    ax.set_title(k)

        return self._sdata

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
