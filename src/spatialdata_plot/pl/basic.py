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

    def imshow(self, ax=None, ncols=4, width=4, height=3, **kwargs) -> sd.SpatialData:
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

    def test_plot(self):
        plt.plot(np.arange(10), np.arange(10))

    def scatter(self):
        plt.scatter(np.random.randn(20), np.random.randn(20))
