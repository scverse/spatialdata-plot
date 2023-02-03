from typing import Union

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from ..accessor import register_spatial_data_accessor


@register_spatial_data_accessor("pl")
class PlotAccessor:
    def __init__(self, sdata):
        self._sdata = sdata

    def imshow(self, ax=None, ncols=4, width=4, height=3, **kwargs):
        image_data = self._sdata.images
        num_images = len(image_data)

        if num_images == 1:
            ax = ax or plt.gca()
            key = [k for k in image_data.keys()][0]
            ax.imshow(image_data[key].values.T)
            ax.set_title(key)
        else:
            nrows, reminder = divmod(num_images, ncols)
            if reminder > 0:
                nrows += 1

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * width, nrows * height))

            for i, (ax, (k, v)) in enumerate(zip(np.ravel(axes), image_data.items())):
                if i < num_images:
                    ax.imshow(v.values.T)
                    ax.set_title(k)

            # get rid of the empty axes
            for i in range(num_images, ncols * nrows):
                axes.ravel()[i].axis("off")

        return self._sdata

    def render_polygon(
        self,
        ax: Union[matplotlib.axes.Axes, list[matplotlib.axes.Axes]] = None,
        cmap=plt.cm.viridis,
        alpha_boundary: float = 1.0,
        alpha_fill: float = 0.3,
        split_by=True,
        **kwargs,
    ):

        if ax is not None:

            if not (isinstance(ax, matplotlib.axes.Axes) or all([isinstance(a, matplotlib.axes.Axes) for a in ax])):

                raise TypeError("Parameter 'ax' must be one or more objects of of type 'matplotlib.axes.Axes'.")

        if not isinstance(cmap, matplotlib.colors.Colormap):

            raise TypeError("Parameter 'cmap' must be of type 'matplotlib.colors.Colormap'.")

        if not isinstance(alpha_boundary, (int, float)):

            raise TypeError("Parameter 'alpha_boundary' must be numeric.")

        if not (0 <= alpha_boundary <= 1):

            raise ValueError("Parameter 'alpha_boundary' must be between 0 and 1.")

        if not (0 <= alpha_fill <= 1):

            raise ValueError("Parameter 'alpha_fill' must be between 0 and 1.")

        if not isinstance(alpha_fill, (int, float)):

            raise TypeError("Parameter 'alpha_fill' must be numeric.")

        if not isinstance(split_by, bool):

            raise TypeError("Parameter 'split_by' must be of type 'bool'.")

        # TODO(ttreis): figure out nesting of geometries
        # TODO(ttreis): figure out how to handle multiple polygon colours
        # TODO(ttreis): include cmap
        
        if split_by:
            for key, value in self._sdata.polygons.items():
                ax = ax or plt.gca()
                for geometry in value.geometry:
                    (
                        x,
                        y,
                    ) = geometry.exterior.xy
                    ax.plot(x, y, alpha=alpha_boundary, **kwargs)

                    (
                        x,
                        y,
                    ) = geometry.exterior.xy
                    ax.fill(x, y, alpha=alpha_fill, **kwargs)
                    

        return self._sdata
