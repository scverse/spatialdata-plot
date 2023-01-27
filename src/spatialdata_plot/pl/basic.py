import numpy as np
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

    def test_plot(self):
        plt.plot(np.arange(10), np.arange(10))

    def scatter(self):
        plt.scatter(np.random.randn(20), np.random.randn(20))
