from ..accessor import register_spatial_data_accessor
from matplotlib import pyplot as plt
import numpy as np
from anndata import AnnData


@register_spatial_data_accessor("pl")
class PlotAccessor:
    
    def __init__(self, sdata):
        self._sdata = sdata
        
    def imshow(self, ax=None, ncols=4, width=4, height=3, **kwargs):
        image_data = self._sdata.im.get_selection()
        num_images = len(image_data)
        
        if num_images == 1:
            ax = ax or plt.gca()
            key = [ k for k in image_data.keys()] [0]
            ax.imshow(image_data[key].values.T)        
            ax.set_title(key)
        else:
            nrows, reminder = divmod(num_images, ncols)
            if reminder > 0:
                nrows += 1
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*width, nrows*height))
            
            for i, (ax, (k, v)) in enumerate(zip(np.ravel(axes), image_data.items())):
                if i < num_images:
                    ax.imshow(v.values.T)
                    ax.set_title(k)
            
            # get rid of the empty axes
            for i in range(num_images, ncols*nrows):
                axes.ravel()[i].axis("off")
                            

        return self._sdata
        
    def test_plot(self):
        plt.plot(np.arange(10), np.arange(10))
        
    def scatter(self):
        plt.scatter(np.random.randn(20), np.random.randn(20))
        


def basic_plot(adata: AnnData) -> int:
    """Generate a basic plot for an AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    print("Import matplotlib and implement a plotting function here.")
    return 0


class BasicClass:
    """A basic class.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.
    """

    my_attribute: str = "Some attribute."
    my_other_attribute: int = 0

    def __init__(self, adata: AnnData):
        print("Implement a class here.")

    def my_method(self, param: int) -> int:
        """A basic method.

        Parameters
        ----------
        param
            A parameter.

        Returns
        -------
        Some integer value.
        """
        print("Implement a method here.")
        return 0

    def my_other_method(self, param: str) -> str:
        """Another basic method.

        Parameters
        ----------
        param
            A parameter.

        Returns
        -------
        Some integer value.
        """
        print("Implement a method here.")
        return ""
