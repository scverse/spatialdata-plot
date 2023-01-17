from ..accessor import register_spatial_data_accessor
from matplotlib import pyplot as plt
import numpy as np
from anndata import AnnData


@register_spatial_data_accessor("pl")
class PlotAccessor:
    
    def __init__(self, spatialdata_obj):
        self._obj = spatialdata_obj
        
    def imshow(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        
        
        # get selection
        sel = self._obj.table.uns['sel'] 
        
        # unpack selection
        image_key = sel['image_key']
        c_slice = sel['c_slice']
        y_slice = sel['y_slice']
        x_slice = sel['x_slice']
        
        ax.imshow(self._obj.images[image_key][c_slice, y_slice, x_slice])
        
        return self._obj
        
    def test_plot(self):
        plt.plot(np.arange(10), np.arange(10))
        


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
