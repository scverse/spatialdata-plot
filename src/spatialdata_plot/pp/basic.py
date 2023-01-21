from anndata import AnnData
from ..accessor import register_spatial_data_accessor

@register_spatial_data_accessor("qu")
class QueryAccessor:
    
    def __init__(self, sdata):
        self._sdata = sdata
        
        # pull information from the AnnData object
        self._images = self._get_images()
        
        self._image_key = None
        self._x = None
        

    def _get_images(self):
        return list(self._sdata.images.keys())
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value
    
    @property
    def image_key(self):
        return self._image_key
    
    @image_key.setter
    def image_key(self, value):
        assert(value in self._images), "Image key not found."
        self._image_key = value
        
        
    


def basic_preproc(adata: AnnData) -> int:
    """Run a basic preprocessing on the AnnData :cite:p:`Wolf2018` object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    print("Implement a preprocessing function here.")
    return 0
