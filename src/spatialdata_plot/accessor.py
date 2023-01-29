from spatialdata import SpatialData
from xarray.core.extensions import _register_accessor


def register_spatial_data_accessor(name):
    """Hijacks xarray _register_accessor to register a SpatialData accessor.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    See Also
    --------
    register_dataset_accessor
    """
    return _register_accessor(name, SpatialData)
