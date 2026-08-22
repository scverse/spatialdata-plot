from importlib.metadata import version

from . import pl
from ._logging import set_verbosity
from ._settings import Verbosity
from .pl._color import PercentileNormalize


def enable_anndata_accessor() -> None:
    """PROTOTYPE (opt-in): attach a ``.pl`` accessor to :class:`anndata.AnnData`.

    Lets an AnnData carrying per-cell coordinates in ``obsm["spatial"]`` be
    plotted via ``adata.pl.render_points(...).pl.show()`` by adapting it to a
    one-points SpatialData and reusing the SpatialData plotting engine. Not
    enabled by default so importing ``spatialdata_plot`` never patches AnnData
    globally unless the caller asks for it.
    """
    from ._anndata_accessor import enable_anndata_accessor as _enable

    _enable()


__all__ = ["PercentileNormalize", "Verbosity", "enable_anndata_accessor", "pl", "set_verbosity"]

__version__ = version("spatialdata-plot")
