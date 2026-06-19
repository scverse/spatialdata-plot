from importlib.metadata import version

from . import pl
from ._logging import set_verbosity
from ._settings import Verbosity
from .pl._color import PercentileNormalize

__all__ = ["PercentileNormalize", "Verbosity", "pl", "set_verbosity"]

__version__ = version("spatialdata-plot")
