from importlib.metadata import version

from . import pl
from ._logging import set_verbosity
from ._settings import Verbosity

__all__ = ["pl", "set_verbosity", "Verbosity"]

__version__ = version("spatialdata-plot")
