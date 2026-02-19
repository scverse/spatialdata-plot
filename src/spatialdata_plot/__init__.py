from importlib.metadata import version

from . import pl
from ._logging import set_verbosity

__all__ = ["pl", "set_verbosity"]

__version__ = version("spatialdata-plot")
