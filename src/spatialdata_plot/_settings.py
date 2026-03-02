"""Settings for spatialdata-plot, mirroring scanpy's verbosity pattern."""

from __future__ import annotations

import logging
from enum import IntEnum


class Verbosity(IntEnum):
    """Verbosity levels, mirroring scanpy's convention.

    ========  =====  =================
    Level     Value  Logging level
    ========  =====  =================
    error       0    ``logging.ERROR``
    warning     1    ``logging.WARNING``
    info        2    ``logging.INFO``
    debug       3    ``logging.DEBUG``
    ========  =====  =================
    """

    error = 0
    warning = 1
    info = 2
    debug = 3


_VERBOSITY_TO_LOGLEVEL: dict[Verbosity, int] = {
    Verbosity.error: logging.ERROR,
    Verbosity.warning: logging.WARNING,
    Verbosity.info: logging.INFO,
    Verbosity.debug: logging.DEBUG,
}
