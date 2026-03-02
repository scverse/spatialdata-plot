# from https://github.com/scverse/spatialdata/blob/main/src/spatialdata/_logging.py

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from ._settings import _VERBOSITY_TO_LOGLEVEL, Verbosity

if TYPE_CHECKING:  # pragma: no cover
    from _pytest.logging import LogCaptureFixture

# Holds the public-facing function name (e.g. "render_shapes") for log messages.
# Set at the top of each _render_* entry point so that all downstream helpers
# report the user-visible origin rather than internal function names.
_log_context: ContextVar[str] = ContextVar("_log_context", default="")


class _ContextFilter(logging.Filter):
    """Inject the public function name from ``_log_context`` into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _log_context.get()
        if ctx:
            record.funcName = ctx
        return True


def _setup_logger() -> logging.Logger:
    from rich.console import Console
    from rich.logging import RichHandler

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    console = Console(force_terminal=True)
    if console.is_jupyter is True:
        console.is_jupyter = False
    ch = RichHandler(show_path=False, console=console, show_time=False)
    ch.setFormatter(logging.Formatter("%(funcName)s: %(message)s"))
    ch.addFilter(_ContextFilter())
    logger.addHandler(ch)

    # this prevents double outputs
    logger.propagate = False
    return logger


logger = _setup_logger()


def set_verbosity(verbosity: Verbosity | int | str) -> None:
    """Set the verbosity level of the spatialdata-plot logger.

    Mirrors scanpy's verbosity convention.

    Parameters
    ----------
    verbosity
        The verbosity level. Accepts a :class:`Verbosity` enum member,
        an ``int`` (0–3), or a ``str`` (e.g. ``"warning"``, ``"info"``).
    """
    if isinstance(verbosity, str):
        try:
            verbosity = Verbosity[verbosity.lower()]
        except KeyError:
            msg = f"Cannot set verbosity to {verbosity!r}. Accepted string values are: {list(Verbosity.__members__)}"
            raise ValueError(msg) from None
    else:
        verbosity = Verbosity(verbosity)
    logger.setLevel(_VERBOSITY_TO_LOGLEVEL[verbosity])


@contextmanager
def logger_warns(
    caplog: LogCaptureFixture,
    logger: logging.Logger,
    match: str | None = None,
    level: int = logging.WARNING,
) -> Iterator[None]:
    """
    Context manager similar to pytest.warns, but for logging.Logger.

    Usage:
        with logger_warns(caplog, logger, match="Found 1 NaN"):
            call_code_that_logs()
    """
    # Store initial record count to only check new records
    initial_record_count = len(caplog.records)

    # Add caplog's handler directly to the logger to capture logs even if propagate=False
    handler = caplog.handler
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(level)

    # Use caplog.at_level to ensure proper capture setup
    with caplog.at_level(level, logger=logger.name):
        try:
            yield
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

    # Only check records that were added during this context
    records = [r for r in caplog.records[initial_record_count:] if r.levelno >= level]

    if match is not None:
        pattern = re.compile(match)
        if not any(pattern.search(r.getMessage()) for r in records):
            msgs = [r.getMessage() for r in records]
            raise AssertionError(f"Did not find log matching {match!r} in records: {msgs!r}")
