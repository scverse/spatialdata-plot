# from https://github.com/scverse/spatialdata/blob/main/src/spatialdata/_logging.py

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager

from _pytest.logging import LogCaptureFixture


def _setup_logger() -> "logging.Logger":
    from rich.console import Console
    from rich.logging import RichHandler

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console = Console(force_terminal=True)
    if console.is_jupyter is True:
        console.is_jupyter = False
    ch = RichHandler(show_path=False, console=console, show_time=False)
    logger.addHandler(ch)

    # this prevents double outputs
    logger.propagate = False
    return logger


logger = _setup_logger()


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
    with caplog.at_level(level, logger=logger.name):
        yield

    records = [r for r in caplog.records if r.levelno >= level]

    if match is not None:
        pattern = re.compile(match)
        if not any(pattern.search(r.getMessage()) for r in records):
            msgs = [r.getMessage() for r in records]
            raise AssertionError(f"Did not find log matching {match!r} in records: {msgs!r}")
