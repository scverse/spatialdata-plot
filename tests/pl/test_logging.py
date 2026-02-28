import logging

import pytest

import spatialdata_plot
from spatialdata_plot._logging import logger
from spatialdata_plot._settings import Verbosity


class TestSetVerbosity:
    @pytest.fixture(autouse=True)
    def _restore_verbosity(self):
        """Restore default verbosity after each test."""
        yield
        spatialdata_plot.set_verbosity(Verbosity.warning)

    def test_default_level_is_warning(self):
        assert logger.level == logging.WARNING

    @pytest.mark.parametrize(
        ("input_value", "expected_level"),
        [
            (Verbosity.error, logging.ERROR),
            (Verbosity.warning, logging.WARNING),
            (Verbosity.info, logging.INFO),
            (Verbosity.debug, logging.DEBUG),
        ],
    )
    def test_set_verbosity_with_enum(self, input_value, expected_level):
        spatialdata_plot.set_verbosity(input_value)
        assert logger.level == expected_level

    @pytest.mark.parametrize(
        ("input_value", "expected_level"),
        [
            (0, logging.ERROR),
            (1, logging.WARNING),
            (2, logging.INFO),
            (3, logging.DEBUG),
        ],
    )
    def test_set_verbosity_with_int(self, input_value, expected_level):
        spatialdata_plot.set_verbosity(input_value)
        assert logger.level == expected_level

    @pytest.mark.parametrize(
        ("input_value", "expected_level"),
        [
            ("error", logging.ERROR),
            ("WARNING", logging.WARNING),
            ("Info", logging.INFO),
            ("debug", logging.DEBUG),
        ],
    )
    def test_set_verbosity_with_string(self, input_value, expected_level):
        spatialdata_plot.set_verbosity(input_value)
        assert logger.level == expected_level

    def test_set_verbosity_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Cannot set verbosity"):
            spatialdata_plot.set_verbosity("verbose")

    def test_set_verbosity_invalid_int_raises(self):
        with pytest.raises(ValueError):
            spatialdata_plot.set_verbosity(99)
