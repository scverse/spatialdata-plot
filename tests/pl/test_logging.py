import io
import logging

from spatialdata import SpatialData

import spatialdata_plot
from tests.conftest import PlotTester, PlotTesterMeta


class TestLogging(PlotTester, metaclass=PlotTesterMeta):
    def test_default_verbosity_hides_info(self, sdata_blobs: SpatialData):
        """INFO logs should be hidden by default."""
        spatialdata_plot.set_verbosity(False)  # ensure default verbosity

        # Replace all handlers temporarily
        logger = spatialdata_plot._logging.logger
        original_handlers = logger.handlers[:]
        logger.handlers = []

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        # Run the function
        sdata_blobs.pl.render_shapes("blobs_circles", method="datashader").pl.show()
        
        # Check captured logs — should NOT contain the datashader info message
        logs = log_stream.getvalue()
        assert "Using 'datashader' backend" not in logs

        # Restore original handlers
        logger.handlers = original_handlers

    def test_verbose_verbosity_shows_info(self, sdata_blobs):
        spatialdata_plot.set_verbosity(True)
        
        # Replace all handlers temporarily
        logger = spatialdata_plot._logging.logger
        original_handlers = logger.handlers[:]
        logger.handlers = []

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        # Run the function
        sdata_blobs.pl.render_shapes("blobs_circles", method="datashader").pl.show()
        
        # Check captured logs
        logs = log_stream.getvalue()
        assert "Using 'datashader' backend" in logs

        # Restore original handlers
        logger.handlers = original_handlers
        spatialdata_plot.set_verbosity(False)