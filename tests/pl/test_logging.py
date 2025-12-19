import logging

import spatialdata_plot
from spatialdata import SpatialData
from tests.conftest import PlotTester, PlotTesterMeta

class TestLogging(PlotTester, metaclass=PlotTesterMeta):
    def test_default_verbosity_hides_info(self, sdata_blobs: SpatialData, caplog):
        """INFO logs should be hidden by default."""
        caplog.set_level(logging.INFO, logger=spatialdata_plot._logging.logger.name)

        # default is verbose=False
        sdata_blobs.pl.render_shapes("blobs_circles", method="datashader").pl.show()

        # make sure no INFO messages were recorded
        assert all(record.levelno != logging.INFO for record in caplog.records)

    def test_verbose_verbosity_shows_info(self, sdata_blobs: SpatialData, caplog):
        """INFO logs should appear when verbose=True."""
        spatialdata_plot.set_verbosity(True)
        caplog.set_level(logging.INFO, logger=spatialdata_plot._logging.logger.name)

        sdata_blobs.pl.render_shapes("blobs_circles", method="datashader").pl.show()

        # at least one INFO record should exist
        assert any(record.levelno == logging.INFO for record in caplog.records)

        # reset verbosity for other tests
        spatialdata_plot.set_verbosity(False)
