import matplotlib
import scanpy as sc
import spatialdata_plot  # noqa: F401
from spatialdata import SpatialData

from tests.conftest import PlotTester, PlotTesterMeta

sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=40, color_map="viridis")
matplotlib.use("agg")  # same as GitHub action runner
_ = spatialdata_plot

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be changed, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestShow(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_pad_extent_adds_padding(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image").pl.show(pad_extent=100)
