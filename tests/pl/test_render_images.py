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


class TestImages(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_image(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image").pl.show()

    def test_plot_can_pass_cmap_to_render_images(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image", cmap="seismic").pl.show()

    def test_plot_can_render_a_single_channel_from_image(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image", channel=0).pl.show()

    def test_plot_can_render_two_channels_from_image(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image", channel=[0, 1]).pl.show()

    def test_plot_can_pass_color_to_single_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image", channel=1, palette="red").pl.show()

    def test_plot_can_pass_cmap_to_single_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image", channel=1, cmap="Reds").pl.show()

    def test_plot_can_pass_color_to_each_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(
            elements="blobs_image", channel=[0, 1, 2], palette=["red", "green", "blue"]
        ).pl.show()

    def test_plot_can_pass_cmap_to_each_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(
            elements="blobs_image", channel=[0, 1, 2], cmap=["Reds", "Greens", "Blues"]
        ).pl.show()

    def test_plot_can_normalize_image(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image", quantiles_for_norm=(5, 90)).pl.show()
