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


class TestShapes(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_circles(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_circles").pl.show()

    def test_plot_can_render_circles_with_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_circles", outline=True).pl.show()

    def test_plot_can_render_circles_with_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_circles", outline=True, outline_color="red").pl.show()

    def test_plot_can_render_polygons(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_polygons").pl.show()

    def test_plot_can_render_polygons_with_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_polygons", outline=True).pl.show()

    def test_plot_can_render_polygons_with_str_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_polygons", outline=True, outline_color="red").pl.show()

    def test_plot_can_render_polygons_with_rgb_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            elements="blobs_polygons", outline=True, outline_color=(0.0, 0.0, 1.0, 1.0)
        ).pl.show()

    def test_plot_can_render_polygons_with_rgba_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            elements="blobs_polygons", outline=True, outline_color=(0.0, 1.0, 0.0, 1.0)
        ).pl.show()
