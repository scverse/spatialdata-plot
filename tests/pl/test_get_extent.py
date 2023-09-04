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


class TestExtent(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_extent_of_img_full_canvas(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image").pl.show()

    def test_plot_extent_of_points_partial_canvas(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points().pl.show()

    def test_plot_extent_of_partial_canvas_on_full_canvas(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image").pl.render_points().pl.show()

    def test_plot_extent_calculation_respects_element_selection_circles(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_circles").pl.show()

    def test_plot_extent_calculation_respects_element_selection_polygons(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_polygons").pl.show()

    def test_plot_extent_calculation_respects_element_selection_circles_and_polygons(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements=["blobs_circles", "blobs_polygons"]).pl.show()

    def test_plot_extent_of_img_is_correct_after_spatial_query(self, sdata_blobs: SpatialData):
        cropped_blobs = sdata_blobs.pp.get_elements(["blobs_image"]).query.bounding_box(
            axes=["x", "y"], min_coordinate=[100, 100], max_coordinate=[400, 400], target_coordinate_system="global"
        )
        cropped_blobs.pl.render_images().pl.show()
