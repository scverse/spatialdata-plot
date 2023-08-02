import matplotlib
import matplotlib.pyplot as plt
import scanpy as sc
import spatialdata_plot  # noqa: F401
from spatialdata import SpatialData
from spatialdata.transformations import (
    MapAxis,
    Scale,
    set_transformation,
)

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


class TestNotebooksTransformations(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_transformations_raccoon_split(self, sdata_raccoon: SpatialData):
        _, axs = plt.subplots(ncols=3, figsize=(12, 3))

        sdata_raccoon.pl.render_images().pl.show(ax=axs[0])
        sdata_raccoon.pl.render_labels().pl.show(ax=axs[1])
        sdata_raccoon.pl.render_shapes().pl.show(ax=axs[2])

    def test_plot_can_render_transformations_raccoon_overlay(self, sdata_raccoon: SpatialData):
        sdata_raccoon.pl.render_images().pl.render_labels().pl.render_shapes().pl.show()

    def test_plot_can_render_transformations_raccoon_scale(self, sdata_raccoon: SpatialData):
        scale = Scale([2.0], axes=("x",))
        set_transformation(sdata_raccoon.images["raccoon"], scale, to_coordinate_system="global")

        sdata_raccoon.pl.render_images().pl.render_labels().pl.render_shapes().pl.show()

    def test_plot_can_render_transformations_raccoon_mapaxis(self, sdata_raccoon: SpatialData):
        map_axis = MapAxis({"x": "y", "y": "x"})
        set_transformation(sdata_raccoon.images["raccoon"], map_axis, to_coordinate_system="global")

        sdata_raccoon.pl.render_images().pl.render_labels().pl.render_shapes().pl.show()


def test_plot_can_render_blobs_images(sdata_blobs: SpatialData):
    sdata_blobs.pl.render_images().pl.show()


def test_plot_can_render_blobs_points(sdata_blobs: SpatialData):
    sdata_blobs.pl.render_points().pl.show()


def test_plot_can_render_blobs_labels(sdata_blobs: SpatialData):
    sdata_blobs.pl.render_labels().pl.show()


def test_plot_can_render_blobs_shapes(sdata_blobs: SpatialData):
    sdata_blobs.pl.render_shapes().pl.show()
