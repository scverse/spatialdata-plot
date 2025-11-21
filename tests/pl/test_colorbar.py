import matplotlib
import numpy as np
import scanpy as sc
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

import spatialdata_plot  # noqa: F401
from tests.conftest import DPI, PlotTester, PlotTesterMeta

sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=DPI, color_map="viridis")
matplotlib.use("agg")  # same as GitHub action runner
_ = spatialdata_plot

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be changed, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestColorbarControls(PlotTester, metaclass=PlotTesterMeta):
    def _make_multi_image_single_channel_sdata(self) -> SpatialData:
        img1 = Image2DModel.parse(np.linspace(0, 1, 2500, dtype=float).reshape(1, 50, 50), dims=("c", "y", "x"))
        img2 = Image2DModel.parse(np.linspace(2, 0, 2500, dtype=float).reshape(1, 50, 50), dims=("c", "y", "x"))
        labels = Labels2DModel.parse(np.arange(2500, dtype=int).reshape(50, 50), dims=("y", "x"))
        return SpatialData(images={"img1": img1, "img2": img2}, labels={"lab": labels})

    def test_plot_image_auto_colorbar_for_single_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images("blobs_image", element="img").pl.show()

    def test_plot_colorbar_img_default_location(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images("blobs_image", channel=0, cmap="Reds").pl.show()

    def test_plot_colorbar_img_bottom(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images("blobs_image", channel=0, cmap="Reds", colorbar_params={"loc": "bottom"}).pl.show()

    def test_plot_colorbar_img_left(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images("blobs_image", channel=0, cmap="Reds", colorbar_params={"loc": "left"}).pl.show()

    def test_plot_colorbar_img_top(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images("blobs_image", channel=0, cmap="Reds", colorbar_params={"loc": "top"}).pl.show()

    def test_plot_colorbar_can_adjust_width(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images("blobs_image", channel=0, cmap="Reds", colorbar_params={"width": 0.4}).pl.show()

    def test_plot_colorbar_can_adjust_title(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(
            "blobs_image", channel=0, cmap="Reds", colorbar_params={"label": "Intensity"}
        ).pl.show()

    def test_plot_colorbar_can_adjust_pad(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images("blobs_image", channel=0, cmap="Reds", colorbar_params={"pad": 0.4}).pl.show()

    def test_plot_colorbar_can_have_colorbars_on_different_sides(self, sdata_blobs: SpatialData):
        (
            sdata_blobs.pl.render_images("blobs_image", channel=0, cmap="Reds", colorbar_params={"loc": "top"})
            .pl.render_labels("blobs_labels", color="instance_id", colorbar_params={"loc": "bottom"})
            .pl.show()
        )

    def test_plot_colorbar_can_have_two_colorbars_on_same_side(self, sdata_blobs: SpatialData):
        (
            sdata_blobs.pl.render_images("blobs_image", channel=0, cmap="Reds")
            .pl.render_labels("blobs_labels", color="instance_id")
            .pl.show()
        )

    def test_plot_colorbar_can_have_colorbars_on_all_sides(self, sdata_blobs: SpatialData):
        # primarily shows that spacing is correct between colorbars and plot elements
        shared_params = {
            "element": "blobs_image",
            "channel": 0,
            "cmap": "Reds",
        }
        (
            sdata_blobs.pl.render_images(**shared_params, colorbar_params={"loc": "top", "label": "top_1"})
            .pl.render_images(**shared_params, colorbar_params={"loc": "right", "label": "right_1"})
            .pl.render_images(**shared_params, colorbar_params={"loc": "left", "label": "left_1"})
            .pl.render_images(**shared_params, colorbar_params={"loc": "bottom", "label": "bottom_1"})
            .pl.render_images(**shared_params, colorbar_params={"loc": "top", "label": "top_2"})
            .pl.render_images(**shared_params, colorbar_params={"loc": "right", "label": "right_2"})
            .pl.render_images(**shared_params, colorbar_params={"loc": "left", "label": "left_2"})
            .pl.render_images(**shared_params, colorbar_params={"loc": "bottom", "label": "bottom_2"})
            .pl.show()
        )

    def test_plot_multiple_images_in_one_cs_result_in_multiple_colorbars(self):
        sdata = self._make_multi_image_single_channel_sdata()
        sdata.pl.render_images(channel=0, cmap="Reds").pl.show()

    def test_plot_can_globally_turn_off_colorbars(self):
        # adresses https://github.com/scverse/spatialdata-plot/issues/431
        sdata = self._make_multi_image_single_channel_sdata()
        sdata.pl.render_images(channel=0, cmap="Reds").pl.show(colorbar=False)

    def test_plot_single_channel_default_channel_name_omits_label(self):
        sdata = self._make_multi_image_single_channel_sdata()
        sdata.pl.render_images(element="img1", channel=0, cmap="Reds").pl.show()
