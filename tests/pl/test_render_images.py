import dask.array as da
import matplotlib
import scanpy as sc
import spatialdata_plot  # noqa: F401
from spatial_image import to_spatial_image
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

    def test_plot_can_pass_str_cmap(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image", cmap="seismic").pl.show()

    def test_plot_can_pass_cmap(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image", cmap=matplotlib.colormaps["seismic"]).pl.show()

    def test_plot_can_pass_str_cmap_list(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(elements="blobs_image", cmap=["seismic", "Reds", "Blues"]).pl.show()

    def test_plot_can_pass_cmap_list(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(
            elements="blobs_image",
            cmap=[matplotlib.colormaps["seismic"], matplotlib.colormaps["Reds"], matplotlib.colormaps["Blues"]],
        ).pl.show()

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

    def test_plot_can_render_multiscale_image(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images("blobs_multiscale_image").pl.show()

    def test_plot_can_render_given_scale_of_multiscale_image(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images("blobs_multiscale_image", scale="scale2").pl.show()

    def test_plot_can_do_rasterization(self, sdata_blobs: SpatialData):
        temp = sdata_blobs["blobs_image"].data.copy()
        temp = da.concatenate([temp] * 6, axis=1)
        temp = da.concatenate([temp] * 6, axis=2)
        img = to_spatial_image(temp, dims=("c", "y", "x"))
        img.attrs["transform"] = sdata_blobs["blobs_image"].transform
        sdata_blobs["blobs_giant_image"] = img

        sdata_blobs.pl.render_images("blobs_giant_image").pl.show()

    def test_plot_can_stop_rasterization_with_scale_full(self, sdata_blobs: SpatialData):
        temp = sdata_blobs["blobs_image"].data.copy()
        temp = da.concatenate([temp] * 6, axis=1)
        temp = da.concatenate([temp] * 6, axis=2)
        img = to_spatial_image(temp, dims=("c", "y", "x"))
        img.attrs["transform"] = sdata_blobs["blobs_image"].transform
        sdata_blobs["blobs_giant_image"] = img

        sdata_blobs.pl.render_images("blobs_giant_image", scale="full").pl.show()

    def test_plot_can_stack_render_images(self, sdata_blobs: SpatialData):
        (
            sdata_blobs.pl.render_images(elements="blobs_image", channel=0, palette="red", alpha=0.5)
            .pl.render_images(elements="blobs_image", channel=1, palette="blue", alpha=0.5)
            .pl.show()
        )
