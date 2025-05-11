import dask.array as da
import matplotlib
import numpy as np
import pytest
import scanpy as sc
from matplotlib.colors import Normalize
from spatial_image import to_spatial_image
from spatialdata import SpatialData

import spatialdata_plot  # noqa: F401
from tests.conftest import DPI, PlotTester, PlotTesterMeta, _viridis_with_under_over

RNG = np.random.default_rng(seed=42)
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


class TestImages(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_image(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image").pl.show()

    def test_plot_can_pass_str_cmap(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image", cmap="seismic").pl.show()

    def test_plot_can_pass_cmap(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image", cmap=matplotlib.colormaps["seismic"]).pl.show()

    def test_plot_can_pass_str_cmap_list(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image", cmap=["seismic", "Reds", "Blues"]).pl.show()

    def test_plot_can_pass_cmap_list(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(
            element="blobs_image",
            cmap=[matplotlib.colormaps["seismic"], matplotlib.colormaps["Reds"], matplotlib.colormaps["Blues"]],
        ).pl.show()

    def test_plot_can_render_a_single_channel_from_multiscale_image(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_multiscale_image", channel=0).pl.show()

    def test_plot_can_render_a_single_channel_str_from_image(self, sdata_blobs_str: SpatialData):
        sdata_blobs_str.pl.render_images(element="blobs_image", channel="c1").pl.show()

    def test_plot_can_render_a_single_channel_str_from_multiscale_image(self, sdata_blobs_str: SpatialData):
        sdata_blobs_str.pl.render_images(element="blobs_multiscale_image", channel="c1").pl.show()

    def test_plot_can_render_two_channels_from_image(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image", channel=[0, 1]).pl.show()

    def test_plot_can_render_two_channels_from_multiscale_image(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_multiscale_image", channel=[0, 1]).pl.show()

    def test_plot_can_render_two_channels_str_from_image(self, sdata_blobs_str: SpatialData):
        sdata_blobs_str.pl.render_images(element="blobs_image", channel=["c1", "c2"]).pl.show()

    def test_plot_can_render_two_channels_str_from_multiscale_image(self, sdata_blobs_str: SpatialData):
        sdata_blobs_str.pl.render_images(element="blobs_multiscale_image", channel=["c1", "c2"]).pl.show()

    def test_plot_can_pass_normalize_clip_True(self, sdata_blobs: SpatialData):
        norm = Normalize(vmin=0.1, vmax=0.5, clip=True)
        sdata_blobs.pl.render_images(
            element="blobs_image", channel=0, norm=norm, cmap=_viridis_with_under_over()
        ).pl.show()

    def test_plot_can_pass_normalize_clip_False(self, sdata_blobs: SpatialData):
        norm = Normalize(vmin=0.1, vmax=0.5, clip=False)
        sdata_blobs.pl.render_images(
            element="blobs_image", channel=0, norm=norm, cmap=_viridis_with_under_over()
        ).pl.show()

    def test_plot_can_pass_color_to_single_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image", channel=1, palette="red").pl.show()

    def test_plot_can_pass_cmap_to_single_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image", channel=1, cmap="Reds").pl.show()

    def test_plot_can_pass_cmap_to_each_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(
            element="blobs_image", channel=[0, 1, 2], cmap=["Reds", "Greens", "Blues"]
        ).pl.show()

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
            sdata_blobs.pl.render_images(element="blobs_image", channel=0, palette="red", alpha=0.5)
            .pl.render_images(element="blobs_image", channel=1, palette="blue", alpha=0.5)
            .pl.show()
        )

    def test_plot_can_stick_to_zorder(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes().pl.render_images().pl.show()

    def test_plot_can_render_multiscale_image_with_custom_cmap(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images("blobs_multiscale_image", channel=0, scale="scale2", cmap="Greys").pl.show()

    def test_plot_can_handle_one_palette_per_img_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image", palette=["red", "green", "blue"]).pl.show()

    def test_plot_can_handle_one_palette_per_user_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(
            element="blobs_image", channel=[0, 1, 2], palette=["red", "green", "blue"]
        ).pl.show()

    def test_plot_can_handle_mixed_channel_order(self, sdata_blobs: SpatialData):
        """Test that channels can be specified in any order and are correctly matched with their palette colors"""
        sdata_blobs.pl.render_images(
            element="blobs_image", channel=[2, 0, 1], palette=["blue", "red", "green"]
        ).pl.show()

    def test_plot_can_handle_single_channel_default_color(self, sdata_blobs: SpatialData):
        """Test that a single channel without palette uses default color mapping"""
        sdata_blobs.pl.render_images(element="blobs_image", channel=0).pl.show()

    def test_plot_can_handle_single_channel_with_cmap(self, sdata_blobs: SpatialData):
        """Test that a single channel can use a cmap instead of a palette color"""
        sdata_blobs.pl.render_images(element="blobs_image", channel=0, cmap="Reds").pl.show()

    def test_plot_can_handle_mixed_color_types(self, sdata_blobs: SpatialData):
        """Test that different channels can use different color types (palette colors and cmaps)"""
        sdata_blobs.pl.render_images(
            element="blobs_image", channel=[0, 1, 2], cmap=["viridis", None, "Reds"], palette=[None, "green", None]
        ).pl.show()

    def test_plot_can_handle_one_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image", channel=[0]).pl.show()

    def test_plot_can_handle_subset_of_channels(self, sdata_blobs: SpatialData):
        """Test case 2A: 3 channels with default RGB mapping"""
        sdata_blobs.pl.render_images(element="blobs_image", channel=[0, 2]).pl.show()

    def test_plot_can_handle_actual_number_of_channels(self, sdata_blobs: SpatialData):
        """Test case 2A: 3 channels with default RGB mapping"""
        sdata_blobs.pl.render_images(element="blobs_image", channel=[0, 1, 2]).pl.show()

    def test_plot_can_handle_scrambled_channels(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image", channel=[0, 2, 1]).pl.show()

    def test_plot_can_handle_three_channels_single_cmap(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image", channel=[0, 1, 2], cmap="viridis").pl.show()

    def test_plot_can_handle_multiple_channels_stack_strategy(self, sdata_multichannel: SpatialData):
        sdata_multichannel.pl.render_images(element="multichannel_image", multichannel_strategy="stack").pl.show()

    def test_plot_can_handle_multiple_channels_pca_strategy(self, sdata_multichannel: SpatialData):
        sdata_multichannel.pl.render_images(element="multichannel_image", multichannel_strategy="pca").pl.show()

    def test_plot_can_handle_multiple_cmaps(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(
            element="blobs_image", channel=[0, 1, 2], cmap=["viridis", "Reds", "Blues"]
        ).pl.show()


def test_fails_with_palette_and_multiple_cmaps(self, sdata_blobs: SpatialData):
    """Test error case: Cannot provide both palette and multiple cmaps"""
    with pytest.raises(ValueError, match="If 'palette' is provided"):
        sdata_blobs.pl.render_images(
            element="blobs_image",
            channel=[0, 1, 2],
            palette=["red", "green", "blue"],
            cmap=["viridis", "Reds", "Blues"],
        ).pl.show()


def test_fail_when_len_palette_is_not_equal_to_len_img_channels(sdata_blobs: SpatialData):
    with pytest.raises(ValueError, match="Palette length"):
        sdata_blobs.pl.render_images(element="blobs_image", palette=["red", "green"]).pl.show()

def test_fail_when_len_palette_is_not_equal_to_len_user_channels(sdata_blobs: SpatialData):
    with pytest.raises(ValueError, match="Palette length"):
        sdata_blobs.pl.render_images(element="blobs_image", channel=[0, 1, 2], palette=["red", "green"]).pl.show()

def test_fail_when_len_cmap_not_equal_len_img_channels(sdata_blobs):
    with pytest.raises(ValueError, match="Cmap length"):
        sdata_blobs.pl.render_images(element="blobs_image", cmap=["Reds", "Blues"]).pl.show()

def test_fail_when_len_cmap_not_equal_len_user_channels(sdata_blobs):
    with pytest.raises(ValueError, match="Cmap length"):
        sdata_blobs.pl.render_images(element="blobs_image", channel=[0,1,2], cmap=["viridis", "Reds"]).pl.show()

def test_fail_invalid_multichannel_strategy(sdata_multichannel):
    with pytest.raises(ValueError, match="Invalid multichannel_strategy"):
        sdata_multichannel.pl.render_images(element="multichannel_image", multichannel_strategy="foo").pl.show()

def test_fail_channel_index_out_of_range(sdata_blobs):
    with pytest.raises(IndexError, match="channel index"):
        sdata_blobs.pl.render_images(element="blobs_image", channel=10).pl.show()
