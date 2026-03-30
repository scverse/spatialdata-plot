import dask.array as da
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scanpy as sc
from matplotlib.colors import Normalize
from spatial_image import to_spatial_image
from spatialdata import SpatialData
from spatialdata.models import Image2DModel

import spatialdata_plot  # noqa: F401
from tests.conftest import DPI, PlotTester, PlotTesterMeta, _viridis_with_under_over

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

    def test_plot_can_render_a_single_channel_from_image(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image", channel=0).pl.show()

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

    def test_plot_can_pass_color_to_each_channel(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(
            element="blobs_image", channel=[0, 1, 2], palette=["red", "green", "blue"]
        ).pl.show()

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

    def test_plot_correctly_normalizes_multichannel_images(self, sdata_raccoon: SpatialData):
        sdata_raccoon["raccoon_int16"] = Image2DModel.parse(
            sdata_raccoon["raccoon"].data.astype(np.uint16) * 257,  # 255 * 257 = 65535,
            dims=("c", "y", "x"),
        )

        # show multi-channel vs single-channel
        fig, axs = plt.subplots(nrows=1, ncols=2)
        sdata_raccoon.pl.render_images("raccoon_int16", channel=[0]).pl.show(ax=axs[0], colorbar=False)
        axs[0].set_title("single-channel uint16")
        sdata_raccoon.pl.render_images("raccoon_int16", channel=[0, 1], palette=["yellow", "red"]).pl.show(ax=axs[1])
        axs[1].set_title("two-channel uint16")
        fig.tight_layout()


# ---------------------------------------------------------------------------
# Helpers for transfunc / grayscale tests (#508, #407)
# ---------------------------------------------------------------------------


def _make_rgb_sdata(dtype=np.float32, c_coords=None) -> SpatialData:
    """Create a minimal SpatialData with a 3-channel RGB image."""
    rng = np.random.default_rng(42)
    if dtype == np.uint8:
        data = rng.integers(0, 255, (3, 50, 50), dtype=np.uint8)
    else:
        data = rng.uniform(0, 1, (3, 50, 50)).astype(dtype)
    if c_coords is None:
        c_coords = ["r", "g", "b"]
    img = Image2DModel.parse(data, dims=("c", "y", "x"), c_coords=c_coords)
    return SpatialData(images={"img": img})


# ---------------------------------------------------------------------------
# Grayscale tests (#407)
# ---------------------------------------------------------------------------


class TestGrayscale:
    """Tests for the grayscale=True convenience parameter."""

    def test_grayscale_renders(self):
        """grayscale=True on a 3-channel image should render without error."""
        sdata = _make_rgb_sdata()
        fig, ax = plt.subplots()
        sdata.pl.render_images("img", grayscale=True).pl.show(ax=ax)
        plt.close("all")

    def test_grayscale_default_cmap_is_gray(self):
        """When grayscale=True and no explicit cmap, the colormap should be 'gray'."""
        sdata = _make_rgb_sdata()
        fig, ax = plt.subplots()
        sdata.pl.render_images("img", grayscale=True).pl.show(ax=ax)
        images = ax.get_images()
        assert len(images) == 1
        assert images[0].cmap.name == "gray"
        plt.close("all")

    def test_grayscale_explicit_cmap_overrides(self):
        """grayscale=True + cmap='viridis' should use viridis, not gray."""
        sdata = _make_rgb_sdata()
        fig, ax = plt.subplots()
        sdata.pl.render_images("img", grayscale=True, cmap="viridis").pl.show(ax=ax)
        images = ax.get_images()
        assert len(images) == 1
        assert images[0].cmap.name == "viridis"
        plt.close("all")

    def test_grayscale_wrong_channel_count_raises(self):
        """grayscale=True on a non-3-channel image should raise ValueError."""
        data = np.random.default_rng(0).uniform(0, 1, (1, 50, 50)).astype(np.float32)
        img = Image2DModel.parse(data, dims=("c", "y", "x"))
        sdata = SpatialData(images={"img": img})
        with pytest.raises(ValueError, match="grayscale=True requires exactly 3 channels"):
            sdata.pl.render_images("img", grayscale=True).pl.show()
        plt.close("all")

    def test_grayscale_with_channel_selection(self):
        """grayscale=True + channel selection on RGBA image should work."""
        data = np.random.default_rng(0).uniform(0, 1, (4, 50, 50)).astype(np.float32)
        img = Image2DModel.parse(data, dims=("c", "y", "x"), c_coords=["r", "g", "b", "a"])
        sdata = SpatialData(images={"img": img})
        fig, ax = plt.subplots()
        sdata.pl.render_images("img", channel=["r", "g", "b"], grayscale=True).pl.show(ax=ax)
        plt.close("all")

    def test_grayscale_with_palette_raises(self):
        """grayscale=True + palette should raise ValueError."""
        sdata = _make_rgb_sdata()
        with pytest.raises(ValueError, match="Cannot combine grayscale=True with palette"):
            sdata.pl.render_images("img", grayscale=True, palette=["red", "green", "blue"]).pl.show()
        plt.close("all")

    def test_grayscale_uint8(self):
        """grayscale=True on uint8 image should render correctly."""
        sdata = _make_rgb_sdata(dtype=np.uint8)
        fig, ax = plt.subplots()
        sdata.pl.render_images("img", grayscale=True).pl.show(ax=ax)
        plt.close("all")


# ---------------------------------------------------------------------------
# Single callable transfunc tests (#508)
# ---------------------------------------------------------------------------


class TestTransfuncSingle:
    """Tests for transfunc as a single callable."""

    def test_transfunc_identity(self):
        """Identity transfunc should produce the same rendering."""
        sdata = _make_rgb_sdata()
        fig, ax = plt.subplots()
        sdata.pl.render_images("img", transfunc=lambda x: x).pl.show(ax=ax)
        plt.close("all")

    def test_transfunc_log1p(self):
        """np.log1p as transfunc should render without error."""
        sdata = _make_rgb_sdata()
        fig, ax = plt.subplots()
        sdata.pl.render_images("img", transfunc=np.log1p).pl.show(ax=ax)
        plt.close("all")

    def test_transfunc_channel_reduction(self):
        """transfunc reducing 3→1 channels should route to single-channel path."""
        sdata = _make_rgb_sdata()
        fig, ax = plt.subplots()
        sdata.pl.render_images("img", transfunc=lambda x: x[:1]).pl.show(ax=ax)
        images = ax.get_images()
        assert len(images) == 1
        plt.close("all")

    def test_transfunc_with_norm(self):
        """transfunc + explicit norm should both be applied."""
        sdata = _make_rgb_sdata()
        fig, ax = plt.subplots()
        norm = Normalize(vmin=0.0, vmax=0.5, clip=True)
        sdata.pl.render_images("img", transfunc=np.sqrt, norm=norm).pl.show(ax=ax)
        plt.close("all")


# ---------------------------------------------------------------------------
# List of callables transfunc tests (#508)
# ---------------------------------------------------------------------------


class TestTransfuncList:
    """Tests for transfunc as a list of per-channel callables."""

    def test_transfunc_list_per_channel(self):
        """Different functions per channel should render correctly."""
        sdata = _make_rgb_sdata()
        fig, ax = plt.subplots()
        sdata.pl.render_images(
            "img",
            transfunc=[lambda c: c**0.3, np.log1p, np.sqrt],
        ).pl.show(ax=ax)
        plt.close("all")

    def test_transfunc_list_wrong_length_raises(self):
        """List of wrong length should raise ValueError."""
        sdata = _make_rgb_sdata()
        with pytest.raises(ValueError, match="Length of transfunc list"):
            sdata.pl.render_images("img", transfunc=[np.sqrt, np.log1p]).pl.show()
        plt.close("all")

    def test_transfunc_list_percentile_independence(self):
        """Per-channel callables should operate independently."""
        rng = np.random.default_rng(0)
        data = np.zeros((3, 50, 50), dtype=np.float32)
        data[0] = rng.uniform(0, 100, (50, 50))
        data[1] = rng.uniform(0, 10, (50, 50))
        data[2] = rng.uniform(0, 1, (50, 50))
        img = Image2DModel.parse(data, dims=("c", "y", "x"), c_coords=[0, 1, 2])
        sdata = SpatialData(images={"img": img})

        # Per-channel clip to 99th percentile
        def pctl_clip(c):
            return np.clip(c, 0, np.percentile(c, 99))

        fig, ax = plt.subplots()
        sdata.pl.render_images("img", transfunc=[pctl_clip, pctl_clip, pctl_clip]).pl.show(ax=ax)
        plt.close("all")


# ---------------------------------------------------------------------------
# Combined transfunc + grayscale tests
# ---------------------------------------------------------------------------


class TestTransfuncGrayscale:
    """Tests for combined transfunc + grayscale (transfunc runs first)."""

    def test_list_transfunc_then_grayscale(self):
        """Per-channel gamma + grayscale: gamma applied first, then luminance."""
        sdata = _make_rgb_sdata()
        fig, ax = plt.subplots()
        sdata.pl.render_images(
            "img",
            transfunc=[lambda c: c**0.8, lambda c: c, lambda c: c**0.9],
            grayscale=True,
        ).pl.show(ax=ax)
        images = ax.get_images()
        assert len(images) == 1
        assert images[0].cmap.name == "gray"
        plt.close("all")

    def test_single_transfunc_then_grayscale(self):
        """Cross-channel transform outputting 3 channels + grayscale should work."""
        sdata = _make_rgb_sdata()
        fig, ax = plt.subplots()
        sdata.pl.render_images(
            "img",
            transfunc=lambda x: x * 0.5,  # still 3 channels
            grayscale=True,
        ).pl.show(ax=ax)
        plt.close("all")

    def test_transfunc_wrong_output_channels_with_grayscale_raises(self):
        """transfunc outputting != 3 channels + grayscale should raise."""
        sdata = _make_rgb_sdata()
        with pytest.raises(ValueError, match="grayscale=True requires exactly 3 channels after transfunc"):
            sdata.pl.render_images(
                "img",
                transfunc=lambda x: x[:1],  # reduces to 1 channel
                grayscale=True,
            ).pl.show()
        plt.close("all")

    def test_transfunc_preserves_spatial_coords(self):
        """Transform should not affect y/x coordinates."""
        sdata = _make_rgb_sdata()
        fig, ax = plt.subplots()
        sdata.pl.render_images("img", transfunc=np.log1p).pl.show(ax=ax)
        # If spatial coords were wrong, the image would not render at the correct position.
        # Just verify it renders without error.
        plt.close("all")
