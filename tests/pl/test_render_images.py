import dask.array as da
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.colors import Normalize
from spatial_image import to_spatial_image
from spatialdata import SpatialData
from spatialdata.models import Image2DModel

import spatialdata_plot  # noqa: F401
from spatialdata_plot.pl.render import _is_rgb_image
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


# Regression tests for #406: RGBA image support
class TestRGBDetection:
    """Unit tests for _is_rgb_image helper."""

    def test_rgb_lowercase(self):
        assert _is_rgb_image(["r", "g", "b"]) == (True, False)

    def test_rgba_lowercase(self):
        assert _is_rgb_image(["r", "g", "b", "a"]) == (True, True)

    def test_rgb_uppercase(self):
        assert _is_rgb_image(["R", "G", "B"]) == (True, False)

    def test_rgba_mixed_case(self):
        assert _is_rgb_image(["R", "g", "B", "a"]) == (True, True)

    def test_integer_channels_not_rgb(self):
        assert _is_rgb_image([0, 1, 2]) == (False, False)

    def test_arbitrary_names_not_rgb(self):
        assert _is_rgb_image(["DAPI", "GFP", "mCherry"]) == (False, False)

    def test_four_arbitrary_channels_not_rgb(self):
        assert _is_rgb_image(["c1", "c2", "c3", "c4"]) == (False, False)

    def test_partial_rgb_not_detected(self):
        assert _is_rgb_image(["r", "g"]) == (False, False)

    def test_rgb_with_extra_channel_not_detected(self):
        assert _is_rgb_image(["r", "g", "b", "x"]) == (False, False)

    def test_duplicate_channel_names_not_detected(self):
        assert _is_rgb_image(["r", "g", "b", "b"]) == (False, False)


class TestRGBARendering:
    """Regression tests for #406: RGBA images rendered correctly."""

    @staticmethod
    def _make_rgba_sdata(c_coords: list, alpha_val: float = 1.0) -> SpatialData:
        data = np.zeros((4, 50, 50), dtype=np.float64)
        data[0] = 0.8
        data[1] = 0.2
        data[2] = 0.1
        data[3] = alpha_val
        img = Image2DModel.parse(data, dims=("c", "y", "x"), c_coords=c_coords)
        return SpatialData(images={"img": img})

    def test_rgba_named_channels_renders(self):
        """RGBA image with r,g,b,a channel names should render without error."""
        sdata = self._make_rgba_sdata(["r", "g", "b", "a"])
        fig, ax = plt.subplots()
        sdata.pl.render_images("img").pl.show(ax=ax)
        plt.close("all")

    def test_rgba_uppercase_channels_renders(self):
        """Case-insensitive detection."""
        sdata = self._make_rgba_sdata(["R", "G", "B", "A"])
        fig, ax = plt.subplots()
        sdata.pl.render_images("img").pl.show(ax=ax)
        plt.close("all")

    def test_rgb_named_channels_renders(self):
        """RGB image (no alpha) with r,g,b channel names should render without error."""
        data = np.zeros((3, 50, 50), dtype=np.float64)
        data[0] = 0.8
        data[1] = 0.2
        data[2] = 0.1
        img = Image2DModel.parse(data, dims=("c", "y", "x"), c_coords=["r", "g", "b"])
        sdata = SpatialData(images={"img": img})
        fig, ax = plt.subplots()
        sdata.pl.render_images("img").pl.show(ax=ax)
        plt.close("all")

    def test_integer_channels_not_treated_as_rgba(self):
        """4-channel image with integer names should use multi-channel path, not RGBA."""
        sdata = self._make_rgba_sdata([0, 1, 2, 3])
        fig, ax = plt.subplots()
        # Should not raise, but goes through multi-channel path
        sdata.pl.render_images("img").pl.show(ax=ax)
        plt.close("all")

    def test_palette_overrides_rgba_detection(self):
        """Explicit palette should skip RGBA path."""
        sdata = self._make_rgba_sdata(["r", "g", "b", "a"])
        fig, ax = plt.subplots()
        sdata.pl.render_images("img", palette=["#ff0000", "#00ff00", "#0000ff", "#ffffff"]).pl.show(ax=ax)
        plt.close("all")

    def test_explicit_alpha_overrides_per_pixel(self):
        """User-specified alpha should override the image's alpha channel."""
        sdata = self._make_rgba_sdata(["r", "g", "b", "a"], alpha_val=0.5)
        fig, ax = plt.subplots()
        sdata.pl.render_images("img", alpha=0.3).pl.show(ax=ax)
        plt.close("all")

    def test_norm_applied_to_rgba(self):
        """User-provided norm should be applied per channel on RGB(A) images."""
        sdata = self._make_rgba_sdata(["r", "g", "b", "a"])
        fig, ax = plt.subplots()
        norm = Normalize(vmin=0.0, vmax=0.5, clip=True)
        sdata.pl.render_images("img", norm=norm).pl.show(ax=ax)
        plt.close("all")

    def test_uint8_rgb_renders(self):
        """uint8 RGB image should be normalized to [0, 1] and render correctly."""
        data = np.zeros((3, 50, 50), dtype=np.uint8)
        data[0] = 200
        data[1] = 100
        data[2] = 50
        img = Image2DModel.parse(data, dims=("c", "y", "x"), c_coords=["r", "g", "b"])
        sdata = SpatialData(images={"img": img})
        fig, ax = plt.subplots()
        sdata.pl.render_images("img").pl.show(ax=ax)
        plt.close("all")

    def test_uint16_rgba_renders(self):
        """uint16 RGBA image should be normalized and render correctly."""
        data = np.zeros((4, 50, 50), dtype=np.uint16)
        data[0] = 50000
        data[1] = 30000
        data[2] = 10000
        data[3] = 65535  # fully opaque
        img = Image2DModel.parse(data, dims=("c", "y", "x"), c_coords=["r", "g", "b", "a"])
        sdata = SpatialData(images={"img": img})
        fig, ax = plt.subplots()
        sdata.pl.render_images("img").pl.show(ax=ax)
        plt.close("all")


class TestMultiChannelClipping:
    """Regression tests: multi-channel compositing should not produce clipping warnings."""

    def test_no_clipping_warning_3channel_different_ranges(self):
        """3-channel image with different value ranges per channel should not clip."""
        import warnings

        rng = np.random.default_rng(42)
        data = np.zeros((3, 50, 50), dtype=np.float32)
        data[0] = rng.uniform(0.0, 0.5, (50, 50))
        data[1] = rng.uniform(0.0, 1.0, (50, 50))
        data[2] = rng.uniform(0.0, 1.5, (50, 50))
        img = Image2DModel.parse(data, dims=("c", "y", "x"), c_coords=[0, 1, 2])
        sdata = SpatialData(images={"img": img})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = plt.subplots()
            sdata.pl.render_images("img").pl.show(ax=ax)
            plt.close("all")
            clip_warns = [x for x in w if "Clipping input data" in str(x.message)]
            assert len(clip_warns) == 0, f"Got unexpected clipping warning: {clip_warns[0].message}"

    def test_no_clipping_warning_palette_compositing(self):
        """Palette compositing should not produce clipping warnings."""
        import warnings

        rng = np.random.default_rng(42)
        data = np.zeros((3, 50, 50), dtype=np.float32)
        data[0] = rng.uniform(0.0, 0.5, (50, 50))
        data[1] = rng.uniform(0.0, 1.0, (50, 50))
        data[2] = rng.uniform(0.0, 1.5, (50, 50))
        img = Image2DModel.parse(data, dims=("c", "y", "x"), c_coords=[0, 1, 2])
        sdata = SpatialData(images={"img": img})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = plt.subplots()
            sdata.pl.render_images("img", palette=["red", "green", "blue"]).pl.show(ax=ax)
            plt.close("all")
            clip_warns = [x for x in w if "Clipping input data" in str(x.message)]
            assert len(clip_warns) == 0, f"Got unexpected clipping warning: {clip_warns[0].message}"
