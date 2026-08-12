import warnings
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from matplotlib.figure import Figure
from spatialdata import SpatialData
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity, set_transformation

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


class TestShow(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_pad_extent_adds_padding(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(pad_extent=100)

    def test_plot_crop_image(self, sdata_blobs: SpatialData):
        """Visual test: crop_coord windows an image to the box (#764)."""
        sdata_blobs.pl.render_images("blobs_image").pl.show(crop_coord=(150, 400, 150, 400))

    def test_plot_crop_points(self, sdata_blobs: SpatialData):
        """Visual test: crop_coord subsets points to the box, colours from the full element (#764)."""
        sdata_blobs.pl.render_points("blobs_points", color="genes", size=20).pl.show(crop_coord=(150, 400, 150, 400))

    def test_plot_crop_shapes(self, sdata_blobs: SpatialData):
        """Visual test: crop_coord subsets polygons to the box (#764)."""
        sdata_blobs.pl.render_shapes("blobs_polygons").pl.show(crop_coord=(150, 400, 150, 400))

    def test_plot_crop_circles(self, sdata_blobs: SpatialData):
        """Visual test: crop_coord keeps circles whose body overlaps the box (radius-aware, #764)."""
        sdata_blobs.pl.render_shapes("blobs_circles").pl.show(crop_coord=(150, 400, 150, 400))

    def test_plot_crop_labels(self, sdata_blobs: SpatialData):
        """Visual test: crop_coord windows a labels layer to the box (#764)."""
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum").pl.show(crop_coord=(150, 400, 150, 400))

    def test_plot_crop_layered_elements(self, sdata_blobs: SpatialData):
        """Visual test: layered image + labels both clip to the same crop box (#764)."""
        (
            sdata_blobs.pl.render_images("blobs_image")
            .pl.render_labels("blobs_labels", fill_alpha=0.5)
            .pl.show(crop_coord=(150, 400, 150, 400))
        )

    def test_plot_frameon_false_single_panel(self, sdata_blobs: SpatialData):
        """Visual test: frameon=False hides axes decorations on a single panel (regression for #204)."""
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(frameon=False)

    def test_plot_frameon_false_multi_panel(self, sdata_blobs: SpatialData):
        """Visual test: frameon=False hides axes decorations on all panels (regression for #204)."""
        set_transformation(sdata_blobs["blobs_image"], Identity(), "second_cs")
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(frameon=False, title="")

    def test_plot_no_decorations(self, sdata_blobs: SpatialData):
        """Visual test: frameon=False + title='' produces just the plot content (regression for #204)."""
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(frameon=False, title="", colorbar=False)

    def test_plot_scalebar_default(self, sdata_blobs: SpatialData):
        """Visual test: scalebar_dx attaches a default scalebar (regression for #614)."""
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(scalebar_dx=1.0)

    def test_plot_scalebar_styled(self, sdata_blobs: SpatialData):
        """Visual test: scalebar_params overrides location and color (regression for #614)."""
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(
            scalebar_dx=1.0,
            scalebar_units="um",
            scalebar_params={"location": "lower right", "color": "white", "box_alpha": 0.6},
        )

    def test_plot_scalebar_no_frame(self, sdata_blobs: SpatialData):
        """Visual test: frameon=False drops the surrounding box."""
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(
            scalebar_dx=1.0,
            scalebar_params={"frameon": False, "color": "white"},
        )

    def test_plot_scalebar_compact(self, sdata_blobs: SpatialData):
        """Visual test: padding and length_fraction shrink the scalebar footprint."""
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(
            scalebar_dx=1.0,
            scalebar_params={"length_fraction": 0.15, "pad": 0.1, "border_pad": 0.1},
        )

    def test_plot_scalebar_fixed_value_label(self, sdata_blobs: SpatialData):
        """Visual test: fixed_value pins the bar length and label overrides the displayed text."""
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(
            scalebar_dx=1.0,
            scalebar_params={"fixed_value": 200, "label": "200 um"},
        )

    def test_plot_user_ax_dpi_preserved(self, sdata_blobs: SpatialData):
        """Visual test: low DPI produces visibly pixelated rasterization (regression for #310).

        Uses dpi=15 so the 512x512 blobs image is downsampled to ~96x72.
        If the bug regresses and DPI is overridden to the default (~100),
        no rasterization occurs and the sharper render fails comparison.
        """
        fig, ax = plt.subplots(dpi=15)
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(ax=ax)

    def test_no_plt_show_when_ax_provided(self, sdata_blobs: SpatialData):
        """plt.show() must not be called when the user supplies ax= (regression for #362)."""
        _, ax = plt.subplots()
        with patch("spatialdata_plot.pl.basic.plt.show") as mock_show:
            sdata_blobs.pl.render_images(element="blobs_image").pl.show(ax=ax)
            mock_show.assert_not_called()
        plt.close("all")

    def test_plt_show_when_ax_provided_and_show_true(self, sdata_blobs: SpatialData):
        """Explicit show=True still calls plt.show() even with ax=."""
        _, ax = plt.subplots()
        with patch("spatialdata_plot.pl.basic.plt.show") as mock_show:
            sdata_blobs.pl.render_images(element="blobs_image").pl.show(ax=ax, show=True)
            mock_show.assert_called_once()
        plt.close("all")

    @pytest.mark.parametrize("interactive,expected_calls", [(False, 1), (True, 0)])
    def test_show_default_keys_off_is_interactive(
        self, sdata_blobs: SpatialData, monkeypatch, interactive: bool, expected_calls: int
    ):
        """show=None calls plt.show() iff matplotlib is non-interactive, ignoring sys.ps1.

        sys.ps1 is set in both cases to simulate a REPL; only matplotlib.is_interactive() may
        decide, so a plain (non-interactive) REPL still displays the figure (regression for #68).
        """
        monkeypatch.setattr("sys.ps1", ">>> ", raising=False)
        with (
            patch("spatialdata_plot.pl.basic.matplotlib.is_interactive", return_value=interactive),
            patch("spatialdata_plot.pl.basic.plt.show") as mock_show,
        ):
            sdata_blobs.pl.render_images(element="blobs_image").pl.show()
            assert mock_show.call_count == expected_calls
        plt.close("all")

    def test_frameon_false_hides_axes_decorations(self, sdata_blobs: SpatialData):
        """frameon=False should turn off axes decorations (regression for #204)."""
        ax = sdata_blobs.pl.render_images(element="blobs_image").pl.show(frameon=False, return_ax=True, show=False)
        assert not ax.axison
        plt.close("all")

    def test_frameon_none_keeps_axes_decorations(self, sdata_blobs: SpatialData):
        """Default frameon=None should keep axes decorations visible."""
        ax = sdata_blobs.pl.render_images(element="blobs_image").pl.show(frameon=None, return_ax=True, show=False)
        assert ax.axison
        plt.close("all")

    def test_title_empty_string_suppresses_title(self, sdata_blobs: SpatialData):
        """title='' should suppress the default coordinate system title (regression for #204)."""
        ax = sdata_blobs.pl.render_images(element="blobs_image").pl.show(title="", return_ax=True, show=False)
        assert ax.get_title() == ""
        plt.close("all")


def test_crop_sets_exact_axis_limits(sdata_blobs: SpatialData):
    """crop_coord=(xmin, xmax, ymin, ymax) pins the view to the box; y is inverted (top-left origin)."""
    ax = sdata_blobs.pl.render_points().pl.show(crop_coord=(100, 300, 120, 260), return_ax=True, show=False)
    assert ax.get_xlim() == pytest.approx((100, 300))
    assert ax.get_ylim() == pytest.approx((260, 120))  # set_ylim(ymax, ymin)
    plt.close("all")


def test_crop_ignores_pad_extent(sdata_blobs: SpatialData):
    """pad_extent must not widen a crop box (the view is exactly the box)."""
    ax = sdata_blobs.pl.render_points().pl.show(crop_coord=(100, 300, 120, 260), pad_extent=50, return_ax=True, show=False)
    assert ax.get_xlim() == pytest.approx((100, 300))
    assert ax.get_ylim() == pytest.approx((260, 120))
    plt.close("all")


def test_crop_reduces_points_drawn(sdata_blobs: SpatialData):
    """The fast subset draws fewer points than the full render."""

    def n_offsets(ax):
        return sum(len(c.get_offsets()) for c in ax.collections if hasattr(c, "get_offsets"))

    full = sdata_blobs.pl.render_points().pl.show(return_ax=True, show=False)
    cropped = sdata_blobs.pl.render_points().pl.show(crop_coord=(100, 300, 120, 260), return_ax=True, show=False)
    assert 0 < n_offsets(cropped) < n_offsets(full)
    plt.close("all")


def test_crop_continuous_color_domain_from_full_element():
    """Auto-scaled color range must come from the full element, so cropped colors match uncropped."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.uniform(0, 100, 2000), "y": rng.uniform(0, 100, 2000), "val": rng.uniform(0, 1, 2000)})
    df.loc[0, ["x", "y", "val"]] = [90, 90, 10.0]  # an extreme value far outside the crop box
    sdata = SpatialData(points={"p": PointsModel.parse(df, transformations={"global": Identity()})})

    def vrange(ax):
        for c in ax.collections:
            if getattr(c, "norm", None) is not None and c.norm.vmax is not None:
                return (c.norm.vmin, c.norm.vmax)
        return None

    full = sdata.pl.render_points("p", color="val").pl.show(return_ax=True, show=False)
    cropped = sdata.pl.render_points("p", color="val").pl.show(crop_coord=(20, 50, 30, 60), return_ax=True, show=False)
    assert vrange(cropped) == pytest.approx(vrange(full))
    plt.close("all")


def test_crop_transfunc_norm_matches_uncropped():
    """The pinned norm must use the transfunc'd full-element range, so crop+transfunc matches uncropped."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.uniform(0, 100, 3000), "y": rng.uniform(0, 100, 3000), "val": rng.uniform(0, 1, 3000)})
    df.loc[0, ["x", "y", "val"]] = [90, 90, 10.0]  # extreme value outside the crop box
    sdata = SpatialData(points={"p": PointsModel.parse(df, transformations={"global": Identity()})})

    def vrange(ax):
        for c in ax.collections:
            if getattr(c, "norm", None) is not None and c.norm.vmax is not None:
                return (c.norm.vmin, c.norm.vmax)
        return None

    full = sdata.pl.render_points("p", color="val", transfunc=np.log1p, method="matplotlib").pl.show(
        return_ax=True, show=False
    )
    cropped = sdata.pl.render_points("p", color="val", transfunc=np.log1p, method="matplotlib").pl.show(
        crop_coord=(20, 50, 30, 60), return_ax=True, show=False
    )
    assert vrange(cropped) == pytest.approx(vrange(full))  # log1p range, not the raw range
    plt.close("all")


def test_crop_datashader_autoscales_over_window():
    """Datashader crop autoscales over the visible window: a value far outside the box can't recolor it."""
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"x": rng.uniform(20, 50, 12000), "y": rng.uniform(30, 60, 12000), "val": rng.uniform(0, 1, 12000)})
    outside = pd.DataFrame({"x": [200.0], "y": [200.0], "val": [1000.0]})  # far outside the crop window
    s_a = SpatialData(points={"p": PointsModel.parse(base, transformations={"global": Identity()})})
    s_b = SpatialData(
        points={"p": PointsModel.parse(pd.concat([base, outside], ignore_index=True), transformations={"global": Identity()})}
    )

    def raster(s):
        ax = s.pl.render_points("p", color="val", method="datashader").pl.show(
            crop_coord=(20, 50, 30, 60), return_ax=True, show=False
        )
        (im,) = ax.get_images()
        arr = np.asarray(im.get_array()).copy()
        plt.close("all")
        return arr

    # If the norm leaked the full-data range (old bug), the extreme value would squish the window's colors.
    np.testing.assert_array_equal(raster(s_a), raster(s_b))


def test_crop_multiscale_selects_finer_level():
    """A crop must pick a pyramid level fine enough for the WINDOW, not the whole image (Visium HD)."""
    from spatialdata.models import Image2DModel

    from spatialdata_plot.pl.utils import _multiscale_to_spatial_image

    n = 800
    rng = np.random.default_rng(0)
    tree = Image2DModel.parse(rng.random((1, n, n), dtype=np.float32), dims=("c", "y", "x"), scale_factors=[2, 2])
    extent = {"x": (0.0, float(n)), "y": (0.0, float(n))}
    coarse = _multiscale_to_spatial_image(tree, dpi=10, width=5, height=5)  # target ~50px over the full image
    fine = _multiscale_to_spatial_image(
        tree, dpi=10, width=5, height=5, crop=(0.0, 0.0, 80.0, 80.0), extent=extent  # 10% window -> 10x boost
    )
    assert fine.shape[-1] > coarse.shape[-1]


def test_crop_image_rasterizes_only_window():
    """A cropped large image is rasterized to the window at figure resolution, not the full image then clipped.

    Regression for #764: rasterize() maps the crop bbox through the element transform, so placement is
    correct even under a Scale+Translation (which a naive .sel would mis-place) and only the window is read.
    """
    from spatialdata.models import Image2DModel
    from spatialdata.transformations import Scale, Sequence, Translation

    n = 3000
    rng = np.random.default_rng(0)
    transform = Sequence([Scale([2.0, 2.0], axes=("x", "y")), Translation([1000.0, 500.0], axes=("x", "y"))])
    img = Image2DModel.parse(
        rng.random((1, n, n), dtype=np.float32), dims=("c", "y", "x"), transformations={"global": transform}
    )
    sdata = SpatialData(images={"img": img})
    # full world extent x=(1000, 7000), y=(500, 6500); crop a small central window
    ax = sdata.pl.render_images("img").pl.show(crop_coord=(3500, 3900, 3500, 3900), return_ax=True, show=False)

    assert ax.get_xlim() == pytest.approx((3500, 3900))
    assert ax.get_ylim() == pytest.approx((3900, 3500))  # inverted y
    # the rendered raster covers the window at ~figure resolution, not the 3000-px source
    (im,) = ax.get_images()
    assert max(im.get_array().shape[:2]) < n // 2
    plt.close("all")


def test_crop_datashader_image_rasterizes_only_window():
    """method='datashader' images are windowed under crop too, not full-rendered then clipped (#764 F4)."""
    from spatialdata.models import Image2DModel
    from spatialdata.transformations import Scale, Sequence, Translation

    n = 3000
    rng = np.random.default_rng(0)
    transform = Sequence([Scale([2.0, 2.0], axes=("x", "y")), Translation([1000.0, 500.0], axes=("x", "y"))])
    img = Image2DModel.parse(
        rng.random((1, n, n), dtype=np.float32), dims=("c", "y", "x"), transformations={"global": transform}
    )
    sdata = SpatialData(images={"img": img})
    ax = sdata.pl.render_images("img", method="datashader").pl.show(
        crop_coord=(3500, 3900, 3500, 3900), return_ax=True, show=False
    )
    assert ax.get_xlim() == pytest.approx((3500, 3900))
    assert ax.get_ylim() == pytest.approx((3900, 3500))  # inverted y
    (im,) = ax.get_images()
    assert max(im.get_array().shape[:2]) < n // 2  # window at figure resolution, not the full source
    plt.close("all")


def _grid_labels_sdata():
    """A 2000x2000 label raster of 400 block-instances with a Scale+Translation, plus a table with a
    categorical and a plain-string colour column. Large enough that rasterize()/windowing engages."""
    import anndata as ad
    from spatialdata.models import Labels2DModel, TableModel
    from spatialdata.transformations import Scale, Sequence, Translation

    n = 2000
    rng = np.random.default_rng(1)
    lab = np.zeros((n, n), dtype=np.int32)
    k = 0
    for i in range(20):
        for j in range(20):
            k += 1
            lab[i * 100 : (i + 1) * 100, j * 100 : (j + 1) * 100] = k
    transform = Sequence([Scale([2.0, 2.0], axes=("x", "y")), Translation([1000.0, 500.0], axes=("x", "y"))])
    labels = Labels2DModel.parse(lab, dims=("y", "x"), transformations={"global": transform})
    obs = pd.DataFrame(
        {
            "instance_id": np.arange(1, 401),
            "region": pd.Categorical(["labels"] * 400),
            "ct_cat": pd.Categorical(rng.choice(list("ABCDE"), size=400), categories=list("ABCDE")),
            "ct_str": rng.choice(list("ABCDE"), size=400).astype(object),
        }
    )
    table = TableModel.parse(ad.AnnData(obs=obs), region="labels", region_key="region", instance_key="instance_id")
    return SpatialData(labels={"labels": labels}, tables={"table": table})


def _legend_colors(ax):
    leg = ax.get_legend()
    out = {}
    if leg is None:
        return out
    for text, handle in zip(leg.get_texts(), leg.legend_handles):
        for attr in ("get_facecolor", "get_color"):
            try:
                v = np.ravel(getattr(handle, attr)())
                if v.size >= 3:
                    out[text.get_text()] = tuple(np.round(v[:3], 3))
                    break
            except (AttributeError, TypeError):
                pass
    return out


def test_crop_labels_placement_and_empty_window():
    """Labels crop pins the view to the box; a window with no labels renders without crashing."""
    sdata = _grid_labels_sdata()
    ax = sdata.pl.render_labels("labels", color="ct_cat").pl.show(
        crop_coord=(2600, 3000, 2200, 2600), return_ax=True, show=False
    )
    assert ax.get_xlim() == pytest.approx((2600, 3000))
    assert ax.get_ylim() == pytest.approx((2600, 2200))  # inverted y
    plt.close("all")
    # a box far outside the data must not raise (empty-window guard)
    sdata.pl.render_labels("labels", color="ct_cat").pl.show(crop_coord=(99000, 99400, 99000, 99400), show=False)
    plt.close("all")


@pytest.mark.parametrize("col", ["ct_cat", "ct_str"])
def test_crop_labels_no_color_reshuffle(col):
    """Cropped label colours must match the uncropped plot for shared categories (windowed dtype-Categorical;
    plain-string falls back to full render so it stays stable too)."""
    sdata = _grid_labels_sdata()
    full = sdata.pl.render_labels("labels", color=col).pl.show(return_ax=True, show=False)
    full_colors = _legend_colors(full)
    plt.close("all")
    cropped = sdata.pl.render_labels("labels", color=col).pl.show(
        crop_coord=(2600, 3000, 2200, 2600), return_ax=True, show=False
    )
    crop_colors = _legend_colors(cropped)
    plt.close("all")
    shared = [c for c in set(full_colors) & set(crop_colors)]
    assert shared  # the window keeps several categories
    for c in shared:
        assert full_colors[c] == pytest.approx(crop_colors[c], abs=0.02)


def test_crop_invalid_order_raises(sdata_blobs: SpatialData):
    with pytest.raises(ValueError, match="xmin < xmax and ymin < ymax"):
        sdata_blobs.pl.render_points().pl.show(crop_coord=(300, 100, 120, 260), show=False)


def test_crop_wrong_length_raises(sdata_blobs: SpatialData):
    with pytest.raises(TypeError, match="tuple of four numbers"):
        sdata_blobs.pl.render_points().pl.show(crop_coord=(100, 300, 120), show=False)


def test_crop_multiple_coordinate_systems_raises(sdata_blobs: SpatialData):
    """crop is one box in one CS's units; rendering several CS at once is rejected."""
    set_transformation(sdata_blobs["blobs_points"], Identity(), to_coordinate_system="other")
    with pytest.raises(ValueError, match="single coordinate system"):
        sdata_blobs.pl.render_points().pl.show(
            coordinate_systems=["global", "other"], crop_coord=(100, 300, 120, 260), show=False
        )


def test_fig_parameter_emits_deprecation_warning(sdata_blobs: SpatialData):
    """Passing fig= should emit a DeprecationWarning (regression for #204)."""
    fig = Figure()
    with pytest.warns(DeprecationWarning, match="`fig` is being deprecated"):
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(fig=fig, show=False)
    plt.close("all")


def test_fig_parameter_default_no_warning(sdata_blobs: SpatialData):
    """Not passing fig= should not emit a deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(show=False)
    plt.close("all")


def test_title_count_validation(sdata_blobs: SpatialData):
    """title must be length 1 or one-per-panel; mismatches raise up front (regression for #695)."""
    base = sdata_blobs.pl.render_images(element="blobs_image")
    with pytest.raises(ValueError, match="number of titles"):  # single panel, too many
        base.pl.show(title=["a", "b"], show=False)
    plt.close("all")

    set_transformation(sdata_blobs["blobs_image"], Identity(), "second_cs")
    base2 = sdata_blobs.pl.render_images(element="blobs_image")
    with pytest.raises(ValueError, match="number of titles"):  # 2 panels, too many (was silently truncated)
        base2.pl.show(title=["a", "b", "c"], show=False)
    plt.close("all")

    axs = base2.pl.show(title=["left", "right"], return_ax=True, show=False)  # one per panel -> applied
    assert sorted(a.get_title() for a in axs) == ["left", "right"]
    plt.close("all")


def test_fig_parameter_warns_with_ax_list(sdata_blobs: SpatialData):
    """Passing fig= alongside a list of axes should also emit the deprecation (regression for #625)."""
    set_transformation(sdata_blobs["blobs_image"], Identity(), "second_cs")
    fig, axs = plt.subplots(1, 2)
    with pytest.warns(DeprecationWarning, match="`fig` is being deprecated"):
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(fig=fig, ax=list(axs), show=False)
    plt.close("all")


def test_show_ax_list_infers_fig(sdata_blobs: SpatialData):
    """show(ax=[...]) should infer fig from the axes without requiring fig= (regression for #625)."""
    set_transformation(sdata_blobs["blobs_image"], Identity(), "second_cs")
    fig, axs = plt.subplots(1, 2)
    sdata_blobs.pl.render_images(element="blobs_image").pl.show(ax=list(axs), show=False)
    for ax in axs:
        assert ax.get_figure() is fig
        assert len(ax.get_images()) > 0
    plt.close(fig)


def test_show_single_panel_accepts_ax_list(sdata_blobs: SpatialData):
    """show(ax=[ax]) for a single coordinate system should be accepted (regression for #625)."""
    fig, ax = plt.subplots()
    sdata_blobs.pl.render_images(element="blobs_image").pl.show(ax=[ax], show=False)
    assert ax.get_figure() is fig
    assert len(ax.get_images()) > 0
    plt.close(fig)


def test_frameon_false_multi_panel(sdata_blobs: SpatialData):
    """frameon=False should apply to all panels in a multi-panel plot (regression for #204)."""
    set_transformation(sdata_blobs["blobs_image"], Identity(), "second_cs")
    axs = sdata_blobs.pl.render_images(element="blobs_image").pl.show(frameon=False, return_ax=True, show=False)
    for ax in axs:
        assert not ax.axison
    plt.close("all")


def test_user_figure_dpi_preserved_when_ax_provided(sdata_blobs: SpatialData):
    """User's figure DPI must not be overridden when ax is passed without explicit dpi (regression for #310)."""
    fig, ax = plt.subplots(dpi=300)
    sdata_blobs.pl.render_images(element="blobs_image").pl.show(ax=ax, show=False)
    assert fig.get_dpi() == 300
    plt.close(fig)


def test_explicit_dpi_overrides_figure_dpi(sdata_blobs: SpatialData):
    """Explicit dpi= in show() should override the figure's DPI."""
    fig, ax = plt.subplots(dpi=300)
    sdata_blobs.pl.render_images(element="blobs_image").pl.show(ax=ax, dpi=150, show=False)
    assert fig.get_dpi() == 150
    plt.close(fig)


def test_dpi_default_used_when_no_ax(sdata_blobs: SpatialData):
    """When no ax is provided and dpi is not set, rcParams default should be used."""
    ax = sdata_blobs.pl.render_images(element="blobs_image").pl.show(return_ax=True, show=False)
    fig = ax.get_figure()
    assert fig.get_dpi() == matplotlib.rcParams["figure.dpi"]
    plt.close(fig)


def _scalebars_on(ax):
    from matplotlib_scalebar.scalebar import ScaleBar

    return [c for c in ax.get_children() if isinstance(c, ScaleBar)]


def test_scalebar_default_off(sdata_blobs: SpatialData):
    """Without scalebar_dx, no ScaleBar artist is attached (preserves existing behavior)."""
    ax = sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(return_ax=True, show=False)
    assert _scalebars_on(ax) == []
    plt.close("all")


def test_scalebar_dx_attaches_one_scalebar(sdata_blobs: SpatialData):
    """show(scalebar_dx=...) attaches exactly one ScaleBar to the axes (regression for #614)."""
    ax = sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(
        scalebar_dx=1.0, scalebar_units="um", return_ax=True, show=False
    )
    sbs = _scalebars_on(ax)
    assert len(sbs) == 1
    assert sbs[0].units == "um"
    plt.close("all")


def test_scalebar_units_default_is_um(sdata_blobs: SpatialData):
    """Omitting scalebar_units falls back to 'um' (matches scanpy/squidpy convention)."""
    ax = sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(scalebar_dx=2.5, return_ax=True, show=False)
    sbs = _scalebars_on(ax)
    assert len(sbs) == 1
    assert sbs[0].units == "um"
    plt.close("all")


def test_scalebar_params_passthrough(sdata_blobs: SpatialData):
    """scalebar_params keys are forwarded verbatim to matplotlib_scalebar.ScaleBar."""
    ax = sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(
        scalebar_dx=1.0,
        scalebar_params={"location": "lower right", "color": "red", "box_alpha": 0.5},
        return_ax=True,
        show=False,
    )
    sbs = _scalebars_on(ax)
    assert len(sbs) == 1
    # ScaleBar normalizes "lower right" to its integer code (4); just verify the constructor accepted it
    # by checking attributes that survive verbatim.
    assert sbs[0].color == "red"
    assert sbs[0].box_alpha == 0.5
    plt.close("all")


def test_scalebar_single_panel_multi_layer_attaches_one(sdata_blobs: SpatialData):
    """Stacking render_images + render_shapes on one axis must produce exactly one scalebar.

    The pre-fix code drew the scalebar inside per-layer decoration logic, so a multi-layer
    plot would have attached duplicates. The fix moves drawing to the per-axis tail of show().
    """
    ax = (
        sdata_blobs.pl.render_images(element="blobs_image")
        .pl.render_shapes(element="blobs_circles")
        .pl.show(scalebar_dx=1.0, return_ax=True, show=False)
    )
    assert len(_scalebars_on(ax)) == 1
    plt.close("all")


def test_scalebar_multi_panel_attaches_one_per_axis(sdata_blobs: SpatialData):
    """Each panel in a multi-panel plot gets its own ScaleBar."""
    set_transformation(sdata_blobs["blobs_image"], Identity(), "second_cs")
    axs = sdata_blobs.pl.render_images(element="blobs_image").pl.show(scalebar_dx=1.0, return_ax=True, show=False)
    for ax in axs:
        assert len(_scalebars_on(ax)) == 1
    plt.close("all")


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"scalebar_dx": "bad"}, TypeError),
        ({"scalebar_dx": True}, TypeError),  # bool is rejected even though it is an int
        ({"scalebar_dx": 0}, ValueError),
        ({"scalebar_dx": -1.5}, ValueError),
        ({"scalebar_dx": 1.0, "scalebar_units": 42}, TypeError),
        ({"scalebar_dx": 1.0, "scalebar_params": []}, TypeError),
    ],
)
def test_scalebar_validation_rejects_bad_inputs(sdata_blobs: SpatialData, kwargs, exc):
    """_validate_show_parameters surfaces actionable errors for bad scalebar inputs."""
    with pytest.raises(exc):
        sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(show=False, **kwargs)
    plt.close("all")


def test_legend_params_dict_form(sdata_blobs: SpatialData):
    """legend_params dict form is accepted and applied (additive sugar around flat legend_* kwargs)."""
    ax = sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(
        legend_params={"loc": "upper right", "fontsize": 14},
        return_ax=True,
        show=False,
    )
    legend = ax.get_legend()
    if legend is not None:
        # When a legend is rendered, fontsize was forwarded.
        for text in legend.get_texts():
            assert text.get_fontsize() == 14
    plt.close("all")


def test_legend_params_overrides_flat_kwarg(sdata_blobs: SpatialData):
    """When the same option is set as both flat kwarg and dict key, the dict wins."""
    ax = sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(
        legend_fontsize=10,
        legend_params={"fontsize": 18},
        return_ax=True,
        show=False,
    )
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            assert text.get_fontsize() == 18
    plt.close("all")


def test_legend_params_default_none_is_noop(sdata_blobs: SpatialData):
    """legend_params=None preserves identical behavior to omitting the kwarg."""
    ax_a = sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(return_ax=True, show=False)
    plt.close("all")
    ax_b = sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(legend_params=None, return_ax=True, show=False)
    assert (ax_a.get_legend() is None) == (ax_b.get_legend() is None)
    plt.close("all")


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"legend_params": []}, TypeError),
        ({"legend_params": "loc=upper right"}, TypeError),
        ({"legend_params": {"loc": "upper right", "frameon": True}}, ValueError),
        ({"legend_params": {"locaton": "upper right"}}, ValueError),  # typo of "location"
    ],
)
def test_legend_params_validation_rejects_bad_inputs(sdata_blobs: SpatialData, kwargs, exc):
    """_validate_show_parameters surfaces actionable errors for bad legend_params inputs."""
    with pytest.raises(exc):
        sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(show=False, **kwargs)
    plt.close("all")


def test_legend_params_location_alias_for_loc(sdata_blobs: SpatialData):
    """legend_params accepts both 'location' (canonical) and 'loc' (matplotlib-native alias)."""
    sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(
        legend_params={"loc": "upper right"}, return_ax=True, show=False
    )
    sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(
        legend_params={"location": "upper right"}, return_ax=True, show=False
    )
    sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show(
        legend_params={"loc": "upper left", "location": "lower right"}, return_ax=True, show=False
    )
    plt.close("all")
