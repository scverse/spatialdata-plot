import warnings
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pytest
import scanpy as sc
from matplotlib.figure import Figure
from spatialdata import SpatialData
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
        ({"legend_params": {"loc": "upper right", "unknown_key": True}}, ValueError),
        ({"legend_params": {"locaton": "upper right"}}, ValueError),  # typo of "location"
        ({"legend_params": {"ncols": 0}}, ValueError),  # must be positive
        ({"legend_params": {"ncols": 1.5}}, ValueError),  # must be an int
        ({"legend_params": {"markerscale": -1}}, ValueError),  # must be positive
        ({"legend_params": {"frameon": "yes"}}, TypeError),  # must be a bool
        ({"legend_params": {"framealpha": 2.0}}, ValueError),  # must be in [0, 1]
        ({"legend_params": {"title_fontsize": True}}, TypeError),  # bool is not a valid size
        ({"legend_params": {"ncol": 2, "ncols": 3}}, ValueError),  # conflicting aliases
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


def _categorical_labels_legend(sdata_blobs: SpatialData, legend_params, n_groups: int = 20):
    """Render ``blobs_labels`` coloured by a fresh ``n_groups``-category obs column, return the legend."""
    import pandas as pd

    adata = sdata_blobs["table"]
    adata.obs["_cat"] = pd.Categorical([f"g{i % n_groups}" for i in range(adata.n_obs)])
    ax = sdata_blobs.pl.render_labels(element="blobs_labels", color="_cat").pl.show(
        legend_params=legend_params, return_ax=True, show=False
    )
    return ax.get_legend()


def test_legend_params_ncols_override(sdata_blobs: SpatialData):
    """Regression #770: a forced ncols sticks on the categorical legend (default would be 2 for 20 groups)."""
    leg_default = _categorical_labels_legend(sdata_blobs, None)
    assert leg_default._ncols == 2
    plt.close("all")

    leg_forced = _categorical_labels_legend(sdata_blobs, {"ncols": 1})
    assert leg_forced._ncols == 1
    plt.close("all")


def test_legend_params_ncol_alias(sdata_blobs: SpatialData):
    """Regression #770: 'ncol' (matplotlib pre-3.6 spelling) is accepted as an alias of 'ncols'."""
    leg = _categorical_labels_legend(sdata_blobs, {"ncol": 3})
    assert leg._ncols == 3
    plt.close("all")


def test_legend_params_override_with_non_margin_loc(sdata_blobs: SpatialData):
    """Regression #770: an override on a non-'right margin' loc rebuilds via the else-branch.

    The default is legend_loc='right margin'; a custom loc must still honour the override and keep
    the frame matplotlib gives that loc (frameon=True) rather than the right-margin default.
    """
    leg = _categorical_labels_legend(sdata_blobs, {"loc": "upper left", "ncols": 2})
    assert leg._ncols == 2
    assert leg.get_frame_on() is True  # mpl default for a non-margin loc, preserved by the rebuild
    plt.close("all")


def test_legend_params_frame_overrides(sdata_blobs: SpatialData):
    """Regression #770: frameon/framealpha reach the categorical legend (default is frameon=False)."""
    leg_default = _categorical_labels_legend(sdata_blobs, None)
    assert leg_default.get_frame_on() is False
    plt.close("all")

    leg = _categorical_labels_legend(sdata_blobs, {"frameon": True, "framealpha": 0.5})
    assert leg.get_frame_on() is True
    assert leg.get_frame().get_alpha() == 0.5
    plt.close("all")

    # framealpha alone implies frameon, else it would be invisible on the default frameless legend.
    leg_alpha = _categorical_labels_legend(sdata_blobs, {"framealpha": 0.3})
    assert leg_alpha.get_frame_on() is True
    assert leg_alpha.get_frame().get_alpha() == 0.3
    plt.close("all")

    # An explicit frameon=False wins over framealpha (matplotlib ignores alpha on a hidden frame).
    leg_off = _categorical_labels_legend(sdata_blobs, {"frameon": False, "framealpha": 0.5})
    assert leg_off.get_frame_on() is False
    plt.close("all")


def test_legend_params_markerscale_override(sdata_blobs: SpatialData):
    """Regression #770: markerscale is forwarded to the categorical legend (default 1.0)."""
    leg_default = _categorical_labels_legend(sdata_blobs, None)
    assert leg_default.markerscale == 1.0
    plt.close("all")

    leg = _categorical_labels_legend(sdata_blobs, {"markerscale": 2.0})
    assert leg.markerscale == 2.0
    plt.close("all")
