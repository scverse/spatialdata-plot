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


def test_fig_parameter_no_warning_with_ax_list(sdata_blobs: SpatialData):
    """Passing fig= with a list of axes should not warn (fig is still required there)."""
    set_transformation(sdata_blobs["blobs_image"], Identity(), "second_cs")
    fig, axs = plt.subplots(1, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sdata_blobs.pl.render_images(element="blobs_image").pl.show(fig=fig, ax=list(axs), show=False)
    plt.close("all")


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
