import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import xarray as xr
from spatialdata import SpatialData

import spatialdata_plot
from spatialdata_plot.pl.utils import (
    _datashader_map_aggregate_to_color,
    _get_subplots,
    _mask_transparent_cmap_entries,
    set_zero_in_cmap_to_transparent,
)
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

# replace with
# from spatialdata._types import ColorLike
# once https://github.com/scverse/spatialdata/pull/689/ is in a release
ColorLike = tuple[float, ...] | str


class TestUtils(PlotTester, metaclass=PlotTesterMeta):
    @pytest.mark.parametrize(
        "outline_color",
        [
            (0.0, 1.0, 0.0, 1.0),
            "#00ff00",
        ],
    )
    def test_plot_set_outline_accepts_str_or_float_or_list_thereof(self, sdata_blobs: SpatialData, outline_color):
        sdata_blobs.pl.render_shapes(element="blobs_polygons", outline_alpha=1, outline_color=outline_color).pl.show()

    @pytest.mark.parametrize(
        "colname",
        ["0", "0.5", "1"],
    )
    def test_plot_colnames_that_are_valid_matplotlib_greyscale_colors_are_not_evaluated_as_colors(
        self, sdata_blobs: SpatialData, colname: str
    ):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"][colname] = [1, 2, 3, 5, 20]
        sdata_blobs.pl.render_shapes("blobs_polygons", color=colname).pl.show()

    def test_plot_can_set_zero_in_cmap_to_transparent(self, sdata_blobs: SpatialData):
        # set up figure and modify the data to add 0s
        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")
        sdata_blobs.tables["table"].obs["my_var"] = list(range(len(sdata_blobs.tables["table"].obs)))
        sdata_blobs.tables["table"].obs["my_var"] += 2  # shift the values to not have 0s

        new_cmap = set_zero_in_cmap_to_transparent(cmap="viridis")

        # baseline img
        sdata_blobs.pl.render_labels("blobs_labels", color="my_var", cmap="viridis", table="table").pl.show(
            ax=axs[0], colorbar=False
        )

        sdata_blobs.tables["table"].obs.iloc[8:12, 2] = 0

        # image with 0s as transparent, so some labels are "missing"
        sdata_blobs.pl.render_labels("blobs_labels", color="my_var", cmap=new_cmap, table="table").pl.show(
            ax=axs[1], colorbar=False
        )

    def _render_transparent_cmap_shapes(self, sdata_blobs: SpatialData, method: str):
        new_cmap = set_zero_in_cmap_to_transparent(cmap="viridis")
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"]["value"] = [0.0, 2.0, 3.0, 4.0, 5.0]
        sdata_blobs.pl.render_images("blobs_image").pl.render_shapes(
            "blobs_polygons", color="value", cmap=new_cmap, method=method
        ).pl.show(colorbar=False)

    def test_plot_transparent_cmap_shapes_matplotlib(self, sdata_blobs: SpatialData):
        self._render_transparent_cmap_shapes(sdata_blobs, method="matplotlib")

    def test_plot_transparent_cmap_shapes_datashader(self, sdata_blobs: SpatialData):
        self._render_transparent_cmap_shapes(sdata_blobs, method="datashader")


@pytest.mark.parametrize(
    "color_result",
    [
        ("0", False),
        ("0.5", False),
        ("1", False),
        ("#00ff00", True),
        ((0.0, 1.0, 0.0, 1.0), True),
    ],
)
def test_is_color_like(color_result: tuple[ColorLike, bool]):
    color, result = color_result

    assert spatialdata_plot.pl.utils._is_color_like(color) == result


class TestMaskTransparentCmapEntries:
    """Regression tests for #376: set_zero_in_cmap_to_transparent with datashader."""

    def test_masks_zero_values_when_cmap_has_transparent_entry(self):
        cmap = set_zero_in_cmap_to_transparent("viridis")
        data = np.array([[0.0, 1.0, 5.0], [0.0, 2.0, 10.0]])
        agg = xr.DataArray(data, dims=["y", "x"])

        masked = _mask_transparent_cmap_entries(agg, cmap, span=[0.0, 10.0])

        assert np.isnan(masked.values[0, 0])
        assert np.isnan(masked.values[1, 0])
        assert masked.values[0, 1] == 1.0
        assert masked.values[0, 2] == 5.0

    def test_no_effect_for_opaque_cmap(self):
        cmap = plt.get_cmap("viridis")
        data = np.array([[0.0, 5.0, 10.0]])
        agg = xr.DataArray(data, dims=["y", "x"])

        masked = _mask_transparent_cmap_entries(agg, cmap, span=[0.0, 10.0])
        np.testing.assert_array_equal(masked.values, data)

    def test_no_effect_for_string_cmap(self):
        data = np.array([[0.0, 5.0, 10.0]])
        agg = xr.DataArray(data, dims=["y", "x"])

        masked = _mask_transparent_cmap_entries(agg, "viridis", span=[0.0, 10.0])
        np.testing.assert_array_equal(masked.values, data)

    def test_datashader_shade_respects_transparent_cmap(self):
        """End-to-end: _datashader_map_aggregate_to_color produces alpha=0 for transparent cmap entries."""
        cmap = set_zero_in_cmap_to_transparent("viridis")
        data = np.array([[0.0, 5.0, 10.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])

        result = _datashader_map_aggregate_to_color(agg, cmap=cmap, min_alpha=254, span=[0.0, 10.0])
        img = result.values if hasattr(result, "values") else result

        alpha_at_zero = (int(img[0, 0]) >> 24) & 0xFF
        alpha_at_five = (int(img[0, 1]) >> 24) & 0xFF

        assert alpha_at_zero == 0, f"Expected alpha=0 at value=0.0, got {alpha_at_zero}"
        assert alpha_at_five > 0, f"Expected non-zero alpha at value=5.0, got {alpha_at_five}"

    def test_span_none_with_zeros(self):
        """Masking works when span is inferred from the aggregate (span=None)."""
        cmap = set_zero_in_cmap_to_transparent("viridis")
        data = np.array([[0.0, 3.0, 10.0]])
        agg = xr.DataArray(data, dims=["y", "x"])

        masked = _mask_transparent_cmap_entries(agg, cmap, span=None)

        assert np.isnan(masked.values[0, 0])
        assert masked.values[0, 1] == 3.0
        assert masked.values[0, 2] == 10.0

    def test_all_nan_aggregate(self):
        """All-NaN aggregate is returned unchanged."""

        cmap = set_zero_in_cmap_to_transparent("viridis")
        data = np.array([[np.nan, np.nan]])
        agg = xr.DataArray(data, dims=["y", "x"])

        masked = _mask_transparent_cmap_entries(agg, cmap, span=None)
        np.testing.assert_array_equal(np.isnan(masked.values), np.isnan(data))


def test_extract_scalar_value():
    """Test the new _extract_scalar_value function for robust numeric conversion."""

    from spatialdata_plot.pl.utils import _extract_scalar_value

    # Test basic functionality
    assert _extract_scalar_value(3.14) == 3.14
    assert _extract_scalar_value(42) == 42.0

    # Test with collections
    assert _extract_scalar_value(pd.Series([1.0, 2.0, 3.0])) == 1.0
    assert _extract_scalar_value([1.0, 2.0, 3.0]) == 1.0

    # Test edge cases
    assert _extract_scalar_value(np.nan) == 0.0
    assert _extract_scalar_value("invalid") == 0.0
    assert _extract_scalar_value([], default=1.0) == 1.0


def test_plot_can_handle_rgba_color_specifications(sdata_blobs: SpatialData):
    """Test handling of RGBA color specifications."""
    # Test with RGBA tuple
    sdata_blobs.pl.render_shapes(element="blobs_circles", color=(1.0, 0.0, 0.0, 0.8)).pl.show()

    # Test with RGB tuple (no alpha)
    sdata_blobs.pl.render_shapes(element="blobs_circles", color=(0.0, 1.0, 0.0)).pl.show()

    # Test with string color
    sdata_blobs.pl.render_shapes(element="blobs_circles", color="blue").pl.show()


@pytest.mark.parametrize(
    "input_output",
    [
        (1, 4, 1, [True]),
        (4, 4, 4, [True, True, True, True]),
        (6, 4, 8, [True, True, True, True, True, True, False, False]),  # 2 rows with 4 columns
    ],
)
def test_utils_get_subplots_produces_correct_axs_layout(input_output):
    num_images, ncols, len_axs, axs_visible = input_output

    _, axs = _get_subplots(num_images=num_images, ncols=ncols)

    assert len_axs == len(axs.flatten())
    assert axs_visible == [ax.axison for ax in axs.flatten()]
