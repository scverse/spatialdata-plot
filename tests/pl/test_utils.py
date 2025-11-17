import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from shapely.geometry import Polygon
from spatialdata import SpatialData

import spatialdata_plot
from spatialdata_plot.pl.utils import _get_subplots, _validate_polygons
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
        from spatialdata_plot.pl.utils import set_zero_in_cmap_to_transparent

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


def test_validate_polygons_converts_holed_polygons(caplog):
    shell = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    hole = [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2), (0.2, 0.2)]
    holed_polygon = Polygon(shell, [hole])
    plain_polygon = Polygon([(2.0, 0.0), (2.0, 1.0), (3.0, 1.0), (3.0, 0.0), (2.0, 0.0)])
    shapes = gpd.GeoDataFrame({"geometry": [holed_polygon, plain_polygon]})

    with caplog.at_level("INFO"):
        validated = _validate_polygons(shapes.copy())

    assert validated.iloc[0].geometry.geom_type == "MultiPolygon"
    assert validated.iloc[1].geometry.geom_type == "Polygon"
    assert "Converted 1 Polygon" in caplog.text
