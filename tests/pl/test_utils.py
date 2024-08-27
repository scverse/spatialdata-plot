import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scanpy as sc
import spatialdata_plot
from spatialdata import SpatialData

from tests.conftest import DPI, PlotTester, PlotTesterMeta

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


class TestUtils(PlotTester, metaclass=PlotTesterMeta):
    @pytest.mark.parametrize(
        "outline_color",
        [
            (0.0, 1.0, 0.0, 1.0),
            "#00ff00",
        ],
    )
    def test_plot_set_outline_accepts_str_or_float_or_list_thereof(self, sdata_blobs: SpatialData, outline_color):
        sdata_blobs.pl.render_shapes(element="blobs_polygons", outline=True, outline_color=outline_color).pl.show()

    @pytest.mark.parametrize(
        "colname",
        ["0", "0.5", "1"],
    )
    def test_plot_colnames_that_are_valid_matplotlib_greyscale_colors_are_not_evaluated_as_colors(
        self, sdata_blobs: SpatialData, colname: str
    ):
        sdata_blobs["table"].obs["region"] = ["blobs_polygons"] * sdata_blobs["table"].n_obs
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"][colname] = [1, 2, 3, 5, 20]
        sdata_blobs.pl.render_shapes("blobs_polygons", color=colname).pl.show()

    def test_plot_can_set_zero_in_cmap_to_transparent(self, sdata_blobs: SpatialData):
        from spatialdata_plot.pl.utils import set_zero_in_cmap_to_transparent

        # set up figure and modify the data to add 0s
        fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
        table = sdata_blobs.table.copy()
        x = table.X.todense()
        x[:10, 0] = 0
        table.X = x
        sdata_blobs.tables["modified_table"] = table

        # create a new cmap with 0 as transparent
        new_cmap = set_zero_in_cmap_to_transparent(cmap="plasma")

        # baseline img
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum", cmap="viridis", table="table").pl.show(
            ax=axs[0], colorbar=False
        )

        # image with 0s as transparent, so some labels are "missing"
        sdata_blobs.pl.render_labels(
            "blobs_labels", color="channel_0_sum", cmap=new_cmap, table="modified_table"
        ).pl.show(ax=axs[1], colorbar=False)
