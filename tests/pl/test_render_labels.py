import dask.array as da
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from spatial_image import to_spatial_image
from spatialdata import SpatialData, deepcopy
from spatialdata.models import TableModel

import spatialdata_plot  # noqa: F401
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


class TestLabels(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_labels(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels(element="blobs_labels").pl.show()

    def test_plot_can_render_multiscale_labels(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = "blobs_multiscale_labels"
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        sdata_blobs.pl.render_labels("blobs_multiscale_labels").pl.show()

    def test_plot_can_render_given_scale_of_multiscale_labels(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = "blobs_multiscale_labels"
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        sdata_blobs.pl.render_labels("blobs_multiscale_labels", scale="scale1").pl.show()

    def test_plot_can_do_rasterization(self, sdata_blobs: SpatialData):
        temp = sdata_blobs["blobs_labels"].data.copy()
        temp = da.concatenate([temp] * 6, axis=0)
        temp = da.concatenate([temp] * 6, axis=1)
        img = to_spatial_image(temp, dims=("y", "x"))
        img.attrs["transform"] = sdata_blobs["blobs_labels"].transform
        sdata_blobs["blobs_giant_labels"] = img

        sdata_blobs["table"].obs["region"] = "blobs_giant_labels"
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_giant_labels"

        sdata_blobs.pl.render_labels("blobs_giant_labels").pl.show()

    def test_plot_can_stop_rasterization_with_scale_full(self, sdata_blobs: SpatialData):
        temp = sdata_blobs["blobs_labels"].data.copy()
        temp = da.concatenate([temp] * 6, axis=0)
        temp = da.concatenate([temp] * 6, axis=1)
        img = to_spatial_image(temp, dims=("y", "x"))
        img.attrs["transform"] = sdata_blobs["blobs_labels"].transform
        sdata_blobs["blobs_giant_labels"] = img

        sdata_blobs["table"].obs["region"] = "blobs_giant_labels"
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_giant_labels"

        sdata_blobs.pl.render_labels("blobs_giant_labels", scale="full").pl.show()

    def test_plot_can_stack_render_labels(self, sdata_blobs: SpatialData):
        (
            sdata_blobs.pl.render_labels(
                element="blobs_labels",
                na_color="red",
                fill_alpha=1,
                outline_alpha=0,
            )
            .pl.render_labels(element="blobs_labels", na_color="blue", fill_alpha=0, outline_alpha=1, contour_px=15)
            .pl.show()
        )

    def test_plot_can_color_labels_by_continuous_variable(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum").pl.show()

    def test_plot_can_color_labels_by_categorical_variable(self, sdata_blobs: SpatialData):
        max_col = sdata_blobs.table.to_df().idxmax(axis=1)
        max_col = pd.Categorical(max_col, categories=sdata_blobs.table.to_df().columns, ordered=True)
        sdata_blobs.table.obs["which_max"] = max_col

        sdata_blobs.pl.render_labels("blobs_labels", color="which_max").pl.show()

    @pytest.mark.parametrize(
        "label",
        [
            "blobs_labels",
            "blobs_multiscale_labels",
        ],
    )
    def test_plot_can_color_labels_by_categorical_variable_in_other_table(self, sdata_blobs: SpatialData, label: str):

        def _make_tablemodel_with_categorical_labels(sdata_blobs, label):

            adata = sdata_blobs.tables["table"].copy()
            max_col = adata.to_df().idxmax(axis=1)
            max_col = max_col.str.replace("channel_", "ch").str.replace("_sum", "")
            max_col = pd.Categorical(max_col, categories=set(max_col), ordered=True)
            adata.obs["which_max"] = max_col
            adata.obs["region"] = label
            del adata.uns["spatialdata_attrs"]
            table = TableModel.parse(
                adata=adata,
                region_key="region",
                instance_key="instance_id",
                region=label,
            )
            sdata_blobs.tables["other_table"] = table

            _, axs = plt.subplots(nrows=1, ncols=3, layout="tight")

            sdata_blobs.pl.render_labels(label, color="channel_1_sum", table="other_table", scale="scale0").pl.show(
                ax=axs[0], title="ch_1_sum", colorbar=False
            )
            sdata_blobs.pl.render_labels(label, color="channel_2_sum", table="other_table", scale="scale0").pl.show(
                ax=axs[1], title="ch_2_sum", colorbar=False
            )
            sdata_blobs.pl.render_labels(label, color="which_max", table="other_table", scale="scale0").pl.show(
                ax=axs[2], legend_fontsize=6
            )

        # we're modifying the data here, so we need an independent copy
        sdata_blobs_local = deepcopy(sdata_blobs)

        _make_tablemodel_with_categorical_labels(sdata_blobs_local, label)

    def test_plot_two_calls_with_coloring_result_in_two_colorbars(self, sdata_blobs: SpatialData):
        # we're modifying the data here so we need an independent copy
        sdata_blobs_local = deepcopy(sdata_blobs)

        table = sdata_blobs_local["table"].copy()
        table.obs["region"] = "blobs_multiscale_labels"
        table.uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        table = table[:, ~table.var_names.isin(["channel_0_sum"])]
        sdata_blobs_local["multi_table"] = table
        sdata_blobs_local.pl.render_labels("blobs_labels", color="channel_0_sum", table_name="table").pl.render_labels(
            "blobs_multiscale_labels", color="channel_1_sum", table_name="multi_table"
        ).pl.show()

    def test_plot_can_control_label_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels(
            "blobs_labels", color="channel_0_sum", outline_alpha=0.4, fill_alpha=0.0, contour_px=15
        ).pl.show()

    def test_plot_can_control_label_infill(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels(
            "blobs_labels",
            color="channel_0_sum",
            outline_alpha=0.0,
            fill_alpha=0.4,
        ).pl.show()

    def test_plot_label_colorbar_uses_alpha_of_less_transparent_infill(
        self,
        sdata_blobs: SpatialData,
    ):

        sdata_blobs.pl.render_labels(
            "blobs_labels", color="channel_0_sum", fill_alpha=0.1, outline_alpha=0.7, contour_px=15
        ).pl.show()

    def test_plot_label_colorbar_uses_alpha_of_less_transparent_outline(
        self,
        sdata_blobs: SpatialData,
    ):

        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum", fill_alpha=0.7, outline_alpha=0.1).pl.show()

    def test_can_plot_with_one_element_color_table(self, sdata_blobs: SpatialData):
        table = sdata_blobs["table"].copy()
        table.obs["region"] = "blobs_multiscale_labels"
        table.uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        table = table[:, ~table.var_names.isin(["channel_0_sum"])]
        sdata_blobs["multi_table"] = table
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum", table_name="table").pl.render_labels(
            "blobs_multiscale_labels", color="channel_1_sum", table_name="multi_table"
        ).pl.show()

    def test_plot_subset_categorical_label_maintains_order(self, sdata_blobs: SpatialData):
        max_col = sdata_blobs.table.to_df().idxmax(axis=1)
        max_col = pd.Categorical(max_col, categories=sdata_blobs.table.to_df().columns, ordered=True)
        sdata_blobs.table.obs["which_max"] = max_col

        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")

        sdata_blobs.pl.render_labels("blobs_labels", color="which_max").pl.show(ax=axs[0], legend_fontsize=6)
        sdata_blobs.pl.render_labels(
            "blobs_labels",
            color="which_max",
            groups=["channel_0_sum"],
        ).pl.show(ax=axs[1])

    def test_plot_subset_categorical_label_maintains_order_when_palette_overwrite(self, sdata_blobs: SpatialData):
        max_col = sdata_blobs.table.to_df().idxmax(axis=1)
        max_col = pd.Categorical(max_col, categories=sdata_blobs.table.to_df().columns, ordered=True)
        sdata_blobs.table.obs["which_max"] = max_col

        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")

        sdata_blobs.pl.render_labels("blobs_labels", color="which_max").pl.show(ax=axs[0], legend_fontsize=6)
        sdata_blobs.pl.render_labels(
            "blobs_labels", color="which_max", groups=["channel_0_sum"], palette="red"
        ).pl.show(ax=axs[1])
