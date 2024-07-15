import dask.array as da
import matplotlib
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import spatialdata_plot  # noqa: F401
from anndata import AnnData
from spatial_image import to_spatial_image
from spatialdata import SpatialData, get_element_instances
from spatialdata.models import TableModel

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
                element="blobs_labels", na_color="red", fill_alpha=1, outline_alpha=0, outline=False
            )
            .pl.render_labels(
                element="blobs_labels", na_color="blue", fill_alpha=0, outline_alpha=1, outline=True, contour_px=15
            )
            .pl.show()
        )

    def test_plot_can_color_labels_by_continuous_variable(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum").pl.show()

    def test_plot_can_color_labels(self, sdata_blobs: SpatialData):
        table = sdata_blobs["table"].copy()
        table.obs["region"] = "blobs_multiscale_labels"
        table.uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        table = table[:, ~table.var_names.isin(["channel_0_sum"])]
        sdata_blobs["multi_table"] = table
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum", table_name="table").pl.render_labels(
            "blobs_multiscale_labels", color="channel_1_sum", table_name="multi_table"
        ).pl.show()

    def test_can_plot_with_one_element_color_table(self, sdata_blobs: SpatialData):
        table = sdata_blobs["table"].copy()
        table.obs["region"] = "blobs_multiscale_labels"
        table.uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        table = table[:, ~table.var_names.isin(["channel_0_sum"])]
        sdata_blobs["multi_table"] = table
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum", table_name="table").pl.render_labels(
            "blobs_multiscale_labels", color="channel_1_sum", table_name="multi_table"
        ).pl.show()

    @pytest.mark.parametrize(
        "label",
        [
            "blobs_labels",
            "blobs_multiscale_labels",
        ],
    )
    def test_plot_label_categorical_color(self, sdata_blobs: SpatialData, label: str):
        self._make_tablemodel_with_categorical_labels(sdata_blobs, label)

    def _make_tablemodel_with_categorical_labels(self, sdata_blobs, label):

        n_obs = max(get_element_instances(sdata_blobs[label]))
        vals = np.arange(n_obs)
        obs = pd.DataFrame({"a": vals, "b": vals + 0.3, "c": vals + 0.7})

        adata = AnnData(vals.reshape(-1, 1), obs=obs)
        adata.obs["instance_id"] = vals
        adata.obs["category"] = list(["a", "b", "c"] * ((n_obs // 3) + 1))[:n_obs]
        adata.obs["region"] = label
        table = TableModel.parse(
            adata=adata,
            region_key="region",
            instance_key="instance_id",
            region=label,
        )
        sdata_blobs["other_table"] = table
        sdata_blobs["other_table"].obs["category"] = sdata_blobs["other_table"].obs["category"].astype("category")
        sdata_blobs.pl.render_labels(label, color="category").pl.show()
