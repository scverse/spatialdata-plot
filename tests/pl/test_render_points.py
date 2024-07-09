import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata_plot  # noqa: F401
from anndata import AnnData
from spatialdata import SpatialData
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


class TestPoints(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_points(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(element="blobs_points").pl.show()

    def test_plot_can_filter_with_groups(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = ["blobs_points"] * sdata_blobs["table"].n_obs
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"
        sdata_blobs.pl.render_points(color="genes", groups="gene_b", palette="orange").pl.show()

    def test_plot_can_filter_with_groups_default_palette(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = ["blobs_points"] * sdata_blobs["table"].n_obs
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"
        sdata_blobs.pl.render_points(color="genes", groups="gene_b").pl.show()

    def test_plot_coloring_with_palette(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = ["blobs_points"] * sdata_blobs["table"].n_obs
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"
        sdata_blobs.pl.render_points(
            color="genes", groups=["gene_a", "gene_b"], palette=["lightgreen", "darkblue"]
        ).pl.show()

    def test_plot_coloring_with_cmap(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = ["blobs_points"] * sdata_blobs["table"].n_obs
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"
        sdata_blobs.pl.render_points(color="genes", cmap="rainbow").pl.show()

    def test_plot_can_stack_render_points(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = ["blobs_points"] * sdata_blobs["table"].n_obs
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"
        (
            sdata_blobs.pl.render_points(element="blobs_points", na_color="red", size=30)
            .pl.render_points(element="blobs_points", na_color="blue", size=10)
            .pl.show()
        )

    def test_plot_color_recognises_actual_color_as_color(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(element="blobs_points", color="red").pl.show()

    def test_plot_points_coercable_categorical_color(self, sdata_blobs: SpatialData):
        n_obs = len(sdata_blobs["blobs_points"])
        adata = AnnData(
            RNG.normal(size=(n_obs, 10)), obs=pd.DataFrame(RNG.normal(size=(n_obs, 3)), columns=["a", "b", "c"])
        )
        adata.obs["instance_id"] = np.arange(adata.n_obs)
        adata.obs["category"] = RNG.choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs["instance_id"] = list(range(adata.n_obs))
        adata.obs["region"] = "blobs_points"
        table = TableModel.parse(adata=adata, region_key="region", instance_key="instance_id", region="blobs_points")
        sdata_blobs["other_table"] = table

        sdata_blobs.pl.render_points("blobs_points", color="category").pl.show()

    def test_plot_points_categorical_color(self, sdata_blobs: SpatialData):
        n_obs = len(sdata_blobs["blobs_points"])
        adata = AnnData(
            RNG.normal(size=(n_obs, 10)), obs=pd.DataFrame(RNG.normal(size=(n_obs, 3)), columns=["a", "b", "c"])
        )
        adata.obs["instance_id"] = np.arange(adata.n_obs)
        adata.obs["category"] = RNG.choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs["instance_id"] = list(range(adata.n_obs))
        adata.obs["region"] = "blobs_points"
        table = TableModel.parse(adata=adata, region_key="region", instance_key="instance_id", region="blobs_points")
        sdata_blobs["other_table"] = table

        sdata_blobs["other_table"].obs["category"] = sdata_blobs["other_table"].obs["category"].astype("category")
        sdata_blobs.pl.render_points("blobs_points", color="category").pl.show()

    def test_plot_datashader_continuous_color(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points", size=40, color="instance_id", alpha=0.6, method="datashader"
        ).pl.show()

    def test_plot_datashader_matplotlib_stack(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points", size=40, color="red", method="datashader"
        ).pl.render_points(element="blobs_points", size=10, color="blue").pl.show()

    def test_plot_datashader_can_color_by_category(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            color="genes", groups="gene_b", palette="lightgreen", size=20, method="datashader"
        ).pl.show()
