import math

import dask.dataframe
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.colors import Normalize
from spatialdata import SpatialData, deepcopy
from spatialdata.models import PointsModel, TableModel
from spatialdata.transformations import Affine, Identity, MapAxis, Scale, Sequence, Translation
from spatialdata.transformations._utils import _set_transformations

import spatialdata_plot  # noqa: F401
from tests.conftest import DPI, PlotTester, PlotTesterMeta, _viridis_with_under_over

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

    def test_plot_can_filter_with_groups_default_palette(self, sdata_blobs: SpatialData):
        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")

        sdata_blobs["table"].obs["region"] = ["blobs_points"] * sdata_blobs["table"].n_obs
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"

        sdata_blobs.pl.render_points(color="genes", size=10).pl.show(ax=axs[0], legend_fontsize=6)
        sdata_blobs.pl.render_points(color="genes", groups="gene_b", size=10).pl.show(ax=axs[1], legend_fontsize=6)

    def test_plot_can_filter_with_groups_custom_palette(self, sdata_blobs: SpatialData):
        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")

        sdata_blobs["table"].obs["region"] = ["blobs_points"] * sdata_blobs["table"].n_obs
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"

        sdata_blobs.pl.render_points(color="genes", size=10).pl.show(ax=axs[0], legend_fontsize=6)
        sdata_blobs.pl.render_points(color="genes", groups="gene_b", size=10, palette="red").pl.show(
            ax=axs[1], legend_fontsize=6
        )

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

    def test_plot_points_categorical_color_column_matplotlib(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points("blobs_points", color="genes", method="matplotlib").pl.show()

    def test_plot_points_categorical_color_column_datashader(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points("blobs_points", color="genes", method="datashader").pl.show()

    def test_plot_points_continuous_color_column_matplotlib(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points("blobs_points", color="instance_id", method="matplotlib").pl.show()

    def test_plot_points_continuous_color_column_datashader(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points("blobs_points", color="instance_id", method="datashader").pl.show()

    def test_plot_datashader_matplotlib_stack(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points", size=40, color="red", method="datashader"
        ).pl.render_points(element="blobs_points", size=10, color="blue").pl.show()

    def test_plot_datashader_can_color_by_category(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            color="genes", groups="gene_b", palette="lightgreen", size=20, method="datashader"
        ).pl.show()

    def test_plot_datashader_can_use_sum_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points", size=40, color="instance_id", method="datashader", datashader_reduction="sum"
        ).pl.show()

    def test_plot_datashader_can_use_mean_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points", size=40, color="instance_id", method="datashader", datashader_reduction="mean"
        ).pl.show()

    def test_plot_datashader_can_use_any_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points", size=40, color="instance_id", method="datashader", datashader_reduction="any"
        ).pl.show()

    def test_plot_datashader_can_use_count_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points", size=40, color="instance_id", method="datashader", datashader_reduction="count"
        ).pl.show()

    def test_plot_datashader_can_use_std_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points", size=40, color="instance_id", method="datashader", datashader_reduction="std"
        ).pl.show()

    def test_plot_datashader_can_use_std_as_reduction_not_all_zero(self, sdata_blobs: SpatialData):
        # originally, all resulting std values are 0, here we alter the points to get at least one actual value
        blob = deepcopy(sdata_blobs)
        temp = blob["blobs_points"].compute()
        temp.loc[195, "x"] = 144
        temp.loc[195, "y"] = 159
        temp.loc[195, "instance_id"] = 13
        blob["blobs_points"] = PointsModel.parse(dask.dataframe.from_pandas(temp, 1), coordinates={"x": "x", "y": "y"})
        blob.pl.render_points(
            element="blobs_points", size=40, color="instance_id", method="datashader", datashader_reduction="std"
        ).pl.show()

    def test_plot_datashader_can_use_var_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points", size=40, color="instance_id", method="datashader", datashader_reduction="var"
        ).pl.show()

    def test_plot_datashader_can_use_max_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points", size=40, color="instance_id", method="datashader", datashader_reduction="max"
        ).pl.show()

    def test_plot_datashader_can_use_min_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points", size=40, color="instance_id", method="datashader", datashader_reduction="min"
        ).pl.show()

    def test_plot_mpl_and_datashader_point_sizes_agree_after_altered_dpi(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(element="blobs_points", size=400, color="blue").pl.render_points(
            element="blobs_points", size=400, color="yellow", method="datashader", alpha=0.8
        ).pl.show(dpi=200)

    def test_plot_points_transformed_ds_agrees_with_mpl(self):
        sdata = SpatialData(
            points={
                "points1": PointsModel.parse(
                    pd.DataFrame({"y": [0, 0, 10, 10, 4, 6, 4, 6], "x": [0, 10, 10, 0, 4, 6, 6, 4]}),
                    transformations={"global": Scale([2, 2], ("y", "x"))},
                )
            },
        )
        sdata.pl.render_points("points1", method="matplotlib", size=50, color="lightgrey").pl.render_points(
            "points1", method="datashader", size=10, color="red"
        ).pl.show()

    def test_plot_datashader_can_transform_points(self, sdata_blobs: SpatialData):
        theta = math.pi / 1.7
        rotation = Affine(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ],
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )

        scale = Scale([-1.3, 1.8], axes=("x", "y"))
        identity = Identity()
        mapaxis = MapAxis({"x": "y", "y": "x"})
        translation = Translation([20, -65], ("x", "y"))
        seq = Sequence([mapaxis, scale, identity, translation, rotation])

        _set_transformations(sdata_blobs["blobs_points"], {"global": seq})

        sdata_blobs.pl.render_points("blobs_points", method="datashader", color="black", size=5).pl.show()

    def test_plot_can_use_norm_with_clip(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            color="instance_id", size=40, norm=Normalize(3, 7, clip=True), cmap=_viridis_with_under_over()
        ).pl.show()

    def test_plot_can_use_norm_without_clip(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            color="instance_id", size=40, norm=Normalize(3, 7, clip=False), cmap=_viridis_with_under_over()
        ).pl.show()

    def test_plot_datashader_can_use_norm_with_clip(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            color="instance_id",
            size=40,
            norm=Normalize(3, 7, clip=True),
            cmap=_viridis_with_under_over(),
            method="datashader",
            datashader_reduction="max",
        ).pl.show()

    def test_plot_datashader_can_use_norm_without_clip(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            color="instance_id",
            size=40,
            norm=Normalize(3, 7, clip=False),
            cmap=_viridis_with_under_over(),
            method="datashader",
            datashader_reduction="max",
        ).pl.show()

    def test_plot_datashader_norm_vmin_eq_vmax_with_clip(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            color="instance_id",
            size=40,
            norm=Normalize(5, 5, clip=True),
            cmap=_viridis_with_under_over(),
            method="datashader",
            datashader_reduction="max",
        ).pl.show()

    def test_plot_datashader_norm_vmin_eq_vmax_without_clip(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            color="instance_id",
            size=40,
            norm=Normalize(5, 5, clip=False),
            cmap=_viridis_with_under_over(),
            method="datashader",
            datashader_reduction="max",
        ).pl.show()

    def test_plot_can_annotate_points_with_table_obs(self, sdata_blobs: SpatialData):
        nrows, ncols = 200, 3
        feature_matrix = RNG.random((nrows, ncols))
        var_names = [f"feature{i}" for i in range(ncols)]

        obs_indices = sdata_blobs["blobs_points"].index

        obs = pd.DataFrame()
        obs["instance_id"] = obs_indices
        obs["region"] = "blobs_points"
        obs["region"].astype("category")
        obs["extra_feature"] = [1, 2] * 100

        table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
        table = TableModel.parse(table, region="blobs_points", region_key="region", instance_key="instance_id")
        sdata_blobs["points_table"] = table

        sdata_blobs.pl.render_points("blobs_points", color="extra_feature", size=10).pl.show()

    def test_plot_can_annotate_points_with_table_X(self, sdata_blobs: SpatialData):
        nrows, ncols = 200, 3
        feature_matrix = RNG.random((nrows, ncols))
        var_names = [f"feature{i}" for i in range(ncols)]

        obs_indices = sdata_blobs["blobs_points"].index

        obs = pd.DataFrame()
        obs["instance_id"] = obs_indices
        obs["region"] = "blobs_points"
        obs["region"].astype("category")

        table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
        table = TableModel.parse(table, region="blobs_points", region_key="region", instance_key="instance_id")
        sdata_blobs["points_table"] = table

        sdata_blobs.pl.render_points("blobs_points", color="feature0", size=10).pl.show()

    def test_plot_can_annotate_points_with_table_and_groups(self, sdata_blobs: SpatialData):
        nrows, ncols = 200, 3
        feature_matrix = RNG.random((nrows, ncols))
        var_names = [f"feature{i}" for i in range(ncols)]

        obs_indices = sdata_blobs["blobs_points"].index

        obs = pd.DataFrame()
        obs["instance_id"] = obs_indices
        obs["region"] = "blobs_points"
        obs["region"].astype("category")
        obs["extra_feature_cat"] = ["one", "two"] * 100

        table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
        table = TableModel.parse(table, region="blobs_points", region_key="region", instance_key="instance_id")
        sdata_blobs["points_table"] = table

        sdata_blobs.pl.render_points("blobs_points", color="extra_feature_cat", groups="two", size=10).pl.show()

    def test_plot_can_annotate_points_with_table_layer(self, sdata_blobs: SpatialData):
        nrows, ncols = 200, 3
        feature_matrix = RNG.random((nrows, ncols))
        var_names = [f"feature{i}" for i in range(ncols)]

        obs_indices = sdata_blobs["blobs_points"].index

        obs = pd.DataFrame()
        obs["instance_id"] = obs_indices
        obs["region"] = "blobs_points"
        obs["region"].astype("category")

        table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
        table = TableModel.parse(table, region="blobs_points", region_key="region", instance_key="instance_id")
        sdata_blobs["points_table"] = table
        sdata_blobs["points_table"].layers["normalized"] = RNG.random((nrows, ncols))

        sdata_blobs.pl.render_points("blobs_points", color="feature0", size=10, table_layer="normalized").pl.show()
