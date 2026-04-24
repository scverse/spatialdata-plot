import math

import dask.dataframe
import datashader as ds
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from matplotlib.colors import Normalize
from spatialdata import SpatialData, deepcopy
from spatialdata.models import PointsModel, TableModel
from spatialdata.transformations import (
    Affine,
    Identity,
    MapAxis,
    Scale,
    Sequence,
    Translation,
)
from spatialdata.transformations._utils import _set_transformations

import spatialdata_plot  # noqa: F401
from spatialdata_plot._logging import logger, logger_no_warns, logger_warns
from spatialdata_plot.pl._datashader import (
    _build_datashader_color_key,
    _ds_aggregate,
    _ds_shade_categorical,
)
from spatialdata_plot.pl.render import _warn_groups_ignored_continuous
from tests.conftest import DPI, PlotTester, PlotTesterMeta, _viridis_with_under_over, get_standard_RNG

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

        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_points"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"

        sdata_blobs.pl.render_points(color="genes", size=10).pl.show(ax=axs[0], legend_fontsize=6)
        sdata_blobs.pl.render_points(color="genes", groups="gene_b", size=10).pl.show(ax=axs[1], legend_fontsize=6)

    def test_plot_can_filter_with_groups_custom_palette(self, sdata_blobs: SpatialData):
        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")

        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_points"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"

        sdata_blobs.pl.render_points(color="genes", size=10).pl.show(ax=axs[0], legend_fontsize=6)
        sdata_blobs.pl.render_points(color="genes", groups="gene_b", size=10, palette="red").pl.show(
            ax=axs[1], legend_fontsize=6
        )

    def test_plot_coloring_with_palette(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_points"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"
        sdata_blobs.pl.render_points(
            color="genes",
            groups=["gene_a", "gene_b"],
            palette=["lightgreen", "darkblue"],
        ).pl.show()

    def test_plot_respects_custom_colors_from_uns_for_points(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_points"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"

        # set a custom palette in `.uns` for the categorical column
        sdata_blobs["table"].uns["genes_colors"] = ["#800080", "#008000", "#FFFF00"]

        sdata_blobs.pl.render_points(
            element="blobs_points",
            color="genes",
        ).pl.show()

    def test_plot_coloring_with_cmap(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_points"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"
        sdata_blobs.pl.render_points(color="genes", cmap="rainbow").pl.show()

    def test_plot_can_stack_render_points(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_points"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"
        (
            sdata_blobs.pl.render_points(element="blobs_points", na_color="red", size=30)
            .pl.render_points(element="blobs_points", na_color="blue", size=10)
            .pl.show()
        )

    def test_plot_can_color_by_color_name(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(element="blobs_points", color="red").pl.show()

    def test_plot_can_color_by_rgb_array(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(element="blobs_points", color=[0.5, 0.5, 1.0]).pl.show()

    def test_plot_can_color_by_rgba_array(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(element="blobs_points", color=[0.5, 0.5, 1.0, 0.5]).pl.show()

    def test_plot_can_color_by_hex(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(element="blobs_points", color="#88a136").pl.show()

    def test_plot_can_color_by_hex_with_alpha(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(element="blobs_points", color="#88a13688").pl.show()

    def test_plot_alpha_overwrites_opacity_from_color(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(element="blobs_points", color=[0.5, 0.5, 1.0, 0.5], alpha=1.0).pl.show()

    def test_plot_points_coercable_categorical_color(self, sdata_blobs: SpatialData):
        n_obs = len(sdata_blobs["blobs_points"])
        adata = AnnData(
            get_standard_RNG().normal(size=(n_obs, 10)),
            obs=pd.DataFrame(get_standard_RNG().normal(size=(n_obs, 3)), columns=["a", "b", "c"]),
        )
        adata.obs["instance_id"] = np.arange(adata.n_obs)
        adata.obs["category"] = get_standard_RNG().choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs["instance_id"] = list(range(adata.n_obs))
        adata.obs["region"] = "blobs_points"
        table = TableModel.parse(
            adata=adata,
            region_key="region",
            instance_key="instance_id",
            region="blobs_points",
        )
        sdata_blobs["other_table"] = table

        sdata_blobs.pl.render_points("blobs_points", color="category").pl.show()

    def test_plot_points_categorical_color(self, sdata_blobs: SpatialData):
        n_obs = len(sdata_blobs["blobs_points"])
        adata = AnnData(
            get_standard_RNG().normal(size=(n_obs, 10)),
            obs=pd.DataFrame(get_standard_RNG().normal(size=(n_obs, 3)), columns=["a", "b", "c"]),
        )
        adata.obs["instance_id"] = np.arange(adata.n_obs)
        adata.obs["category"] = get_standard_RNG().choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs["instance_id"] = list(range(adata.n_obs))
        adata.obs["region"] = "blobs_points"
        table = TableModel.parse(
            adata=adata,
            region_key="region",
            instance_key="instance_id",
            region="blobs_points",
        )
        sdata_blobs["other_table"] = table

        sdata_blobs["other_table"].obs["category"] = sdata_blobs["other_table"].obs["category"].astype("category")
        sdata_blobs.pl.render_points("blobs_points", color="category").pl.show()

    def test_plot_datashader_continuous_color(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points",
            size=40,
            color="instance_id",
            alpha=0.6,
            method="datashader",
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
            color="genes",
            groups="gene_b",
            palette="lightgreen",
            size=20,
            method="datashader",
        ).pl.show()

    def test_render_points_missing_color_column_raises_key_error(self, sdata_blobs: SpatialData) -> None:
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_points"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"
        with pytest.raises(KeyError, match="does_not_exist"):
            sdata_blobs.pl.render_points(element="blobs_points", color="does_not_exist")

    def test_render_points_missing_region_for_table_raises_key_error(self, sdata_blobs: SpatialData) -> None:
        blob = deepcopy(sdata_blobs)
        blob["table"].obs["region"] = pd.Categorical(["blobs_points"] * blob["table"].n_obs)
        blob["table"].uns["spatialdata_attrs"]["region"] = "blobs_points"
        blob["table"].obs["table_value"] = np.arange(blob["table"].n_obs)
        other_table = blob["table"].copy()
        other_table.obs["region"] = pd.Categorical(["other"] * other_table.n_obs)
        other_table.uns["spatialdata_attrs"]["region"] = "other"
        blob["other_table"] = other_table
        with pytest.raises(KeyError, match="does not annotate element"):
            blob.pl.render_points(element="blobs_points", color="table_value", table_name="other_table")

    def test_plot_datashader_colors_from_table_obs(self, sdata_blobs: SpatialData):
        n_obs = len(sdata_blobs["blobs_points"])
        obs = pd.DataFrame(
            {
                "instance_id": np.arange(n_obs),
                "region": pd.Categorical(["blobs_points"] * n_obs),
                "foo": pd.Categorical(np.where(np.arange(n_obs) % 2 == 0, "a", "b")),
            }
        )

        table = TableModel.parse(
            adata=AnnData(get_standard_RNG().normal(size=(n_obs, 3)), obs=obs),
            region="blobs_points",
            region_key="region",
            instance_key="instance_id",
        )
        sdata_blobs["datashader_table"] = table

        sdata_blobs.pl.render_points(
            "blobs_points",
            color="foo",
            table_name="datashader_table",
            method="datashader",
            size=5,
        ).pl.show()

    def test_plot_datashader_can_use_sum_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points",
            size=40,
            color="instance_id",
            method="datashader",
            datashader_reduction="sum",
        ).pl.show()

    def test_plot_datashader_can_use_mean_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points",
            size=40,
            color="instance_id",
            method="datashader",
            datashader_reduction="mean",
        ).pl.show()

    def test_plot_datashader_can_use_any_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points",
            size=40,
            color="instance_id",
            method="datashader",
            datashader_reduction="any",
        ).pl.show()

    def test_plot_datashader_can_use_count_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points",
            size=40,
            color="instance_id",
            method="datashader",
            datashader_reduction="count",
        ).pl.show()

    def test_plot_datashader_can_use_std_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points",
            size=40,
            color="instance_id",
            method="datashader",
            datashader_reduction="std",
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
            element="blobs_points",
            size=40,
            color="instance_id",
            method="datashader",
            datashader_reduction="std",
        ).pl.show()

    def test_plot_datashader_can_use_var_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points",
            size=40,
            color="instance_id",
            method="datashader",
            datashader_reduction="var",
        ).pl.show()

    def test_plot_datashader_can_use_max_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points",
            size=40,
            color="instance_id",
            method="datashader",
            datashader_reduction="max",
        ).pl.show()

    def test_plot_datashader_can_use_min_as_reduction(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            element="blobs_points",
            size=40,
            color="instance_id",
            method="datashader",
            datashader_reduction="min",
        ).pl.show()

    def test_plot_mpl_and_datashader_point_sizes_agree_after_altered_dpi(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(element="blobs_points", size=400, color="blue").pl.render_points(
            element="blobs_points",
            size=400,
            color="yellow",
            method="datashader",
            alpha=0.8,
        ).pl.show(dpi=200)

    def test_plot_points_transformed_ds_agrees_with_mpl(self):
        sdata = SpatialData(
            points={
                "points1": PointsModel.parse(
                    pd.DataFrame(
                        {
                            "y": [0, 0, 10, 10, 4, 6, 4, 6],
                            "x": [0, 10, 10, 0, 4, 6, 6, 4],
                        }
                    ),
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
            color="instance_id",
            size=40,
            norm=Normalize(3, 7, clip=True),
            cmap=_viridis_with_under_over(),
        ).pl.show()

    def test_plot_can_use_norm_without_clip(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            color="instance_id",
            size=40,
            norm=Normalize(3, 7, clip=False),
            cmap=_viridis_with_under_over(),
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
        feature_matrix = get_standard_RNG().random((nrows, ncols))
        var_names = [f"feature{i}" for i in range(ncols)]

        obs_indices = sdata_blobs["blobs_points"].index

        obs = pd.DataFrame()
        obs["instance_id"] = obs_indices
        obs["region"] = "blobs_points"
        obs["region"].astype("category")
        obs["extra_feature"] = [1, 2] * 100

        table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
        table = TableModel.parse(
            table,
            region="blobs_points",
            region_key="region",
            instance_key="instance_id",
        )
        sdata_blobs["points_table"] = table

        sdata_blobs.pl.render_points("blobs_points", color="extra_feature", size=10).pl.show()

    def test_plot_can_annotate_points_with_table_X(self, sdata_blobs: SpatialData):
        nrows, ncols = 200, 3
        feature_matrix = get_standard_RNG().random((nrows, ncols))
        var_names = [f"feature{i}" for i in range(ncols)]

        obs_indices = sdata_blobs["blobs_points"].index

        obs = pd.DataFrame()
        obs["instance_id"] = obs_indices
        obs["region"] = "blobs_points"
        obs["region"].astype("category")

        table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
        table = TableModel.parse(
            table,
            region="blobs_points",
            region_key="region",
            instance_key="instance_id",
        )
        sdata_blobs["points_table"] = table

        sdata_blobs.pl.render_points("blobs_points", color="feature0", size=10).pl.show()

    def test_plot_can_annotate_points_with_table_and_groups(self, sdata_blobs: SpatialData):
        nrows, ncols = 200, 3
        feature_matrix = get_standard_RNG().random((nrows, ncols))
        var_names = [f"feature{i}" for i in range(ncols)]

        obs_indices = sdata_blobs["blobs_points"].index

        obs = pd.DataFrame()
        obs["instance_id"] = obs_indices
        obs["region"] = "blobs_points"
        obs["region"].astype("category")
        obs["extra_feature_cat"] = ["one", "two"] * 100

        table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
        table = TableModel.parse(
            table,
            region="blobs_points",
            region_key="region",
            instance_key="instance_id",
        )
        sdata_blobs["points_table"] = table

        sdata_blobs.pl.render_points("blobs_points", color="extra_feature_cat", groups="two", size=10).pl.show()

    def test_plot_can_annotate_points_with_table_layer(self, sdata_blobs: SpatialData):
        nrows, ncols = 200, 3
        feature_matrix = get_standard_RNG().random((nrows, ncols))
        var_names = [f"feature{i}" for i in range(ncols)]

        obs_indices = sdata_blobs["blobs_points"].index

        obs = pd.DataFrame()
        obs["instance_id"] = obs_indices
        obs["region"] = "blobs_points"
        obs["region"].astype("category")

        table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
        table = TableModel.parse(
            table,
            region="blobs_points",
            region_key="region",
            instance_key="instance_id",
        )
        sdata_blobs["points_table"] = table
        sdata_blobs["points_table"].layers["normalized"] = get_standard_RNG().random((nrows, ncols))

        sdata_blobs.pl.render_points("blobs_points", color="feature0", size=10, table_layer="normalized").pl.show()

    def test_plot_can_annotate_points_with_nan_in_table_obs_categorical_matplotlib(
        self, sdata_blobs_points_with_nans_in_table: SpatialData
    ):
        sdata_blobs_points_with_nans_in_table.pl.render_points(
            "blobs_points", color="category", size=40, method="matplotlib"
        ).pl.show()

    def test_plot_can_annotate_points_with_nan_in_table_obs_categorical_datashader(
        self, sdata_blobs_points_with_nans_in_table: SpatialData
    ):
        sdata_blobs_points_with_nans_in_table.pl.render_points(
            "blobs_points", color="category", size=40, method="datashader"
        ).pl.show()

    def test_plot_can_annotate_points_with_nan_in_table_obs_continuous(
        self, sdata_blobs_points_with_nans_in_table: SpatialData
    ):
        sdata_blobs_points_with_nans_in_table.pl.render_points("blobs_points", color="col_a", size=30).pl.show()

    def test_plot_can_annotate_points_with_nan_in_table_obs_continuous_datashader(
        self, sdata_blobs_points_with_nans_in_table: SpatialData
    ):
        sdata_blobs_points_with_nans_in_table.pl.render_points(
            "blobs_points", color="col_a", size=40, method="datashader"
        ).pl.show()

    def test_plot_can_annotate_points_with_nan_in_table_X_continuous(
        self, sdata_blobs_points_with_nans_in_table: SpatialData
    ):
        sdata_blobs_points_with_nans_in_table.pl.render_points("blobs_points", color="col1", size=30).pl.show()

    def test_plot_can_annotate_points_with_nan_in_table_X_continuous_datashader(
        self, sdata_blobs_points_with_nans_in_table: SpatialData
    ):
        sdata_blobs_points_with_nans_in_table.pl.render_points(
            "blobs_points", color="col1", size=40, method="datashader"
        ).pl.show()

    def test_plot_can_annotate_points_with_nan_in_df_categorical(self, sdata_blobs: SpatialData):
        sdata_blobs["blobs_points"]["cat_color"] = pd.Series([np.nan, "a", "b", "c"] * 50, dtype="category")
        sdata_blobs.pl.render_points("blobs_points", color="cat_color", size=30).pl.show()

    def test_plot_can_annotate_points_with_nan_in_df_categorical_datashader(self, sdata_blobs: SpatialData):
        sdata_blobs["blobs_points"]["cat_color"] = pd.Series([np.nan, "a", "b", "c"] * 50, dtype="category")
        sdata_blobs.pl.render_points("blobs_points", color="cat_color", size=40, method="datashader").pl.show()

    def test_plot_can_annotate_points_with_nan_in_df_continuous(self, sdata_blobs: SpatialData):
        sdata_blobs["blobs_points"]["cont_color"] = pd.Series([np.nan, 2, 9, 13] * 50)
        sdata_blobs.pl.render_points("blobs_points", color="cont_color", size=30).pl.show()

    def test_plot_can_annotate_points_with_nan_in_df_continuous_datashader(self, sdata_blobs: SpatialData):
        sdata_blobs["blobs_points"]["cont_color"] = pd.Series([np.nan, 2, 9, 13] * 50)
        sdata_blobs.pl.render_points("blobs_points", color="cont_color", size=40, method="datashader").pl.show()

    def test_plot_groups_na_color_none_filters_points(self, sdata_blobs: SpatialData):
        """With groups, non-matching points are filtered by default; na_color='red' keeps them visible."""
        sdata_blobs["blobs_points"]["cat_color"] = pd.Series(["a", "b", "c", "a"] * 50, dtype="category")
        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")
        sdata_blobs.pl.render_points("blobs_points", color="cat_color", groups=["a"], na_color="red", size=30).pl.show(
            ax=axs[0], title="na_color='red'"
        )
        sdata_blobs.pl.render_points("blobs_points", color="cat_color", groups=["a"], size=30).pl.show(
            ax=axs[1], title="default (filtered)"
        )

    def test_plot_groups_na_color_none_filters_points_datashader(self, sdata_blobs: SpatialData):
        """With groups + datashader, non-matching points are filtered by default."""
        sdata_blobs["blobs_points"]["cat_color"] = pd.Series(["a", "b", "c", "a"] * 50, dtype="category")
        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")
        sdata_blobs.pl.render_points(
            "blobs_points", color="cat_color", groups=["a"], na_color="red", size=30, method="datashader"
        ).pl.show(ax=axs[0], title="na_color='red'")
        sdata_blobs.pl.render_points(
            "blobs_points", color="cat_color", groups=["a"], size=30, method="datashader"
        ).pl.show(ax=axs[1], title="default (filtered)")

    @staticmethod
    def _make_sampled_sdata() -> SpatialData:
        """Points with two spatially separated clusters, shuffled via .sample() (#358)."""
        rng = get_standard_RNG()
        n = 100
        x = np.concatenate([rng.uniform(0, 10, n // 2), rng.uniform(90, 100, n // 2)])
        y = np.concatenate([rng.uniform(0, 10, n // 2), rng.uniform(90, 100, n // 2)])
        df = pd.DataFrame({"x": x, "y": y, "cluster": pd.Categorical(["A"] * (n // 2) + ["B"] * (n // 2))})
        sdata = SpatialData(points={"pts": PointsModel.parse(df)})
        sampled = sdata.points["pts"].compute().sample(frac=0.8, random_state=42)
        sdata.points["pts"] = PointsModel.parse(sampled)
        return sdata

    def test_plot_sampled_points_categorical_color_matplotlib(self):
        """Regression test for #358: .sample() must not shuffle categorical colors."""
        self._make_sampled_sdata().pl.render_points("pts", color="cluster", method="matplotlib").pl.show()

    def test_plot_sampled_points_categorical_color_datashader(self):
        """Regression test for #358: .sample() must not shuffle categorical colors."""
        self._make_sampled_sdata().pl.render_points("pts", color="cluster", method="datashader").pl.show()

    def test_plot_can_color_points_by_gene_symbols(self, sdata_blobs: SpatialData):
        """Color points by gene symbol alias instead of var_name (#247)."""
        rng = get_standard_RNG()
        pts = sdata_blobs["blobs_points"].compute()
        n_obs = len(pts)
        # Assign unique instance IDs to each point
        pts["instance_id"] = np.arange(n_obs)
        sdata_blobs["blobs_points"] = PointsModel.parse(pts)
        adata = AnnData(
            X=rng.random((n_obs, 3)),
            var=pd.DataFrame({"gene_symbol": ["GeneA", "GeneB", "GeneC"]}, index=["f0", "f1", "f2"]),
        )
        adata.obs["region"] = pd.Categorical(["blobs_points"] * n_obs)
        adata.obs["instance_id"] = np.arange(n_obs)
        table = TableModel.parse(adata, region="blobs_points", region_key="region", instance_key="instance_id")
        sdata_blobs["table"] = table
        sdata_blobs.pl.render_points(
            "blobs_points", color="GeneA", table_name="table", gene_symbols="gene_symbol", size=10
        ).pl.show()


def test_groups_na_color_none_no_match_points(sdata_blobs: SpatialData):
    """When no elements match the groups, the plot should render without error."""
    sdata_blobs["blobs_points"]["cat_color"] = pd.Series(["a", "b", "c", "a"] * 50, dtype="category")
    sdata_blobs.pl.render_points(
        "blobs_points", color="cat_color", groups=["nonexistent"], na_color=None, size=30
    ).pl.show()


@pytest.mark.parametrize("na_color", [None, "red"])
def test_groups_warns_when_no_groups_match_points(sdata_blobs: SpatialData, caplog, na_color):
    """Warning fires regardless of na_color when no groups match."""
    sdata_blobs["blobs_points"]["cat_color"] = pd.Series(["a", "b", "c", "a"] * 50, dtype="category")
    with logger_warns(caplog, logger, match="None of the requested groups"):
        sdata_blobs.pl.render_points(
            "blobs_points", color="cat_color", groups=["nonexistent"], na_color=na_color, size=30
        ).pl.show()


@pytest.mark.parametrize("na_color", [None, "red"])
def test_groups_warns_when_some_groups_missing_points(sdata_blobs: SpatialData, caplog, na_color):
    """Warning fires regardless of na_color when some groups are missing."""
    sdata_blobs["blobs_points"]["cat_color"] = pd.Series(["a", "b", "c", "a"] * 50, dtype="category")
    with logger_warns(caplog, logger, match="were not found in"):
        sdata_blobs.pl.render_points(
            "blobs_points", color="cat_color", groups=["a", "nonexistent"], na_color=na_color, size=30
        ).pl.show()


def test_raises_when_table_does_not_annotate_element(sdata_blobs: SpatialData):
    # Work on an independent copy since we mutate tables
    sdata_blobs_local = deepcopy(sdata_blobs)

    # Create a table that annotates a DIFFERENT element than the one we will render
    other_table = sdata_blobs_local["table"].copy()
    other_table.obs["region"] = pd.Categorical(["blobs_labels"] * other_table.n_obs)  # Different from blobs_points
    other_table.uns["spatialdata_attrs"]["region"] = "blobs_labels"
    sdata_blobs_local["other_table"] = other_table

    # Rendering "blobs_points" with a table that annotates "blobs_labels"
    # should now raise to alert the user about the mismatch.
    with pytest.raises(
        KeyError,
        match="Table 'other_table' does not annotate element 'blobs_points'",
    ):
        sdata_blobs_local.pl.render_points(
            "blobs_points",
            color="channel_0_sum",
            table_name="other_table",
        ).pl.show()


def test_datashader_colors_points_from_table_obs(sdata_blobs: SpatialData):
    # Fast regression for https://github.com/scverse/spatialdata-plot/issues/479.
    n_obs = len(sdata_blobs["blobs_points"])
    obs = pd.DataFrame(
        {
            "instance_id": np.arange(n_obs),
            "region": pd.Categorical(["blobs_points"] * n_obs),
            "foo": pd.Categorical(np.where(np.arange(n_obs) % 2 == 0, "a", "b")),
        }
    )

    table = TableModel.parse(
        adata=AnnData(get_standard_RNG().normal(size=(n_obs, 3)), obs=obs),
        region="blobs_points",
        region_key="region",
        instance_key="instance_id",
    )
    sdata_blobs["datashader_table"] = table

    sdata_blobs.pl.render_points(
        "blobs_points",
        color="foo",
        table_name="datashader_table",
        method="datashader",
        size=5,
    ).pl.show()


def test_plot_datashader_single_category_points(sdata_blobs: SpatialData):
    """Datashader with a single-category Categorical must not raise.

    Regression test for https://github.com/scverse/spatialdata-plot/issues/483.
    Before the fix, color_key was None when there was only 1 category, but ds.by()
    still produced a 3D DataArray, causing datashader to raise:
        ValueError: Color key must be provided, with at least as many colors as
        there are categorical fields
    """
    n_obs = len(sdata_blobs["blobs_points"])
    obs = pd.DataFrame(
        {
            "instance_id": np.arange(n_obs),
            "region": pd.Categorical(["blobs_points"] * n_obs),
            "foo": pd.Categorical(["only_cat"] * n_obs),
        }
    )
    table = TableModel.parse(
        adata=AnnData(get_standard_RNG().normal(size=(n_obs, 3)), obs=obs),
        region="blobs_points",
        region_key="region",
        instance_key="instance_id",
    )
    sdata_blobs["single_cat_table"] = table

    sdata_blobs.pl.render_points(
        "blobs_points",
        color="foo",
        table_name="single_cat_table",
        method="datashader",
        size=5,
    ).pl.show()


def test_datashader_points_visible_with_nonuniform_scale(sdata_blobs: SpatialData):
    """Datashader points must remain visible when data has a non-square aspect ratio.

    Regression test for https://github.com/scverse/spatialdata-plot/issues/445.
    Before the fix, the datashader canvas was oversized on the longer axis, causing
    spread(px) to be downscaled to sub-pixel size on display.
    """
    _set_transformations(sdata_blobs["blobs_points"], {"global": Scale([1, 5], axes=("x", "y"))})
    sdata_blobs.pl.render_points("blobs_points", method="datashader", color="black").pl.show()


def test_datashader_alpha_not_applied_twice(sdata_blobs: SpatialData):
    """Datashader alpha must not be applied twice (once in shade, once in imshow).

    Regression test for https://github.com/scverse/spatialdata-plot/issues/367.
    Before the fix, alpha was passed both to ds.tf.shade(min_alpha=...) and to
    ax.imshow(alpha=...), resulting in effective transparency of alpha**2.
    """
    fig, ax = plt.subplots()
    sdata_blobs.pl.render_points(method="datashader", alpha=0.5, color="red").pl.show(ax=ax)

    axes_images = [c for c in ax.get_children() if isinstance(c, matplotlib.image.AxesImage)]
    for img in axes_images:
        assert img.get_alpha() is None, (
            f"Datashader AxesImage has alpha={img.get_alpha()}, which would be applied "
            "on top of the alpha already in the RGBA channels — causing double transparency."
        )
    plt.close(fig)


def _make_ds_canvas_and_df(n=500, seed=42):
    """Small datashader Canvas + DataFrame with x, y, cat, val columns."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x": rng.uniform(-10, 10, n),
            "y": rng.uniform(-10, 10, n),
            "cat": pd.Categorical(rng.choice(["A", "B", "C"], n)),
            "val": rng.normal(0, 1, n),
        }
    )
    cvs = ds.Canvas(plot_width=50, plot_height=50, x_range=(-10, 10), y_range=(-10, 10))
    return cvs, df


def test_datashader_points_categorical_with_nan(sdata_blobs: SpatialData):
    """Datashader must handle categorical coloring with NaN values.

    Regression test for https://github.com/scverse/spatialdata-plot/issues/379.
    Exercises the optimised aggregation and color-key paths (pandas DataFrame
    instead of dask, early-exit in _build_datashader_color_key).
    """
    n = 200
    rng = get_standard_RNG()
    cats = pd.Categorical(rng.choice(["A", "B", None], n))
    points = sdata_blobs["blobs_points"].compute().head(n).copy()
    points["cat"] = cats.astype("object")  # force object so PointsModel accepts it

    sdata_blobs.points["test_pts"] = PointsModel.parse(points)

    fig, ax = plt.subplots()
    sdata_blobs.pl.render_points("test_pts", method="datashader", color="cat").pl.show(ax=ax)

    axes_images = [c for c in ax.get_children() if isinstance(c, matplotlib.image.AxesImage)]
    assert len(axes_images) > 0, "Datashader should produce at least one AxesImage"
    plt.close(fig)


def test_ds_aggregate_default_reduction_is_forwarded():
    """default_reduction must affect the actual aggregation, not just the log message."""
    cvs, df = _make_ds_canvas_and_df()
    agg_sum, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, None, "sum", "points")
    agg_max, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, None, "max", "points")
    assert not np.allclose(
        np.nan_to_num(agg_sum.values, nan=0),
        np.nan_to_num(agg_max.values, nan=0),
    )


def test_ds_aggregate_default_reduction_equals_explicit():
    """default_reduction='max' with ds_reduction=None must equal explicit ds_reduction='max'."""
    cvs, df = _make_ds_canvas_and_df()
    agg_default, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, None, "max", "points")
    agg_explicit, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, "max", "max", "points")
    np.testing.assert_array_equal(
        np.nan_to_num(agg_default.values, nan=0),
        np.nan_to_num(agg_explicit.values, nan=0),
    )


def test_ds_aggregate_explicit_overrides_default():
    """Explicit ds_reduction takes precedence over default_reduction."""
    cvs, df = _make_ds_canvas_and_df()
    agg, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, "max", "sum", "points")
    agg_max, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, "max", "max", "points")
    np.testing.assert_array_equal(
        np.nan_to_num(agg.values, nan=0),
        np.nan_to_num(agg_max.values, nan=0),
    )


def test_ds_reduction_ignored_for_categorical(caplog):
    """Categorical aggregation always uses ds.count(); a warning is emitted when ds_reduction is set."""
    cvs, df = _make_ds_canvas_and_df()
    with logger_warns(caplog, logger, match="ignored.*categorical"):
        _ds_aggregate(cvs, df.copy(), "cat", True, "mean", "mean", "points")


def test_ds_reduction_no_warning_when_none(caplog):
    """No spurious warning when ds_reduction is None (the default)."""
    cvs, df = _make_ds_canvas_and_df()
    with logger_no_warns(caplog, logger, match="ignored"):
        _ds_aggregate(cvs, df.copy(), "cat", True, None, "sum", "points")


@pytest.mark.parametrize("reduction", ["mean", "max", "min", "count", "std", "var"])
def test_ds_reduction_categorical_always_uses_count(reduction):
    """Categorical aggregation always uses ds.count(), regardless of ds_reduction (by design)."""
    cvs, df = _make_ds_canvas_and_df()
    base, _, _ = _ds_aggregate(cvs, df.copy(), "cat", True, "sum", "sum", "points")
    agg, _, _ = _ds_aggregate(cvs, df.copy(), "cat", True, reduction, reduction, "points")
    np.testing.assert_array_equal(agg.values, base.values)


def test_groups_warns_when_continuous_points(sdata_blobs: SpatialData, caplog):
    """Using groups with a continuous color column should warn."""
    n = len(sdata_blobs["blobs_points"])
    sdata_blobs["blobs_points"]["cont_val"] = pd.Series(list(range(n)), dtype=float)
    with logger_warns(caplog, logger, match="groups.*ignored.*continuous"):
        sdata_blobs.pl.render_points("blobs_points", color="cont_val", groups=["nonexistent"]).pl.show()


def test_warn_groups_ignored_continuous_emits(caplog):
    """_warn_groups_ignored_continuous emits when groups is set but data is continuous."""
    with logger_warns(caplog, logger, match="ignored.*continuous"):
        _warn_groups_ignored_continuous(["A"], None, "my_col")


def test_warn_groups_ignored_continuous_silent_for_categorical(caplog):
    """No warning when color_source_vector is present (categorical)."""
    with logger_no_warns(caplog, logger, match="ignored"):
        _warn_groups_ignored_continuous(["A"], pd.Categorical(["A", "B"]), "cat_col")


def test_color_key_warns_on_short_color_vector(caplog):
    """Warning when color_vector is shorter than categorical series."""
    cat = pd.Categorical(["A", "B", "C", "A", "B", "C", "A"])
    with logger_warns(caplog, logger, match="color_vector length"):
        result = _build_datashader_color_key(cat, ["#ff0000", "#00ff00", "#0000ff", "#ff0000", "#00ff00"], "#cccccc")
    assert "A" in result and "B" in result and "C" in result


def test_color_key_warns_on_long_color_vector(caplog):
    """Warning when color_vector is longer than categorical series."""
    cat = pd.Categorical(["A", "B"])
    with logger_warns(caplog, logger, match="color_vector length"):
        _build_datashader_color_key(cat, ["#ff0000", "#00ff00", "#0000ff", "#ffff00"], "#cccccc")


def test_color_key_no_warning_when_lengths_match(caplog):
    """No warning when lengths match."""
    cat = pd.Categorical(["A", "B", "C"])
    with logger_no_warns(caplog, logger, match="color_vector length"):
        _build_datashader_color_key(cat, ["#ff0000", "#00ff00", "#0000ff"], "#cccccc")


def test_color_key_unseen_category_gets_na_color(caplog):
    """Categories only appearing after the truncation point get na_color."""
    cat = pd.Categorical(["A", "B", "A", "B", "A", "D"])
    with logger_warns(caplog, logger, match="color_vector length"):
        result = _build_datashader_color_key(cat, ["#ff0000", "#00ff00", "#ff0000", "#00ff00"], "#cccccc")
    assert result["D"] == "#cccccc"


def test_shade_categorical_color_key_overrides_cmap():
    """When color_key is provided, different color_vector[0] values must produce identical output."""
    cvs, df = _make_ds_canvas_and_df(n=100)
    agg = cvs.points(df, "x", "y", agg=ds.by("cat", ds.count()))
    color_key = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}

    shaded1 = _ds_shade_categorical(agg, color_key, np.array(["#ff0000"] * 100), alpha=1.0)
    shaded2 = _ds_shade_categorical(agg, color_key, np.array(["#0000ff"] * 100), alpha=1.0)
    np.testing.assert_array_equal(np.asarray(shaded1), np.asarray(shaded2))


def test_shade_categorical_cmap_used_when_no_color_key():
    """When color_key is None (no color column), cmap from color_vector[0] affects output."""
    cvs, df = _make_ds_canvas_and_df(n=100)
    agg = cvs.points(df, "x", "y", agg=ds.count())
    shaded_red = _ds_shade_categorical(agg, None, np.array(["#ff0000"] * 100), alpha=1.0)
    shaded_blue = _ds_shade_categorical(agg, None, np.array(["#0000ff"] * 100), alpha=1.0)
    # Different color_vector[0] values should produce different shaded output
    assert not np.array_equal(np.asarray(shaded_red), np.asarray(shaded_blue))


def test_datashader_na_color_none_no_nan_overlay_points(sdata_blobs: SpatialData):
    """NaN overlay is skipped when na_color is fully transparent (#565)."""
    pts = sdata_blobs.points["blobs_points"].compute()
    n = len(pts)
    values = np.full(n, np.nan)
    values[: n // 2] = np.random.default_rng(0).uniform(0, 100, n // 2)
    pts["val"] = values
    sdata_blobs.points["blobs_points"] = PointsModel.parse(pts)

    fig, ax = plt.subplots()
    sdata_blobs.pl.render_points("blobs_points", color="val", na_color=None, method="datashader").pl.show(ax=ax)

    assert len(ax.get_images()) == 1, (
        f"Expected 1 image (no NaN overlay), got {len(ax.get_images())}; "
        "datashader is still rendering an opaque NaN overlay despite na_color=None"
    )
    plt.close(fig)


def test_no_table_fallback_warning_for_element_column(caplog):
    """No spurious 'using X as fallback for color mapping' warning when coloring by an element column (#562)."""
    n = 50
    rng = np.random.default_rng(42)
    coords = pd.DataFrame({"x": rng.uniform(0, 100, n), "y": rng.uniform(0, 100, n)})
    coords["cell_type"] = pd.Categorical(rng.choice(["A", "B", "C"], n))
    points = PointsModel.parse(coords)

    # Two tables so the multi-table fallback path is triggered
    obs1 = pd.DataFrame(index=pd.RangeIndex(10).astype(str))
    obs1["region"] = "s1"
    obs1["instance_id"] = range(10)
    table1 = AnnData(obs=obs1, X=np.zeros((10, 1)))
    table1 = TableModel.parse(table1, region="s1", region_key="region", instance_key="instance_id")

    obs2 = pd.DataFrame(index=pd.RangeIndex(10).astype(str))
    obs2["region"] = "s2"
    obs2["instance_id"] = range(10)
    table2 = AnnData(obs=obs2, X=np.zeros((10, 1)))
    table2 = TableModel.parse(table2, region="s2", region_key="region", instance_key="instance_id")

    sdata = SpatialData(
        points={"pts": points},
        tables={"RNA": table1, "protein": table2},
    )

    with logger_no_warns(caplog, logger, match="fallback for color mapping"):
        sdata.pl.render_points("pts", color="cell_type").pl.show()
    plt.close("all")


def test_render_points_registers_artist_once():
    # Regression test for #594: ax.scatter already registers the PathCollection internally;
    # the previous ax.add_collection(_cax) call added a second reference to the same object.
    rng = np.random.default_rng(0)
    n = 20
    coords = pd.DataFrame({"x": rng.uniform(0, 10, n), "y": rng.uniform(0, 10, n)})
    points = PointsModel.parse(coords)
    sdata = SpatialData(points={"pts": points})

    fig, ax = plt.subplots()
    sdata.pl.render_points("pts").pl.show(ax=ax)
    assert len(ax.collections) == 1, f"Expected 1 collection artist, got {len(ax.collections)}"
    plt.close(fig)
