import dask.array as da
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from matplotlib.colors import Normalize
from matplotlib.legend import Legend
from spatial_image import to_spatial_image
from spatialdata import SpatialData, deepcopy, get_element_instances
from spatialdata.models import Labels2DModel, Labels3DModel, TableModel

import spatialdata_plot  # noqa: F401
from spatialdata_plot._logging import logger, logger_warns
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


def _annotate_labels_with_outline_columns(sdata: SpatialData) -> SpatialData:
    """Patch the shared blobs fixture so its table annotates ``blobs_labels`` with categorical columns."""
    sdata["table"].obs["region"] = pd.Categorical(["blobs_labels"] * sdata["table"].n_obs)
    sdata["table"].uns["spatialdata_attrs"]["region"] = "blobs_labels"
    n = sdata["table"].n_obs
    sdata["table"].obs["cluster"] = pd.Categorical((["c1", "c2"] * ((n + 1) // 2))[:n])
    sdata["table"].obs["stage"] = pd.Categorical((["s1", "s2"] * ((n + 1) // 2))[:n])
    return sdata


class TestLabels(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_labels(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels(element="blobs_labels").pl.show()

    def test_plot_can_render_multiscale_labels(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_multiscale_labels"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        sdata_blobs.pl.render_labels("blobs_multiscale_labels").pl.show()

    def test_plot_can_render_given_scale_of_multiscale_labels(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_multiscale_labels"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        sdata_blobs.pl.render_labels("blobs_multiscale_labels", scale="scale1").pl.show()

    def test_plot_can_do_rasterization(self, sdata_blobs: SpatialData):
        temp = sdata_blobs["blobs_labels"].data.copy()
        temp = da.concatenate([temp] * 6, axis=0)
        temp = da.concatenate([temp] * 6, axis=1)
        img = to_spatial_image(temp, dims=("y", "x"))
        img.attrs["transform"] = sdata_blobs["blobs_labels"].transform
        sdata_blobs["blobs_giant_labels"] = img

        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_giant_labels"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_giant_labels"

        sdata_blobs.pl.render_labels("blobs_giant_labels").pl.show()

    def test_plot_can_stop_rasterization_with_scale_full(self, sdata_blobs: SpatialData):
        temp = sdata_blobs["blobs_labels"].data.copy()
        temp = da.concatenate([temp] * 6, axis=0)
        temp = da.concatenate([temp] * 6, axis=1)
        img = to_spatial_image(temp, dims=("y", "x"))
        img.attrs["transform"] = sdata_blobs["blobs_labels"].transform
        sdata_blobs["blobs_giant_labels"] = img

        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_giant_labels"] * sdata_blobs["table"].n_obs)
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
            .pl.render_labels(
                element="blobs_labels",
                na_color="blue",
                fill_alpha=0,
                outline_alpha=1,
                contour_px=15,
            )
            .pl.show()
        )

    def test_two_categorical_label_renders_make_two_distinct_legends(self, sdata_blobs: SpatialData):
        # Regression test for #364: two render_labels calls coloring by two categorical columns
        # must produce two separate, column-titled legends (not one merged/replaced legend), and
        # the second render must not reuse the first's colors. State-based (no image comparison).
        n = sdata_blobs["table"].n_obs
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_labels"] * n)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_labels"
        sdata_blobs["table"].obs["cat0"] = pd.Categorical((["A", "B"] * ((n + 1) // 2))[:n])
        sdata_blobs["table"].obs["cat1"] = pd.Categorical((["C", "D"] * ((n + 1) // 2))[:n])

        (
            sdata_blobs.pl.render_labels("blobs_labels", color="cat0")
            .pl.render_labels("blobs_labels", color="cat1")
            .pl.show()
        )

        fig = plt.gcf()
        ax = fig.axes[0]
        legends = [c for c in ax.get_children() if isinstance(c, Legend)]
        assert len(legends) == 2
        entries = {leg.get_title().get_text(): {t.get_text() for t in leg.get_texts()} for leg in legends}
        assert entries == {"cat0": {"A", "B"}, "cat1": {"C", "D"}}

        # palette offset: the second categorical render must not reuse the first's colors
        c0 = set(sdata_blobs["table"].uns["cat0_colors"])
        c1 = set(sdata_blobs["table"].uns["cat1_colors"])
        assert c0.isdisjoint(c1)

        # legends are placed side-by-side in the right margin: top-aligned, non-overlapping in x (#364)
        fig.canvas.draw()
        inv = ax.transAxes.inverted()
        boxes = sorted((leg.get_window_extent().transformed(inv) for leg in legends), key=lambda b: b.x0)
        assert boxes[0].x1 <= boxes[1].x0  # no horizontal overlap
        assert abs(boxes[0].y1 - boxes[1].y1) < 0.01  # tops aligned
        plt.close()

    def test_three_categorical_label_renders_make_three_legends(self, sdata_blobs: SpatialData):
        # Regression test for #364: re-adding prior legends must not duplicate them; three renders
        # yield exactly three distinct legends (not four with a repeat).
        n = sdata_blobs["table"].n_obs
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_labels"] * n)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_labels"
        for col, (a, b) in {"cat0": ("A", "B"), "cat1": ("C", "D"), "cat2": ("E", "F")}.items():
            sdata_blobs["table"].obs[col] = pd.Categorical(([a, b] * ((n + 1) // 2))[:n])

        (
            sdata_blobs.pl.render_labels("blobs_labels", color="cat0")
            .pl.render_labels("blobs_labels", color="cat1")
            .pl.render_labels("blobs_labels", color="cat2")
            .pl.show()
        )

        ax = plt.gcf().axes[0]
        titles = sorted(c.get_title().get_text() for c in ax.get_children() if isinstance(c, Legend))
        assert titles == ["cat0", "cat1", "cat2"]

        # palette offset accumulates: every render skips all colors used by earlier ones, so all
        # three palettes are mutually disjoint (not just cat0 vs cat1).
        c0, c1, c2 = (set(sdata_blobs["table"].uns[f"cat{i}_colors"]) for i in range(3))
        assert c0.isdisjoint(c1) and c1.isdisjoint(c2) and c0.isdisjoint(c2)
        plt.close()

    def test_single_categorical_label_render_legend_has_no_title(self, sdata_blobs: SpatialData):
        # A lone categorical render produces exactly one, untitled legend: the column title is only
        # added to disambiguate 2+ legends on an axis (#364).
        n = sdata_blobs["table"].n_obs
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_labels"] * n)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_labels"
        sdata_blobs["table"].obs["cat0"] = pd.Categorical((["A", "B"] * ((n + 1) // 2))[:n])

        sdata_blobs.pl.render_labels("blobs_labels", color="cat0").pl.show()

        ax = plt.gcf().axes[0]
        legends = [c for c in ax.get_children() if isinstance(c, Legend)]
        assert len(legends) == 1
        assert legends[0].get_title().get_text() == ""
        plt.close()

    def test_two_legend_plot_saves_to_vector_backend(self, sdata_blobs: SpatialData, tmp_path):
        # Regression for #364: the side-by-side legend layout runs on every draw, so it must use the
        # draw event's renderer (valid on PDF/SVG) — not the Agg-only fig.canvas.get_renderer().
        n = sdata_blobs["table"].n_obs
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_labels"] * n)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_labels"
        sdata_blobs["table"].obs["cat0"] = pd.Categorical((["A", "B"] * ((n + 1) // 2))[:n])
        sdata_blobs["table"].obs["cat1"] = pd.Categorical((["C", "D"] * ((n + 1) // 2))[:n])

        sdata_blobs.pl.render_labels("blobs_labels", color="cat0").pl.render_labels(
            "blobs_labels", color="cat1"
        ).pl.show()
        plt.gcf().savefig(tmp_path / "out.pdf")  # must not raise on the Pdf canvas
        plt.close()

    def test_plot_can_color_by_rgba_array(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color=[0.5, 0.5, 1.0, 0.5]).pl.show()

    def test_plot_can_color_by_hex(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color="#88a136").pl.show()

    def test_plot_can_color_by_hex_with_alpha(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color="#88a13688").pl.show()

    def test_plot_alpha_overwrites_opacity_from_color(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color=[0.5, 0.5, 1.0, 0.5], fill_alpha=1.0).pl.show()

    def test_plot_can_render_outline_color(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels(
            "blobs_labels", outline_alpha=1, fill_alpha=0, outline_color="red", contour_px=10
        ).pl.show()

    def test_plot_can_render_outline_with_fill(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels(
            "blobs_labels", outline_alpha=1, fill_alpha=0.3, outline_color="blue", contour_px=10
        ).pl.show()

    def test_plot_outline_inherits_literal_color(self, sdata_blobs: SpatialData):
        """Literal color= should be used for outlines when outline_color is not set (#462)."""
        sdata_blobs.pl.render_labels(
            "blobs_labels", color="white", outline_alpha=1, fill_alpha=0, contour_px=10
        ).pl.show()

    def test_plot_outline_uses_data_driven_colors(self, sdata_blobs: SpatialData):
        """Data-driven color should produce per-label outline colors when outline_color is None."""
        sdata_blobs.pl.render_labels(
            "blobs_labels", color="channel_0_sum", outline_alpha=1, fill_alpha=0, contour_px=10
        ).pl.show()

    def test_plot_outline_color_by_categorical_obs_labels(self, sdata_blobs: SpatialData):
        sdata_blobs = _annotate_labels_with_outline_columns(sdata_blobs)
        sdata_blobs.pl.render_labels(
            "blobs_labels", fill_alpha=0, outline_alpha=1, outline_color="cluster", contour_px=10
        ).pl.show()

    def test_plot_can_color_labels_by_continuous_variable(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum").pl.show()

    def test_plot_can_color_labels_by_categorical_variable(self, sdata_blobs: SpatialData):
        max_col = sdata_blobs["table"].to_df().idxmax(axis=1)
        max_col = pd.Categorical(max_col, categories=sdata_blobs["table"].to_df().columns, ordered=True)
        sdata_blobs["table"].obs["which_max"] = max_col

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
            adata.obs["region"] = pd.Categorical([label] * adata.n_obs)
            del adata.uns["spatialdata_attrs"]
            table = TableModel.parse(
                adata=adata,
                region_key="region",
                instance_key="instance_id",
                region=label,
            )
            sdata_blobs.tables["other_table"] = table

            _, axs = plt.subplots(nrows=1, ncols=3, layout="tight")

            sdata_blobs.pl.render_labels(
                label, color="channel_1_sum", table_name="other_table", scale="scale0"
            ).pl.show(ax=axs[0], title="ch_1_sum", colorbar=False)
            sdata_blobs.pl.render_labels(
                label, color="channel_2_sum", table_name="other_table", scale="scale0"
            ).pl.show(ax=axs[1], title="ch_2_sum", colorbar=False)
            sdata_blobs.pl.render_labels(label, color="which_max", table_name="other_table", scale="scale0").pl.show(
                ax=axs[2], legend_fontsize=6
            )

        # we're modifying the data here, so we need an independent copy
        sdata_blobs_local = deepcopy(sdata_blobs)

        _make_tablemodel_with_categorical_labels(sdata_blobs_local, label)

    def test_plot_two_calls_with_coloring_result_in_two_colorbars(self, sdata_blobs: SpatialData):
        # we're modifying the data here so we need an independent copy
        sdata_blobs_local = deepcopy(sdata_blobs)

        table = sdata_blobs_local["table"].copy()
        table.obs["region"] = pd.Categorical(["blobs_multiscale_labels"] * table.n_obs)
        table.uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        table = table[:, ~table.var_names.isin(["channel_0_sum"])]
        sdata_blobs_local["multi_table"] = table
        sdata_blobs_local.pl.render_labels("blobs_labels", color="channel_0_sum", table_name="table").pl.render_labels(
            "blobs_multiscale_labels", color="channel_1_sum", table_name="multi_table"
        ).pl.show()

    def test_plot_can_control_label_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels(
            "blobs_labels",
            color="channel_0_sum",
            outline_alpha=0.4,
            fill_alpha=0.0,
            contour_px=15,
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
            "blobs_labels",
            color="channel_0_sum",
            fill_alpha=0.1,
            outline_alpha=0.7,
            contour_px=15,
        ).pl.show()

    def test_plot_label_colorbar_uses_alpha_of_less_transparent_outline(
        self,
        sdata_blobs: SpatialData,
    ):
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum", fill_alpha=0.7, outline_alpha=0.1).pl.show()

    def test_can_plot_with_one_element_color_table(self, sdata_blobs: SpatialData):
        table = sdata_blobs["table"].copy()
        table.obs["region"] = pd.Categorical(["blobs_multiscale_labels"] * table.n_obs)
        table.uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        table = table[:, ~table.var_names.isin(["channel_0_sum"])]
        sdata_blobs["multi_table"] = table
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum", table_name="table").pl.render_labels(
            "blobs_multiscale_labels", color="channel_1_sum", table_name="multi_table"
        ).pl.show()

    def test_plot_subset_categorical_label_maintains_order(self, sdata_blobs: SpatialData):
        max_col = sdata_blobs["table"].to_df().idxmax(axis=1)
        max_col = pd.Categorical(max_col, categories=sdata_blobs["table"].to_df().columns, ordered=True)
        sdata_blobs["table"].obs["which_max"] = max_col

        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")

        sdata_blobs.pl.render_labels("blobs_labels", color="which_max").pl.show(ax=axs[0], legend_fontsize=6)
        sdata_blobs.pl.render_labels(
            "blobs_labels",
            color="which_max",
            groups=["channel_0_sum"],
        ).pl.show(ax=axs[1])

    def test_plot_subset_categorical_label_maintains_order_when_palette_overwrite(self, sdata_blobs: SpatialData):
        max_col = sdata_blobs["table"].to_df().idxmax(axis=1)
        max_col = pd.Categorical(max_col, categories=sdata_blobs["table"].to_df().columns, ordered=True)
        sdata_blobs["table"].obs["which_max"] = max_col

        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")

        sdata_blobs.pl.render_labels("blobs_labels", color="which_max").pl.show(ax=axs[0], legend_fontsize=6)
        sdata_blobs.pl.render_labels(
            "blobs_labels", color="which_max", groups=["channel_0_sum"], palette="red"
        ).pl.show(ax=axs[1])

    def test_plot_label_categorical_color(self, sdata_blobs: SpatialData):
        self._make_tablemodel_with_categorical_labels(sdata_blobs, labels_name="blobs_labels")
        sdata_blobs.pl.render_labels("blobs_labels", color="category").pl.show()

    def _make_tablemodel_with_categorical_labels(self, sdata_blobs, labels_name: str):
        instances = get_element_instances(sdata_blobs[labels_name])
        n_obs = len(instances)
        adata = AnnData(
            get_standard_RNG().normal(size=(n_obs, 10)),
            obs=pd.DataFrame(get_standard_RNG().normal(size=(n_obs, 3)), columns=["a", "b", "c"]),
        )
        adata.obs["instance_id"] = instances.values
        adata.obs["category"] = get_standard_RNG().choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs.loc[adata.obs.index[:3], "category"] = ["a", "b", "c"]
        adata.obs["region"] = labels_name
        table = TableModel.parse(
            adata=adata,
            region_key="region",
            instance_key="instance_id",
            region=labels_name,
        )
        sdata_blobs["other_table"] = table
        sdata_blobs["other_table"].obs["category"] = sdata_blobs["other_table"].obs["category"].astype("category")

    def test_plot_can_color_with_norm_and_clipping(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels(
            "blobs_labels",
            color="channel_0_sum",
            norm=Normalize(400, 1000, clip=True),
            cmap=_viridis_with_under_over(),
        ).pl.show()

    def test_plot_can_color_with_norm_no_clipping(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels(
            "blobs_labels",
            color="channel_0_sum",
            norm=Normalize(400, 1000, clip=False),
            cmap=_viridis_with_under_over(),
        ).pl.show()

    def test_plot_transfunc_applied_to_continuous_labels(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum", transfunc=lambda x: x * 100).pl.show(
            title="transfunc: x * 100"
        )

    def test_plot_can_annotate_labels_with_table_layer(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].layers["normalized"] = get_standard_RNG().random(sdata_blobs["table"].X.shape)
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum", table_layer="normalized").pl.show()

    def _prepare_labels_with_small_objects(self, sdata_blobs: SpatialData) -> SpatialData:
        # add a categorical column
        adata = sdata_blobs["table"]
        sdata_blobs["table"].obs["category"] = ["a"] * 10 + ["b"] * 10 + ["c"] * 6

        sdata_blobs["table"].obs["category"] = sdata_blobs["table"].obs["category"].astype("category")

        labels = sdata_blobs["blobs_labels"].data.compute()

        # make label 1 small
        mask = labels == 1
        labels[mask] = 0
        labels[200, 200] = 1

        sdata_blobs["blobs_labels"] = Labels2DModel.parse(labels)

        # tile the labels object
        arr = da.tile(sdata_blobs["blobs_labels"], (4, 4))
        sdata_blobs["blobs_labels_large"] = Labels2DModel.parse(arr)

        adata.obs["region"] = "blobs_labels_large"
        sdata_blobs.set_table_annotates_spatialelement("table", region="blobs_labels_large")
        return sdata_blobs

    def test_plot_can_handle_dropping_small_labels_after_rasterize_continuous(self, sdata_blobs: SpatialData):
        # reported here https://github.com/scverse/spatialdata-plot/issues/443
        sdata_blobs = self._prepare_labels_with_small_objects(sdata_blobs)

        sdata_blobs.pl.render_labels("blobs_labels_large", color="channel_0_sum", table_name="table").pl.show()

    def test_plot_can_handle_dropping_small_labels_after_rasterize_categorical(self, sdata_blobs: SpatialData):
        sdata_blobs = self._prepare_labels_with_small_objects(sdata_blobs)

        sdata_blobs.pl.render_labels("blobs_labels_large", color="category", table_name="table").pl.show()

    def test_plot_respects_custom_colors_from_uns(self, sdata_blobs: SpatialData):
        labels_name = "blobs_labels"
        instances = get_element_instances(sdata_blobs[labels_name])
        n_obs = len(instances)
        adata = AnnData(
            get_standard_RNG().normal(size=(n_obs, 10)),
            obs=pd.DataFrame(get_standard_RNG().normal(size=(n_obs, 3)), columns=["a", "b", "c"]),
        )
        adata.obs["instance_id"] = instances.values
        adata.obs["category"] = get_standard_RNG().choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs.loc[adata.obs.index[:3], "category"] = ["a", "b", "c"]
        adata.obs["region"] = labels_name
        table = TableModel.parse(
            adata=adata,
            region_key="region",
            instance_key="instance_id",
            region=labels_name,
        )
        sdata_blobs["other_table"] = table
        sdata_blobs["other_table"].obs["category"] = sdata_blobs["other_table"].obs["category"].astype("category")
        sdata_blobs["other_table"].uns["category_colors"] = ["red", "green", "blue"]  # purple, green ,yellow

        sdata_blobs.pl.render_labels("blobs_labels", color="category").pl.show()

    def test_plot_respects_custom_colors_from_uns_with_groups_and_palette(
        self,
        sdata_blobs: SpatialData,
    ):
        labels_name = "blobs_labels"
        instances = get_element_instances(sdata_blobs[labels_name])
        n_obs = len(instances)
        adata = AnnData(
            get_standard_RNG().normal(size=(n_obs, 10)),
            obs=pd.DataFrame(get_standard_RNG().normal(size=(n_obs, 3)), columns=["a", "b", "c"]),
        )
        adata.obs["instance_id"] = instances.values
        adata.obs["category"] = get_standard_RNG().choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs.loc[adata.obs.index[:3], "category"] = ["a", "b", "c"]
        adata.obs["region"] = labels_name
        table = TableModel.parse(
            adata=adata,
            region_key="region",
            instance_key="instance_id",
            region=labels_name,
        )
        sdata_blobs["other_table"] = table
        sdata_blobs["other_table"].obs["category"] = sdata_blobs["other_table"].obs["category"].astype("category")
        sdata_blobs["other_table"].uns["category_colors"] = {
            "a": "red",
            "b": "green",
            "c": "blue",
        }

        # palette overwrites uns colors
        sdata_blobs.pl.render_labels(
            "blobs_labels",
            color="category",
            groups=["a", "b"],
            palette=["yellow", "cyan"],
        ).pl.show()

    def test_plot_can_annotate_labels_with_nan_in_table_obs_categorical(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["cat_color"] = pd.Categorical(["a", "b", "b", "a", "b"] * 5 + [np.nan])
        sdata_blobs.pl.render_labels("blobs_labels", color="cat_color").pl.show()

    def test_plot_can_annotate_labels_with_nan_in_table_obs_continuous(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["cont_color"] = [np.nan, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] * 2
        sdata_blobs.pl.render_labels("blobs_labels", color="cont_color").pl.show()

    def test_plot_can_annotate_labels_with_nan_in_table_X_continuous(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].X[0:5, 0] = np.nan
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum").pl.show()

    def test_plot_can_color_labels_by_gene_symbols(self, sdata_blobs: SpatialData):
        """Color labels by gene symbol alias instead of var_name (#247)."""
        sdata_blobs["table"].var["gene_symbol"] = ["GeneA", "GeneB", "GeneC"]
        sdata_blobs.pl.render_labels(
            "blobs_labels", color="GeneA", table_name="table", gene_symbols="gene_symbol"
        ).pl.show()

    @staticmethod
    def _as_points(sdata_blobs: SpatialData, method: str, color: str = "instance_id"):
        # identical params for both backends; non-overlapping size so the only place the engines differ
        # (overlap: matplotlib stacks markers, datashader aggregates) doesn't enter the comparison
        return sdata_blobs.pl.render_labels("blobs_labels", color=color, as_points=True, method=method, size=120)

    @staticmethod
    def _add_categorical_color(sdata_blobs: SpatialData) -> str:
        max_col = sdata_blobs["table"].to_df().idxmax(axis=1)
        sdata_blobs["table"].obs["which_max"] = pd.Categorical(
            max_col, categories=sdata_blobs["table"].to_df().columns, ordered=True
        )
        return "which_max"

    def test_plot_labels_as_points_matplotlib(self, sdata_blobs: SpatialData):
        """as_points draws one colored dot per label at its centroid (matplotlib backend)."""
        self._as_points(sdata_blobs, "matplotlib").pl.show()

    def test_plot_labels_as_points_datashader(self, sdata_blobs: SpatialData):
        """Same render via datashader; should look maximally similar to the matplotlib baseline."""
        self._as_points(sdata_blobs, "datashader").pl.show()

    def test_plot_labels_as_points_categorical_matplotlib(self, sdata_blobs: SpatialData):
        """Categorical-coloured as_points with a legend (matplotlib backend)."""
        self._as_points(sdata_blobs, "matplotlib", color=self._add_categorical_color(sdata_blobs)).pl.show()

    def test_plot_labels_as_points_categorical_datashader(self, sdata_blobs: SpatialData):
        """Same categorical render via datashader (color_source_vector + legend path)."""
        self._as_points(sdata_blobs, "datashader", color=self._add_categorical_color(sdata_blobs)).pl.show()


def test_raises_when_table_does_not_annotate_element(sdata_blobs: SpatialData):
    # Work on an independent copy since we mutate tables
    sdata_blobs_local = deepcopy(sdata_blobs)

    # Create a table that annotates a DIFFERENT element than the one we will render
    other_table = sdata_blobs_local["table"].copy()
    other_table.obs["region"] = pd.Categorical(["blobs_multiscale_labels"] * other_table.n_obs)
    other_table.uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
    sdata_blobs_local["other_table"] = other_table

    # Rendering "blobs_labels" with a table that annotates "blobs_multiscale_labels"
    # should now raise to alert the user about the mismatch.
    with pytest.raises(
        KeyError,
        match="Table 'other_table' does not annotate element 'blobs_labels'",
    ):
        sdata_blobs_local.pl.render_labels(
            "blobs_labels",
            color="channel_0_sum",
            table_name="other_table",
        ).pl.show()


def test_groups_warns_when_no_groups_match_labels(sdata_blobs: SpatialData, caplog):
    """Warning fires when no groups match label color categories."""
    labels_name = "blobs_labels"
    instances = get_element_instances(sdata_blobs[labels_name])
    n_obs = len(instances)
    adata = AnnData(np.zeros((n_obs, 1)))
    adata.obs["instance_id"] = instances.values
    adata.obs["cat"] = pd.Categorical(["a", "b"] * (n_obs // 2) + ["a"] * (n_obs % 2))
    adata.obs["region"] = labels_name
    sdata_blobs["label_table"] = TableModel.parse(
        adata=adata, region_key="region", instance_key="instance_id", region=labels_name
    )
    with logger_warns(caplog, logger, match="None of the requested groups"):
        sdata_blobs.pl.render_labels(
            labels_name, color="cat", groups=["nonexistent"], table_name="label_table", na_color=None
        ).pl.show()


def test_transfunc_is_applied_for_continuous_labels(sdata_blobs: SpatialData):
    called = []

    def track(x):
        called.append(True)
        return x

    fig, ax = plt.subplots()
    sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum", transfunc=track).pl.show(ax=ax)
    plt.close(fig)

    assert called, "transfunc was not called for continuous labels data"


def test_render_labels_all_nan_color_renders_under_rasterize(sdata_blobs: SpatialData):
    # Regression: an all-NaN color column is the "none" colortype (na-array source, not None).
    # Under default rasterize the per-instance mask path used to trip an
    # `assert color_source_vector is None`; it must now just render.
    labels_name = "blobs_labels"
    instances = get_element_instances(sdata_blobs[labels_name])
    n_obs = len(instances)
    adata = AnnData(np.zeros((n_obs, 1)))
    adata.obs["instance_id"] = instances.values
    adata.obs["nanvals"] = np.full(n_obs, np.nan)
    adata.obs["region"] = labels_name
    sdata_blobs["label_table"] = TableModel.parse(
        adata=adata, region_key="region", instance_key="instance_id", region=labels_name
    )
    fig, ax = plt.subplots()
    sdata_blobs.pl.render_labels(labels_name, color="nanvals", table_name="label_table").pl.show(ax=ax)
    plt.close(fig)


def test_render_labels_as_points_all_nan_color_does_not_crash(sdata_blobs: SpatialData):
    # Regression: as_points rebuilds the centroid ColorSpec; an all-NaN column is the "none" colortype
    # and must not be re-classified "categorical" (which would later call Categorical-only methods on
    # the na-array source). It must just render the centroids in na_color.
    labels_name = "blobs_labels"
    instances = get_element_instances(sdata_blobs[labels_name])
    n_obs = len(instances)
    adata = AnnData(np.zeros((n_obs, 1)))
    adata.obs["instance_id"] = instances.values
    adata.obs["nanvals"] = np.full(n_obs, np.nan)
    adata.obs["region"] = labels_name
    sdata_blobs["label_table"] = TableModel.parse(
        adata=adata, region_key="region", instance_key="instance_id", region=labels_name
    )
    fig, ax = plt.subplots()
    sdata_blobs.pl.render_labels(labels_name, color="nanvals", table_name="label_table", as_points=True).pl.show(ax=ax)
    plt.close(fig)


def test_render_labels_lognorm_with_zeros_does_not_crash(sdata_blobs: SpatialData):
    # Regression: a continuous LogNorm column containing 0 must derive a positive vmin instead of a
    # LogNorm(vmin=0) that raises "Invalid vmin or vmax" when the segmentation colors are mapped.
    from matplotlib.colors import LogNorm

    labels_name = "blobs_labels"
    instances = get_element_instances(sdata_blobs[labels_name])
    n_obs = len(instances)
    adata = AnnData(np.zeros((n_obs, 1)))
    adata.obs["instance_id"] = instances.values
    adata.obs["counts"] = np.linspace(0.0, 10.0, n_obs)  # includes 0
    adata.obs["region"] = labels_name
    sdata_blobs["label_table"] = TableModel.parse(
        adata=adata, region_key="region", instance_key="instance_id", region=labels_name
    )
    fig, ax = plt.subplots()
    sdata_blobs.pl.render_labels(labels_name, color="counts", table_name="label_table", norm=LogNorm()).pl.show(ax=ax)
    plt.close(fig)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_render_labels_rejects_float_dtype(dtype):
    # Regression test for #606: float-dtype labels must raise a clear
    # ValueError naming the element and dtype, not a cryptic skimage TypeError.
    arr = np.zeros((20, 20), dtype=dtype)
    arr[3:8, 3:8] = 1
    arr[12:17, 12:17] = 2
    sdata = SpatialData(labels={"lbl": Labels2DModel.parse(arr, dims=["y", "x"])})

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match=r"Label element 'lbl'.*integer dtype"):
            sdata.pl.render_labels("lbl").pl.show(ax=ax)
    finally:
        plt.close(fig)


def test_render_labels_rejects_background_instance_id_in_table():
    # Regression test for #607: table row with instance_id=0 (background)
    # used to crash with obnscure error.
    labels_data = np.zeros((20, 20), dtype=np.int32)
    labels_data[3:8, 3:8] = 1
    labels_data[12:17, 12:17] = 2
    labels = Labels2DModel.parse(labels_data, dims=["y", "x"])

    obs = pd.DataFrame(
        {
            "region": pd.Categorical(["lbl"] * 3),
            "instance_id": [0, 1, 2],
            "score": [99.0, 1.0, 2.0],
        }
    )
    table = TableModel.parse(
        AnnData(X=np.zeros((3, 1)), obs=obs),
        region="lbl",
        region_key="region",
        instance_key="instance_id",
    )
    sdata = SpatialData(labels={"lbl": labels}, tables={"t": table})

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match=r"instance_id=0.*background"):
            sdata.pl.render_labels("lbl", color="score", table_name="t").pl.show(ax=ax)
    finally:
        plt.close(fig)


def test_render_labels_disjoint_instance_ids_clear_error():
    # regression test for #603: disjoint instance_id values must raise a clear ValueError
    arr = np.zeros((20, 20), dtype=np.int32)
    arr[3:8, 3:8] = 1
    arr[12:17, 12:17] = 2
    obs = pd.DataFrame(
        {
            "instance_id": [99, 100],  # label has IDs 1, 2 (no overlap)
            "region": pd.Categorical(["lbl"] * 2),
            "cat": pd.Categorical(["A", "B"]),
        }
    )
    obs.index = obs.index.astype(str)
    table = TableModel.parse(
        AnnData(X=np.zeros((2, 1)), obs=obs),
        region=["lbl"],
        region_key="region",
        instance_key="instance_id",
    )
    sdata = SpatialData(labels={"lbl": Labels2DModel.parse(arr, dims=["y", "x"])}, tables={"t": table})

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match=r"No instance IDs overlap.*table 't'.*element 'lbl'"):
            sdata.pl.render_labels("lbl", color="cat", table_name="t").pl.show(ax=ax)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("scale_factors", [None, [2]])
def test_render_labels_raises_on_3d(scale_factors):
    # Regression test for #608: 3D labels must raise a clear ValueError, not crash
    # deep in numpy with an opaque concatenation error.
    arr = np.random.default_rng(0).integers(0, 5, size=(4, 16, 16), dtype=np.int32)
    labels3d = Labels3DModel.parse(arr, dims=["z", "y", "x"], scale_factors=scale_factors)
    sdata = SpatialData(labels={"lbl3d": labels3d})
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match=r"render_labels does not support 3D.*lbl3d.*z.*4"):
            sdata.pl.render_labels("lbl3d").pl.show(ax=ax)
    finally:
        plt.close(fig)


def test_labels_outline_color_groups_filter_aligns(sdata_blobs: SpatialData):
    """When `groups` filters the fill labels, the outline vector must be masked alongside it."""
    sdata_blobs = _annotate_labels_with_outline_columns(sdata_blobs)
    fig, ax = plt.subplots()
    sdata_blobs.pl.render_labels(
        "blobs_labels",
        color="cluster",
        groups=["c1"],
        outline_alpha=1.0,
        outline_color="stage",
    ).pl.show(ax=ax)
    plt.close(fig)


def test_render_labels_color_list_creates_one_panel_per_key(sdata_blobs: SpatialData):
    """A list of color keys produces one panel per key, titled by the key (#611)."""
    # the default blobs table annotates blobs_labels with channel_*_sum vars
    axs = sdata_blobs.pl.render_labels("blobs_labels", color=["channel_0_sum", "channel_1_sum"]).pl.show(return_ax=True)
    assert isinstance(axs, list)
    assert len(axs) == 2
    assert [ax.get_title() for ax in axs] == ["channel_0_sum", "channel_1_sum"]
    plt.close("all")


def test_render_labels_as_points_renders_centroids(sdata_blobs: SpatialData):
    """as_points draws one dot per label near its centroid. Centroids are computed on the rendered
    (possibly downsampled) raster, so positions are checked in display space within a few-pixel
    rasterization tolerance rather than against the exact full-resolution centroid."""
    import spatialdata as sd

    fig, ax = plt.subplots()
    sdata_blobs.pl.render_labels("blobs_labels", color="instance_id", as_points=True, size=50).pl.show(ax=ax)
    coll = ax.collections[0]
    dots = coll.get_offset_transform().transform(np.asarray(coll.get_offsets()))  # display px
    ref_world = sd.get_centroids(sdata_blobs["blobs_labels"], coordinate_system="global").compute()[["x", "y"]]
    ref = ax.transData.transform(ref_world.to_numpy())
    assert len(dots) == len(ref)
    od, oe = np.lexsort((dots[:, 1], dots[:, 0])), np.lexsort((ref[:, 1], ref[:, 0]))
    assert np.allclose(dots[od], ref[oe], atol=3.0)  # within a few display px of the true centroid
    plt.close(fig)


def test_render_labels_as_points_without_color(sdata_blobs: SpatialData):
    """as_points must not crash without a color column; the background label (0) is excluded."""
    import spatialdata as sd

    fig, ax = plt.subplots()
    sdata_blobs.pl.render_labels("blobs_labels", as_points=True).pl.show(ax=ax)
    offsets = np.asarray(ax.collections[0].get_offsets())
    n_cells = len(sd.get_centroids(sdata_blobs["blobs_labels"], coordinate_system="global").compute())
    assert len(offsets) == n_cells  # one dot per cell, no spurious background point
    # without a color column, cells get distinct random colours (like the mask path), not one na_color
    facecolors = ax.collections[0].get_facecolors()
    assert len({tuple(np.round(c, 4)) for c in facecolors}) > 1
    plt.close(fig)


def test_render_labels_as_points_applies_non_identity_transform(sdata_blobs: SpatialData):
    """Regression guard: under a non-identity element->CS transform the dots must land at the cells'
    coordinate-system positions (in display space). A wrong transform would be off by the scale
    factor (hundreds of px); the rendered-raster centroid is correct within a few px."""
    import spatialdata as sd
    from spatialdata.transformations import Scale, set_transformation

    set_transformation(sdata_blobs["blobs_labels"], Scale([2.0, 3.0], axes=("x", "y")), "scaled")
    fig, ax = plt.subplots()
    sdata_blobs.pl.render_labels("blobs_labels", color="instance_id", as_points=True, size=50).pl.show(
        ax=ax, coordinate_systems="scaled"
    )
    coll = ax.collections[0]
    dots_display = coll.get_offset_transform().transform(np.asarray(coll.get_offsets()))
    cs = sd.get_centroids(sdata_blobs["blobs_labels"], coordinate_system="scaled").compute()[["x", "y"]].to_numpy()
    expected_display = ax.transData.transform(cs)  # where the cells truly are, in display pixels
    order_d = np.lexsort((dots_display[:, 1], dots_display[:, 0]))
    order_e = np.lexsort((expected_display[:, 1], expected_display[:, 0]))
    assert np.allclose(dots_display[order_d], expected_display[order_e], atol=3.0)
    plt.close(fig)


def test_render_labels_as_points_method_datashader_renders_image(sdata_blobs: SpatialData):
    """method='datashader' under as_points (with a color column) draws a datashaded raster."""
    fig, ax = plt.subplots()
    sdata_blobs.pl.render_labels(
        "blobs_labels", color="instance_id", as_points=True, method="datashader", size=50
    ).pl.show(ax=ax)
    assert len(ax.images) >= 1
    assert len(ax.collections) == 0
    plt.close(fig)


def test_render_labels_as_points_no_color_forces_matplotlib(sdata_blobs: SpatialData, caplog):
    """No-color labels get one random colour per cell, which datashader cannot represent; even
    method='datashader' must fall back to a matplotlib scatter (with a warning)."""
    fig, ax = plt.subplots()
    with logger_warns(caplog, logger, match="cannot use datashader"):
        sdata_blobs.pl.render_labels("blobs_labels", as_points=True, method="datashader").pl.show(ax=ax)
    assert len(ax.collections) == 1  # matplotlib scatter, not a datashader image
    assert len(ax.images) == 0
    plt.close(fig)


def test_resolve_as_points_method_threshold_and_fallback():
    """Backend selection: explicit honored, no-color/empty force matplotlib, auto switches past the cap."""
    from types import SimpleNamespace

    from spatialdata_plot.pl.render import AS_POINTS_DS_AUTO, _resolve_as_points_method

    rp_auto = SimpleNamespace(method=None)
    rp_ds = SimpleNamespace(method="datashader")
    rp_mpl = SimpleNamespace(method="matplotlib")
    # auto: matplotlib below the cap, datashader above
    assert _resolve_as_points_method(rp_auto, n=1000, allow_datashader=True) == "matplotlib"
    assert _resolve_as_points_method(rp_auto, n=AS_POINTS_DS_AUTO + 1, allow_datashader=True) == "datashader"
    # explicit datashader honored only when the colouring allows it
    assert _resolve_as_points_method(rp_ds, n=10, allow_datashader=True) == "datashader"
    assert _resolve_as_points_method(rp_ds, n=10, allow_datashader=False) == "matplotlib"
    # explicit matplotlib always matplotlib; empty always matplotlib
    assert _resolve_as_points_method(rp_mpl, n=10**9, allow_datashader=True) == "matplotlib"
    assert _resolve_as_points_method(rp_auto, n=0, allow_datashader=True) == "matplotlib"
