import math

import anndata
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from matplotlib.colors import Normalize
from shapely.geometry import MultiPolygon, Point, Polygon
from spatialdata import SpatialData, deepcopy
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import Affine, Identity, MapAxis, Scale, Sequence, Translation
from spatialdata.transformations._utils import _set_transformations

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


class TestShapes(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_circles(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show()

    def test_plot_can_render_circles_with_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", outline_alpha=1).pl.show()

    def test_plot_can_render_circles_with_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", outline_alpha=1, outline_color="red").pl.show()

    def test_plot_can_render_polygons(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_polygons").pl.show()

    def test_plot_can_render_polygons_with_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_polygons", outline_alpha=1).pl.show()

    def test_plot_can_render_polygons_with_str_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_polygons", outline_alpha=1, outline_color="red").pl.show()

    def test_plot_can_render_polygons_with_rgb_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            element="blobs_polygons", outline_alpha=1, outline_color=(0.0, 0.0, 1.0, 1.0)
        ).pl.show()

    def test_plot_can_render_polygons_with_rgba_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            element="blobs_polygons", outline_alpha=1, outline_color=(0.0, 1.0, 0.0, 1.0)
        ).pl.show()

    def test_plot_can_render_empty_geometry(self, sdata_blobs: SpatialData):
        sdata_blobs.shapes["blobs_circles"].at[0, "geometry"] = gpd.points_from_xy([None], [None])[0]
        sdata_blobs.pl.render_shapes().pl.show()

    def test_plot_can_render_circles_with_default_outline_width(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", outline_alpha=1).pl.show()

    def test_plot_can_render_circles_with_specified_outline_width(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", outline_alpha=1, outline_width=3.0).pl.show()

    def test_plot_can_render_multipolygons(self):
        def _make_multi():
            hole = MultiPolygon(
                [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)), [((0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2))])]
            )
            overlap = MultiPolygon(
                [
                    Polygon([(2.0, 0.0), (2.0, 0.8), (2.8, 0.8), (2.8, 0.0)]),
                    Polygon([(2.2, 0.2), (2.2, 1.0), (3.0, 1.0), (3.0, 0.2)]),
                ]
            )
            poly = Polygon([(4.0, 0.0), (4.0, 1.0), (5.0, 1.0), (5.0, 0.0)])
            circ = Point(6.0, 0.5)
            polygon_series = gpd.GeoSeries([hole, overlap, poly, circ])
            cell_polygon_table = gpd.GeoDataFrame(geometry=polygon_series)
            sd_polygons = ShapesModel.parse(cell_polygon_table)
            sd_polygons.loc[:, "radius"] = [None, None, None, 0.3]

            return sd_polygons

        sdata = SpatialData(shapes={"p": _make_multi()})
        adata = anndata.AnnData(pd.DataFrame({"p": ["hole", "overlap", "square", "circle"]}))
        adata.obs.loc[:, "region"] = "p"
        adata.obs.loc[:, "val"] = [0, 1, 2, 3]
        table = TableModel.parse(adata, region="p", region_key="region", instance_key="val")
        sdata["table"] = table
        sdata.pl.render_shapes(color="val", fill_alpha=0.3).pl.show()

    def test_plot_can_render_multipolygons_with_multiple_holes(self):
        square = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]
        first_hole = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0), (1.0, 1.0)]
        second_hole = [(3.0, 3.0), (4.0, 3.0), (4.0, 4.0), (3.0, 4.0), (3.0, 3.0)]
        multipoly = MultiPolygon([Polygon(square, holes=[first_hole, second_hole])])
        cell_polygon_table = gpd.GeoDataFrame(geometry=gpd.GeoSeries([multipoly]))
        sd_polygons = ShapesModel.parse(cell_polygon_table)
        sdata = SpatialData(shapes={"two_holes": sd_polygons})

        fig, ax = plt.subplots()
        sdata.pl.render_shapes(element="two_holes").pl.show(ax=ax)
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 6)

        fig.tight_layout()

    def test_plot_can_color_multipolygons_with_multiple_holes(self):
        square = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]
        first_hole = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0), (1.0, 1.0)]
        second_hole = [(3.0, 3.0), (4.0, 3.0), (4.0, 4.0), (3.0, 4.0), (3.0, 3.0)]
        multipoly = MultiPolygon([Polygon(square, holes=[first_hole, second_hole])])
        cell_polygon_table = gpd.GeoDataFrame(geometry=gpd.GeoSeries([multipoly]))
        cell_polygon_table["instance_id"] = [0]
        sd_polygons = ShapesModel.parse(cell_polygon_table)

        adata = anndata.AnnData(pd.DataFrame({"value": [1]}))
        adata.obs["region"] = pd.Categorical(["two_holes"] * adata.n_obs)
        adata.obs["instance_id"] = [0]
        adata.obs["category"] = ["holey"]
        table = TableModel.parse(adata, region="two_holes", region_key="region", instance_key="instance_id")

        sdata = SpatialData(shapes={"two_holes": sd_polygons}, tables={"table": table})

        fig, ax = plt.subplots()
        sdata.pl.render_shapes(element="two_holes", color="category", table_name="table").pl.show(ax=ax)
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 6)

        fig.tight_layout()

    def test_plot_can_color_from_geodataframe(self, sdata_blobs: SpatialData):
        blob = deepcopy(sdata_blobs)
        blob["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * blob["table"].n_obs)
        blob["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        blob.shapes["blobs_polygons"]["value"] = [1, 10, 1, 20, 1]
        blob.pl.render_shapes(
            element="blobs_polygons",
            color="value",
        ).pl.show()

    def test_plot_can_scale_shapes(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", scale=0.5).pl.show()

    def test_plot_can_filter_with_groups(self, sdata_blobs: SpatialData):
        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")

        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"]["cluster"] = "c1"
        sdata_blobs.shapes["blobs_polygons"].iloc[3:5, 1] = "c2"
        sdata_blobs.shapes["blobs_polygons"]["cluster"] = sdata_blobs.shapes["blobs_polygons"]["cluster"].astype(
            "category"
        )

        sdata_blobs.pl.render_shapes("blobs_polygons", color="cluster").pl.show(ax=axs[0], legend_fontsize=6)
        sdata_blobs.pl.render_shapes("blobs_polygons", color="cluster", groups="c1").pl.show(
            ax=axs[1], legend_fontsize=6
        )

    def test_plot_coloring_with_palette(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"]["cluster"] = "c1"
        sdata_blobs.shapes["blobs_polygons"].iloc[3:5, 1] = "c2"
        sdata_blobs.shapes["blobs_polygons"]["cluster"] = sdata_blobs.shapes["blobs_polygons"]["cluster"].astype(
            "category"
        )

        sdata_blobs.pl.render_shapes(
            "blobs_polygons", color="cluster", groups=["c2", "c1"], palette=["green", "yellow"]
        ).pl.show()

    def test_plot_colorbar_respects_input_limits(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"]["cluster"] = [1, 2, 3, 5, 20]
        sdata_blobs.pl.render_shapes("blobs_polygons", color="cluster").pl.show()

    def test_plot_colorbar_can_be_normalised(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"]["cluster"] = [1, 2, 3, 5, 20]
        norm = Normalize(vmin=0, vmax=5, clip=True)
        sdata_blobs.pl.render_shapes("blobs_polygons", color="cluster", groups=["c1"], norm=norm).pl.show()

    def test_render_shapes_raises_when_color_key_missing(self, sdata_blobs_shapes_annotated: SpatialData):
        missing_col = "__non_existent_column__"
        with pytest.raises(KeyError, match=f"Unable to locate color key '{missing_col}'"):
            sdata_blobs_shapes_annotated.pl.render_shapes(
                element="blobs_polygons",
                color=missing_col,
            ).pl.show()

    def test_render_shapes_raises_for_invalid_table_name(self, sdata_blobs_shapes_annotated: SpatialData):
        table = sdata_blobs_shapes_annotated["table"]
        table.obs["region"] = pd.Categorical(["blobs_polygons"] * table.n_obs)
        table.uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        table.obs["valid_col"] = np.arange(table.n_obs)

        with pytest.raises(KeyError, match="Table 'not_a_table' does not annotate element 'blobs_polygons'"):
            sdata_blobs_shapes_annotated.pl.render_shapes(
                element="blobs_polygons", color="valid_col", table_name="not_a_table"
            )

    def test_render_shapes_raises_for_missing_column_in_table(self, sdata_blobs_shapes_annotated: SpatialData):
        table = sdata_blobs_shapes_annotated["table"]
        table.obs["region"] = pd.Categorical(["blobs_polygons"] * table.n_obs)
        table.uns["spatialdata_attrs"]["region"] = "blobs_polygons"

        with pytest.raises(
            KeyError, match="Column 'not_a_column' not found in obs/var of table 'table' for element 'blobs_polygons'"
        ):
            sdata_blobs_shapes_annotated.pl.render_shapes(
                element="blobs_polygons", color="not_a_column", table_name="table"
            )

    def test_plot_can_plot_shapes_after_spatial_query(self, sdata_blobs: SpatialData):
        # subset to only shapes, should be unnecessary after rasterizeation of multiscale images is included
        blob = SpatialData.init_from_elements(
            {
                "blobs_circles": sdata_blobs.shapes["blobs_circles"],
                "blobs_multipolygons": sdata_blobs.shapes["blobs_multipolygons"],
                "blobs_polygons": sdata_blobs.shapes["blobs_polygons"],
            }
        )
        cropped_blob = blob.query.bounding_box(
            axes=["x", "y"], min_coordinate=[100, 100], max_coordinate=[300, 300], target_coordinate_system="global"
        )
        cropped_blob.pl.render_shapes().pl.show()

    def test_plot_can_plot_with_annotation_despite_random_shuffling(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_circles"] * sdata_blobs["table"].n_obs)
        new_table = sdata_blobs["table"][:5]
        new_table.uns["spatialdata_attrs"]["region"] = "blobs_circles"
        new_table.obs["instance_id"] = np.array(range(5))

        new_table.obs["annotation"] = ["a", "b", "c", "d", "e"]
        new_table.obs["annotation"] = new_table.obs["annotation"].astype("category")

        sdata_blobs["table"] = new_table

        # random permutation of table and shapes
        sdata_blobs["table"].obs = sdata_blobs["table"].obs.sample(frac=1, random_state=83)
        temp = sdata_blobs["blobs_circles"].sample(frac=1, random_state=47)
        del sdata_blobs.shapes["blobs_circles"]
        sdata_blobs["blobs_circles"] = temp

        sdata_blobs.pl.render_shapes("blobs_circles", color="annotation").pl.show()

    def test_plot_can_plot_queried_with_annotation_despite_random_shuffling(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_circles"] * sdata_blobs["table"].n_obs)
        new_table = sdata_blobs["table"][:5].copy()
        new_table.uns["spatialdata_attrs"]["region"] = "blobs_circles"
        new_table.obs["instance_id"] = np.array(range(5))

        new_table.obs["annotation"] = ["a", "b", "c", "d", "e"]
        new_table.obs["annotation"] = new_table.obs["annotation"].astype("category")

        sdata_blobs["table"] = new_table

        # random permutation of table and shapes
        sdata_blobs["table"].obs = sdata_blobs["table"].obs.sample(frac=1, random_state=83)
        temp = sdata_blobs["blobs_circles"].sample(frac=1, random_state=47)
        del sdata_blobs.shapes["blobs_circles"]
        sdata_blobs["blobs_circles"] = temp

        # subsetting the data
        sdata_cropped = sdata_blobs.query.bounding_box(
            axes=("x", "y"),
            min_coordinate=[100, 150],
            max_coordinate=[400, 250],
            target_coordinate_system="global",
            filter_table=True,
        )

        sdata_cropped.pl.render_shapes("blobs_circles", color="annotation").pl.show()

    def test_plot_can_color_two_shapes_elements_by_annotation(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_circles"] * sdata_blobs["table"].n_obs)
        new_table = sdata_blobs["table"][:10].copy()
        new_table.uns["spatialdata_attrs"]["region"] = ["blobs_circles", "blobs_polygons"]
        new_table.obs["instance_id"] = np.concatenate((np.array(range(5)), np.array(range(5))))

        new_table.obs["region"] = new_table.obs["region"].cat.add_categories(["blobs_polygons"])
        new_table.obs.loc[5 * [False] + 5 * [True], "region"] = "blobs_polygons"
        new_table.obs["annotation"] = ["a", "b", "c", "d", "e", "v", "w", "x", "y", "z"]
        new_table.obs["annotation"] = new_table.obs["annotation"].astype("category")

        sdata_blobs["table"] = new_table

        sdata_blobs.pl.render_shapes("blobs_circles", color="annotation").pl.render_shapes(
            "blobs_polygons", color="annotation"
        ).pl.show()

    def test_plot_can_color_two_queried_shapes_elements_by_annotation(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_circles"] * sdata_blobs["table"].n_obs)
        new_table = sdata_blobs["table"][:10].copy()
        new_table.uns["spatialdata_attrs"]["region"] = ["blobs_circles", "blobs_polygons"]
        new_table.obs["instance_id"] = np.concatenate((np.array(range(5)), np.array(range(5))))

        new_table.obs["region"] = new_table.obs["region"].cat.add_categories(["blobs_polygons"])
        new_table.obs.loc[5 * [False] + 5 * [True], "region"] = "blobs_polygons"
        new_table.obs["annotation"] = ["a", "b", "c", "d", "e", "v", "w", "x", "y", "z"]
        new_table.obs["annotation"] = new_table.obs["annotation"].astype("category")

        sdata_blobs["table"] = new_table
        sdata_blobs["table"].obs = sdata_blobs["table"].obs.sample(frac=1, random_state=83)
        temp = sdata_blobs["blobs_circles"].sample(frac=1, random_state=47)
        sdata_blobs["blobs_circles"] = temp
        temp = sdata_blobs["blobs_polygons"].sample(frac=1, random_state=71)
        sdata_blobs["blobs_polygons"] = temp

        # subsetting the data
        sdata_cropped = sdata_blobs.query.bounding_box(
            axes=("x", "y"),
            min_coordinate=[100, 150],
            max_coordinate=[350, 300],
            target_coordinate_system="global",
            filter_table=True,
        )

        sdata_cropped.pl.render_shapes("blobs_circles", color="annotation").pl.render_shapes(
            "blobs_polygons", color="annotation"
        ).pl.show()

    def test_plot_can_stack_render_shapes(self, sdata_blobs: SpatialData):
        (
            sdata_blobs.pl.render_shapes(element="blobs_circles", na_color="red", fill_alpha=0.5)
            .pl.render_shapes(element="blobs_polygons", na_color="blue", fill_alpha=0.5)
            .pl.show()
        )

    def test_plot_can_color_by_color_name(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", color="red").pl.show()

    def test_plot_can_color_by_rgb_array(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", color=[0.5, 0.5, 1.0]).pl.show()

    def test_plot_can_color_by_rgba_array(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", color=[0.5, 0.5, 1.0, 0.5]).pl.show()

    def test_plot_can_color_by_hex(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", color="#88a136").pl.show()

    def test_plot_can_color_by_hex_with_alpha(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", color="#88a13688").pl.show()

    def test_plot_alpha_overwrites_opacity_from_color(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", color=[0.5, 0.5, 1.0, 0.5], fill_alpha=1.0).pl.show()

    def test_plot_shapes_coercable_categorical_color(self, sdata_blobs: SpatialData):
        n_obs = len(sdata_blobs["blobs_polygons"])
        adata = AnnData(get_standard_RNG().normal(size=(n_obs, 10)))
        adata.obs = pd.DataFrame(get_standard_RNG().normal(size=(n_obs, 3)), columns=["a", "b", "c"])
        adata.obs["instance_id"] = np.arange(adata.n_obs)
        adata.obs["category"] = get_standard_RNG().choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs["instance_id"] = list(range(adata.n_obs))
        adata.obs["region"] = "blobs_polygons"
        table = TableModel.parse(adata=adata, region_key="region", instance_key="instance_id", region="blobs_polygons")
        sdata_blobs["table"] = table

        sdata_blobs.pl.render_shapes("blobs_polygons", color="category").pl.show()

    def test_plot_shapes_categorical_color(self, sdata_blobs: SpatialData):
        n_obs = len(sdata_blobs["blobs_polygons"])
        adata = AnnData(get_standard_RNG().normal(size=(n_obs, 10)))
        adata.obs = pd.DataFrame(get_standard_RNG().normal(size=(n_obs, 3)), columns=["a", "b", "c"])
        adata.obs["instance_id"] = np.arange(adata.n_obs)
        adata.obs["category"] = get_standard_RNG().choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs["instance_id"] = list(range(adata.n_obs))
        adata.obs["region"] = "blobs_polygons"
        table = TableModel.parse(adata=adata, region_key="region", instance_key="instance_id", region="blobs_polygons")
        sdata_blobs["table"] = table

        sdata_blobs["table"].obs["category"] = sdata_blobs["table"].obs["category"].astype("category")
        sdata_blobs.pl.render_shapes("blobs_polygons", color="category").pl.show()

    def test_plot_datashader_can_render_shapes(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(method="datashader").pl.show()

    def test_plot_datashader_can_render_colored_shapes(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(method="datashader", color="red").pl.show()

    def test_plot_datashader_can_render_with_different_alpha(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(method="datashader", fill_alpha=0.7).pl.show()

    def test_plot_datashader_can_color_by_category(self, sdata_blobs: SpatialData):
        n_obs = len(sdata_blobs["blobs_polygons"])
        adata = AnnData(get_standard_RNG().normal(size=(n_obs, 10)))
        adata.obs = pd.DataFrame(get_standard_RNG().normal(size=(n_obs, 3)), columns=["a", "b", "c"])
        adata.obs["category"] = get_standard_RNG().choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs["instance_id"] = list(range(adata.n_obs))
        adata.obs["region"] = "blobs_polygons"
        table = TableModel.parse(
            adata=adata,
            region_key="region",
            instance_key="instance_id",
            region="blobs_polygons",
        )
        sdata_blobs["table"] = table

        sdata_blobs.pl.render_shapes(element="blobs_polygons", color="category", method="datashader").pl.show()

    def test_plot_datashader_can_color_by_category_with_cmap(self, sdata_blobs: SpatialData):
        n_obs = len(sdata_blobs["blobs_polygons"])
        adata = AnnData(get_standard_RNG().normal(size=(n_obs, 10)))
        adata.obs = pd.DataFrame(get_standard_RNG().normal(size=(n_obs, 3)), columns=["a", "b", "c"])
        adata.obs["category"] = get_standard_RNG().choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs["instance_id"] = list(range(adata.n_obs))
        adata.obs["region"] = "blobs_polygons"
        table = TableModel.parse(
            adata=adata,
            region_key="region",
            instance_key="instance_id",
            region="blobs_polygons",
        )
        sdata_blobs["table"] = table

        sdata_blobs.pl.render_shapes(
            element="blobs_polygons", color="category", method="datashader", cmap="cool"
        ).pl.show()

    def test_plot_can_color_by_category_with_cmap(self, sdata_blobs: SpatialData):
        n_obs = len(sdata_blobs["blobs_polygons"])
        adata = AnnData(get_standard_RNG().normal(size=(n_obs, 10)))
        adata.obs = pd.DataFrame(get_standard_RNG().normal(size=(n_obs, 3)), columns=["a", "b", "c"])
        adata.obs["category"] = get_standard_RNG().choice(["a", "b", "c"], size=adata.n_obs)
        adata.obs["instance_id"] = list(range(adata.n_obs))
        adata.obs["region"] = "blobs_polygons"
        table = TableModel.parse(
            adata=adata,
            region_key="region",
            instance_key="instance_id",
            region="blobs_polygons",
        )
        sdata_blobs["table"] = table

        sdata_blobs.pl.render_shapes(element="blobs_polygons", color="category", cmap="cool").pl.show()

    def test_plot_datashader_can_color_by_value(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"]["value"] = [1, 10, 1, 20, 1]
        sdata_blobs.pl.render_shapes(element="blobs_polygons", color="value", method="datashader").pl.show()

    def test_plot_datashader_can_color_by_identical_value(self, sdata_blobs: SpatialData):
        """
        We test this, because datashader internally scales the values, so when all shapes have the same value,
        the scaling would lead to all of them being assigned an alpha of 0, so we wouldn't see anything
        """
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"]["value"] = [1, 1, 1, 1, 1]
        sdata_blobs.pl.render_shapes(element="blobs_polygons", color="value", method="datashader").pl.show()

    def test_plot_datashader_shades_with_linear_cmap(self, sdata_blobs: SpatialData):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"]["value"] = [1, 2, 1, 20, 1]
        sdata_blobs.pl.render_shapes(element="blobs_polygons", color="value", method="datashader").pl.show()

    def test_plot_datashader_can_render_with_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(method="datashader", element="blobs_polygons", outline_alpha=1).pl.show()

    def test_plot_datashader_can_render_with_diff_alpha_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(method="datashader", element="blobs_polygons", outline_alpha=0.5).pl.show()

    def test_plot_datashader_can_render_with_diff_width_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            method="datashader", element="blobs_polygons", outline_alpha=1.0, outline_width=5.0
        ).pl.show()

    def test_plot_datashader_can_render_with_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            method="datashader", element="blobs_polygons", outline_alpha=1, outline_color="red"
        ).pl.show()

    def test_plot_datashader_can_render_with_rgb_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            method="datashader", element="blobs_polygons", outline_alpha=1, outline_color=(0.0, 0.0, 1.0)
        ).pl.show()

    def test_plot_datashader_can_render_with_rgba_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            method="datashader", element="blobs_polygons", outline_alpha=1, outline_color=(0.0, 1.0, 0.0, 1.0)
        ).pl.show()

    def test_plot_can_set_clims_clip(self, sdata_blobs: SpatialData):
        table_shapes = sdata_blobs["table"][:5].copy()
        table_shapes.obs.instance_id = list(range(5))
        table_shapes.obs["region"] = pd.Categorical(["blobs_circles"] * table_shapes.n_obs)
        table_shapes.obs["dummy_gene_expression"] = [i * 10 for i in range(5)]
        table_shapes.uns["spatialdata_attrs"]["region"] = "blobs_circles"
        sdata_blobs["new_table"] = table_shapes

        norm = Normalize(vmin=20, vmax=40, clip=True)
        sdata_blobs.pl.render_shapes(
            "blobs_circles", color="dummy_gene_expression", norm=norm, table_name="new_table"
        ).pl.show()

    def test_plot_datashader_can_transform_polygons(self, sdata_blobs: SpatialData):
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

        _set_transformations(sdata_blobs["blobs_polygons"], {"global": seq})

        sdata_blobs.pl.render_shapes("blobs_polygons", method="datashader", outline_alpha=1.0).pl.show()

    def test_plot_datashader_can_transform_multipolygons(self, sdata_blobs: SpatialData):
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

        _set_transformations(sdata_blobs["blobs_multipolygons"], {"global": seq})

        sdata_blobs.pl.render_shapes("blobs_multipolygons", method="datashader", outline_alpha=1.0).pl.show()

    def test_plot_datashader_can_transform_circles(self, sdata_blobs: SpatialData):
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

        _set_transformations(sdata_blobs["blobs_circles"], {"global": seq})

        sdata_blobs.pl.render_shapes("blobs_circles", method="datashader", outline_alpha=1.0).pl.show()

    def test_plot_can_do_non_matching_table(self, sdata_blobs: SpatialData):
        table_shapes = sdata_blobs["table"][:3].copy()
        table_shapes.obs.instance_id = list(range(3))
        table_shapes.obs["region"] = pd.Categorical(["blobs_circles"] * table_shapes.n_obs)
        table_shapes.uns["spatialdata_attrs"]["region"] = "blobs_circles"
        sdata_blobs["new_table"] = table_shapes

        sdata_blobs.pl.render_shapes("blobs_circles", color="instance_id").pl.show()

    def test_plot_can_color_with_norm_no_clipping(self, sdata_blobs_shapes_annotated: SpatialData):
        sdata_blobs_shapes_annotated.pl.render_shapes(
            element="blobs_polygons", color="value", norm=Normalize(2, 4, clip=False), cmap=_viridis_with_under_over()
        ).pl.show()

    def test_plot_datashader_can_color_with_norm_and_clipping(self, sdata_blobs_shapes_annotated: SpatialData):
        sdata_blobs_shapes_annotated.pl.render_shapes(
            element="blobs_polygons",
            color="value",
            norm=Normalize(2, 4, clip=True),
            cmap=_viridis_with_under_over(),
            method="datashader",
            datashader_reduction="max",
        ).pl.show()

    def test_plot_datashader_can_color_with_norm_no_clipping(self, sdata_blobs_shapes_annotated: SpatialData):
        sdata_blobs_shapes_annotated.pl.render_shapes(
            element="blobs_polygons",
            color="value",
            norm=Normalize(2, 4, clip=False),
            cmap=_viridis_with_under_over(),
            method="datashader",
            datashader_reduction="max",
        ).pl.show()

    def test_plot_datashader_norm_vmin_eq_vmax_without_clip(self, sdata_blobs_shapes_annotated: SpatialData):
        sdata_blobs_shapes_annotated.pl.render_shapes(
            element="blobs_polygons",
            color="value",
            norm=Normalize(3, 3, clip=False),
            cmap=_viridis_with_under_over(),
            method="datashader",
            datashader_reduction="max",
        ).pl.show()

    def test_plot_datashader_norm_vmin_eq_vmax_with_clip(self, sdata_blobs_shapes_annotated: SpatialData):
        sdata_blobs_shapes_annotated.pl.render_shapes(
            element="blobs_polygons",
            color="value",
            norm=Normalize(3, 3, clip=True),
            cmap=_viridis_with_under_over(),
            method="datashader",
            datashader_reduction="max",
        ).pl.show()

    def test_plot_can_annotate_shapes_with_table_layer(self, sdata_blobs: SpatialData):
        nrows, ncols = 5, 3
        feature_matrix = get_standard_RNG().random((nrows, ncols))
        var_names = [f"feature{i}" for i in range(ncols)]

        obs_indices = sdata_blobs["blobs_circles"].index

        obs = pd.DataFrame()
        obs["instance_id"] = obs_indices
        obs["region"] = "blobs_circles"
        obs["region"].astype("category")

        table = AnnData(X=feature_matrix, var=pd.DataFrame(index=var_names), obs=obs)
        table = TableModel.parse(table, region="blobs_circles", region_key="region", instance_key="instance_id")
        sdata_blobs["circle_table"] = table
        sdata_blobs["circle_table"].layers["normalized"] = get_standard_RNG().random((nrows, ncols))

        sdata_blobs.pl.render_shapes("blobs_circles", color="feature0", table_layer="normalized").pl.show()

    def test_plot_can_render_circles_to_hex(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", shape="hex").pl.show()

    def test_plot_can_render_circles_to_square(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", shape="square").pl.show()

    def test_plot_can_render_polygons_to_hex(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_polygons", shape="hex").pl.show()

    def test_plot_can_render_polygons_to_square(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_polygons", shape="square").pl.show()

    def test_plot_can_render_polygons_to_circle(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_polygons", shape="circle").pl.show()

    def test_plot_can_render_multipolygons_to_hex(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_multipolygons", shape="hex").pl.show()

    def test_plot_can_render_multipolygons_to_square(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_multipolygons", shape="square").pl.show()

    def test_plot_can_render_multipolygons_to_circle(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_multipolygons", shape="circle").pl.show()

    def test_plot_visium_hex_hexagonal_grid(self, sdata_hexagonal_grid_spots: SpatialData):
        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")

        sdata_hexagonal_grid_spots.pl.render_shapes(element="spots", shape="circle").pl.show(ax=axs[0])
        sdata_hexagonal_grid_spots.pl.render_shapes(element="spots", shape="visium_hex").pl.show(ax=axs[1])

    def test_plot_datashader_can_render_circles_to_hex(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", shape="hex", method="datashader").pl.show()

    def test_plot_datashader_can_render_circles_to_square(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles", shape="square", method="datashader").pl.show()

    def test_plot_datashader_can_render_polygons_to_hex(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_polygons", shape="hex", method="datashader").pl.show()

    def test_plot_datashader_can_render_polygons_to_square(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_polygons", shape="square", method="datashader").pl.show()

    def test_plot_datashader_can_render_polygons_to_circle(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_polygons", shape="circle", method="datashader").pl.show()

    def test_plot_datashader_can_render_multipolygons_to_hex(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_multipolygons", shape="hex", method="datashader").pl.show()

    def test_plot_datashader_can_render_multipolygons_to_square(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_multipolygons", shape="square", method="datashader").pl.show()

    def test_plot_datashader_can_render_multipolygons_to_circle(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_multipolygons", shape="circle", method="datashader").pl.show()

    def test_plot_can_render_shapes_with_double_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes("blobs_circles", outline_width=(10.0, 5.0)).pl.show()

    def test_plot_can_render_shapes_with_colored_double_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            "blobs_polygons", outline_width=(10.0, 5.0), outline_color=("purple", "orange")
        ).pl.show()

    def test_plot_can_render_double_outline_with_diff_alpha(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            element="blobs_circles", outline_color=("red", "blue"), outline_alpha=(0.7, 0.3), outline_width=(20, 10)
        ).pl.show()

    def test_plot_outline_alpha_takes_precedence(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            element="blobs_circles",
            outline_color=("#ff660033", "#33aa0066"),
            outline_width=(20, 10),
            outline_alpha=(1.0, 1.0),
        ).pl.show()

    def test_plot_datashader_can_render_shapes_with_double_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes("blobs_circles", outline_width=(10.0, 5.0), method="datashader").pl.show()

    def test_plot_datashader_can_render_shapes_with_colored_double_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            "blobs_polygons",
            outline_width=(10.0, 5.0),
            outline_color=("purple", "orange"),
            method="datashader",
        ).pl.show()

    def test_raises_when_table_does_not_annotate_element(self, sdata_blobs: SpatialData):
        # Work on an independent copy since we mutate tables
        sdata_blobs_local = deepcopy(sdata_blobs)

        # Create a table that annotates a DIFFERENT element than the one we will render
        other_table = sdata_blobs_local["table"].copy()
        other_table.obs["region"] = pd.Categorical(["blobs_points"] * other_table.n_obs)  # Different region
        other_table.uns["spatialdata_attrs"]["region"] = "blobs_points"
        sdata_blobs_local["other_table"] = other_table

        # Rendering "blobs_circles" with a table that annotates "blobs_points"
        # should now raise to alert the user about the mismatch.
        with pytest.raises(
            KeyError,
            match="Table 'other_table' does not annotate element 'blobs_circles'",
        ):
            sdata_blobs_local.pl.render_shapes(
                "blobs_circles",
                color="channel_0_sum",
                table_name="other_table",
            ).pl.show()

    def test_plot_can_handle_nan_values_in_color_data(self, sdata_blobs: SpatialData, caplog):
        """Test that NaN values in color data are handled gracefully."""
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_circles"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_circles"

        # Add color column with NaN values
        sdata_blobs.shapes["blobs_circles"]["color_with_nan"] = [1.0, 2.0, np.nan, 4.0, 5.0]

        with logger_warns(caplog, logger, match="Found 1 NaN values in color data"):
            sdata_blobs.pl.render_shapes(element="blobs_circles", color="color_with_nan", na_color="red").pl.show()

    def test_plot_colorbar_normalization_with_nan_values(self, sdata_blobs: SpatialData):
        """Test that colorbar normalization works correctly with NaN values."""
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"

        sdata_blobs.shapes["blobs_polygons"]["color_with_nan"] = [1.0, 2.0, np.nan, 4.0, 5.0]

        # Test colorbar with NaN values - should use nanmin/nanmax
        sdata_blobs.pl.render_shapes(element="blobs_polygons", color="color_with_nan", na_color="gray").pl.show()

    def test_plot_can_handle_non_numeric_radius_values(self, sdata_blobs: SpatialData):
        """Test that non-numeric radius values are handled gracefully."""
        sdata_blobs.shapes["blobs_circles"]["radius_mixed"] = [1.0, "invalid", 3.0, np.nan, 5.0]

        sdata_blobs.pl.render_shapes(element="blobs_circles", color="red").pl.show()

    def test_plot_can_handle_mixed_numeric_and_color_data(self, sdata_blobs: SpatialData):
        """Test handling of mixed numeric and color-like data."""
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_circles"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_circles"

        sdata_blobs.shapes["blobs_circles"]["mixed_data"] = [1.0, 2.0, np.nan, "red", 5.0]

        sdata_blobs.pl.render_shapes(element="blobs_circles", color="mixed_data", na_color="gray").pl.show()
