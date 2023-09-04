import anndata
import geopandas as gpd
import matplotlib
import pandas as pd
import scanpy as sc
import spatialdata_plot  # noqa: F401
from shapely.geometry import MultiPolygon, Point, Polygon
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

from tests.conftest import PlotTester, PlotTesterMeta

sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=40, color_map="viridis")
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
        sdata_blobs.pl.render_shapes(elements="blobs_circles").pl.show()

    def test_plot_can_render_circles_with_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_circles", outline=True).pl.show()

    def test_plot_can_render_circles_with_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_circles", outline=True, outline_color="red").pl.show()

    def test_plot_can_render_polygons(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_polygons").pl.show()

    def test_plot_can_render_polygons_with_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_polygons", outline=True).pl.show()

    def test_plot_can_render_polygons_with_str_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_polygons", outline=True, outline_color="red").pl.show()

    def test_plot_can_render_polygons_with_rgb_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            elements="blobs_polygons", outline=True, outline_color=(0.0, 0.0, 1.0, 1.0)
        ).pl.show()

    def test_plot_can_render_polygons_with_rgba_colored_outline(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(
            elements="blobs_polygons", outline=True, outline_color=(0.0, 1.0, 0.0, 1.0)
        ).pl.show()

    def test_plot_can_render_empty_geometry(self, sdata_blobs: SpatialData):
        sdata_blobs.shapes["blobs_circles"].at[0, "geometry"] = gpd.points_from_xy([None], [None])[0]
        sdata_blobs.pl.render_shapes().pl.show()

    def test_plot_can_render_circles_with_default_outline_width(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_circles", outline=True).pl.show()

    def test_plot_can_render_circles_with_specified_outline_width(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_circles", outline=True, outline_width=3.0).pl.show()

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
        sdata.table = table
        sdata.pl.render_shapes(color="val", outline=True, fill_alpha=0.3).pl.show()

    def test_plot_can_color_from_geodataframe(self, sdata_blobs: SpatialData):
        blob = sdata_blobs
        blob.shapes["blobs_polygons"]["value"] = [1, 10, 1, 20, 1]
        blob.pl.render_shapes(
            elements="blobs_polygons",
            color="value",
        ).pl.show()

    def test_plot_can_scale_shapes(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(elements="blobs_circles", scale=0.5).pl.show()

    def test_plot_colorbar_respects_input_limits(self, sdata_blobs: SpatialData):
        sdata_blobs.shapes["blobs_polygons"]["cluster"] = [1, 2, 3, 5, 20]
        sdata_blobs.pl.render_shapes("blobs_polygons", color="cluster", groups=["c1"]).pl.show()

    def test_plot_colorbar_can_be_normalised(self, sdata_blobs: SpatialData):
        sdata_blobs.shapes["blobs_polygons"]["cluster"] = [1, 2, 3, 5, 20]
        sdata_blobs.pl.render_shapes("blobs_polygons", color="cluster", groups=["c1"], norm=True).pl.show()
