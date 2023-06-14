import matplotlib
import scanpy as sc
import spatialdata_plot  # noqa: F401
from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd

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
        sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show()

    def test_plot_can_render_polygons(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_polygons").pl.show()

    def test_plot_can_render_multipolygons(self):

        def _make_multi():
            hole = MultiPolygon([(
                ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                [((0.2,0.2), (0.2,0.8), (0.8,0.8), (0.8,0.2))]
            )])
            overlap = MultiPolygon([
                Polygon([(2.0, 0.0), (2.0, 0.8), (2.8, 0.8), (2.8, 0.0)]),
                Polygon([(2.2, 0.2), (2.2, 1.0), (3.0, 1.0), (3.0, 0.2)])
            ])
            poly = Polygon([(4.0, 0.0), (4.0, 1.0), (5.0, 1.0), (5.0, 0.0)])
            polygon_series = gpd.GeoSeries([hole, overlap, poly])
            cell_polygon_table = gpd.GeoDataFrame(geometry=polygon_series)
            sd_polygons = ShapesModel.parse(cell_polygon_table)
            return sd_polygons

        sdata = SpatialData(shapes={"p": _make_multi()})
        sdata.pl.render_shapes(outline=True, fill_alpha=0.3).pl.show()
