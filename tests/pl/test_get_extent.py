import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Point, Polygon
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Affine, set_transformation

import spatialdata_plot  # noqa: F401
from tests.conftest import DPI, PlotTester, PlotTesterMeta

RNG = np.random.default_rng(seed=42)
sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=DPI, color_map="viridis")
matplotlib.use("agg")  # same as GitHub action runner
_ = spatialdata_plot
plt.tight_layout()

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be changed, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestExtent(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_extent_of_img_full_canvas(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image").pl.show()

    def test_plot_extent_of_points_partial_canvas(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points().pl.show()

    def test_plot_extent_of_partial_canvas_on_full_canvas(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_images(element="blobs_image").pl.render_points().pl.show()

    def test_plot_extent_calculation_respects_element_selection_circles(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_circles").pl.show()

    def test_plot_extent_calculation_respects_element_selection_polygons(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes(element="blobs_polygons").pl.show()

    def test_plot_extent_calculation_respects_element_selection_circles_and_polygons(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_shapes("blobs_circles").pl.render_shapes("blobs_polygons").pl.show()

    def test_plot_extent_of_img_is_correct_after_spatial_query(self, sdata_blobs: SpatialData):
        cropped_blobs = sdata_blobs.query.bounding_box(
            axes=["x", "y"], min_coordinate=[100, 100], max_coordinate=[400, 400], target_coordinate_system="global"
        )
        cropped_blobs.pl.render_images().pl.show()

    def test_plot_correct_plot_after_transformations(self):
        # inspired by https://github.com/scverse/spatialdata/blob/ef0a2dc7f9af8d4c84f15eec503177f1d08c3d46/tests/core/test_data_extent.py#L125

        circles = [Point(p) for p in [[0.5, 0.1], [0.9, 0.5], [0.5, 0.9], [0.1, 0.5]]]
        circles_gdf = GeoDataFrame(geometry=circles)
        circles_gdf["radius"] = 0.1
        circles_gdf = ShapesModel.parse(circles_gdf)

        polygons = [Polygon([(0.5, 0.5), (0.5, 0), (0.6, 0.1), (0.5, 0.5)])]
        polygons.append(Polygon([(0.5, 0.5), (1, 0.5), (0.9, 0.6), (0.5, 0.5)]))
        polygons.append(Polygon([(0.5, 0.5), (0.5, 1), (0.4, 0.9), (0.5, 0.5)]))
        polygons.append(Polygon([(0.5, 0.5), (0, 0.5), (0.1, 0.4), (0.5, 0.5)]))
        polygons_gdf = GeoDataFrame(geometry=polygons)
        polygons_gdf = ShapesModel.parse(polygons_gdf)

        multipolygons = [
            MultiPolygon(
                [
                    polygons[0],
                    Polygon([(0.7, 0.1), (0.9, 0.1), (0.9, 0.3), (0.7, 0.1)]),
                ]
            )
        ]
        multipolygons.append(MultiPolygon([polygons[1], Polygon([(0.9, 0.7), (0.9, 0.9), (0.7, 0.9), (0.9, 0.7)])]))
        multipolygons.append(MultiPolygon([polygons[2], Polygon([(0.3, 0.9), (0.1, 0.9), (0.1, 0.7), (0.3, 0.9)])]))
        multipolygons.append(MultiPolygon([polygons[3], Polygon([(0.1, 0.3), (0.1, 0.1), (0.3, 0.1), (0.1, 0.3)])]))
        multipolygons_gdf = GeoDataFrame(geometry=multipolygons)
        multipolygons_gdf = ShapesModel.parse(multipolygons_gdf)

        points_df = PointsModel.parse(np.array([[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]]))

        sdata = SpatialData(
            shapes={
                "circles": circles_gdf,
                "polygons": polygons_gdf,
                "multipolygons": multipolygons_gdf,
                "circles_pi3": circles_gdf,
                "polygons_pi3": polygons_gdf,
                "multipolygons_pi3": multipolygons_gdf,
                "circles_pi4": circles_gdf,
                "polygons_pi4": polygons_gdf,
                "multipolygons_pi4": multipolygons_gdf,
            },
            points={"points": points_df, "points_pi3": points_df, "points_pi4": points_df},
        )

        for i in [3, 4]:
            theta = math.pi / i
            rotation = Affine(
                [
                    [math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1],
                ],
                input_axes=("x", "y"),
                output_axes=("x", "y"),
            )
            for element_name in [f"circles_pi{i}", f"polygons_pi{i}", f"multipolygons_pi{i}", f"points_pi{i}"]:
                set_transformation(element=sdata[element_name], transformation=rotation, to_coordinate_system=f"pi{i}")

        _, axs = plt.subplots(ncols=3, nrows=4, figsize=(7, 9))

        for cs_idx, cs in enumerate(["global", "pi3", "pi4"]):
            if cs == "global":
                circles_name = "circles"
                polygons_name = "polygons"
                multipolygons_name = "multipolygons"
                points_name = "points"
            elif cs == "pi3":
                circles_name = "circles_pi3"
                polygons_name = "polygons_pi3"
                multipolygons_name = "multipolygons_pi3"
                points_name = "points_pi3"
            else:
                circles_name = "circles_pi4"
                polygons_name = "polygons_pi4"
                multipolygons_name = "multipolygons_pi4"
                points_name = "points_pi4"

            sdata.pl.render_shapes(element=circles_name).pl.show(coordinate_systems=cs, ax=axs[0, cs_idx], title="")
            sdata.pl.render_shapes(element=polygons_name).pl.show(coordinate_systems=cs, ax=axs[1, cs_idx], title="")
            sdata.pl.render_shapes(element=multipolygons_name).pl.show(
                coordinate_systems=cs, ax=axs[2, cs_idx], title=""
            )
            sdata.pl.render_points(element=points_name, size=10).pl.show(
                coordinate_systems=cs, ax=axs[3, cs_idx], title="", pad_extent=0.02
            )

        plt.tight_layout()
