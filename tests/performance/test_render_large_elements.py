import numpy as np
import pytest
import scanpy as sc
import spatialdata_plot  # noqa: F401

RNG = np.random.default_rng(seed=42)
sc.pl.set_rcParams_defaults()
# matplotlib.use("agg")  # same as GitHub action runner
_ = spatialdata_plot

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from numpy.random import default_rng
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel


def create_large_spatialdata(
    c: int,
    y: int,
    x: int,
    scale_factors: list[int],
    n_points: int,
    n_circles: int,
    n_polygons: int,
    n_multipolygons: int,
) -> SpatialData:
    rng = default_rng(seed=0)

    # create random image cxy
    image_data = rng.random((c, y, x))
    image = Image2DModel.parse(image_data, dims=["c", "y", "x"])

    # create random labels yx
    labels_data = rng.integers(0, 256, size=(y, x)).astype(np.uint8)
    labels = Labels2DModel.parse(labels_data, dims=["y", "x"])

    # create multiscale versions
    multiscale_image = Image2DModel.parse(image_data, dims=["c", "y", "x"], scale_factors=scale_factors)
    multiscale_labels = Labels2DModel.parse(labels_data, dims=["y", "x"], scale_factors=scale_factors)

    # create random xy points
    points_data = rng.random((n_points, 2)) * [x, y]
    points_df = pd.DataFrame(points_data, columns=["x", "y"])
    points = PointsModel.parse(points_df.to_numpy(), feature_key="x", instance_key="y")

    # create random circles
    circles = ShapesModel.parse(points_df.to_numpy(), geometry=0, radius=10)

    def generate_random_polygons(n: int, bbox: tuple[int, int]) -> list[Polygon]:
        minx = miny = bbox[0]
        maxx = maxy = bbox[1]
        polygons: list[Polygon] = []
        for i in range(n):
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            poly = Polygon(
                [
                    (x + rng.uniform(0, maxx // 4), y + rng.uniform(0, maxy // 4)),
                    (x + rng.uniform(0, maxx // 4), y),
                    (x, y + rng.uniform(0, maxy // 4)),
                ]
            )
            polygons.append(poly)
        return polygons

    # create random polygons
    polygons = GeoDataFrame(geometry=generate_random_polygons(n_polygons, (0, max(x, y))))
    polygons = ShapesModel.parse(polygons)

    def generate_random_multipolygons(n: int, bbox: tuple[int, int]) -> list[MultiPolygon]:
        minx = miny = bbox[0]
        maxx = maxy = bbox[1]
        multipolygons: list[MultiPolygon] = []
        for i in range(n):
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            poly1 = Polygon(
                [
                    (x + rng.uniform(0, maxx // 4), y + rng.uniform(0, maxy // 4)),
                    (x + rng.uniform(0, maxx // 4), y),
                    (x, y + rng.uniform(0, maxy // 4)),
                ]
            )
            poly2 = translate(poly1, xoff=maxx // 4, yoff=maxy // 4)
            multipolygons.append(MultiPolygon([poly1, poly2]))
        return multipolygons

    # create random multipolygons (2 polygons each)
    multipolygons = GeoDataFrame(geometry=generate_random_multipolygons(n_multipolygons, (0, max(x, y))))
    multipolygons = ShapesModel.parse(multipolygons)

    return SpatialData(
        images={"image": image, "multiscale_image": multiscale_image},
        labels={"labels": labels, "multiscale_labels": multiscale_labels},
        points={"points": points},
        shapes={"circles": circles, "polygons": polygons, "multipolygons": multipolygons},
    )


sdata = create_large_spatialdata(
    c=2,
    y=10000,
    x=10000,
    scale_factors=[2, 2, 2],
    n_points=1000,
    n_circles=1000,
    n_polygons=1000,
    n_multipolygons=1000,
)


@pytest.mark.parametrize("element", ["image", "multiscale_image"])
@pytest.mark.benchmark
def test_plot_can_render_large_image(element: str):
    sdata.pl.render_images(element=element).pl.show()


@pytest.mark.parametrize("element", ["labels", "multiscale_labels"])
@pytest.mark.benchmark
def test_plot_can_render_large_labels(element: str):
    sdata.pl.render_labels(element=element).pl.show()


@pytest.mark.parametrize("method", ["matplotlib", "datashader"])
@pytest.mark.benchmark
def test_plot_can_render_large_circles(method: str):
    sdata.pl.render_shapes(element="circles", method=method).pl.show()


@pytest.mark.parametrize("method", ["matplotlib", "datashader"])
@pytest.mark.benchmark
def test_plot_can_render_large_polygons(method: str):
    sdata.pl.render_shapes(element="polygons", method=method).pl.show()


@pytest.mark.parametrize("method", ["matplotlib", "datashader"])
@pytest.mark.benchmark
def test_plot_can_render_large_multipolygons(method: str):
    sdata.pl.render_shapes(element="multipolygons", method=method).pl.show()


@pytest.mark.parametrize("method", ["matplotlib", "datashader"])
@pytest.mark.benchmark
def test_plot_can_render_large_points(method: str):
    sdata.pl.render_points(element="points", method=method).pl.show()
