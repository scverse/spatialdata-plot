from abc import ABC, ABCMeta
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import spatialdata as sd
from anndata import AnnData
from geopandas import GeoDataFrame
from matplotlib.testing.compare import compare_images
from shapely.geometry import MultiPolygon, Polygon
from spatialdata import SpatialData
from spatialdata.datasets import blobs, raccoon
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from xarray import DataArray, DataTree

import spatialdata_plot  # noqa: F401

HERE: Path = Path(__file__).parent

EXPECTED = HERE / "_images"
ACTUAL = HERE / "figures"
TOL = 15
DPI = 80

RNG = np.random.default_rng(seed=42)


@pytest.fixture()
def full_sdata() -> SpatialData:
    return SpatialData(
        images=_get_images(),
        labels=_get_labels(),
        shapes=_get_shapes(),
        points=_get_points(),
        table=_get_table(region="sample1"),
    )


@pytest.fixture()
def sdata_blobs() -> SpatialData:
    return blobs()


@pytest.fixture()
def sdata_blobs_str() -> SpatialData:
    return blobs(n_channels=5, c_coords=["c1", "c2", "c3", "c4", "c5"])


@pytest.fixture()
def sdata_raccoon() -> SpatialData:
    return raccoon()


@pytest.fixture
def test_sdata_single_image():
    """Creates a simple sdata object."""
    images = {
        "data1_image": sd.models.Image2DModel.parse(
            np.zeros((1, 10, 10)), dims=("c", "y", "x"), transformations={"data1": sd.transformations.Identity()}
        )
    }
    sdata = sd.SpatialData(images=images)
    return sdata


@pytest.fixture
def test_sdata_single_image_with_label():
    """Creates a simple sdata object."""
    images = {"data1": sd.models.Image2DModel.parse(np.zeros((1, 10, 10)), dims=("c", "y", "x"))}
    labels = {"label1": sd.models.Labels2DModel.parse(np.zeros((10, 10)), dims=("y", "x"))}
    sdata = sd.SpatialData(images=images, labels=labels)
    return sdata


@pytest.fixture
def test_sdata_multiple_images():
    """Creates an sdata object with multiple images."""
    images = {
        "data1_image": sd.models.Image2DModel.parse(
            np.zeros((1, 10, 10)), dims=("c", "y", "x"), transformations={"data1": sd.transformations.Identity()}
        ),
        "data2_image": sd.models.Image2DModel.parse(
            np.zeros((1, 10, 10)), dims=("c", "y", "x"), transformations={"data1": sd.transformations.Identity()}
        ),
        "data3_image": sd.models.Image2DModel.parse(
            np.zeros((1, 10, 10)), dims=("c", "y", "x"), transformations={"data1": sd.transformations.Identity()}
        ),
    }
    sdata = sd.SpatialData(images=images)
    return sdata


@pytest.fixture
def test_sdata_multiple_images_with_table():
    """Creates an sdata object with multiple images."""
    images = {
        "data1": sd.models.Image2DModel.parse(np.zeros((1, 10, 10)), dims=("c", "y", "x")),
        "data2": sd.models.Image2DModel.parse(np.zeros((1, 10, 10)), dims=("c", "y", "x")),
        "data3": sd.models.Image2DModel.parse(np.zeros((1, 10, 10)), dims=("c", "y", "x")),
    }

    instance_key = "instance_id"
    region_key = "annotated_region"

    adata = AnnData(RNG.normal(size=(30, 10)), obs=pd.DataFrame(RNG.normal(size=(30, 3)), columns=["a", "b", "c"]))
    adata.obs[instance_key] = list(range(3)) + list(range(7)) + list(range(20))
    adata.obs[region_key] = ["data1"] * 3 + ["data2"] * 7 + ["data3"] * 20
    table = TableModel.parse(
        adata=adata, region=adata.obs[region_key].unique().tolist(), instance_key=instance_key, region_key=region_key
    )
    sdata = sd.SpatialData(images=images, tables={"table": table})
    return sdata


@pytest.fixture
def test_sdata_multiple_images_dims():
    """Creates an sdata object with multiple images."""
    images = {
        "data1": sd.models.Image2DModel.parse(np.zeros((3, 10, 10)), dims=("c", "y", "x")),
        "data2": sd.models.Image2DModel.parse(np.zeros((3, 10, 10)), dims=("c", "y", "x")),
        "data3": sd.models.Image2DModel.parse(np.zeros((3, 10, 10)), dims=("c", "y", "x")),
    }
    sdata = sd.SpatialData(images=images)
    return sdata


@pytest.fixture
def test_sdata_multiple_images_diverging_dims():
    """Creates an sdata object with multiple images."""
    images = {
        "data1": sd.models.Image2DModel.parse(np.zeros((3, 10, 10)), dims=("c", "y", "x")),
        "data2": sd.models.Image2DModel.parse(np.zeros((6, 10, 10)), dims=("c", "y", "x")),
        "data3": sd.models.Image2DModel.parse(np.zeros((3, 10, 10)), dims=("c", "y", "x")),
    }
    sdata = sd.SpatialData(images=images)
    return sdata


@pytest.fixture
def sdata_blobs_shapes_annotated() -> SpatialData:
    """Get blobs sdata with continuous annotation of polygons."""
    blob = blobs()
    blob["table"].obs["region"] = "blobs_polygons"
    blob["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
    blob.shapes["blobs_polygons"]["value"] = [1, 2, 3, 4, 5]
    return blob


def _viridis_with_under_over() -> matplotlib.colors.ListedColormap:
    cmap = matplotlib.colormaps["viridis"]
    cmap.set_under("black")
    cmap.set_over("grey")
    return cmap


# Code below taken from spatialdata main repo


@pytest.fixture()
def images() -> SpatialData:
    return SpatialData(images=_get_images())


@pytest.fixture()
def labels() -> SpatialData:
    return SpatialData(labels=_get_labels())


@pytest.fixture()
def polygons() -> SpatialData:
    return SpatialData(polygons=_get_polygons())


@pytest.fixture()
def shapes() -> SpatialData:
    return SpatialData(shapes=_get_shapes())


@pytest.fixture()
def element() -> SpatialData:
    return SpatialData(points=_get_points())


@pytest.fixture()
def table_single_annotation() -> SpatialData:
    return SpatialData(table=_get_table(region="sample1"))


@pytest.fixture()
def table_multiple_annotations() -> SpatialData:
    return SpatialData(table=_get_table(region=["sample1", "sample2"]))


@pytest.fixture()
def empty_table() -> SpatialData:
    adata = AnnData(shape=(0, 0))
    adata = TableModel.parse(adata=adata)
    return SpatialData(table=adata)


@pytest.fixture(
    # params=["labels"]
    params=["full", "empty"] + ["images", "labels", "points", "table_single_annotation", "table_multiple_annotations"]
    # + ["empty_" + x for x in ["table"]] # TODO: empty table not supported yet
)
def sdata(request) -> SpatialData:
    if request.param == "full":
        s = SpatialData(
            images=_get_images(),
            labels=_get_labels(),
            shapes=_get_shapes(),
            points=_get_points(),
            table=_get_table("sample1"),
        )
    elif request.param == "empty":
        s = SpatialData()
    else:
        s = request.getfixturevalue(request.param)
    return s


def _get_images() -> dict[str, DataArray | DataTree]:
    out = {}
    dims_2d = ("c", "y", "x")
    dims_3d = ("z", "y", "x", "c")
    out["image2d"] = Image2DModel.parse(RNG.normal(size=(3, 64, 64)), dims=dims_2d, c_coords=["r", "g", "b"])
    out["image2d_multiscale"] = Image2DModel.parse(
        RNG.normal(size=(3, 64, 64)), scale_factors=[2, 2], dims=dims_2d, c_coords=["r", "g", "b"]
    )
    out["image2d_xarray"] = Image2DModel.parse(DataArray(RNG.normal(size=(3, 64, 64)), dims=dims_2d), dims=None)
    out["image2d_multiscale_xarray"] = Image2DModel.parse(
        DataArray(RNG.normal(size=(3, 64, 64)), dims=dims_2d),
        scale_factors=[2, 4],
        dims=None,
    )
    out["image3d_numpy"] = Image3DModel.parse(RNG.normal(size=(2, 64, 64, 3)), dims=dims_3d)
    out["image3d_multiscale_numpy"] = Image3DModel.parse(
        RNG.normal(size=(2, 64, 64, 3)), scale_factors=[2], dims=dims_3d
    )
    out["image3d_xarray"] = Image3DModel.parse(DataArray(RNG.normal(size=(2, 64, 64, 3)), dims=dims_3d), dims=None)
    out["image3d_multiscale_xarray"] = Image3DModel.parse(
        DataArray(RNG.normal(size=(2, 64, 64, 3)), dims=dims_3d),
        scale_factors=[2],
        dims=None,
    )
    return out


def _get_labels() -> dict[str, DataArray | DataTree]:
    out = {}
    dims_2d = ("y", "x")
    dims_3d = ("z", "y", "x")

    out["labels2d"] = Labels2DModel.parse(RNG.integers(0, 100, size=(64, 64)), dims=dims_2d)
    out["labels2d_multiscale"] = Labels2DModel.parse(
        RNG.integers(0, 100, size=(64, 64)), scale_factors=[2, 4], dims=dims_2d
    )
    out["labels2d_xarray"] = Labels2DModel.parse(
        DataArray(RNG.integers(0, 100, size=(64, 64)), dims=dims_2d), dims=None
    )
    out["labels2d_multiscale_xarray"] = Labels2DModel.parse(
        DataArray(RNG.integers(0, 100, size=(64, 64)), dims=dims_2d),
        scale_factors=[2, 4],
        dims=None,
    )
    out["labels3d_numpy"] = Labels3DModel.parse(RNG.integers(0, 100, size=(10, 64, 64)), dims=dims_3d)
    out["labels3d_multiscale_numpy"] = Labels3DModel.parse(
        RNG.integers(0, 100, size=(10, 64, 64)), scale_factors=[2, 4], dims=dims_3d
    )
    out["labels3d_xarray"] = Labels3DModel.parse(
        DataArray(RNG.integers(0, 100, size=(10, 64, 64)), dims=dims_3d), dims=None
    )
    out["labels3d_multiscale_xarray"] = Labels3DModel.parse(
        DataArray(RNG.integers(0, 100, size=(10, 64, 64)), dims=dims_3d),
        scale_factors=[2, 4],
        dims=None,
    )
    return out


def _get_polygons() -> dict[str, GeoDataFrame]:
    # TODO: add polygons from geojson and from ragged arrays since now only the GeoDataFrame initializer is tested.
    out = {}
    poly = GeoDataFrame(
        {
            "geometry": [
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
                Polygon(((0, 0), (0, -1), (-1, -1), (-1, 0))),
                Polygon(((0, 0), (0, 1), (1, 10))),
                Polygon(((0, 0), (0, 1), (1, 1))),
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0), (1, 0))),
            ]
        }
    )

    multipoly = GeoDataFrame(
        {
            "geometry": [
                MultiPolygon(
                    [
                        Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
                        Polygon(((0, 0), (0, -1), (-1, -1), (-1, 0))),
                    ]
                ),
                MultiPolygon(
                    [
                        Polygon(((0, 0), (0, 1), (1, 10))),
                        Polygon(((0, 0), (0, 1), (1, 1))),
                        Polygon(((0, 0), (0, 1), (1, 1), (1, 0), (1, 0))),
                    ]
                ),
            ]
        }
    )

    out["poly"] = ShapesModel.parse(poly, name="poly")
    out["multipoly"] = ShapesModel.parse(multipoly, name="multipoly")

    return out


def _get_shapes() -> dict[str, AnnData]:
    out = {}
    arr = RNG.normal(size=(100, 2))
    out["shapes_0"] = ShapesModel.parse(arr, shape_type="Square", shape_size=3)
    out["shapes_1"] = ShapesModel.parse(arr, shape_type="Circle", shape_size=np.repeat(1, len(arr)))

    return out


def _get_points() -> dict[str, pa.Table]:
    name = "points"
    var_names = [np.arange(3), ["genex", "geney"]]

    out = {}
    for i, v in enumerate(var_names):
        name = f"{name}_{i}"
        arr = RNG.normal(size=(100, 2))
        # randomly assign some values from v to the points
        points_assignment0 = pd.Series(RNG.choice(v, size=arr.shape[0]))
        points_assignment1 = pd.Series(RNG.choice(v, size=arr.shape[0]))
        annotations = pa.table(
            {"points_assignment0": points_assignment0, "points_assignment1": points_assignment1},
        )
        out[name] = PointsModel.parse(coords=arr, annotations=annotations)
    return out


def _get_table(
    region: AnnData | None = None,
    region_key: str | None = None,
    instance_key: str | None = None,
) -> AnnData:
    region_key = region_key or "annotated_region"
    instance_key = instance_key or "instance_id"
    adata = AnnData(RNG.normal(size=(100, 10)), obs=pd.DataFrame(RNG.normal(size=(100, 3)), columns=["a", "b", "c"]))
    adata.obs[instance_key] = np.arange(adata.n_obs)
    if isinstance(region, str):
        table = TableModel.parse(adata=adata, region=region, instance_key=instance_key)
    elif isinstance(region, list):
        adata.obs[region_key] = RNG.choice(region, size=adata.n_obs)
        adata.obs[instance_key] = RNG.integers(0, 10, size=(100,))
        table = TableModel.parse(adata=adata, region=region, region_key=region_key, instance_key=instance_key)
    else:
        table = TableModel.parse(adata=adata, region=region, region_key=region_key, instance_key=instance_key)

    return table


class PlotTesterMeta(ABCMeta):
    def __new__(cls, clsname, superclasses, attributedict):
        for key, value in attributedict.items():
            if callable(value):
                attributedict[key] = _decorate(value, clsname, name=key)
        return super().__new__(cls, clsname, superclasses, attributedict)


class PlotTester(ABC):  # noqa: B024
    @classmethod
    def compare(cls, basename: str, tolerance: float | None = None):
        ACTUAL.mkdir(parents=True, exist_ok=True)
        out_path = ACTUAL / f"{basename}.png"

        width, height = 400, 300  # fixed dimensions so runners don't change
        fig = plt.gcf()
        fig.set_size_inches(width / DPI, height / DPI)
        fig.set_dpi(DPI)

        # Apply constrained layout and save the plot
        fig.set_constrained_layout(True)
        plt.savefig(out_path, dpi=DPI)
        plt.close()

        if tolerance is None:
            # see https://github.com/scverse/squidpy/pull/302
            tolerance = 2 * TOL if "Napari" in basename else TOL

        res = compare_images(str(EXPECTED / f"{basename}.png"), str(out_path), tolerance)

        assert res is None, res


def _decorate(fn: Callable, clsname: str, name: str | None = None) -> Callable:
    @wraps(fn)
    def save_and_compare(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        self.compare(fig_name)

    if not callable(fn):
        raise TypeError(f"Expected a `callable` for class `{clsname}`, found `{type(fn).__name__}`.")

    name = fn.__name__ if name is None else name

    if not name.startswith("test_plot_") or not clsname.startswith("Test"):
        return fn

    fig_name = f"{clsname[4:]}_{name[10:]}"

    return save_and_compare


@pytest.fixture
def get_sdata_with_multiple_images(request) -> sd.SpatialData:
    """Yields a sdata object with multiple images which may or may not share a coordinate system."""

    def _get_sdata_with_multiple_images(share_coordinate_system: str = "all"):
        if share_coordinate_system == "all":
            images = {
                "data1": sd.models.Image2DModel.parse(np.zeros((1, 10, 10)), dims=("c", "y", "x")),
                "data2": sd.models.Image2DModel.parse(np.zeros((1, 10, 10)), dims=("c", "y", "x")),
                "data3": sd.models.Image2DModel.parse(np.zeros((1, 10, 10)), dims=("c", "y", "x")),
            }

        elif share_coordinate_system == "two":
            images = {
                "data1": sd.models.Image2DModel.parse(
                    np.zeros((1, 10, 10)),
                    dims=("c", "y", "x"),
                    transformations={"coord_sys1": sd.transformations.Identity()},
                ),
                "data2": sd.models.Image2DModel.parse(
                    np.zeros((1, 10, 10)),
                    dims=("c", "y", "x"),
                    transformations={"coord_sys2": sd.transformations.Identity()},
                ),
                "data3": sd.models.Image2DModel.parse(
                    np.zeros((1, 10, 10)),
                    dims=("c", "y", "x"),
                    transformations={"coord_sys1": sd.transformations.Identity()},
                ),
            }

        elif share_coordinate_system == "none":
            images = {
                "data1": sd.models.Image2DModel.parse(
                    np.zeros((1, 10, 10)),
                    dims=("c", "y", "x"),
                    transformations={"coord_sys1": sd.transformations.Identity()},
                ),
                "data2": sd.models.Image2DModel.parse(
                    np.zeros((1, 10, 10)),
                    dims=("c", "y", "x"),
                    transformations={"coord_sys2": sd.transformations.Identity()},
                ),
                "data3": sd.models.Image2DModel.parse(
                    np.zeros((1, 10, 10)),
                    dims=("c", "y", "x"),
                    transformations={"coord_sys3": sd.transformations.Identity()},
                ),
            }

        elif share_coordinate_system == "similar_name":
            images = {
                "data1": sd.models.Image2DModel.parse(
                    np.zeros((1, 10, 10)),
                    dims=("c", "y", "x"),
                    transformations={"coord_sys1": sd.transformations.Identity()},
                ),
                "data2": sd.models.Image2DModel.parse(
                    np.zeros((1, 10, 10)),
                    dims=("c", "y", "x"),
                    transformations={"coord_sys2": sd.transformations.Identity()},
                ),
                "data3": sd.models.Image2DModel.parse(
                    np.zeros((1, 10, 10)),
                    dims=("c", "y", "x"),
                    transformations={"coord_sys11": sd.transformations.Identity()},
                ),
            }

        else:
            raise ValueError("Invalid share_coordinate_system value.")

        sdata = sd.SpatialData(images=images)

        return sdata

    return _get_sdata_with_multiple_images
