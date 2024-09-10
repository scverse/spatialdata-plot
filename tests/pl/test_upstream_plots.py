import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from spatialdata import SpatialData
from spatialdata.transformations import (
    Affine,
    MapAxis,
    Scale,
    Sequence,
    Translation,
    set_transformation,
)

import spatialdata_plot  # noqa: F401
from tests.conftest import DPI, PlotTester, PlotTesterMeta

RNG = np.random.default_rng(seed=42)
# sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=DPI, color_map="viridis")
matplotlib.use("agg")  # same as GitHub action runner
_ = spatialdata_plot

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be changed, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestNotebooksTransformations(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_transformations_raccoon_split(self, sdata_raccoon: SpatialData):
        _, axs = plt.subplots(ncols=3, figsize=(12, 3))

        sdata_raccoon.pl.render_images().pl.show(ax=axs[0])
        sdata_raccoon.pl.render_labels().pl.show(ax=axs[1])
        sdata_raccoon.pl.render_shapes().pl.show(ax=axs[2])

    def test_plot_can_render_transformations_raccoon_overlay(self, sdata_raccoon: SpatialData):
        sdata_raccoon.pl.render_images().pl.render_labels().pl.render_shapes().pl.show()

    def test_plot_can_render_transformations_raccoon_scale(self, sdata_raccoon: SpatialData):
        scale = Scale([2.0], axes=("x",))
        set_transformation(sdata_raccoon.images["raccoon"], scale, to_coordinate_system="global")

        sdata_raccoon.pl.render_images().pl.render_labels().pl.render_shapes().pl.show()

    def test_plot_can_render_transformations_raccoon_mapaxis(self, sdata_raccoon: SpatialData):
        map_axis = MapAxis({"x": "y", "y": "x"})
        set_transformation(sdata_raccoon.images["raccoon"], map_axis, to_coordinate_system="global")

        sdata_raccoon.pl.render_images().pl.render_labels().pl.render_shapes().pl.show()

    def test_plot_can_render_transformations_raccoon_rotation(self, sdata_raccoon: SpatialData):
        theta = math.pi / 6
        rotation = Affine(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ],
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )

        set_transformation(sdata_raccoon.images["raccoon"], rotation, to_coordinate_system="global")

        sdata_raccoon.pl.render_images().pl.render_labels().pl.render_shapes().pl.show()

    def test_plot_can_render_transformations_raccoon_translation(self, sdata_raccoon: SpatialData):
        translation = Translation([500, 300], axes=("x", "y"))
        set_transformation(sdata_raccoon.images["raccoon"], translation, to_coordinate_system="global")

        sdata_raccoon.pl.render_images().pl.render_labels().pl.render_shapes().pl.show()

    def test_plot_can_render_transformations_raccoon_affine(self, sdata_raccoon: SpatialData):
        theta = math.pi / 6
        rotation = Affine(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ],
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
        scale = Scale([2.0], axes=("x",))
        sequence = Sequence([rotation, scale])

        set_transformation(sdata_raccoon.images["raccoon"], sequence, to_coordinate_system="global")

        sdata_raccoon.pl.render_images().pl.render_labels().pl.render_shapes().pl.show()

    def test_plot_can_render_transformations_raccoon_composition(self, sdata_raccoon: SpatialData):
        theta = math.pi / 6
        rotation = Affine(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ],
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
        scale = Scale([2.0], axes=("x",))

        set_transformation(sdata_raccoon.images["raccoon"], scale, to_coordinate_system="global")
        set_transformation(sdata_raccoon.shapes["circles"], scale, to_coordinate_system="global")
        set_transformation(sdata_raccoon.labels["segmentation"], rotation, to_coordinate_system="global")

        sdata_raccoon.pl.render_images().pl.render_labels().pl.render_shapes().pl.show()

    def test_plot_can_render_transformations_raccoon_inverse(self, sdata_raccoon: SpatialData):
        theta = math.pi / 6
        rotation = Affine(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ],
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
        scale = Scale([2.0], axes=("x",))
        sequence = Sequence([rotation, rotation.inverse(), scale, scale.inverse()])
        set_transformation(sdata_raccoon.images["raccoon"], sequence, to_coordinate_system="global")

        sdata_raccoon.pl.render_images().pl.render_labels().pl.render_shapes().pl.show()


def test_plot_can_render_blobs_images(sdata_blobs: SpatialData):
    sdata_blobs.pl.render_images().pl.show()


def test_plot_can_render_blobs_points(sdata_blobs: SpatialData):
    sdata_blobs.pl.render_points().pl.show()


def test_plot_can_render_blobs_labels(sdata_blobs: SpatialData):
    sdata_blobs.pl.render_labels().pl.show()


def test_plot_can_render_blobs_shapes(sdata_blobs: SpatialData):
    sdata_blobs.pl.render_shapes().pl.show()
