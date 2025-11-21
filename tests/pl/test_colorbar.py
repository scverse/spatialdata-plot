import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel

import spatialdata_plot  # noqa: F401
from tests.conftest import DPI, PlotTester, PlotTesterMeta

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


def _make_colorbar_sdata() -> SpatialData:
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
        ]
    )
    annotations = pd.DataFrame(
        {
            "value": np.linspace(0, 1, len(coords)),
            "category": np.array(["a", "b", "a", "b", "a", "b"], dtype=object),
        }
    )
    points = PointsModel.parse(coords, annotation=annotations)

    image = Image2DModel.parse(
        np.linspace(0, 1, 25, dtype=float).reshape(1, 5, 5),
        dims=("c", "y", "x"),
        c_coords=["chan0"],
    )

    return SpatialData(images={"img": image}, points={"pts": points})


class TestColorbarControls(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_image_auto_colorbar_for_single_channel(self):
        sdata = _make_colorbar_sdata()
        sdata.pl.render_images(element="img").pl.show()

    def test_plot_colorbars_on_opposite_sides(self):
        sdata = _make_colorbar_sdata()
        (
            sdata.pl.render_images(element="img", colorbar=True)
            .pl.render_points(
                element="pts",
                color="value",
                size=30,
                colorbar=True,
                colorbar_params={"loc": "left", "width": 0.05},
            )
            .pl.show()
        )

    def test_plot_multiple_colorbars_same_side(self):
        sdata = _make_colorbar_sdata()
        (
            sdata.pl.render_images(element="img", colorbar=True, colorbar_params={"width": 0.03})
            .pl.render_points(
                element="pts",
                color="value",
                size=30,
                colorbar=True,
                colorbar_params={"width": 0.03, "pad": 0.02},
            )
            .pl.show()
        )

    def test_plot_categorical_color_skips_colorbar(self):
        sdata = _make_colorbar_sdata()
        sdata.pl.render_points(element="pts", color="category", size=40).pl.show()

    def test_plot_shapes_colorbar(self):
        sdata = _make_colorbar_sdata()
        sdata.pl.render_shapes(element="pts", color="value", colorbar=True).pl.show()

    def test_plot_labels_colorbar_all_sides(self):
        sdata = _make_colorbar_sdata()
        # reuse points for labels-style layout by making a simple label image
        # (PlotTester harness cares about image output, not semantics)
        # top, bottom, left, right
        sdata.pl.render_images(element="img").pl.render_labels(
            element="img",
            colorbar=True,
            colorbar_params={"loc": "top", "label": "topbar"},
        ).pl.render_labels(
            element="img",
            colorbar=True,
            colorbar_params={"loc": "bottom", "label": "botbar"},
        ).pl.render_labels(
            element="img",
            colorbar=True,
            colorbar_params={"loc": "left", "label": "leftbar"},
        ).pl.render_labels(
            element="img",
            colorbar=True,
            colorbar_params={"loc": "right", "label": "rightbar"},
        ).pl.show()

    def test_plot_labels_two_colorbars_each_side(self):
        sdata = _make_colorbar_sdata()
        (
            sdata.pl.render_labels(
                element="img",
                colorbar=True,
                colorbar_params={"loc": "top", "pad": 0.02, "label": "top1"},
            )
            .pl.render_labels(
                element="img",
                colorbar=True,
                colorbar_params={"loc": "top", "pad": 0.05, "label": "top2"},
            )
            .pl.render_labels(
                element="img",
                colorbar=True,
                colorbar_params={"loc": "right", "pad": 0.02, "label": "right1"},
            )
            .pl.render_labels(
                element="img",
                colorbar=True,
                colorbar_params={"loc": "right", "pad": 0.05, "label": "right2"},
            )
            .pl.render_labels(
                element="img",
                colorbar=True,
                colorbar_params={"loc": "left", "pad": 0.02, "label": "left1"},
            )
            .pl.render_labels(
                element="img",
                colorbar=True,
                colorbar_params={"loc": "left", "pad": 0.05, "label": "left2"},
            )
            .pl.render_labels(
                element="img",
                colorbar=True,
                colorbar_params={"loc": "bottom", "pad": 0.02, "label": "bottom1"},
            )
            .pl.render_labels(
                element="img",
                colorbar=True,
                colorbar_params={"loc": "bottom", "pad": 0.05, "label": "bottom2"},
            )
            .pl.show()
        )

    def test_plot_colorbar_size_pad_label(self):
        sdata = _make_colorbar_sdata()
        (
            sdata.pl.render_points(
                element="pts",
                color="value",
                colorbar=True,
                colorbar_params={"width": 0.12, "pad": 0.08, "label": "intensity"},
            ).pl.show()
        )
