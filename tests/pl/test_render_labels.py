import dask.array as da
import matplotlib
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import spatialdata_plot  # noqa: F401
from anndata import AnnData
from matplotlib.colors import LogNorm, Normalize
from spatial_image import to_spatial_image
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, TableModel

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


class TestLabels(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_labels(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels(elements="blobs_labels").pl.show()

    def test_plot_can_render_multiscale_labels(self, sdata_blobs: SpatialData):
        sdata_blobs.table.obs["region"] = "blobs_multiscale_labels"
        sdata_blobs.table.uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        sdata_blobs.pl.render_labels("blobs_multiscale_labels").pl.show()

    def test_plot_can_render_given_scale_of_multiscale_labels(self, sdata_blobs: SpatialData):
        sdata_blobs.table.obs["region"] = "blobs_multiscale_labels"
        sdata_blobs.table.uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
        sdata_blobs.pl.render_labels("blobs_multiscale_labels", scale="scale1").pl.show()

    def test_plot_can_do_rasterization(self, sdata_blobs: SpatialData):
        temp = sdata_blobs["blobs_labels"].data.copy()
        temp = da.concatenate([temp] * 6, axis=0)
        temp = da.concatenate([temp] * 6, axis=1)
        img = to_spatial_image(temp, dims=("y", "x"))
        img.attrs["transform"] = sdata_blobs["blobs_labels"].transform
        sdata_blobs["blobs_giant_labels"] = img

        sdata_blobs.table.obs["region"] = "blobs_giant_labels"
        sdata_blobs.table.uns["spatialdata_attrs"]["region"] = "blobs_giant_labels"

        sdata_blobs.pl.render_labels("blobs_giant_labels").pl.show()

    def test_plot_can_stop_rasterization_with_scale_full(self, sdata_blobs: SpatialData):
        temp = sdata_blobs["blobs_labels"].data.copy()
        temp = da.concatenate([temp] * 6, axis=0)
        temp = da.concatenate([temp] * 6, axis=1)
        img = to_spatial_image(temp, dims=("y", "x"))
        img.attrs["transform"] = sdata_blobs["blobs_labels"].transform
        sdata_blobs["blobs_giant_labels"] = img

        sdata_blobs.table.obs["region"] = "blobs_giant_labels"
        sdata_blobs.table.uns["spatialdata_attrs"]["region"] = "blobs_giant_labels"

        sdata_blobs.pl.render_labels("blobs_giant_labels", scale="full").pl.show()

    def test_plot_can_stack_render_labels(self, sdata_blobs: SpatialData):
        (
            sdata_blobs.pl.render_labels(
                elements="blobs_labels", na_color="red", fill_alpha=1, outline_alpha=0, outline=False
            )
            .pl.render_labels(
                elements="blobs_labels", na_color="blue", fill_alpha=0, outline_alpha=1, outline=True, contour_px=10
            )
            .pl.show()
        )

    def test_plot_can_render_no_fill(self, sdata_blobs: SpatialData):
        (
            sdata_blobs.pl.render_labels(
                elements="blobs_labels", fill_alpha=0, outline_alpha=1, outline=True, contour_px=10
            ).pl.show()
        )

    def test_can_render_no_fill_no_outline(self, sdata_blobs: SpatialData):
        # This passes only with outline_alpha=0
        (
            sdata_blobs.pl.render_labels(
                elements="blobs_labels",
                fill_alpha=0,
                outline=False,
            ).pl.show()
        )
        self.compare("Labels_can_render_no_fill_no_outline", tolerance=5)

    def test_plot_can_color_labels_by_continuous_variable(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color="channel_0_sum").pl.show()


@pytest.fixture
def sdata():
    box = np.zeros((10, 10), dtype=int)
    box[2:-2, 2:-2] = 1

    labels_dct = {
        "empty_labels": Labels2DModel.parse(np.zeros((1, 3)), dims=("y", "x")),
        "one_label": Labels2DModel.parse(np.array([[0, 0, 1]]), dims=("y", "x")),
        "labels1": Labels2DModel.parse(np.arange(3).reshape((1, 3)), dims=("y", "x")),
        "skipped_label": Labels2DModel.parse(np.array([[0, 1, 3]]), dims=("y", "x")),
        "no_background": Labels2DModel.parse(np.arange(1, 3 + 1).reshape((1, 3)), dims=("y", "x")),
        "box": Labels2DModel.parse(box, dims=("y", "x")),
    }
    obs = pd.DataFrame(
        [
            ["one_label", 1, np.pi, True, "a", "cat"],
            ["two_labels", 1, 0.1, True, "a", "cat"],
            ["two_labels", 2, 0.9, False, "b", "dog"],
            ["skipped_label", 1, 1.0, True, "a", "rat"],
            ["skipped_label", 3, np.nan, False, "b", "rat"],
            ["no_background", 1, 1.0, True, "a", "dog"],
            ["no_background", 2, 10.0, False, "b", "rat"],
            ["no_background", 3, 100.0, True, "b", "cat"],
            ["box", 1, np.pi, True, "a", "cat"],
        ],
        columns=[
            "region",
            "instance_id",
            "continuous",
            "boolean",
            "string",
            "categorical",
        ],
    )
    obs["categorical"] = obs["categorical"].astype("category")
    return SpatialData(
        labels=labels_dct,
        table=TableModel.parse(
            AnnData(obs=obs),
            region=obs["region"].unique().tolist(),
            region_key="region",
            instance_key="instance_id",
        ),
    )


@pytest.mark.parametrize(
    "elements",
    [
        ["empty_labels"],  # exception
        ["one_label"],  # correct
        ["two_labels"],  # incorrect, empty plot
        ["skipped_label"],  # correct (but different labels plotted with same color)
        ["no_background"],  # internal exception, incorrect plot (label values 2+3 same color)
    ],
)
def test_can_handle_different_labels_images(sdata: SpatialData, elements: list[str]):
    sdata.pl.render_labels(elements=elements).pl.show()
    # TODO: Assertions or comparison with expected image


@pytest.mark.parametrize(
    ("elements", "options"),
    [
        # Render nothing
        ([], dict()),  # correct
        # Render with defaults
        (["two_labels"], dict()),  # empty plot
        # Render labels with color by continuous value, defaults
        (["two_labels"], dict(color="continuous")),  # empty plot
        # Render labels with color by continuous value and NaN
        (["skipped_label"], dict(color="continuous", na_color="magenta")),  # exception: Not all values are color-like.
        (
            ["skipped_label"],
            dict(color="continuous", na_color=(1.0, 0.0, 1.0, 1.0)),
        ),  # exception: Not all values are color-like.
        # Render labels with color by continuous value, fill alpha
        (["two_labels"], dict(color="continuous", fill_alpha=0.0)),  # empty plot
        (["two_labels"], dict(color="continuous", fill_alpha=1.0)),  # empty plot
        # Render labels with color map, defaults
        (["two_labels"], dict(color="continuous", cmap="viridis")),  # empty plot
        (["two_labels"], dict(color="continuous", cmap=matplotlib.colormaps["plasma"])),  # empty plot
        # Render labels with color map, with norm
        (
            ["two_labels"],
            dict(color="continuous", cmap="viridis", norm=Normalize(vmin=0.1, vmax=0.9)),
        ),
        (["two_labels"], dict(color="continuous", cmap="viridis", vmin=0.1, vmax=0.9)),
        (
            ["no_background"],
            dict(color="continuous", cmap="viridis", norm=LogNorm(vmin=1, vmax=100)),
        ),
        # Render labels with color by non-continuous values, defaults
        (["two_labels"], dict(color="boolean")),
        (["two_labels"], dict(color="string")),
        (["two_labels"], dict(color="categorical")),
        # Render labels with color by non-continuous values, defaults
        (["two_labels"], dict(color="boolean")),
        (["two_labels"], dict(color="string")),
        (["two_labels"], dict(color="categorical")),
        # Render only groups from a categorical
        (["no_background"], dict(color="categorical", groups=["cat", "dog"])),
        # Render categorical as continuous with colormap
        (["no_background"], dict(color="categorical", cmap="viridis")),
        # Render categorical with palette, one value
        (["one_label"], dict(color="boolean", palette="red")),
        (["one_label"], dict(color="string", palette="red")),
        # Render categorical with palette, multiple values
        (["two_labels"], dict(color="boolean", palette="red")),
        (["two_labels"], dict(color="boolean", palette=["red", "lime"])),
        (["two_labels"], dict(color="string", palette=["red", "lime"])),
        (["two_labels"], dict(color="categorical", palette=["red", "lime"])),
        # Render categorical with palette, one category
        (["one_label"], dict(color="categorical", palette="red")),
        (["skipped_label"], dict(color="categorical", palette="red")),
        # Render categorical with palette, multiple categories
        (["two_labels"], dict(color="categorical", palette="red")),
        (["no_background"], dict(color="categorical", palette="red")),
    ],
)
def test_can_render_labels_fill_with_options(sdata: SpatialData, elements: list[str], options: dict):
    sdata.pl.render_labels(elements=elements, **options).pl.show()
    # TODO: Assertions or comparison with expected image


@pytest.mark.parametrize(
    ("elements", "options"),
    [
        # Render outline with defaults
        (["box"], dict(outline=True)),  # correct
        # Render outline with no fill
        (["box"], dict(outline=True, fill_alpha=0.0)),  # correct
        # Render outline with different width
        pytest.param(["box"], dict(outline=True, contour_px=0), marks=pytest.mark.xfail(reason="not supported")),
        (["box"], dict(outline=True, contour_px=1)),
        (
            ["box"],
            dict(outline=True, contour_px=5),
        ),  # incorrect, black outline 1, additonal magenta outline (apparently it depends on labels image size)
        # Render outline with alpha
        (["box"], dict(outline=True, outline_alpha=0.0)),  # incorrect, outline shown with ~0.5 alpha
        (["box"], dict(outline=True, outline_alpha=0.5)),  # incorrect, outline shown with ~1.0 alpha
        (["box"], dict(outline=True, outline_alpha=1.0)),  # correct
    ],
)
def test_can_render_labels_outline_with_options(sdata: SpatialData, elements: list[str], options: dict):
    sdata.pl.render_labels(elements=elements, **options).pl.show(dpi=200)
    # TODO: Assertions or comparison with expected image
