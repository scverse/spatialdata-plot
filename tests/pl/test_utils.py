import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import xarray as xr
from anndata import AnnData
from shapely.geometry import Point
from spatialdata import SpatialData, get_centroids
from spatialdata.models import Labels2DModel, PointsModel, ShapesModel, TableModel

import spatialdata_plot
from spatialdata_plot.pl import measure_obs
from spatialdata_plot.pl._datashader import (
    _apply_cmap_alpha_to_datashader_result,
    _datashader_map_aggregate_to_color,
)
from spatialdata_plot.pl.render_params import CmapParams, Color, ColorLike, colormap_with_alpha
from spatialdata_plot.pl._color import _set_outline
from spatialdata_plot.pl.utils import set_zero_in_cmap_to_transparent
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


class TestUtils(PlotTester, metaclass=PlotTesterMeta):
    @pytest.mark.parametrize(
        "outline_color",
        [
            (0.0, 1.0, 0.0, 1.0),
            "#00ff00",
        ],
    )
    def test_plot_set_outline_accepts_str_or_float_or_list_thereof(self, sdata_blobs: SpatialData, outline_color):
        sdata_blobs.pl.render_shapes(element="blobs_polygons", outline_alpha=1, outline_color=outline_color).pl.show()

    @pytest.mark.parametrize(
        "colname",
        ["0", "0.5", "1"],
    )
    def test_plot_colnames_that_are_valid_matplotlib_greyscale_colors_are_not_evaluated_as_colors(
        self, sdata_blobs: SpatialData, colname: str
    ):
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"][colname] = [1, 2, 3, 5, 20]
        sdata_blobs.pl.render_shapes("blobs_polygons", color=colname).pl.show()

    def test_plot_can_set_zero_in_cmap_to_transparent(self, sdata_blobs: SpatialData):
        # set up figure and modify the data to add 0s
        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")
        sdata_blobs.tables["table"].obs["my_var"] = list(range(len(sdata_blobs.tables["table"].obs)))
        sdata_blobs.tables["table"].obs["my_var"] += 2  # shift the values to not have 0s

        new_cmap = set_zero_in_cmap_to_transparent(cmap="viridis")

        # baseline img
        sdata_blobs.pl.render_labels("blobs_labels", color="my_var", cmap="viridis", table_name="table").pl.show(
            ax=axs[0], colorbar=False
        )

        sdata_blobs.tables["table"].obs.iloc[8:12, 2] = 0

        # image with 0s as transparent, so some labels are "missing"
        sdata_blobs.pl.render_labels("blobs_labels", color="my_var", cmap=new_cmap, table_name="table").pl.show(
            ax=axs[1], colorbar=False
        )

    def _render_transparent_cmap_shapes(self, sdata_blobs: SpatialData, method: str):
        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")
        new_cmap = set_zero_in_cmap_to_transparent(cmap="viridis")
        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"]["value"] = [0.0, 2.0, 3.0, 4.0, 5.0]

        # left: baseline with standard viridis
        sdata_blobs.pl.render_images("blobs_image").pl.render_shapes(
            "blobs_polygons", color="value", cmap="viridis", method=method
        ).pl.show(ax=axs[0], colorbar=False)

        # right: transparent cmap — shape with value=0 should reveal the image
        sdata_blobs.pl.render_images("blobs_image").pl.render_shapes(
            "blobs_polygons", color="value", cmap=new_cmap, method=method
        ).pl.show(ax=axs[1], colorbar=False)

    def test_plot_transparent_cmap_shapes_matplotlib(self, sdata_blobs: SpatialData):
        self._render_transparent_cmap_shapes(sdata_blobs, method="matplotlib")

    def test_plot_transparent_cmap_shapes_datashader(self, sdata_blobs: SpatialData):
        self._render_transparent_cmap_shapes(sdata_blobs, method="datashader")

    def test_plot_transparent_cmap_shapes_clip_false(self, sdata_blobs: SpatialData):
        """Transparent cmap with clip=False norm (3-part shading path)."""
        from matplotlib.colors import Normalize

        _, axs = plt.subplots(nrows=1, ncols=2, layout="tight")
        new_cmap = set_zero_in_cmap_to_transparent(cmap="viridis")
        norm = Normalize(vmin=0, vmax=5, clip=False)

        sdata_blobs["table"].obs["region"] = pd.Categorical(["blobs_polygons"] * sdata_blobs["table"].n_obs)
        sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = "blobs_polygons"
        sdata_blobs.shapes["blobs_polygons"]["value"] = [0.0, 2.0, 3.0, 4.0, 5.0]

        sdata_blobs.pl.render_images("blobs_image").pl.render_shapes(
            "blobs_polygons", color="value", cmap="viridis", norm=norm, method="datashader"
        ).pl.show(ax=axs[0], colorbar=False)

        sdata_blobs.pl.render_images("blobs_image").pl.render_shapes(
            "blobs_polygons", color="value", cmap=new_cmap, norm=norm, method="datashader"
        ).pl.show(ax=axs[1], colorbar=False)


@pytest.mark.parametrize(
    "color_result",
    [
        # greyscale strings rejected
        ("0", False),
        ("0.5", False),
        ("1", False),
        # valid full-form colors accepted
        ("#00ff00", True),
        ("#00ff00aa", True),
        ((0.0, 1.0, 0.0, 1.0), True),
        ("red", True),
        ("blue", True),
        # short hex rejected
        ("#f00", False),
        ("#f00a", False),
        # single-letter shortcuts rejected (#211)
        ("b", False),
        ("g", False),
        ("r", False),
        ("c", False),
        ("m", False),
        ("y", False),
        ("k", False),
        ("w", False),
        # CN cycle notation rejected (#211)
        ("C0", False),
        ("C1", False),
        ("C10", False),
        # tab: prefixed rejected (#211)
        ("tab:blue", False),
        ("tab:orange", False),
        # xkcd: prefixed rejected (#211)
        ("xkcd:sky blue", False),
        ("xkcd:red", False),
    ],
)
def test_is_color_like(color_result: tuple[ColorLike, bool]):
    color, result = color_result

    assert spatialdata_plot.pl._color._is_color_like(color) == result


@pytest.mark.parametrize(
    ("outline_alpha", "outline_color", "expected"),
    [
        (0.0, Color("#ff0000"), (0.0, 0.0)),
        (0, Color("#ff0000"), (0.0, 0.0)),
        ((0.0, 0.0), Color("#ff0000"), (0.0, 0.0)),
        (0.5, Color("#ff0000"), (0.5, 0.0)),
        (1.0, Color("#ff0000"), (1.0, 0.0)),
    ],
)
def test_set_outline_respects_zero_alpha(outline_alpha, outline_color, expected):
    """outline_alpha=0 must yield (0.0, 0.0) even when outline_color is set (#617 follow-up)."""
    alpha, _ = _set_outline(outline_alpha=outline_alpha, outline_width=None, outline_color=outline_color)
    assert alpha == expected


class TestCmapAlphaDatashader:
    """Regression tests for #376: set_zero_in_cmap_to_transparent with datashader."""

    def test_transparent_pixels_get_alpha_zero(self):
        """Post-processing sets alpha=0 for pixels mapping to transparent cmap entries."""
        import datashader as ds

        cmap = set_zero_in_cmap_to_transparent("viridis")
        data = np.array([[0.0, 5.0, 10.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])

        shaded = ds.tf.shade(agg, cmap=cmap, min_alpha=254, how="linear")
        result = _apply_cmap_alpha_to_datashader_result(shaded, agg, cmap, span=[0.0, 10.0])
        rgba = result.to_numpy().base if hasattr(result, "to_numpy") else result

        assert rgba[0, 0, 3] == 0, f"Expected alpha=0 at value=0.0, got {rgba[0, 0, 3]}"
        assert rgba[0, 1, 3] > 0, "Expected non-zero alpha at value=5.0"
        assert rgba[0, 2, 3] > 0, "Expected non-zero alpha at value=10.0"

    def test_opaque_cmap_unchanged(self):
        """Post-processing is a no-op for fully opaque cmaps."""
        import datashader as ds

        cmap = plt.get_cmap("viridis")
        data = np.array([[0.0, 5.0, 10.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])

        shaded = ds.tf.shade(agg, cmap=cmap, min_alpha=254, how="linear")
        rgba_before = shaded.to_numpy().base.copy()
        result = _apply_cmap_alpha_to_datashader_result(shaded, agg, cmap, span=[0.0, 10.0])
        rgba_after = result.to_numpy().base if hasattr(result, "to_numpy") else result
        np.testing.assert_array_equal(rgba_before, rgba_after)

    def test_string_cmap_passthrough(self):
        """Post-processing is a no-op for string cmaps (early return)."""
        dummy_rgba = np.zeros((2, 3, 4), dtype=np.uint8)
        dummy_rgba[:, :, 3] = 200
        data = np.array([[0.0, 5.0, 10.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])

        result = _apply_cmap_alpha_to_datashader_result(dummy_rgba, agg, "viridis", span=[0.0, 10.0])
        np.testing.assert_array_equal(result, dummy_rgba)

    def test_end_to_end_datashader_map(self):
        """_datashader_map_aggregate_to_color produces alpha=0 for transparent cmap entries."""
        cmap = set_zero_in_cmap_to_transparent("viridis")
        data = np.array([[0.0, 5.0, 10.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])

        result = _datashader_map_aggregate_to_color(agg, cmap=cmap, min_alpha=254, span=[0.0, 10.0])
        img = result.to_numpy().base if hasattr(result, "to_numpy") else result

        assert img[0, 0, 3] == 0, f"Expected alpha=0 at value=0.0, got {img[0, 0, 3]}"
        assert img[0, 1, 3] > 0, "Expected non-zero alpha at value=5.0"

    def test_span_none_preserves_colors(self):
        """With span=None, non-transparent shapes keep their correct colors."""
        cmap = set_zero_in_cmap_to_transparent("viridis")
        data = np.array([[0.0, 5.0, 10.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])

        result = _datashader_map_aggregate_to_color(agg, cmap=cmap, min_alpha=254)
        img = result.to_numpy().base if hasattr(result, "to_numpy") else result

        # value=0 should be transparent
        assert img[0, 0, 3] == 0
        # value=5 and value=10 should be opaque with correct viridis colors (not white)
        assert img[0, 1, 3] > 0
        assert img[0, 2, 3] > 0
        # The non-transparent pixels should NOT be white (R=255,G=255,B=255)
        assert not (img[0, 1, 0] == 255 and img[0, 1, 1] == 255 and img[0, 1, 2] == 255)


def test_extract_scalar_value():
    """Test the new _extract_scalar_value function for robust numeric conversion."""

    from spatialdata_plot.pl.utils import _extract_scalar_value

    # Test basic functionality
    assert _extract_scalar_value(3.14) == 3.14
    assert _extract_scalar_value(42) == 42.0

    # Test with collections
    assert _extract_scalar_value(pd.Series([1.0, 2.0, 3.0])) == 1.0
    assert _extract_scalar_value([1.0, 2.0, 3.0]) == 1.0

    # Test edge cases
    assert _extract_scalar_value(np.nan) == 0.0
    assert _extract_scalar_value("invalid") == 0.0
    assert _extract_scalar_value([], default=1.0) == 1.0


def test_plot_can_handle_rgba_color_specifications(sdata_blobs: SpatialData):
    """Test handling of RGBA color specifications."""
    # Test with RGBA tuple
    sdata_blobs.pl.render_shapes(element="blobs_circles", color=(1.0, 0.0, 0.0, 0.8)).pl.show()

    # Test with RGB tuple (no alpha)
    sdata_blobs.pl.render_shapes(element="blobs_circles", color=(0.0, 1.0, 0.0)).pl.show()

    # Test with string color
    sdata_blobs.pl.render_shapes(element="blobs_circles", color="blue").pl.show()


class TestMultiscaleToSpatialImage:
    """Regression tests for #589: multiscale resolution selection."""

    @staticmethod
    def _make_multiscale(shape, scale_factors):
        from spatialdata.models import Image2DModel

        rng = np.random.default_rng(42)
        return Image2DModel.parse(
            rng.normal(size=shape),
            scale_factors=scale_factors,
            dims=("c", "y", "x"),
            c_coords=["r", "g", "b"],
        )

    def test_larger_figure_never_picks_lower_resolution(self):
        """Increasing figure size must select equal or higher resolution."""
        from spatialdata_plot.pl.utils import _multiscale_to_spatial_image

        multiscale = self._make_multiscale((3, 1024, 1024), [2, 2])
        dpi = 100.0
        prev_x = 0
        for size in [3, 4, 5, 6, 7, 8, 10, 12]:
            result = _multiscale_to_spatial_image(multiscale, dpi, float(size), float(size))
            cur_x = result.sizes["x"]
            assert cur_x >= prev_x, (
                f"figsize {size} selected x={cur_x} which is lower than x={prev_x} from a smaller figure"
            )
            prev_x = cur_x

    def test_asymmetric_image_picks_sufficient_resolution(self):
        """When image aspect ratio differs from figure, both axes must be covered."""
        from spatialdata_plot.pl.utils import _multiscale_to_spatial_image

        multiscale = self._make_multiscale((3, 400, 1200), [2, 2])
        scales_info = {
            leaf.name: (multiscale[leaf.name].dims["x"], multiscale[leaf.name].dims["y"]) for leaf in multiscale.leaves
        }
        max_x = max(x for x, _ in scales_info.values())
        max_y = max(y for _, y in scales_info.values())

        dpi = 100.0
        for w, h in [(5, 5), (3, 10), (10, 3), (7, 4)]:
            result = _multiscale_to_spatial_image(multiscale, dpi, float(w), float(h))
            sel_x, sel_y = result.sizes["x"], result.sizes["y"]
            opt_x, opt_y = w * dpi, h * dpi
            assert sel_x >= opt_x or sel_x == max_x, (
                f"figsize {w}x{h}: x={sel_x} < optimal {opt_x} and not the maximum available"
            )
            assert sel_y >= opt_y or sel_y == max_y, (
                f"figsize {w}x{h}: y={sel_y} < optimal {opt_y} and not the maximum available"
            )

    def test_all_scales_too_small_picks_highest_resolution(self):
        """When no scale is large enough, the highest resolution is selected."""
        from spatialdata_plot.pl.utils import _multiscale_to_spatial_image

        multiscale = self._make_multiscale((3, 64, 64), [2, 2])
        result = _multiscale_to_spatial_image(multiscale, dpi=100.0, width=20.0, height=20.0)
        assert result.sizes["x"] == 64

    def test_single_scale_level(self):
        """A single-level multiscale image always returns that level."""
        from spatialdata_plot.pl.utils import _multiscale_to_spatial_image

        multiscale = self._make_multiscale((3, 512, 512), [2])
        for size in [2, 5, 10]:
            result = _multiscale_to_spatial_image(multiscale, dpi=100.0, width=float(size), height=float(size))
            assert result.sizes["x"] in (512, 256)

    def test_exact_match_selects_that_scale(self):
        """When optimal pixels exactly match a scale's dimensions, that scale is selected."""
        from spatialdata_plot.pl.utils import _multiscale_to_spatial_image

        multiscale = self._make_multiscale((3, 500, 500), [2, 2])
        result = _multiscale_to_spatial_image(multiscale, dpi=100.0, width=2.5, height=2.5)
        assert result.sizes["x"] == 250


def test_color_column_collision_on_element_columns_raises():
    # regression test for #619, element-column path: points with an "orange" column + color="orange".
    points = PointsModel.parse(pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "orange": [0.1, 0.2, 0.3]}))
    sdata = SpatialData(points={"pts": points})

    with pytest.raises(ValueError, match=r"color='orange'.*ambiguous.*element 'pts'"):
        sdata.pl.render_points("pts", color="orange")

    sdata.pl.render_points("pts", color="#ffa500")
    sdata.pl.render_points("pts", color=(1.0, 0.65, 0.0))


def test_color_column_collision_on_annotating_table_raises():
    # regression test for #619, table path: element has no "orange" column but its annotating table does.
    shapes = ShapesModel.parse(gpd.GeoDataFrame({"geometry": [Point(i, 0) for i in range(3)], "radius": [0.5] * 3}))
    obs = pd.DataFrame(
        {
            "region": pd.Categorical(["s"] * 3),
            "instance_id": list(range(3)),
            "orange": pd.Categorical(["A", "B", "A"]),
        }
    )
    table = TableModel.parse(
        AnnData(X=np.zeros((3, 1)), obs=obs),
        region="s",
        region_key="region",
        instance_key="instance_id",
    )
    sdata = SpatialData(shapes={"s": shapes}, tables={"t": table})

    with pytest.raises(ValueError, match=r"color='orange'.*ambiguous.*table 't'"):
        sdata.pl.render_shapes("s", color="orange")

    sdata.pl.render_shapes("s", color="#ffa500")


def test_color_key_obs_var_shadow_raises():
    # regression test for #621
    pts = PointsModel.parse(pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]}))
    obs = pd.DataFrame({"instance_id": [0, 1], "region": ["pts"] * 2, "GeneA": [0.9, 0.6]}, index=["0", "1"])
    table = TableModel.parse(
        AnnData(X=np.zeros((2, 1)), obs=obs, var=pd.DataFrame(index=["GeneA"])),
        region=["pts"],
        region_key="region",
        instance_key="instance_id",
    )
    sdata = SpatialData(points={"pts": pts}, tables={"t": table})

    with pytest.raises(ValueError, match=r"'GeneA'.*ambiguous.*obs\.columns.*var_names"):
        sdata.pl.render_points("pts", color="GeneA", table_name="t").pl.show()


def test_explicit_table_name_honored_when_element_has_same_column():
    # regression test for #620: explicit table_name= must not be silently
    # discarded when the element has a same-named column with different values.
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "geometry": [Point(5, 5), Point(15, 5)],
                "radius": [2.0, 2.0],
                "cat": pd.Categorical(["X", "Y"]),
            }
        )
    )
    obs = pd.DataFrame(
        {
            "instance_id": [0, 1],
            "region": pd.Categorical(["s1", "s1"]),
            "cat": pd.Categorical(["A", "B"]),
        }
    )
    table = TableModel.parse(
        AnnData(X=np.zeros((2, 1)), obs=obs),
        region=["s1"],
        region_key="region",
        instance_key="instance_id",
    )
    sdata = SpatialData(shapes={"s1": shapes}, tables={"t": table})

    fig, ax = plt.subplots()
    sdata.pl.render_shapes("s1", color="cat", table_name="t").pl.show(ax=ax)
    assert sorted(t.get_text() for t in ax.get_legend().get_texts()) == ["A", "B"]
    plt.close(fig)

    fig, ax = plt.subplots()
    sdata.pl.render_shapes("s1", color="cat").pl.show(ax=ax)
    assert sorted(t.get_text() for t in ax.get_legend().get_texts()) == ["X", "Y"]
    plt.close(fig)


def test_rasterize_target_unit_to_pixels_uses_world_extent(monkeypatch):
    # regression test for #668: _rasterize_if_necessary must compute
    # target_unit_to_pixels in world units (target_px / world_unit), not in
    # intrinsic pixels (target_px / source_px). For Scale=0.5 the two differ
    # by a factor of 2.
    from spatialdata.models import Image2DModel
    from spatialdata.transformations import Scale, Sequence, Translation, set_transformation

    from spatialdata_plot.pl import utils as plut

    arr = np.zeros((3, 8000, 8000), dtype=np.uint8)  # large enough to trigger rasterization
    img = Image2DModel.parse(arr, dims=("c", "y", "x"))
    set_transformation(
        img,
        Sequence([Scale([0.5, 0.5], axes=("x", "y")), Translation([100.0, 200.0], axes=("x", "y"))]),
        to_coordinate_system="global",
    )

    captured: dict[str, float] = {}

    def fake_rasterize(*args, **kwargs):
        captured["target_unit_to_pixels"] = kwargs["target_unit_to_pixels"]
        return img  # return is unused for the assertion

    monkeypatch.setattr(plut, "rasterize", fake_rasterize)

    # Source intrinsic is 8000 px; Scale=0.5 → world extent 4000 wu.
    # Display target = dpi*width = 100*6 = 600 px.
    # Expected (world-unit basis): 600 / 4000 = 0.15
    # Old (intrinsic basis):       600 / 8000 = 0.075  (wrong by 2x)
    plut._rasterize_if_necessary(
        image=img,
        dpi=100,
        width=6,
        height=6,
        coordinate_system="global",
        extent={"x": (100.0, 4100.0), "y": (200.0, 4200.0)},
    )

    assert "target_unit_to_pixels" in captured, "rasterize was not called"
    assert captured["target_unit_to_pixels"] == pytest.approx(0.15, rel=1e-6), (
        f"Expected world-unit basis (0.15), got {captured['target_unit_to_pixels']}"
    )


def _add_shapes_table(sdata: SpatialData, element: str = "blobs_polygons", name: str = "shapes_table") -> SpatialData:
    """Add a table annotating a shapes element so it can be measured."""
    gdf = sdata[element]
    adata = AnnData(np.zeros((len(gdf), 1), dtype=np.float32))
    adata.obs["instance_id"] = list(gdf.index)
    adata.obs["region"] = element
    sdata[name] = TableModel.parse(adata, region_key="region", instance_key="instance_id", region=element)
    return sdata


def _labels_sdata(arr: np.ndarray, name: str = "lab", table: str = "t") -> SpatialData:
    """Build a SpatialData with a single 2D-labels element annotated by a table."""
    ids = np.unique(arr)
    ids = ids[ids != 0].astype(int)
    adata = AnnData(np.zeros((len(ids), 1), dtype=np.float32))
    adata.obs["instance_id"] = ids
    adata.obs["region"] = name
    return SpatialData(
        labels={name: Labels2DModel.parse(arr, dims=("y", "x"))},
        tables={table: TableModel.parse(adata, region_key="region", instance_key="instance_id", region=name)},
    )


class TestMeasureObs:
    """`measure_obs` materializes centroids/area/equivalent diameter into the annotating table."""

    def test_writes_centroid_area_diameter_for_labels(self, sdata_blobs: SpatialData) -> None:
        ret = measure_obs(sdata_blobs, "blobs_labels")
        assert ret is None  # inplace default
        table = sdata_blobs["table"]

        assert "spatial" in table.obsm
        coords = table.obsm["spatial"]
        assert coords.shape == (table.n_obs, 2)
        assert np.isfinite(coords).all()
        assert "area" in table.obs and "equivalent_diameter" in table.obs

        # centroids match spatialdata's get_centroids (blobs_labels has the identity transform,
        # so global == intrinsic here)
        gc = get_centroids(sdata_blobs["blobs_labels"], coordinate_system="global").compute()
        expected = gc.reindex(table.obs["instance_id"].to_numpy())[["x", "y"]].to_numpy()
        np.testing.assert_allclose(coords, expected, rtol=1e-6)

        # area is the pixel count (positive integers); diameter = 2*sqrt(area/pi)
        area = table.obs["area"].to_numpy()
        assert (area > 0).all()
        np.testing.assert_allclose(table.obs["equivalent_diameter"].to_numpy(), 2.0 * np.sqrt(area / np.pi), rtol=1e-12)

    def test_writes_for_shapes(self, sdata_blobs: SpatialData) -> None:
        _add_shapes_table(sdata_blobs, "blobs_polygons")
        measure_obs(sdata_blobs, "blobs_polygons", table_name="shapes_table")
        table = sdata_blobs["shapes_table"]

        gdf = sdata_blobs["blobs_polygons"]
        expected_xy = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
        order = table.obs["instance_id"].to_numpy()
        expected_xy = expected_xy[[list(gdf.index).index(i) for i in order]]
        np.testing.assert_allclose(table.obsm["spatial"], expected_xy, rtol=1e-9)
        np.testing.assert_allclose(
            table.obs["area"].to_numpy(),
            gdf.geometry.area.to_numpy()[[list(gdf.index).index(i) for i in order]],
            rtol=1e-9,
        )

    def test_circles_area_is_pi_r_squared(self, sdata_blobs: SpatialData) -> None:
        # Circles are stored as Point geometries (radius column); shapely .area is 0 for them,
        # so area must be pi * r**2 and equivalent diameter must equal the true diameter 2*r.
        _add_shapes_table(sdata_blobs, "blobs_circles", name="circles_table")
        measure_obs(sdata_blobs, "blobs_circles", table_name="circles_table")
        table = sdata_blobs["circles_table"]
        gdf = sdata_blobs["blobs_circles"]
        order = table.obs["instance_id"].to_numpy()
        r = gdf["radius"].to_numpy()[[list(gdf.index).index(i) for i in order]]
        np.testing.assert_allclose(table.obs["area"].to_numpy(), np.pi * r**2, rtol=1e-9)
        np.testing.assert_allclose(table.obs["equivalent_diameter"].to_numpy(), 2.0 * r, rtol=1e-9)

    def test_inplace_false_leaves_original_untouched(self, sdata_blobs: SpatialData) -> None:
        out = measure_obs(sdata_blobs, "blobs_labels", inplace=False)
        assert isinstance(out, SpatialData)
        assert "spatial" in out["table"].obsm
        assert "spatial" not in sdata_blobs["table"].obsm  # original not mutated

    def test_existing_centroids_not_clobbered(self, sdata_blobs: SpatialData) -> None:
        # #1: a populated obsm["spatial"] (reader- or prior-call-provided) is not overwritten.
        table = sdata_blobs["table"]
        sentinel = np.arange(table.n_obs * 2, dtype=float).reshape(table.n_obs, 2)
        table.obsm["spatial"] = sentinel.copy()
        with pytest.warns(UserWarning, match="already populated"):
            measure_obs(sdata_blobs, "blobs_labels")
        np.testing.assert_array_equal(table.obsm["spatial"], sentinel)  # centroids untouched
        assert "area" in table.obs  # area/diameter still written

    def test_centroids_false_keeps_existing_obsm(self, sdata_blobs: SpatialData) -> None:
        table = sdata_blobs["table"]
        sentinel = np.arange(table.n_obs * 2, dtype=float).reshape(table.n_obs, 2)
        table.obsm["spatial"] = sentinel.copy()
        measure_obs(sdata_blobs, "blobs_labels", centroids=False)  # only area/diameter written
        np.testing.assert_array_equal(table.obsm["spatial"], sentinel)
        assert "area" in table.obs

    def test_incompatible_obsm_shape_raises(self, sdata_blobs: SpatialData) -> None:
        table = sdata_blobs["table"]
        table.obsm["spatial"] = np.zeros((table.n_obs, 3))  # e.g. xyz; cannot write 2D centroids over it
        with pytest.raises(ValueError, match="Refusing to overwrite"):
            measure_obs(sdata_blobs, "blobs_labels")

    def test_unmatched_instance_ids_warn_and_write_nan(self, sdata_blobs: SpatialData) -> None:
        # #2: instance ids that don't match the element (e.g. str vs int) -> warn + NaN, not silent.
        table = sdata_blobs["table"]
        table.obs["instance_id"] = table.obs["instance_id"].astype(str)
        with pytest.warns(UserWarning, match="no match"):
            measure_obs(sdata_blobs, "blobs_labels")
        assert np.isnan(table.obsm["spatial"]).all()

    def test_float_dtype_labels_supported(self, sdata_blobs: SpatialData) -> None:
        # #3: a float-typed (but integer-valued) labels raster must not crash np.bincount.
        arr = np.asarray(sdata_blobs["blobs_labels"].data).astype(np.float32)
        sd = _labels_sdata(arr)
        measure_obs(sd, "lab", table_name="t")
        assert np.isfinite(sd["t"].obsm["spatial"]).all()

    def test_existing_nonnumeric_column_raises_before_any_write(self, sdata_blobs: SpatialData) -> None:
        # #4: a non-numeric collision raises BEFORE obsm is mutated (no half-written table).
        table = sdata_blobs["table"]
        table.obs["area"] = pd.Categorical(["lo"] * table.n_obs)
        with pytest.raises(ValueError, match="not numeric"):
            measure_obs(sdata_blobs, "blobs_labels")
        assert "spatial" not in table.obsm  # atomic: nothing written

    def test_sparse_high_label_ids(self, sdata_blobs: SpatialData) -> None:
        # #5: sparse/high label ids (max id >> n_labels) are measured correctly (dense relabelling).
        arr = np.asarray(sdata_blobs["blobs_labels"].data)
        hi = arr.astype(np.int64) * 1000  # ids become 1000, 2000, ... ; max id is huge, few labels
        measure_obs(sd_hi := _labels_sdata(hi), "lab", table_name="t")
        measure_obs(sd_lo := _labels_sdata(arr.astype(np.int64)), "lab", table_name="t")
        # relabelling values does not move pixels -> identical centroid set
        np.testing.assert_allclose(
            np.sort(sd_hi["t"].obsm["spatial"], axis=0), np.sort(sd_lo["t"].obsm["spatial"], axis=0)
        )

    def test_polygon_with_radius_column_uses_geometric_area(self, sdata_blobs: SpatialData) -> None:
        # #7: dispatch on geometry type, not the "radius" column name -> polygons use geometry.area.
        gdf = sdata_blobs["blobs_polygons"].copy()
        gdf["radius"] = 5.0  # incidental column; must NOT trigger the circle (pi*r**2) branch
        sdata_blobs["pr"] = ShapesModel.parse(gdf)
        _add_shapes_table(sdata_blobs, "pr", name="pt")
        measure_obs(sdata_blobs, "pr", table_name="pt")
        order = sdata_blobs["pt"].obs["instance_id"].to_numpy()
        expected = gdf.geometry.area.to_numpy()[[list(gdf.index).index(i) for i in order]]
        np.testing.assert_allclose(sdata_blobs["pt"].obs["area"].to_numpy(), expected, rtol=1e-9)

    def test_flags_select_outputs(self, sdata_blobs: SpatialData) -> None:
        measure_obs(sdata_blobs, "blobs_labels", area=False, diameter=False)
        table = sdata_blobs["table"]
        assert "spatial" in table.obsm
        assert "area" not in table.obs and "equivalent_diameter" not in table.obs

    def test_no_annotating_table_raises(self, sdata_blobs: SpatialData) -> None:
        # blobs_circles is not annotated by any table
        with pytest.raises(ValueError, match="no annotating table"):
            measure_obs(sdata_blobs, "blobs_circles")

    def test_ambiguous_table_raises(self, sdata_blobs: SpatialData) -> None:
        _add_shapes_table(sdata_blobs, "blobs_polygons", name="table_a")
        _add_shapes_table(sdata_blobs, "blobs_polygons", name="table_b")
        with pytest.raises(ValueError, match="multiple tables"):
            measure_obs(sdata_blobs, "blobs_polygons")

    def test_nothing_to_measure_raises(self, sdata_blobs: SpatialData) -> None:
        with pytest.raises(ValueError, match="at least one"):
            measure_obs(sdata_blobs, "blobs_labels", centroids=False, area=False, diameter=False)

    def test_element_none_measures_single_table_elements(self, sdata_blobs: SpatialData) -> None:
        # default blobs: only blobs_labels has a single annotating table
        measure_obs(sdata_blobs)
        assert "spatial" in sdata_blobs["table"].obsm


class TestGetExtentFast:
    """`_get_extent_fast` matches spatialdata's `get_extent` while skipping the per-geometry transform."""

    @pytest.mark.parametrize(
        ("matrix", "expected"),
        [
            ([[2, 0], [0, 3]], True),  # anisotropic scale
            ([[-1, 0], [0, 1]], True),  # flip
            ([[0, -1], [1, 0]], True),  # 90-degree rotation
            ([[0, 1], [1, 0]], True),  # axis swap
            ([[0.7071, -0.7071], [0.7071, 0.7071]], False),  # 45-degree rotation
            ([[1, 0.5], [0, 1]], False),  # shear
        ],
    )
    def test_is_axis_aligned(self, matrix, expected):
        from spatialdata_plot.pl.utils import _is_axis_aligned

        assert _is_axis_aligned(matrix) is expected

    @pytest.mark.parametrize("element", ["blobs_circles", "blobs_polygons"])
    @pytest.mark.parametrize("kind", ["scale_iso", "scale_aniso", "translate", "flip", "rot90", "rot45", "shear"])
    def test_matches_get_extent(self, sdata_blobs: SpatialData, element: str, kind: str):
        import math

        from spatialdata import get_extent
        from spatialdata.transformations import Affine, Scale, Translation, set_transformation

        from spatialdata_plot.pl.utils import _get_extent_fast

        def _rot(theta: float) -> Affine:
            c, s = math.cos(theta), math.sin(theta)
            return Affine([[c, -s, 0], [s, c, 0], [0, 0, 1]], input_axes=("x", "y"), output_axes=("x", "y"))

        transforms = {
            "scale_iso": Scale([2.0, 2.0], axes=("x", "y")),
            "scale_aniso": Scale([2.0, 3.0], axes=("x", "y")),  # circles fall back here
            "translate": Translation([10.0, 20.0], axes=("x", "y")),
            "flip": Scale([-1.0, 1.0], axes=("x", "y")),
            "rot90": _rot(math.pi / 2),
            "rot45": _rot(math.pi / 4),  # not axis-aligned -> fall back
            "shear": Affine([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]], input_axes=("x", "y"), output_axes=("x", "y")),
        }
        set_transformation(sdata_blobs[element], transforms[kind], "cs")
        sub = SpatialData(shapes={element: sdata_blobs[element]})
        kw = dict(has_images=False, has_labels=False, has_points=False)
        fast = _get_extent_fast(sub, "cs", **kw)
        exact = get_extent(sub, "cs", exact=True, **kw)
        for ax in ("x", "y"):
            np.testing.assert_allclose(fast[ax], exact[ax], atol=1e-6)


class TestExtractColorColumn:
    """`_extract_color_column` matches spatialdata's `get_values` bit-identically without copying the table."""

    @staticmethod
    def _annotated_shapes(n: int = 30, *, shuffle: bool = False, drop: int = 0, seed: int = 0) -> SpatialData:
        rng = np.random.default_rng(seed)
        coords = rng.random((n, 2)) * 100
        geom = gpd.GeoDataFrame(
            {"geometry": [Point(*xy) for xy in coords], "radius": np.ones(n)}, index=pd.Index(range(n))
        )
        inst = (rng.permutation(n) if shuffle else np.arange(n))[drop:]
        adata = AnnData(
            X=rng.random((len(inst), 4)).astype("float32"),
            obs=pd.DataFrame(
                {
                    "region": pd.Categorical(["shapes"] * len(inst)),
                    "instance_id": inst,
                    "num": rng.random(len(inst)),
                    "cat": pd.Categorical(rng.choice(list("abc"), len(inst))),
                }
            ),
        )
        adata.var_names = [f"g{i}" for i in range(4)]
        table = TableModel.parse(adata, region="shapes", region_key="region", instance_key="instance_id")
        return SpatialData(shapes={"shapes": ShapesModel.parse(geom)}, tables={"table": table})

    @pytest.mark.parametrize(("key", "origin"), [("g0", "var"), ("g3", "var"), ("num", "obs"), ("cat", "obs")])
    def test_matches_get_values(self, key: str, origin: str):
        from spatialdata import get_values

        from spatialdata_plot.pl._color import _extract_color_column

        sdata = self._annotated_shapes()
        old = pd.Series(get_values(value_key=key, sdata=sdata, element_name="shapes", table_name="table")[key])
        new = _extract_color_column(sdata["table"], key, origin=origin, element=sdata["shapes"], element_name="shapes")
        assert (old.index == new.index).all()
        if pd.api.types.is_numeric_dtype(old):
            np.testing.assert_allclose(old.to_numpy(float), new.to_numpy(float))
        else:
            assert old.astype(str).equals(new.astype(str))
            assert isinstance(new.dtype, pd.CategoricalDtype)  # preserved for the legend path

    def test_shuffled_table_order_realigns(self):
        from spatialdata import get_values

        from spatialdata_plot.pl._color import _extract_color_column

        sdata = self._annotated_shapes(shuffle=True)
        old = pd.Series(get_values(value_key="g0", sdata=sdata, element_name="shapes", table_name="table")["g0"])
        new = _extract_color_column(sdata["table"], "g0", origin="var", element=sdata["shapes"], element_name="shapes")
        np.testing.assert_allclose(old.to_numpy(float), new.to_numpy(float))

    def test_missing_instances_become_nan(self):
        from spatialdata_plot.pl._color import _extract_color_column

        sdata = self._annotated_shapes(drop=5)  # 5 shapes have no annotating table row
        new = _extract_color_column(sdata["table"], "g0", origin="var", element=sdata["shapes"], element_name="shapes")
        assert len(new) == 30
        assert int(new.isna().sum()) == 5


def test_show_renders_all_coordinate_systems_for_distributed_elements():
    """Regression for #694: element types split across coordinate systems must all render.

    Before the fix, ``_resolve_coordinate_systems`` computed the rendered element set from a
    leaked loop variable (the last validated CS only), so ``_get_valid_cs`` could drop a CS whose
    elements lived elsewhere. ``_get_elements_to_be_rendered`` keys on element *type*, so the bug
    only surfaces when the leftover CS lacks a queued element type: here the image lives in ``cs_a``
    and the labels in ``cs_b``, so the leaked ``cs_b`` (no image) used to drop ``cs_a``.
    """
    from spatialdata.models import Image2DModel
    from spatialdata.transformations import Identity

    rng = np.random.default_rng(0)
    img = Image2DModel.parse(rng.random((3, 16, 16)), transformations={"cs_a": Identity()})
    lab = Labels2DModel.parse(rng.integers(0, 5, (16, 16)), transformations={"cs_b": Identity()})
    sdata = SpatialData(images={"img": img}, labels={"lab": lab})

    axes = sdata.pl.render_images("img").pl.render_labels("lab").pl.show(return_ax=True)
    axes = axes if isinstance(axes, list) else [axes]
    assert {ax.get_title() for ax in axes} == {"cs_a", "cs_b"}
    plt.close("all")


class TestCmapParamsMethods:
    """Unit tests for fresh_norm and the colormap_with_alpha helper."""

    @staticmethod
    def _params(cmap_name: str = "viridis", na: Color | None = None) -> CmapParams:
        from matplotlib import colormaps
        from matplotlib.colors import Normalize

        return CmapParams(cmap=colormaps[cmap_name], norm=Normalize(vmin=0.0, vmax=1.0), na_color=na or Color())

    def test_fresh_norm_is_independent_copy(self):
        params = self._params()
        fresh = params.fresh_norm()
        assert fresh is not params.norm
        assert (fresh.vmin, fresh.vmax) == (params.norm.vmin, params.norm.vmax)
        # autoscaling the copy must not mutate the shared norm (the bug fresh_norm prevents)
        fresh.autoscale(np.array([5.0, 10.0]))
        assert (params.norm.vmin, params.norm.vmax) == (0.0, 1.0)

    def test_colormap_with_alpha_preserves_body_and_sets_alpha(self):
        from matplotlib import colormaps

        out = colormap_with_alpha(colormaps["viridis"], 0.5, Color().get_hex_with_alpha())
        # sample at bin centers so quantization is exact for both colormaps
        xs = (np.arange(out.N) + 0.5) / out.N
        got = out(xs)
        np.testing.assert_array_equal(got[:, :3], colormaps["viridis"](xs)[:, :3])
        np.testing.assert_allclose(got[:, 3], 0.5)

    def test_colormap_with_alpha_bad_color_uses_na_with_requested_alpha(self):
        from matplotlib import colormaps
        from matplotlib.colors import to_rgba

        na = Color()  # default lightgray, fully opaque
        out = colormap_with_alpha(colormaps["viridis"], 0.25, na.get_hex_with_alpha())
        bad = out(np.nan)
        np.testing.assert_allclose(bad[:3], to_rgba(na.get_hex_with_alpha())[:3], atol=1e-6)
        # historical `_lut[:, -1] = alpha` overwrote the bad-row alpha too
        assert bad[3] == 0.25


class TestResolveContinuousNorm:
    """Unit tests for `_resolve_continuous_norm` — the single norm source feeding pixels + colorbar."""

    @staticmethod
    def _params(vmin: float | None = None, vmax: float | None = None) -> CmapParams:
        from matplotlib import colormaps
        from matplotlib.colors import Normalize

        return CmapParams(cmap=colormaps["viridis"], norm=Normalize(vmin=vmin, vmax=vmax, clip=False), na_color=Color())

    def test_honors_explicit_vmin_vmax(self):
        from spatialdata_plot.pl._color import _resolve_continuous_norm

        norm = _resolve_continuous_norm(np.array([0.0, 100.0]), self._params(vmin=10.0, vmax=20.0))
        assert (norm.vmin, norm.vmax) == (10.0, 20.0)

    def test_derives_data_range_ignoring_nan_and_inf(self):
        from spatialdata_plot.pl._color import _resolve_continuous_norm

        norm = _resolve_continuous_norm(np.array([1.0, np.nan, 5.0, np.inf, -np.inf]), self._params())
        assert (norm.vmin, norm.vmax) == (1.0, 5.0)

    def test_all_nan_falls_back_to_unit_range(self):
        from spatialdata_plot.pl._color import _resolve_continuous_norm

        norm = _resolve_continuous_norm(np.array([np.nan, np.nan]), self._params())
        assert (norm.vmin, norm.vmax) == (0.0, 1.0)

    def test_degenerate_range_is_not_expanded(self):
        # behavior-preserving: a single distinct value stays degenerate (no invented +/-0.5)
        from spatialdata_plot.pl._color import _resolve_continuous_norm

        norm = _resolve_continuous_norm(np.array([5.0, 5.0, 5.0]), self._params())
        assert (norm.vmin, norm.vmax) == (5.0, 5.0)

    def test_object_dtype_coerces_color_strings_to_nan(self):
        from spatialdata_plot.pl._color import _resolve_continuous_norm

        norm = _resolve_continuous_norm(np.array([1.0, "red", 9.0], dtype=object), self._params())
        assert (norm.vmin, norm.vmax) == (1.0, 9.0)

    def test_same_values_give_identical_norm_and_never_mutate_shared(self):
        # the core #699 invariant: pixels and colorbar call this with the same vector -> same result,
        # and the shared CmapParams.norm is never autoscaled in place.
        from spatialdata_plot.pl._color import _resolve_continuous_norm

        params = self._params()
        vals = np.array([2.0, 7.0, np.nan, 4.0])
        a = _resolve_continuous_norm(vals, params)
        b = _resolve_continuous_norm(vals, params)
        assert (a.vmin, a.vmax) == (b.vmin, b.vmax) == (2.0, 7.0)
        assert params.norm.vmin is None and params.norm.vmax is None


class TestResolveColor:
    """resolve_color classifies each _set_color_source_vec return branch into a ColorType (#700)."""

    @staticmethod
    def _cmap_params() -> CmapParams:
        from matplotlib import colormaps
        from matplotlib.colors import Normalize

        return CmapParams(cmap=colormaps["viridis"], norm=Normalize(), na_color=Color())

    def _resolve(self, sdata_blobs_shapes_annotated: SpatialData, value_to_plot: str | None):
        from spatialdata_plot.pl._color import resolve_color

        element_name = "blobs_polygons"
        return resolve_color(
            sdata=sdata_blobs_shapes_annotated,
            element=sdata_blobs_shapes_annotated[element_name],
            element_name=element_name,
            value_to_plot=value_to_plot,
            na_color=Color(),
            cmap_params=self._cmap_params(),
        )

    def test_no_value_is_none_colortype(self, sdata_blobs_shapes_annotated: SpatialData):
        spec = self._resolve(sdata_blobs_shapes_annotated, None)
        assert spec.colortype == "none"
        assert spec.source_vector is not None  # na array, not None

    def test_all_nan_column_is_none_colortype(self, sdata_blobs_shapes_annotated: SpatialData):
        sdata_blobs_shapes_annotated["blobs_polygons"]["nanvals"] = [np.nan] * 5
        spec = self._resolve(sdata_blobs_shapes_annotated, "nanvals")
        assert spec.colortype == "none"

    def test_numeric_column_is_continuous(self, sdata_blobs_shapes_annotated: SpatialData):
        spec = self._resolve(sdata_blobs_shapes_annotated, "value")  # fixture's [1..5]
        assert spec.colortype == "continuous"
        assert spec.source_vector is None  # the continuous marker

    def test_categorical_column_is_categorical(self, sdata_blobs_shapes_annotated: SpatialData):
        sdata_blobs_shapes_annotated["blobs_polygons"]["cat"] = ["a", "b", "a", "b", "a"]
        spec = self._resolve(sdata_blobs_shapes_annotated, "cat")
        assert spec.colortype == "categorical"
        assert isinstance(spec.source_vector, pd.Categorical)
