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
from spatialdata.models import PointsModel, ShapesModel, TableModel

import spatialdata_plot
from spatialdata_plot.pl.render_params import Color, ColorLike
from spatialdata_plot.pl.utils import (
    _apply_cmap_alpha_to_datashader_result,
    _datashader_map_aggregate_to_color,
    _set_outline,
    measure_obs,
    set_zero_in_cmap_to_transparent,
)
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

    assert spatialdata_plot.pl.utils._is_color_like(color) == result


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
    sdata[name] = TableModel.parse(
        adata, region_key="region", instance_key="instance_id", region=element
    )
    return sdata


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
        np.testing.assert_allclose(
            table.obs["equivalent_diameter"].to_numpy(), 2.0 * np.sqrt(area / np.pi), rtol=1e-12
        )

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

    def test_recompute_overwrites(self, sdata_blobs: SpatialData) -> None:
        measure_obs(sdata_blobs, "blobs_labels")
        table = sdata_blobs["table"]
        table.obsm["spatial"][0] = [999.0, 999.0]  # corrupt one row
        measure_obs(sdata_blobs, "blobs_labels")  # second call overwrites with the real centroid
        assert tuple(table.obsm["spatial"][0]) != (999.0, 999.0)

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
