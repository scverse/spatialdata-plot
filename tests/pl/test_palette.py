"""Tests for palette generation (issue #210)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from matplotlib.colors import to_hex, to_rgb
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel, TableModel

import spatialdata_plot  # noqa: F401 — registers accessor
from spatialdata_plot.pl._palette import (
    _optimize_assignment,
    _pairwise_oklab_dist,
    _perceptual_distance_matrix,
    _rgb_to_oklab,
    _simulate_cvd,
    _spatial_interlacement,
    make_palette,
    make_palette_from_data,
)
from tests.conftest import DPI, PlotTester, PlotTesterMeta

matplotlib.use("agg")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_clustered_points_sdata(seed: int = 0) -> SpatialData:
    """Create a SpatialData with two spatially interleaved point clusters.

    Cluster layout (deliberately interleaved):
        - "A" cells at (0,0), (1,0), (0,1)
        - "B" cells at (0.5,0.5), (1.5,0.5), (0.5,1.5)
        - "C" cells at (10,10), (11,10), (10,11)  — isolated cluster
    """
    rng = np.random.default_rng(seed)
    coords_a = np.array([[0, 0], [1, 0], [0, 1]], dtype=float) + rng.normal(0, 0.05, (3, 2))
    coords_b = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5]], dtype=float) + rng.normal(0, 0.05, (3, 2))
    coords_c = np.array([[10, 10], [11, 10], [10, 11]], dtype=float) + rng.normal(0, 0.05, (3, 2))

    coords = np.vstack([coords_a, coords_b, coords_c])
    labels = pd.Categorical(["A"] * 3 + ["B"] * 3 + ["C"] * 3)

    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cell_type": labels})
    points = PointsModel.parse(df)
    return SpatialData(points={"cells": points})


def _make_shapes_sdata(seed: int = 0) -> SpatialData:
    """Create a SpatialData with shapes + linked table."""
    from anndata import AnnData
    from geopandas import GeoDataFrame
    from shapely import Point

    rng = np.random.default_rng(seed)
    n = 30
    coords = rng.normal(size=(n, 2)) * 5
    gdf = GeoDataFrame({"radius": np.ones(n)}, geometry=[Point(x, y) for x, y in coords])
    gdf.index = pd.RangeIndex(n)

    labels = pd.Categorical(rng.choice(["X", "Y", "Z"], size=n))
    adata = AnnData(
        np.zeros((n, 1)),
        obs=pd.DataFrame(
            {
                "cell_type": labels,
                "instance_id": np.arange(n),
                "region": ["my_shapes"] * n,
            },
            index=pd.RangeIndex(n).astype(str),
        ),
    )
    adata = TableModel.parse(
        adata=adata,
        region="my_shapes",
        region_key="region",
        instance_key="instance_id",
    )

    shapes = ShapesModel.parse(gdf)
    return SpatialData(shapes={"my_shapes": shapes}, tables={"table": adata})


# ---------------------------------------------------------------------------
# Unit tests: color-space helpers
# ---------------------------------------------------------------------------


class TestOklab:
    def test_black_and_white(self):
        rgb = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        lab = _rgb_to_oklab(rgb)
        assert lab[0, 0] == pytest.approx(0.0, abs=0.01)
        assert lab[1, 0] == pytest.approx(1.0, abs=0.01)

    def test_pairwise_distance_symmetric(self):
        rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        lab = _rgb_to_oklab(rgb)
        d = _pairwise_oklab_dist(lab)
        assert d.shape == (3, 3)
        np.testing.assert_allclose(d, d.T)
        np.testing.assert_allclose(np.diag(d), 0)

    def test_distinct_colors_have_positive_distance(self):
        rgb = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
        lab = _rgb_to_oklab(rgb)
        d = _pairwise_oklab_dist(lab)
        assert d[0, 1] > 0.1


# ---------------------------------------------------------------------------
# Unit tests: CVD simulation
# ---------------------------------------------------------------------------


class TestCVD:
    @pytest.mark.parametrize("cvd_type", ["protanopia", "deuteranopia", "tritanopia"])
    def test_output_in_range(self, cvd_type: str):
        rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        sim = _simulate_cvd(rgb, cvd_type)
        assert sim.shape == (3, 3)
        assert np.all(sim >= 0)
        assert np.all(sim <= 1)

    def test_general_returns_stacked(self):
        rgb = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        sim = _simulate_cvd(rgb, "general")
        assert sim.shape == (3, 2, 3)

    @pytest.mark.parametrize("cvd_type", ["protanopia", "deuteranopia", "tritanopia"])
    def test_red_green_less_distinct(self, cvd_type: str):
        """Under protanopia/deuteranopia, red and green should be less distinct than for normal vision."""
        rgb = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        normal_dist = _perceptual_distance_matrix(rgb)[0, 1]
        cvd_dist = _perceptual_distance_matrix(rgb, colorblind_type=cvd_type)[0, 1]
        if cvd_type in ("protanopia", "deuteranopia"):
            assert cvd_dist < normal_dist


# ---------------------------------------------------------------------------
# Unit tests: spatial interlacement
# ---------------------------------------------------------------------------


class TestSpatialInterlacement:
    def test_interleaved_higher_than_separated(self):
        """Categories that are spatially interleaved should have higher scores."""
        coords = np.array([[0, 0], [1, 0], [0.5, 0.5], [1.5, 0.5], [10, 10], [11, 10]])
        labels = np.array(["A", "B", "A", "B", "C", "C"])
        categories = ["A", "B", "C"]

        mat = _spatial_interlacement(coords, labels, categories, n_neighbors=3)

        assert mat[0, 1] > mat[0, 2]
        assert mat[0, 1] > mat[1, 2]

    def test_diagonal_is_zero(self):
        coords = np.array([[0, 0], [1, 0], [0.5, 0.5]])
        labels = np.array(["A", "B", "A"])
        mat = _spatial_interlacement(coords, labels, ["A", "B"], n_neighbors=2)
        np.testing.assert_allclose(np.diag(mat), 0)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        coords = rng.normal(size=(50, 2))
        labels = np.array(["A", "B", "C", "D", "E"] * 10)
        mat = _spatial_interlacement(coords, labels, ["A", "B", "C", "D", "E"], n_neighbors=5)
        np.testing.assert_allclose(mat, mat.T)


# ---------------------------------------------------------------------------
# Unit tests: optimizer
# ---------------------------------------------------------------------------


class TestOptimizer:
    def test_single_category(self):
        perm = _optimize_assignment(np.zeros((1, 1)), np.zeros((1, 1)))
        assert list(perm) == [0]

    def test_two_categories(self):
        """With 2 categories, there are only 2 permutations — optimizer should pick the better one."""
        inter = np.array([[0, 1], [1, 0]], dtype=float)
        cdist = np.array([[0, 10], [10, 0]], dtype=float)
        rng = np.random.default_rng(0)
        perm = _optimize_assignment(inter, cdist, rng=rng)
        assert set(perm) == {0, 1}

    def test_deterministic_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        inter = np.random.default_rng(0).random((5, 5))
        inter = np.maximum(inter, inter.T)
        np.fill_diagonal(inter, 0)
        cdist = np.random.default_rng(1).random((5, 5))
        cdist = np.maximum(cdist, cdist.T)
        np.fill_diagonal(cdist, 0)

        perm1 = _optimize_assignment(inter, cdist, rng=rng1)
        perm2 = _optimize_assignment(inter, cdist, rng=rng2)
        np.testing.assert_array_equal(perm1, perm2)


# ---------------------------------------------------------------------------
# Tests: make_palette
# ---------------------------------------------------------------------------


class TestMakePalette:
    def test_default_returns_n_colors(self):
        result = make_palette(5)
        assert len(result) == 5
        assert all(c.startswith("#") for c in result)

    def test_returns_list(self):
        result = make_palette(3)
        assert isinstance(result, list)

    def test_named_palette(self):
        result = make_palette(4, palette="okabe_ito")
        assert len(result) == 4

    def test_matplotlib_cmap(self):
        result = make_palette(6, palette="tab10")
        assert len(result) == 6

    def test_custom_list(self):
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        result = make_palette(3, palette=colors)
        assert result == [to_hex(to_rgb(c)) for c in colors]

    def test_contrast_reorders(self):
        """Contrast method should produce a permutation of the input colors."""
        colors = ["#ff0000", "#ff1100", "#0000ff", "#00ff00"]
        result = make_palette(4, palette=colors, method="contrast", seed=42)
        assert set(result) == {to_hex(to_rgb(c)) for c in colors}

    def test_colorblind_reorders(self):
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        result = make_palette(3, palette=colors, method="colorblind", seed=42)
        assert set(result) == {to_hex(to_rgb(c)) for c in colors}

    def test_deuteranopia(self):
        result = make_palette(5, method="deuteranopia", seed=42)
        assert len(result) == 5

    def test_deterministic(self):
        r1 = make_palette(5, method="contrast", seed=42)
        r2 = make_palette(5, method="contrast", seed=42)
        assert r1 == r2

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            make_palette(0)

    def test_too_few_colors_raises(self):
        with pytest.raises(ValueError, match="needed"):
            make_palette(10, palette=["red", "blue"])

    def test_spaco_method_raises(self):
        with pytest.raises(ValueError, match="requires spatial data"):
            make_palette(3, method="spaco")  # type: ignore[arg-type]

    def test_spaco_colorblind_method_raises(self):
        with pytest.raises(ValueError, match="requires spatial data"):
            make_palette(3, method="spaco_colorblind")  # type: ignore[arg-type]

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            make_palette(3, method="invalid")  # type: ignore[arg-type]

    def test_unknown_palette_name_raises(self):
        with pytest.raises(ValueError, match="Unknown palette name"):
            make_palette(3, palette="nonexistent_palette")


# ---------------------------------------------------------------------------
# Tests: make_palette_from_data — default
# ---------------------------------------------------------------------------


class TestMakePaletteFromDataDefault:
    def test_basic(self):
        sdata = _make_clustered_points_sdata()
        result = make_palette_from_data(sdata, "cells", "cell_type")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"A", "B", "C"}
        for v in result.values():
            assert v.startswith("#")

    def test_matches_scanpy_order(self):
        """Default method should assign colors in sorted-category order, matching scanpy."""
        from scanpy.plotting.palettes import default_20

        sdata = _make_clustered_points_sdata()
        result = make_palette_from_data(sdata, "cells", "cell_type", method="default")

        for i, cat in enumerate(sorted(result.keys())):
            assert result[cat] == to_hex(to_rgb(default_20[i]))

    def test_custom_palette(self):
        sdata = _make_clustered_points_sdata()
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        result = make_palette_from_data(sdata, "cells", "cell_type", palette=colors)
        assert list(result.values()) == [to_hex(to_rgb(c)) for c in colors]

    def test_named_palette(self):
        sdata = _make_clustered_points_sdata()
        result = make_palette_from_data(sdata, "cells", "cell_type", palette="okabe_ito")
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_matplotlib_cmap(self):
        sdata = _make_clustered_points_sdata()
        result = make_palette_from_data(sdata, "cells", "cell_type", palette="tab10")
        assert isinstance(result, dict)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: make_palette_from_data — spaco
# ---------------------------------------------------------------------------


class TestMakePaletteFromDataContrast:
    def test_contrast_returns_dict(self):
        sdata = _make_clustered_points_sdata()
        result = make_palette_from_data(sdata, "cells", "cell_type", method="contrast", seed=42)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"A", "B", "C"}

    def test_colorblind_returns_dict(self):
        sdata = _make_clustered_points_sdata()
        result = make_palette_from_data(sdata, "cells", "cell_type", method="colorblind", seed=42)
        assert isinstance(result, dict)
        assert len(result) == 3


class TestMakePaletteFromDataSpaco:
    def test_basic_points(self):
        sdata = _make_clustered_points_sdata()
        result = make_palette_from_data(sdata, "cells", "cell_type", method="spaco", seed=42)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"A", "B", "C"}

    def test_deterministic(self):
        sdata = _make_clustered_points_sdata()
        r1 = make_palette_from_data(sdata, "cells", "cell_type", method="spaco", seed=42)
        r2 = make_palette_from_data(sdata, "cells", "cell_type", method="spaco", seed=42)
        assert r1 == r2

    def test_different_seeds_can_differ(self):
        sdata = _make_clustered_points_sdata()
        r1 = make_palette_from_data(sdata, "cells", "cell_type", method="spaco", seed=0)
        r2 = make_palette_from_data(sdata, "cells", "cell_type", method="spaco", seed=999)
        assert set(r1.keys()) == set(r2.keys())

    def test_custom_palette(self):
        sdata = _make_clustered_points_sdata()
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        result = make_palette_from_data(sdata, "cells", "cell_type", method="spaco", palette=colors, seed=42)
        assert set(result.values()) == {to_hex(to_rgb(c)) for c in colors}

    def test_spaco_colorblind(self):
        sdata = _make_clustered_points_sdata()
        result = make_palette_from_data(sdata, "cells", "cell_type", method="spaco_colorblind", seed=42)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_spaco_deuteranopia(self):
        sdata = _make_clustered_points_sdata()
        result = make_palette_from_data(sdata, "cells", "cell_type", method="spaco_deuteranopia", seed=42)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_spaco_with_named_palette(self):
        sdata = _make_clustered_points_sdata()
        result = make_palette_from_data(sdata, "cells", "cell_type", method="spaco", palette="okabe_ito", seed=42)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_spaco_with_matplotlib_cmap(self):
        sdata = _make_clustered_points_sdata()
        result = make_palette_from_data(sdata, "cells", "cell_type", method="spaco", palette="tab10", seed=42)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_single_category(self):
        coords = np.array([[0, 0], [1, 1]], dtype=float)
        df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "ct": pd.Categorical(["A", "A"])})
        points = PointsModel.parse(df)
        sdata = SpatialData(points={"pts": points})

        result = make_palette_from_data(sdata, "pts", "ct", method="spaco", seed=0)
        assert len(result) == 1
        assert "A" in result

    def test_nan_labels_filtered(self):
        coords = np.array([[0, 0], [1, 0], [0, 1], [10, 10]], dtype=float)
        labels = pd.Categorical(["A", "B", "A", None])
        df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "ct": labels})
        points = PointsModel.parse(df)
        sdata = SpatialData(points={"pts": points})

        result = make_palette_from_data(sdata, "pts", "ct", method="spaco", seed=0)
        assert "A" in result
        assert "B" in result

    def test_shapes_with_table(self):
        sdata = _make_shapes_sdata()
        result = make_palette_from_data(sdata, "my_shapes", "cell_type", method="spaco", seed=42)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"X", "Y", "Z"}

    def test_interleaved_get_distinct_colors(self):
        """Core property: spatially interleaved categories should get the most distinct colors."""
        sdata = _make_clustered_points_sdata(seed=0)
        palette = ["#ff0000", "#ff1100", "#0000ff"]
        result = make_palette_from_data(sdata, "cells", "cell_type", method="spaco", palette=palette, seed=0)

        a_color = result["A"]
        b_color = result["B"]
        assert a_color == "#0000ff" or b_color == "#0000ff"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestMakePaletteFromDataErrors:
    def test_too_few_colors_raises(self):
        sdata = _make_clustered_points_sdata()
        with pytest.raises(ValueError, match="needed"):
            make_palette_from_data(sdata, "cells", "cell_type", method="spaco", palette=["red", "blue"], seed=0)

    def test_missing_element_raises(self):
        sdata = _make_clustered_points_sdata()
        with pytest.raises(KeyError, match="not found"):
            make_palette_from_data(sdata, "nonexistent", "cell_type")

    def test_missing_column_raises(self):
        sdata = _make_clustered_points_sdata()
        with pytest.raises(KeyError, match="not found"):
            make_palette_from_data(sdata, "cells", "nonexistent_col")

    def test_unknown_method_raises(self):
        sdata = _make_clustered_points_sdata()
        with pytest.raises(ValueError, match="Unknown method"):
            make_palette_from_data(sdata, "cells", "cell_type", method="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration tests: dict palette through render pipeline
# ---------------------------------------------------------------------------


class TestDictPalette:
    def test_dict_palette_in_render_points(self, sdata_blobs: SpatialData):
        """Dict palette should flow through render_points without errors."""
        palette = {"0": "#ff0000", "1": "#00ff00"}
        sdata_blobs.pl.render_points("blobs_points", color="genes", palette=palette)

    def test_dict_palette_in_render_labels(self, sdata_blobs: SpatialData):
        """Dict palette should flow through render_labels without errors."""
        palette = {"blobs_labels": "#ff0000"}
        sdata_blobs.pl.render_labels("blobs_labels", color="region", palette=palette)


# ---------------------------------------------------------------------------
# Visual tests: make_palette_from_data → render → show
# ---------------------------------------------------------------------------

sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=DPI, color_map="viridis")


class TestPaletteVisual(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_dict_palette_hex_points(self, sdata_blobs: SpatialData):
        """Visual test: hex dict palette renders points correctly."""
        palette = make_palette_from_data(sdata_blobs, "blobs_points", "genes", palette="okabe_ito")
        sdata_blobs.pl.render_points("blobs_points", color="genes", palette=palette).pl.show()

    def test_plot_dict_palette_hex_shapes(self, sdata_blobs: SpatialData):
        """Visual test: hex dict palette renders shapes correctly."""
        sdata_blobs["blobs_polygons"]["cat_col"] = pd.Series(["a", "b", "a", "b", "a"], dtype="category")
        palette = make_palette_from_data(sdata_blobs, "blobs_polygons", "cat_col", palette="okabe_ito")
        sdata_blobs.pl.render_shapes("blobs_polygons", color="cat_col", palette=palette).pl.show()

    def test_plot_dict_palette_hex_labels(self, sdata_blobs: SpatialData):
        """Visual test: hex dict palette renders labels correctly."""
        palette = make_palette_from_data(sdata_blobs, "blobs_labels", "region", palette="okabe_ito")
        sdata_blobs.pl.render_labels("blobs_labels", color="region", palette=palette).pl.show()

    def test_plot_dict_palette_named_colors_points(self, sdata_blobs: SpatialData):
        """Visual test: named-color dict palette renders points correctly."""
        palette = {"gene_a": "red", "gene_b": "dodgerblue"}
        sdata_blobs.pl.render_points("blobs_points", color="genes", palette=palette).pl.show()

    def test_plot_dict_palette_named_colors_shapes(self, sdata_blobs: SpatialData):
        """Visual test: named-color dict palette renders shapes correctly."""
        sdata_blobs["blobs_polygons"]["cat_col"] = pd.Series(["a", "b", "a", "b", "a"], dtype="category")
        palette = {"a": "forestgreen", "b": "orchid"}
        sdata_blobs.pl.render_shapes("blobs_polygons", color="cat_col", palette=palette).pl.show()

    def test_plot_dict_palette_named_colors_labels(self, sdata_blobs: SpatialData):
        """Visual test: named-color dict palette renders labels correctly."""
        palette = {"blobs_labels": "coral"}
        sdata_blobs.pl.render_labels("blobs_labels", color="region", palette=palette).pl.show()
