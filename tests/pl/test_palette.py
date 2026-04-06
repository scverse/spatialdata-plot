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


def _build_clustered_points_sdata(seed: int = 0) -> SpatialData:
    """SpatialData with interleaved A/B clusters near origin and isolated C far away."""
    rng = np.random.default_rng(seed)
    coords_a = np.array([[0, 0], [1, 0], [0, 1]], dtype=float) + rng.normal(0, 0.05, (3, 2))
    coords_b = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5]], dtype=float) + rng.normal(0, 0.05, (3, 2))
    coords_c = np.array([[10, 10], [11, 10], [10, 11]], dtype=float) + rng.normal(0, 0.05, (3, 2))

    coords = np.vstack([coords_a, coords_b, coords_c])
    labels = pd.Categorical(["A"] * 3 + ["B"] * 3 + ["C"] * 3)
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cell_type": labels})
    return SpatialData(points={"cells": PointsModel.parse(df)})


def _build_shapes_sdata(seed: int = 0) -> SpatialData:
    """SpatialData with shapes + linked table containing categorical labels."""
    from anndata import AnnData
    from geopandas import GeoDataFrame
    from shapely import Point

    rng = np.random.default_rng(seed)
    n = 30
    coords = rng.normal(size=(n, 2)) * 5
    gdf = GeoDataFrame({"radius": np.ones(n)}, geometry=[Point(x, y) for x, y in coords])
    gdf.index = pd.RangeIndex(n)

    adata = AnnData(
        np.zeros((n, 1)),
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(rng.choice(["X", "Y", "Z"], size=n)),
                "instance_id": np.arange(n),
                "region": ["my_shapes"] * n,
            },
            index=pd.RangeIndex(n).astype(str),
        ),
    )
    adata = TableModel.parse(adata=adata, region="my_shapes", region_key="region", instance_key="instance_id")
    return SpatialData(shapes={"my_shapes": ShapesModel.parse(gdf)}, tables={"table": adata})


@pytest.fixture(scope="module")
def clustered_sdata() -> SpatialData:
    return _build_clustered_points_sdata()


@pytest.fixture(scope="module")
def shapes_sdata() -> SpatialData:
    return _build_shapes_sdata()


# ---------------------------------------------------------------------------
# Unit tests: internals
# ---------------------------------------------------------------------------


class TestOklab:
    def test_black_and_white(self):
        lab = _rgb_to_oklab(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        assert lab[0, 0] == pytest.approx(0.0, abs=0.01)
        assert lab[1, 0] == pytest.approx(1.0, abs=0.01)

    def test_pairwise_distance_symmetric(self):
        d = _pairwise_oklab_dist(_rgb_to_oklab(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)))
        assert d.shape == (3, 3)
        np.testing.assert_allclose(d, d.T)
        np.testing.assert_allclose(np.diag(d), 0)

    def test_distinct_colors_have_positive_distance(self):
        d = _pairwise_oklab_dist(_rgb_to_oklab(np.array([[1, 0, 0], [0, 0, 1]], dtype=float)))
        assert d[0, 1] > 0.1


class TestCVD:
    @pytest.mark.parametrize("cvd_type", ["protanopia", "deuteranopia", "tritanopia"])
    def test_output_in_range(self, cvd_type: str):
        sim = _simulate_cvd(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float), cvd_type)
        assert sim.shape == (3, 3)
        assert np.all((sim >= 0) & (sim <= 1))

    def test_general_returns_stacked(self):
        sim = _simulate_cvd(np.array([[1, 0, 0], [0, 1, 0]], dtype=float), "general")
        assert sim.shape == (3, 2, 3)

    @pytest.mark.parametrize("cvd_type", ["protanopia", "deuteranopia"])
    def test_red_green_less_distinct(self, cvd_type: str):
        rgb = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        assert _perceptual_distance_matrix(rgb, colorblind_type=cvd_type)[0, 1] < _perceptual_distance_matrix(rgb)[0, 1]


class TestSpatialInterlacement:
    def test_interleaved_higher_than_separated(self):
        coords = np.array([[0, 0], [1, 0], [0.5, 0.5], [1.5, 0.5], [10, 10], [11, 10]])
        mat = _spatial_interlacement(coords, np.array(["A", "B", "A", "B", "C", "C"]), ["A", "B", "C"], n_neighbors=3)
        assert mat[0, 1] > mat[0, 2]
        assert mat[0, 1] > mat[1, 2]

    def test_diagonal_is_zero(self):
        mat = _spatial_interlacement(np.array([[0, 0], [1, 0], [0.5, 0.5]]), np.array(["A", "B", "A"]), ["A", "B"], 2)
        np.testing.assert_allclose(np.diag(mat), 0)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        mat = _spatial_interlacement(rng.normal(size=(50, 2)), np.array(list("ABCDE") * 10), list("ABCDE"), 5)
        np.testing.assert_allclose(mat, mat.T)


class TestOptimizer:
    def test_single_category(self):
        assert list(_optimize_assignment(np.zeros((1, 1)), np.zeros((1, 1)))) == [0]

    def test_two_categories(self):
        perm = _optimize_assignment(np.array([[0, 1], [1, 0]], dtype=float), np.array([[0, 10], [10, 0]], dtype=float))
        assert set(perm) == {0, 1}

    def test_deterministic_with_seed(self):
        inter = np.random.default_rng(0).random((5, 5))
        inter = np.maximum(inter, inter.T)
        np.fill_diagonal(inter, 0)
        cdist = np.random.default_rng(1).random((5, 5))
        cdist = np.maximum(cdist, cdist.T)
        np.fill_diagonal(cdist, 0)

        p1 = _optimize_assignment(inter, cdist, rng=np.random.default_rng(42))
        p2 = _optimize_assignment(inter, cdist, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Tests: make_palette
# ---------------------------------------------------------------------------


class TestMakePalette:
    def test_default_returns_n_hex_colors(self):
        result = make_palette(5)
        assert len(result) == 5
        assert isinstance(result, list)
        assert all(c.startswith("#") for c in result)

    @pytest.mark.parametrize("palette", ["okabe_ito", "tab10", None])
    def test_palette_sources(self, palette: str | None):
        result = make_palette(4, palette=palette)
        assert len(result) == 4

    def test_custom_list(self):
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        assert make_palette(3, palette=colors) == [to_hex(to_rgb(c)) for c in colors]

    @pytest.mark.parametrize("method", ["contrast", "colorblind", "deuteranopia"])
    def test_optimization_methods_produce_permutation(self, method: str):
        colors = ["#ff0000", "#ff1100", "#0000ff", "#00ff00"]
        result = make_palette(4, palette=colors, method=method, seed=42)
        assert set(result) == {to_hex(to_rgb(c)) for c in colors}

    def test_deterministic(self):
        assert make_palette(5, method="contrast", seed=42) == make_palette(5, method="contrast", seed=42)

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            make_palette(0)

    def test_too_few_colors_raises(self):
        with pytest.raises(ValueError, match="needed"):
            make_palette(10, palette=["red", "blue"])

    @pytest.mark.parametrize("method", ["spaco", "spaco_colorblind"])
    def test_spaco_methods_raise(self, method: str):
        with pytest.raises(ValueError, match="requires spatial data"):
            make_palette(3, method=method)  # type: ignore[arg-type]

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            make_palette(3, method="invalid")  # type: ignore[arg-type]

    def test_unknown_palette_name_raises(self):
        with pytest.raises(ValueError, match="Unknown palette name"):
            make_palette(3, palette="nonexistent_palette")


# ---------------------------------------------------------------------------
# Tests: make_palette_from_data
# ---------------------------------------------------------------------------


class TestMakePaletteFromData:
    def test_default_returns_dict(self, clustered_sdata: SpatialData):
        result = make_palette_from_data(clustered_sdata, "cells", "cell_type")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"A", "B", "C"}
        assert all(v.startswith("#") for v in result.values())

    def test_default_matches_scanpy_order(self, clustered_sdata: SpatialData):
        from scanpy.plotting.palettes import default_20

        result = make_palette_from_data(clustered_sdata, "cells", "cell_type")
        for i, cat in enumerate(sorted(result.keys())):
            assert result[cat] == to_hex(to_rgb(default_20[i]))

    def test_custom_palette(self, clustered_sdata: SpatialData):
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        result = make_palette_from_data(clustered_sdata, "cells", "cell_type", palette=colors)
        assert list(result.values()) == [to_hex(to_rgb(c)) for c in colors]

    @pytest.mark.parametrize("palette", ["okabe_ito", "tab10"])
    def test_named_palette_sources(self, clustered_sdata: SpatialData, palette: str):
        result = make_palette_from_data(clustered_sdata, "cells", "cell_type", palette=palette)
        assert isinstance(result, dict) and len(result) == 3

    @pytest.mark.parametrize(
        "method",
        ["contrast", "colorblind", "spaco", "spaco_colorblind", "spaco_deuteranopia"],
    )
    def test_all_methods_return_valid_dict(self, clustered_sdata: SpatialData, method: str):
        result = make_palette_from_data(clustered_sdata, "cells", "cell_type", method=method, seed=42)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"A", "B", "C"}

    def test_spaco_deterministic(self, clustered_sdata: SpatialData):
        r1 = make_palette_from_data(clustered_sdata, "cells", "cell_type", method="spaco", seed=42)
        r2 = make_palette_from_data(clustered_sdata, "cells", "cell_type", method="spaco", seed=42)
        assert r1 == r2

    def test_spaco_different_seeds_can_differ(self, clustered_sdata: SpatialData):
        r1 = make_palette_from_data(clustered_sdata, "cells", "cell_type", method="spaco", seed=0)
        r2 = make_palette_from_data(clustered_sdata, "cells", "cell_type", method="spaco", seed=999)
        assert set(r1.keys()) == set(r2.keys())

    def test_spaco_custom_palette_is_permutation(self, clustered_sdata: SpatialData):
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        result = make_palette_from_data(clustered_sdata, "cells", "cell_type", method="spaco", palette=colors, seed=42)
        assert set(result.values()) == {to_hex(to_rgb(c)) for c in colors}

    def test_spaco_single_category(self):
        df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "ct": pd.Categorical(["A", "A"])})
        sdata = SpatialData(points={"pts": PointsModel.parse(df)})
        result = make_palette_from_data(sdata, "pts", "ct", method="spaco", seed=0)
        assert len(result) == 1 and "A" in result

    def test_spaco_nan_labels_filtered(self):
        df = pd.DataFrame(
            {"x": [0.0, 1.0, 0.0, 10.0], "y": [0.0, 0.0, 1.0, 10.0], "ct": pd.Categorical(["A", "B", "A", None])}
        )
        sdata = SpatialData(points={"pts": PointsModel.parse(df)})
        result = make_palette_from_data(sdata, "pts", "ct", method="spaco", seed=0)
        assert {"A", "B"} <= set(result.keys())

    def test_shapes_with_table(self, shapes_sdata: SpatialData):
        result = make_palette_from_data(shapes_sdata, "my_shapes", "cell_type", method="spaco", seed=42)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"X", "Y", "Z"}

    def test_interleaved_get_distinct_colors(self):
        sdata = _build_clustered_points_sdata(seed=0)
        palette = ["#ff0000", "#ff1100", "#0000ff"]
        result = make_palette_from_data(sdata, "cells", "cell_type", method="spaco", palette=palette, seed=0)
        # A and B (interleaved) should not both get red-ish colors
        assert result["A"] == "#0000ff" or result["B"] == "#0000ff"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestMakePaletteFromDataErrors:
    def test_too_few_colors(self, clustered_sdata: SpatialData):
        with pytest.raises(ValueError, match="needed"):
            make_palette_from_data(clustered_sdata, "cells", "cell_type", palette=["red", "blue"])

    def test_missing_element(self, clustered_sdata: SpatialData):
        with pytest.raises(KeyError, match="not found"):
            make_palette_from_data(clustered_sdata, "nonexistent", "cell_type")

    def test_missing_column(self, clustered_sdata: SpatialData):
        with pytest.raises(KeyError, match="not found"):
            make_palette_from_data(clustered_sdata, "cells", "nonexistent_col")

    def test_unknown_method(self, clustered_sdata: SpatialData):
        with pytest.raises(ValueError, match="Unknown method"):
            make_palette_from_data(clustered_sdata, "cells", "cell_type", method="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration: dict palette through render pipeline
# ---------------------------------------------------------------------------


class TestDictPalette:
    def test_dict_palette_in_render_points(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points("blobs_points", color="genes", palette={"0": "#ff0000", "1": "#00ff00"})

    def test_dict_palette_in_render_labels(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color="region", palette={"blobs_labels": "#ff0000"})


# ---------------------------------------------------------------------------
# Visual tests
# ---------------------------------------------------------------------------

sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=DPI, color_map="viridis")


class TestPaletteVisual(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_dict_palette_hex_points(self, sdata_blobs: SpatialData):
        palette = make_palette_from_data(sdata_blobs, "blobs_points", "genes", palette="okabe_ito")
        sdata_blobs.pl.render_points("blobs_points", color="genes", palette=palette).pl.show()

    def test_plot_dict_palette_hex_shapes(self, sdata_blobs: SpatialData):
        sdata_blobs["blobs_polygons"]["cat_col"] = pd.Series(["a", "b", "a", "b", "a"], dtype="category")
        palette = make_palette_from_data(sdata_blobs, "blobs_polygons", "cat_col", palette="okabe_ito")
        sdata_blobs.pl.render_shapes("blobs_polygons", color="cat_col", palette=palette).pl.show()

    def test_plot_dict_palette_hex_labels(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color="region", palette={"blobs_labels": "#E69F00"}).pl.show()

    def test_plot_dict_palette_named_colors_points(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_points(
            "blobs_points", color="genes", palette={"gene_a": "red", "gene_b": "dodgerblue"}
        ).pl.show()

    def test_plot_dict_palette_named_colors_shapes(self, sdata_blobs: SpatialData):
        sdata_blobs["blobs_polygons"]["cat_col"] = pd.Series(["a", "b", "a", "b", "a"], dtype="category")
        sdata_blobs.pl.render_shapes(
            "blobs_polygons", color="cat_col", palette={"a": "forestgreen", "b": "orchid"}
        ).pl.show()

    def test_plot_dict_palette_named_colors_labels(self, sdata_blobs: SpatialData):
        sdata_blobs.pl.render_labels("blobs_labels", color="region", palette={"blobs_labels": "coral"}).pl.show()
