"""Tests for render_points fixes: groups handling, datashader method, and ds_agg reductions.

These tests verify correctness at the internal-function level — no visual/image tests.
They check that warnings are emitted, parameters are forwarded, and values are correct.
"""

from __future__ import annotations

import logging

import datashader as ds
import numpy as np
import pandas as pd
import pytest

from spatialdata_plot._logging import logger, logger_warns
from spatialdata_plot.pl._datashader import (
    _build_color_key,
    _build_datashader_color_key,
    _coerce_categorical_source,
    _ds_aggregate,
    _ds_shade_categorical,
    _inject_ds_nan_sentinel,
)
from spatialdata_plot.pl.utils import (
    _datashader_aggregate_with_function,
    _datshader_get_how_kw_for_spread,
    _hex_no_alpha,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_canvas_and_df(n=500, seed=42):
    """Create a small datashader Canvas and a DataFrame with x, y, cat, val columns."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x": rng.uniform(-10, 10, n),
            "y": rng.uniform(-10, 10, n),
            "cat": pd.Categorical(rng.choice(["A", "B", "C"], n)),
            "val": rng.normal(0, 1, n),
        }
    )
    cvs = ds.Canvas(plot_width=50, plot_height=50, x_range=(-10, 10), y_range=(-10, 10))
    return cvs, df


# ===========================================================================
# FIX 1: `default_reduction` parameter in `_ds_aggregate` is now forwarded
# ===========================================================================


class TestDefaultReductionForwarded:
    """The ``default_reduction`` parameter is now forwarded to the aggregation
    call when ``ds_reduction`` is None."""

    def test_default_reduction_is_forwarded(self):
        """Changing default_reduction should change the aggregation result."""
        cvs, df = _make_simple_canvas_and_df()

        agg_sum, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, None, "sum", "points")
        agg_max, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, None, "max", "points")

        # After fix: default_reduction is honored, so these should differ
        assert not np.allclose(
            np.nan_to_num(agg_sum.values, nan=0),
            np.nan_to_num(agg_max.values, nan=0),
        ), "default_reduction='max' should produce different aggregate than 'sum'"

    def test_default_reduction_result_equals_explicit_reduction(self):
        """default_reduction='max' with ds_reduction=None should equal explicit ds_reduction='max'."""
        cvs, df = _make_simple_canvas_and_df()

        agg_default_max, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, None, "max", "points")
        agg_explicit_max, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, "max", "max", "points")

        np.testing.assert_array_equal(
            np.nan_to_num(agg_default_max.values, nan=0),
            np.nan_to_num(agg_explicit_max.values, nan=0),
        )

    def test_explicit_ds_reduction_overrides_default(self):
        """When ds_reduction is explicitly set, it takes precedence over default_reduction."""
        cvs, df = _make_simple_canvas_and_df()

        # default_reduction="sum" but ds_reduction="max" — should use max
        agg, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, "max", "sum", "points")
        agg_max, _, _ = _ds_aggregate(cvs, df.copy(), "val", False, "max", "max", "points")

        np.testing.assert_array_equal(
            np.nan_to_num(agg.values, nan=0),
            np.nan_to_num(agg_max.values, nan=0),
        )


# ===========================================================================
# FIX 2: ds_reduction warns when ignored for categorical data
# ===========================================================================


class TestDsReductionWarnsCategorical:
    """When coloring by a categorical column, ``_ds_aggregate`` always uses
    ``ds.by(col, ds.count())``. A warning is now emitted when the user
    specifies a different ``ds_reduction``."""

    def test_ds_reduction_ignored_categorical(self):
        """Categorical data always uses ds.count(), by design."""
        cvs, df = _make_simple_canvas_and_df()

        agg_count, _, _ = _ds_aggregate(cvs, df.copy(), "cat", True, "count", "count", "points")
        agg_max, _, _ = _ds_aggregate(cvs, df.copy(), "cat", True, "max", "max", "points")

        # By design: categorical always uses ds.by(col, ds.count())
        np.testing.assert_array_equal(agg_count.values, agg_max.values)

    def test_all_reductions_produce_same_categorical_result(self):
        """Every reduction produces the same output for categorical data, by design."""
        cvs, df = _make_simple_canvas_and_df()

        base_agg, _, _ = _ds_aggregate(cvs, df.copy(), "cat", True, "sum", "sum", "points")
        for reduction in ["mean", "max", "min", "count", "std", "var"]:
            agg, _, _ = _ds_aggregate(cvs, df.copy(), "cat", True, reduction, reduction, "points")
            np.testing.assert_array_equal(
                agg.values, base_agg.values, err_msg=f"Categorical always uses count, but '{reduction}' differs"
            )

    def test_warning_for_ignored_reduction(self, caplog):
        """A warning is emitted when ds_reduction is set for categorical data."""
        cvs, df = _make_simple_canvas_and_df()
        with logger_warns(caplog, logger, match="ignored.*categorical"):
            _ds_aggregate(cvs, df.copy(), "cat", True, "mean", "mean", "points")

    def test_no_warning_when_ds_reduction_is_none(self, caplog):
        """No warning when ds_reduction is None (the default)."""
        cvs, df = _make_simple_canvas_and_df()
        with caplog.at_level(logging.WARNING, logger=logger.name):
            logger.addHandler(caplog.handler)
            try:
                _ds_aggregate(cvs, df.copy(), "cat", True, None, "sum", "points")
            finally:
                logger.removeHandler(caplog.handler)

        ignored_warnings = [r for r in caplog.records if "ignored" in r.message.lower()]
        assert len(ignored_warnings) == 0


# ===========================================================================
# ISSUE 3 (documented): Default "sum" + spread = value inflation
# ===========================================================================


class TestSumSpreadInflation:
    """With default reduction='sum' and spread applied, values in dense regions
    get inflated because spread(how='add') sums neighboring aggregates.
    This is documented behavior, not a bug."""

    def test_spread_inflates_sum_aggregate(self):
        """After spread(how='add'), the max value in the aggregate increases."""
        cvs, df = _make_simple_canvas_and_df(n=1000)

        agg_before = _datashader_aggregate_with_function("sum", cvs, df, "val", "points")
        max_before = float(agg_before.max())

        agg_after = ds.tf.spread(agg_before, px=3, how=_datshader_get_how_kw_for_spread("sum"))
        max_after = float(agg_after.max())

        assert max_after > max_before, (
            f"Expected spread(how='add') to inflate sum values: before={max_before}, after={max_after}"
        )

    def test_spread_does_not_inflate_max_aggregate(self):
        """With 'max' reduction and spread(how='max'), values should not increase."""
        cvs, df = _make_simple_canvas_and_df(n=1000)

        agg_before = _datashader_aggregate_with_function("max", cvs, df, "val", "points")
        max_before = float(np.nanmax(agg_before.values))

        agg_after = ds.tf.spread(agg_before, px=3, how=_datshader_get_how_kw_for_spread("max"))
        max_after = float(np.nanmax(agg_after.values))

        assert max_after <= max_before + 1e-10, (
            f"Expected max reduction to not inflate: before={max_before}, after={max_after}"
        )

    def test_sum_default_differs_from_matplotlib_visual(self):
        """Default datashader (sum) gives a wider value range than max."""
        cvs, df = _make_simple_canvas_and_df(n=1000)

        agg_sum = _datashader_aggregate_with_function("sum", cvs, df, "val", "points")
        agg_max = _datashader_aggregate_with_function("max", cvs, df, "val", "points")

        original_range = df["val"].max() - df["val"].min()
        sum_range = float(np.nanmax(agg_sum.values) - np.nanmin(agg_sum.values))
        max_range = float(np.nanmax(agg_max.values) - np.nanmin(agg_max.values))

        assert sum_range > max_range
        sum_deviation = abs(sum_range - original_range) / original_range
        max_deviation = abs(max_range - original_range) / original_range
        assert max_deviation < sum_deviation


# ===========================================================================
# FIX 3: Groups warns when used with continuous data
# ===========================================================================


class TestGroupsWarnsWithContinuous:
    """When ``color_source_vector`` is None (continuous data) and ``groups`` is set,
    a warning is now emitted via ``_warn_groups_ignored_continuous``."""

    def test_warns_when_groups_with_continuous(self, caplog):
        """A warning is emitted when groups is set but data is continuous."""
        from spatialdata_plot.pl.render import _warn_groups_ignored_continuous

        with logger_warns(caplog, logger, match="ignored.*continuous"):
            _warn_groups_ignored_continuous(["A", "B"], None, "my_value_col")

    def test_no_warning_when_categorical(self, caplog):
        """No warning when color_source_vector is present (categorical data)."""
        from spatialdata_plot.pl.render import _warn_groups_ignored_continuous

        with caplog.at_level(logging.WARNING, logger=logger.name):
            logger.addHandler(caplog.handler)
            try:
                _warn_groups_ignored_continuous(["A"], pd.Categorical(["A", "B"]), "cat_col")
            finally:
                logger.removeHandler(caplog.handler)
        assert not any("ignored" in r.message for r in caplog.records)

    def test_no_warning_when_no_groups(self, caplog):
        """No warning when groups is None."""
        from spatialdata_plot.pl.render import _warn_groups_ignored_continuous

        with caplog.at_level(logging.WARNING, logger=logger.name):
            logger.addHandler(caplog.handler)
            try:
                _warn_groups_ignored_continuous(None, None, "val")
            finally:
                logger.removeHandler(caplog.handler)
        assert not any("ignored" in r.message for r in caplog.records)


# ===========================================================================
# FIX 5: _build_datashader_color_key warns on length mismatch
# ===========================================================================


class TestColorKeyAlignment:
    """``_build_datashader_color_key`` now warns when color_vector and cat_series
    have different lengths."""

    def test_mismatched_lengths_emits_warning(self, caplog):
        """When color_vector is shorter than cat_series, a warning is logged."""
        cat = pd.Categorical(["A", "B", "C", "A", "B", "C", "A"])
        color_vector = ["#ff0000", "#00ff00", "#0000ff", "#ff0000", "#00ff00"]
        na_color = "#cccccc"

        with logger_warns(caplog, logger, match="color_vector length"):
            result = _build_datashader_color_key(cat, color_vector, na_color)

        assert "A" in result and "B" in result and "C" in result

    def test_longer_color_vector_emits_warning(self, caplog):
        """Extra colors in color_vector trigger a warning."""
        cat = pd.Categorical(["A", "B"])
        color_vector = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
        na_color = "#cccccc"

        with logger_warns(caplog, logger, match="color_vector length"):
            result = _build_datashader_color_key(cat, color_vector, na_color)

        assert len(result) == 2

    def test_category_not_seen_before_truncation_gets_na_color(self, caplog):
        """If a category only appears after the truncation point, it gets na_color."""
        cat = pd.Categorical(["A", "B", "A", "B", "A", "D"])
        color_vector = ["#ff0000", "#00ff00", "#ff0000", "#00ff00"]
        na_color = "#cccccc"

        with logger_warns(caplog, logger, match="color_vector length"):
            result = _build_datashader_color_key(cat, color_vector, na_color)

        assert result["D"] == na_color

    def test_matching_lengths_no_warning(self, caplog):
        """No warning when lengths match."""
        cat = pd.Categorical(["A", "B", "C"])
        color_vector = ["#ff0000", "#00ff00", "#0000ff"]
        na_color = "#cccccc"

        with caplog.at_level(logging.WARNING, logger=logger.name):
            logger.addHandler(caplog.handler)
            try:
                _build_datashader_color_key(cat, color_vector, na_color)
            finally:
                logger.removeHandler(caplog.handler)

        length_warnings = [r for r in caplog.records if "color_vector length" in r.message]
        assert len(length_warnings) == 0


# ===========================================================================
# FIX 6: _ds_shade_categorical only uses cmap when no color_key
# ===========================================================================


class TestCategoricalCmapFallback:
    """``_ds_shade_categorical`` now only sets cmap from color_vector when
    color_key is None (the no-color-column case)."""

    def test_cmap_not_set_when_color_key_present(self):
        """When color_key is provided, cmap is not set from color_vector."""
        cvs, df = _make_simple_canvas_and_df(n=100)
        agg = cvs.points(df, "x", "y", agg=ds.by("cat", ds.count()))
        color_key = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}

        color_vector1 = np.array(["#ff0000"] * 100)
        color_vector2 = np.array(["#0000ff"] * 100)

        shaded1 = _ds_shade_categorical(agg, color_key, color_vector1, alpha=1.0)
        shaded2 = _ds_shade_categorical(agg, color_key, color_vector2, alpha=1.0)

        img1 = np.array(shaded1) if hasattr(shaded1, "__array__") else shaded1
        img2 = np.array(shaded2) if hasattr(shaded2, "__array__") else shaded2
        np.testing.assert_array_equal(img1, img2, err_msg="color_key should override cmap for categorical data")

    def test_cmap_used_when_no_color_key(self):
        """When color_key is None (no color column), cmap is set from color_vector[0]."""
        cvs, df = _make_simple_canvas_and_df(n=100)
        agg = cvs.points(df, "x", "y", agg=ds.count())

        color_vector = np.array(["#ff0000"] * 100)
        # Should not crash — cmap is set from color_vector[0]
        shaded = _ds_shade_categorical(agg, None, color_vector, alpha=1.0)
        assert shaded is not None


# ===========================================================================
# NaN sentinel tests
# ===========================================================================


class TestNanSentinel:
    """Verify NaN sentinel injection for datashader categorical aggregation."""

    def test_nan_values_get_sentinel(self):
        series = pd.Series(pd.Categorical(["A", "B", None, "A", None]))
        result = _inject_ds_nan_sentinel(series)
        assert "ds_nan" in result.cat.categories
        assert result.isna().sum() == 0
        assert (result == "ds_nan").sum() == 2

    def test_sentinel_preserves_existing_categories(self):
        series = pd.Series(pd.Categorical(["A", "B", "C"]))
        result = _inject_ds_nan_sentinel(series)
        assert set(result.cat.categories) == {"A", "B", "C", "ds_nan"}
        assert (result == "ds_nan").sum() == 0

    def test_sentinel_idempotent(self):
        series = pd.Series(pd.Categorical(["A", None]))
        result = _inject_ds_nan_sentinel(series)
        result2 = _inject_ds_nan_sentinel(result)
        assert list(result2.cat.categories).count("ds_nan") == 1

    def test_non_categorical_input_converted(self):
        series = pd.Series(["A", "B", None, "C"])
        result = _inject_ds_nan_sentinel(series)
        assert isinstance(result.dtype, pd.CategoricalDtype)
        assert "ds_nan" in result.cat.categories


# ===========================================================================
# Comprehensive reduction function tests
# ===========================================================================


class TestReductionFunctions:
    """Test that all reduction functions work correctly with datashader aggregation."""

    @pytest.mark.parametrize("reduction", ["sum", "mean", "max", "min", "count", "std", "var", "any"])
    def test_reduction_produces_valid_output(self, reduction):
        cvs, df = _make_simple_canvas_and_df(n=500)
        agg = _datashader_aggregate_with_function(reduction, cvs, df, "val", "points")
        assert agg.shape == (50, 50)
        assert not np.all(np.isnan(agg.values))

    @pytest.mark.parametrize("reduction", ["sum", "mean", "max", "min", "count", "std", "var", "any"])
    def test_spread_how_mapping_exists(self, reduction):
        how = _datshader_get_how_kw_for_spread(reduction)
        assert isinstance(how, str) and len(how) > 0

    def test_none_reduction_defaults_to_sum(self):
        cvs, df = _make_simple_canvas_and_df()
        agg_none = _datashader_aggregate_with_function(None, cvs, df, "val", "points")
        agg_sum = _datashader_aggregate_with_function("sum", cvs, df, "val", "points")
        np.testing.assert_array_equal(agg_none.values, agg_sum.values)

    def test_invalid_reduction_raises(self):
        cvs, df = _make_simple_canvas_and_df()
        with pytest.raises(ValueError, match="not supported"):
            _datashader_aggregate_with_function("invalid", cvs, df, "val", "points")

    def test_any_reduction_returns_binary(self):
        cvs, df = _make_simple_canvas_and_df(n=500)
        agg = _datashader_aggregate_with_function("any", cvs, df, "val", "points")
        non_nan = agg.values[~np.isnan(agg.values)]
        assert np.all(non_nan == 1.0)

    def test_count_counts_nonnan(self):
        cvs, df = _make_simple_canvas_and_df(n=500)
        agg = _datashader_aggregate_with_function("count", cvs, df, "val", "points")
        non_nan = agg.values[~np.isnan(agg.values)]
        assert np.all(non_nan >= 0)
        assert np.all(non_nan == np.floor(non_nan))


# ===========================================================================
# _coerce_categorical_source edge cases
# ===========================================================================


class TestCoerceCategoricalSource:
    def test_pandas_categorical_passthrough(self):
        series = pd.Series(pd.Categorical(["A", "B", "C"]))
        result = _coerce_categorical_source(series)
        assert isinstance(result, pd.Categorical)
        assert list(result.categories) == ["A", "B", "C"]

    def test_string_series_converted(self):
        series = pd.Series(["A", "B", "C"])
        result = _coerce_categorical_source(series)
        assert isinstance(result, pd.Categorical)

    def test_dask_series_computed(self):
        import dask.dataframe as dd

        pdf = pd.DataFrame({"cat": pd.Categorical(["A", "B", "C"])})
        ddf = dd.from_pandas(pdf, npartitions=1)
        result = _coerce_categorical_source(ddf["cat"])
        assert isinstance(result, pd.Categorical)


# ===========================================================================
# Datashader vs matplotlib value range comparison
# ===========================================================================


class TestDatashaderVsMatplotlib:
    def test_max_preserves_data_range(self):
        cvs, df = _make_simple_canvas_and_df(n=1000)
        agg_max = _datashader_aggregate_with_function("max", cvs, df, "val", "points")
        agg_max_range = float(np.nanmax(agg_max.values) - np.nanmin(agg_max.values))
        original_range = float(df["val"].max() - df["val"].min())
        ratio = agg_max_range / original_range
        assert 0.8 < ratio <= 1.0

    def test_sum_exceeds_data_range(self):
        cvs, df = _make_simple_canvas_and_df(n=2000)
        agg_sum = _datashader_aggregate_with_function("sum", cvs, df, "val", "points")
        agg_sum_range = float(np.nanmax(agg_sum.values) - np.nanmin(agg_sum.values))
        original_range = float(df["val"].max() - df["val"].min())
        assert agg_sum_range > original_range


# ===========================================================================
# _build_color_key dispatch tests
# ===========================================================================


class TestBuildColorKey:
    def test_returns_none_for_non_categorical(self):
        cvs, df = _make_simple_canvas_and_df()
        result = _build_color_key(df, "val", False, ["#ff0000"] * len(df), "#cccccc")
        assert result is None

    def test_returns_none_when_no_color_column(self):
        result = _build_color_key(pd.DataFrame(), None, True, [], "#cccccc")
        assert result is None

    def test_returns_dict_for_categorical(self):
        cvs, df = _make_simple_canvas_and_df(n=100)
        colors = (["#ff0000", "#00ff00", "#0000ff"] * 34)[:100]
        result = _build_color_key(df, "cat", True, colors, "#cccccc")
        assert isinstance(result, dict)
        assert "A" in result and "B" in result and "C" in result


# ===========================================================================
# _hex_no_alpha edge cases
# ===========================================================================


class TestHexNoAlpha:
    def test_strips_alpha(self):
        assert _hex_no_alpha("#ff0000ff") == "#ff0000"

    def test_no_alpha_unchanged(self):
        assert _hex_no_alpha("#ff0000") == "#ff0000"

    def test_short_hex_raises(self):
        with pytest.raises(ValueError, match="Invalid hex color length"):
            _hex_no_alpha("#fff")
