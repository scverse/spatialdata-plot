"""Structural tests for the ``*RenderParams`` dataclass hierarchy.

Locks the ``RenderParams`` base-class refactor: the five renderers share one base holding only the
universal fields, and each subclass must keep exactly the field set (and the universal defaults) it
had before the reparent. A dropped field or a flipped default is invisible to the image-baseline
suite (CI-only), so it is asserted directly here.
"""

import dataclasses

import pytest

from spatialdata_plot.pl.render_params import (
    GraphRenderParams,
    ImageRenderParams,
    LabelsRenderParams,
    PointsRenderParams,
    RenderParams,
    ShapesRenderParams,
)

ALL_SUBCLASSES = [
    ShapesRenderParams,
    PointsRenderParams,
    ImageRenderParams,
    LabelsRenderParams,
    GraphRenderParams,
]

# The only fields with identical type+default across all five renderers; everything else lives on
# the subclasses. ``element`` is required; the other three carry their canonical defaults.
UNIVERSAL_FIELDS = {"element", "zorder", "colorbar", "colorbar_params"}

# Frozen per-subclass field-name snapshot (captured from the pre-refactor dataclasses). The set must
# stay identical through the reparent; adding/removing a renderer field is a deliberate change that
# updates this map.
EXPECTED_FIELD_NAMES = {
    ShapesRenderParams: {
        "cmap_params",
        "outline_params",
        "element",
        "color",
        "col_for_color",
        "col_for_outline_color",
        "outline_table_name",
        "groups",
        "palette",
        "outline_alpha",
        "fill_alpha",
        "scale",
        "transfunc",
        "method",
        "zorder",
        "table_name",
        "table_layer",
        "shape",
        "as_points",
        "size",
        "ds_reduction",
        "colorbar",
        "colorbar_params",
        "panel_key",
    },
    PointsRenderParams: {
        "cmap_params",
        "element",
        "color",
        "col_for_color",
        "groups",
        "palette",
        "alpha",
        "size",
        "transfunc",
        "method",
        "zorder",
        "table_name",
        "table_layer",
        "ds_reduction",
        "colorbar",
        "colorbar_params",
        "density",
        "density_how",
    },
    ImageRenderParams: {
        "cmap_params",
        "element",
        "channel",
        "palette",
        "alpha",
        "scale",
        "zorder",
        "colorbar",
        "colorbar_params",
        "transfunc",
        "grayscale",
        "channels_as_legend",
        "method",
        "ds_reduction",
    },
    LabelsRenderParams: {
        "cmap_params",
        "element",
        "color",
        "col_for_color",
        "col_for_outline_color",
        "outline_table_name",
        "groups",
        "contour_px",
        "palette",
        "outline_alpha",
        "outline_color",
        "fill_alpha",
        "scale",
        "table_name",
        "table_layer",
        "transfunc",
        "zorder",
        "colorbar",
        "colorbar_params",
        "as_points",
        "size",
        "method",
        "panel_key",
    },
    GraphRenderParams: {
        "element",
        "connectivity_obsp_key",
        "table_name",
        "color",
        "obs_col",
        "obsp_key",
        "cmap_params",
        "palette_map",
        "na_color",
        "color_source",
        "groups",
        "group_key",
        "edge_width",
        "edge_alpha",
        "weight_key",
        "linestyle",
        "rasterize",
        "include_self_loops",
        "zorder",
        "colorbar",
        "colorbar_params",
    },
}


@pytest.mark.parametrize("cls", ALL_SUBCLASSES)
def test_is_renderparams_subclass(cls):
    assert issubclass(cls, RenderParams)


def test_base_holds_only_universal_fields():
    assert {f.name for f in dataclasses.fields(RenderParams)} == UNIVERSAL_FIELDS


def test_base_universal_defaults():
    defaults = {f.name: f.default for f in dataclasses.fields(RenderParams)}
    assert defaults["element"] is dataclasses.MISSING  # required
    assert defaults["zorder"] == 0
    assert defaults["colorbar"] == "auto"
    assert defaults["colorbar_params"] is None


@pytest.mark.parametrize("cls", ALL_SUBCLASSES)
def test_field_names_preserved(cls):
    assert {f.name for f in dataclasses.fields(cls)} == EXPECTED_FIELD_NAMES[cls]


@pytest.mark.parametrize("cls", ALL_SUBCLASSES)
def test_universal_fields_inherited_with_defaults(cls):
    names = {f.name for f in dataclasses.fields(cls)}
    assert names >= UNIVERSAL_FIELDS
    defaults = {f.name: f.default for f in dataclasses.fields(cls)}
    assert defaults["zorder"] == 0
    assert defaults["colorbar"] == "auto"
    assert defaults["colorbar_params"] is None


def test_keyword_construction_roundtrip():
    # All construction in basic.py is keyword-only; kw_only=True must not break it, and the inherited
    # universal fields must round-trip through the subclass constructor.
    p = GraphRenderParams(element="graph", zorder=3, colorbar=False)
    assert p.element == "graph"
    assert p.zorder == 3
    assert p.colorbar is False
    assert p.colorbar_params is None  # inherited default
