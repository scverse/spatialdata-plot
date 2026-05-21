"""Tests for the in-memory commit policy."""

from __future__ import annotations

import geopandas as gpd
import spatialdata as sd
from shapely.geometry import Polygon
from spatialdata.models import ShapesModel
from spatialdata.transformations.transformations import Identity

from spatialdata_plot.pl.interactive._persist import commit_to_memory


def _make_shape() -> ShapesModel:
    gdf = gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]})
    return ShapesModel.parse(gdf, transformations={"global": Identity()})


def test_commit_to_memory_stores_under_name():
    sdata = sd.SpatialData()
    target = commit_to_memory(sdata, _make_shape(), "tumor_region")
    assert target == "tumor_region"
    assert "tumor_region" in sdata.shapes


def test_commit_to_memory_overwrites_on_collision():
    sdata = sd.SpatialData()
    first = _make_shape()
    sdata.shapes["tumor_region"] = first
    second = _make_shape()
    target = commit_to_memory(sdata, second, "tumor_region")
    assert target == "tumor_region"
    assert sdata.shapes["tumor_region"] is second
