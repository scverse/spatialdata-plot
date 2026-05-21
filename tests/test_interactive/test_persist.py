"""Tests for the in-memory commit + zarr write policy."""
from __future__ import annotations

import geopandas as gpd
import pytest
import spatialdata as sd
from shapely.geometry import Polygon
from spatialdata.models import ShapesModel

from spatialdata_plot.pl.interactive._persist import commit_to_memory, persist_to_disk


def _make_sdata(tmp_path=None) -> sd.SpatialData:
    sdata = sd.SpatialData()
    if tmp_path is not None:
        sdata.write(tmp_path / "test.zarr")
        sdata = sd.read_zarr(tmp_path / "test.zarr")
    return sdata


def _make_shape() -> ShapesModel:
    gdf = gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]})
    return ShapesModel.parse(gdf, transformations={"global": sd.transformations.Identity()})


def test_commit_to_memory_stores_under_name():
    sdata = _make_sdata()
    target = commit_to_memory(sdata, _make_shape(), "tumor_region")
    assert target == "tumor_region"
    assert "tumor_region" in sdata.shapes


def test_commit_to_memory_renames_on_collision():
    sdata = _make_sdata()
    sdata.shapes["tumor_region"] = _make_shape()
    target = commit_to_memory(sdata, _make_shape(), "tumor_region")
    # Original preserved, new one gets a timestamp suffix.
    assert "tumor_region" in sdata.shapes
    assert target.startswith("tumor_region_")
    assert target != "tumor_region"
    assert target in sdata.shapes


def test_persist_raises_when_not_zarr_backed():
    sdata = _make_sdata()
    sdata.shapes["foo"] = _make_shape()
    with pytest.raises(ValueError, match="not zarr-backed"):
        persist_to_disk(sdata, "foo")


def test_persist_writes_to_zarr(tmp_path):
    sdata = _make_sdata(tmp_path=tmp_path)
    sdata.shapes["foo"] = _make_shape()
    persist_to_disk(sdata, "foo")
    # Re-read from disk and check the element survives.
    sdata2 = sd.read_zarr(tmp_path / "test.zarr")
    assert "foo" in sdata2.shapes
