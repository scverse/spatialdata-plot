"""Coloring by the values of a MuData table, for SpatialData versions supporting multimodal tables."""

import anndata as ad
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import Normalize, to_rgb
from shapely import Point
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

import spatialdata_plot  # noqa: F401
from spatialdata_plot.pl.utils import _table_has_key

md = pytest.importorskip("mudata")


def _spatialdata_supports_mudata() -> bool:
    try:
        TableModel.validate(md.MuData({"rna": ad.AnnData(np.ones((1, 1)))}))
    except TypeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _spatialdata_supports_mudata(), reason="SpatialData does not support MuData tables")

N_CELLS = 6
N_PROT_CELLS = 4


@pytest.fixture
def sdata_mudata() -> SpatialData:
    obs_names = [f"cell_{i}" for i in range(N_CELLS)]
    rna = ad.AnnData(np.arange(2 * N_CELLS, dtype=float).reshape(N_CELLS, 2), obs=pd.DataFrame(index=obs_names))
    rna.var_names = ["CD45", "Gene1"]
    rna.obs["leiden"] = pd.Categorical(["a", "b"] * (N_CELLS // 2))
    prot = ad.AnnData(
        100 + np.arange(2 * N_PROT_CELLS, dtype=float).reshape(N_PROT_CELLS, 2),
        obs=pd.DataFrame(index=obs_names[:N_PROT_CELLS]),
    )
    prot.var_names = ["CD45", "CD68"]
    mdata = md.MuData({"rna": rna, "prot": prot})
    mdata.obs["region"] = pd.Categorical(["cells"] * N_CELLS)
    mdata.obs["instance_id"] = np.arange(N_CELLS)
    mdata.obs["score"] = np.linspace(0, 1, N_CELLS)
    mdata = TableModel.parse(mdata, region="cells", region_key="region", instance_key="instance_id")
    gdf = gpd.GeoDataFrame(geometry=[Point(i, i) for i in range(N_CELLS)], index=np.arange(N_CELLS))
    gdf["radius"] = 0.4
    return SpatialData(shapes={"cells": ShapesModel.parse(gdf)}, tables={"table": mdata})


def test_table_has_key(sdata_mudata: SpatialData):
    table = sdata_mudata["table"]
    assert _table_has_key(table, "score")
    assert _table_has_key(table, "prot:CD68")
    assert _table_has_key(table, "CD68")
    assert _table_has_key(table, "rna:leiden")
    assert not _table_has_key(table, "rna:CD68")
    assert not _table_has_key(table, "not_a_key")


def _plotted_colors(sdata: SpatialData, color: str) -> np.ndarray:
    fig, ax = plt.subplots()
    sdata.pl.render_shapes("cells", color=color, table_name="table").pl.show(ax=ax)
    colors = ax.collections[0].get_facecolor()[:, :3]
    plt.close(fig)
    return colors


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ("score", np.linspace(0, 1, N_CELLS)),
        ("rna:CD45", np.arange(0, 2 * N_CELLS, 2, dtype=float)),
        # only measured in the first N_PROT_CELLS cells: the other cells get the na_color
        ("prot:CD45", [100, 102, 104, 106, np.nan, np.nan]),
        ("CD68", [101, 103, 105, 107, np.nan, np.nan]),
    ],
)
def test_render_shapes_colored_by_mudata_values(sdata_mudata: SpatialData, color: str, expected: list[float]):
    values = np.asarray(expected, dtype=float)
    measured = ~np.isnan(values)
    norm = Normalize(vmin=values[measured].min(), vmax=values[measured].max())
    expected_colors = plt.get_cmap("viridis")(norm(values[measured]))[:, :3]
    colors = _plotted_colors(sdata_mudata, color)
    np.testing.assert_allclose(colors[measured], expected_colors, atol=1e-3)
    np.testing.assert_allclose(colors[~measured], np.broadcast_to(to_rgb("lightgray"), colors[~measured].shape))


def test_render_shapes_colored_by_modality_category(sdata_mudata: SpatialData):
    fig, ax = plt.subplots()
    sdata_mudata.pl.render_shapes("cells", color="rna:leiden", table_name="table").pl.show(ax=ax)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert legend_labels == ["a", "b"]


def test_ambiguous_variable_requires_modality_prefix(sdata_mudata: SpatialData):
    with pytest.raises(ValueError, match="prefix it with the name of the modality"):
        sdata_mudata.pl.render_shapes("cells", color="CD45", table_name="table").pl.show()
