import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import spatialdata as sd
from anndata import AnnData
from matplotlib.collections import CircleCollection, LineCollection
from scipy.sparse import csr_matrix, lil_matrix, triu
from scipy.spatial import KDTree
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.datasets import blobs
from spatialdata.models import ShapesModel, TableModel

import spatialdata_plot  # noqa: F401
from spatialdata_plot._logging import logger, logger_warns
from tests.conftest import DPI, PlotTester, PlotTesterMeta, get_standard_RNG

sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=DPI, color_map="viridis")
matplotlib.use("agg")
_ = spatialdata_plot


def _knn_adjacency(coords: np.ndarray, k: int = 3) -> csr_matrix:
    n = len(coords)
    adj = lil_matrix((n, n))
    tree = KDTree(coords)
    for i in range(n):
        _, neighbors = tree.query(coords[i], k=min(k + 1, n))
        for j in neighbors[1:]:
            adj[i, j] = adj[j, i] = 1.0
    return adj.tocsr()


def _sdata_with_graph_on_shapes() -> SpatialData:
    rng = get_standard_RNG()
    n = 20
    coords = rng.uniform(10, 90, size=(n, 2))
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in coords], data={"radius": np.ones(n) * 2.5})
    )
    adata = AnnData(rng.normal(size=(n, 5)))
    adata.obs["instance_id"] = np.arange(n)
    adata.obs["region"] = "my_shapes"
    adata.obs["cell_type"] = pd.Categorical(rng.choice(["tumor", "immune", "stroma"], size=n))
    adata.obsp["spatial_connectivities"] = _knn_adjacency(coords, k=3)
    table = TableModel.parse(adata, region="my_shapes", region_key="region", instance_key="instance_id")
    return SpatialData(shapes={"my_shapes": shapes}, tables={"table": table})


def _sdata_with_graph_on_labels() -> SpatialData:
    sdata = blobs()
    table = sdata["table"]
    centroids = sd.get_centroids(sdata["blobs_labels"]).compute()
    coords = centroids.loc[table.obs["instance_id"].values, ["x", "y"]].to_numpy()
    table.obsp["spatial_connectivities"] = _knn_adjacency(coords, k=3)
    return sdata


def _sdata_with_weighted_graph() -> SpatialData:
    sdata = _sdata_with_graph_on_shapes()
    adj = sdata["table"].obsp["spatial_connectivities"].copy().astype(float).tolil()
    rng = get_standard_RNG()
    for r, c in zip(*adj.nonzero(), strict=True):
        adj[r, c] = float(rng.uniform(0.1, 5.0))
    sdata["table"].obsp["spatial_distances"] = adj.tocsr()
    return sdata


class TestGraph(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_graph_on_shapes(self):
        sdata = _sdata_with_graph_on_shapes()
        sdata.pl.render_graph("my_shapes", table_name="table").pl.render_shapes("my_shapes").pl.show()

    def test_plot_can_render_graph_on_labels(self):
        sdata = _sdata_with_graph_on_labels()
        (
            sdata.pl.render_images("blobs_image")
            .pl.render_graph("blobs_labels", table_name="table", edge_alpha=0.5)
            .pl.render_labels("blobs_labels")
            .pl.show()
        )

    def test_plot_can_render_graph_with_groups_filter(self):
        sdata = _sdata_with_graph_on_shapes()
        (
            sdata.pl.render_graph("my_shapes", table_name="table", group_key="cell_type", groups=["tumor"])
            .pl.render_shapes("my_shapes", color="cell_type")
            .pl.show()
        )


def test_render_graph_empty_graph_does_not_error():
    sdata = _sdata_with_graph_on_shapes()
    sdata["table"].obsp["spatial_connectivities"] = csr_matrix((20, 20))
    sdata.pl.render_graph("my_shapes", table_name="table").pl.render_shapes("my_shapes").pl.show()


def test_render_graph_auto_discovers_element_and_table():
    sdata = _sdata_with_graph_on_shapes()
    step_key, params = next(iter(sdata.pl.render_graph().plotting_tree.items()))
    assert step_key.endswith("_render_graph")
    assert params.element == "my_shapes" and params.table_name == "table"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"connectivity_key": "nonexistent"}, "not found in `table.obsp`"),
        ({"element": "no_such_element"}, "not found in shapes, points, or labels"),
        ({"groups": ["tumor"]}, "`groups` requires `group_key`"),
        ({"color": None, "obsp_key": "no_such"}, "`obsp_key='no_such'` not found"),
        ({"edge_width": "weight", "weight_key": "no_such"}, "`weight_key='no_such'` not found"),
    ],
)
def test_render_graph_error_paths(kwargs, match):
    sdata = _sdata_with_graph_on_shapes()
    element = kwargs.pop("element", "my_shapes")
    with pytest.raises((KeyError, ValueError), match=match):
        sdata.pl.render_graph(element, table_name="table", **kwargs)


def test_render_graph_rejects_color_and_obsp_key_together():
    sdata = _sdata_with_weighted_graph()
    with pytest.raises(ValueError, match="Cannot set both `color` and `obsp_key`"):
        sdata.pl.render_graph("my_shapes", table_name="table", color="red", obsp_key="spatial_distances")


def test_render_graph_warns_on_groups_not_in_column(caplog):
    sdata = _sdata_with_graph_on_shapes()
    with logger_warns(caplog, logger, match="not_a_real_group"):
        sdata.pl.render_graph("my_shapes", table_name="table", group_key="cell_type", groups=["not_a_real_group"])


def test_render_graph_raises_on_table_without_region_key():
    sdata = _sdata_with_graph_on_shapes()
    sdata["table"].uns["spatialdata_attrs"]["region_key"] = None
    with pytest.raises(ValueError, match="has no `region_key`"):
        sdata.pl.render_graph("my_shapes", table_name="table")


def test_render_graph_obsp_key_populates_edge_values_from_matrix():
    """Edge color array must equal the obsp matrix entries for the rendered edges."""
    sdata = _sdata_with_weighted_graph()
    fig, ax = plt.subplots()
    (
        sdata.pl.render_shapes("my_shapes")
        .pl.render_graph("my_shapes", table_name="table", color=None, obsp_key="spatial_distances", cmap="viridis")
        .pl.show(ax=ax)
    )
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    distances = sdata["table"].obsp["spatial_distances"]
    rows, cols = triu(distances, k=1).nonzero()
    np.testing.assert_allclose(np.asarray(lc.get_array()), distances[rows, cols].A1)
    plt.close(fig)


def test_render_graph_color_by_obs_categorical_with_palette_dict():
    """Same-category edges get the palette colour; cross-category edges get na_color."""
    sdata = _sdata_with_graph_on_shapes()
    palette = {"tumor": "#ff0000", "immune": "#00ff00", "stroma": "#0000ff"}
    fig, ax = plt.subplots()
    (
        sdata.pl.render_shapes("my_shapes")
        .pl.render_graph("my_shapes", table_name="table", color="cell_type", palette=palette, na_color="#888888")
        .pl.show(ax=ax)
    )
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    allowed = {tuple(matplotlib.colors.to_rgba(v)) for v in palette.values()}
    allowed.add(tuple(matplotlib.colors.to_rgba("#888888")))
    for c in lc.get_colors():
        assert tuple(c) in allowed
    plt.close(fig)


def test_render_graph_color_by_obs_continuous_uses_endpoint_mean():
    sdata = _sdata_with_graph_on_shapes()
    rng = get_standard_RNG()
    scores = rng.uniform(0.0, 1.0, size=sdata["table"].n_obs)
    sdata["table"].obs["score"] = scores
    fig, ax = plt.subplots()
    (
        sdata.pl.render_shapes("my_shapes")
        .pl.render_graph("my_shapes", table_name="table", color="score", cmap="magma")
        .pl.show(ax=ax)
    )
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    rows, cols = triu(sdata["table"].obsp["spatial_connectivities"], k=1).nonzero()
    np.testing.assert_allclose(np.asarray(lc.get_array()), 0.5 * (scores[rows] + scores[cols]))
    plt.close(fig)


def test_render_graph_draws_colorbar_for_continuous_coloring():
    sdata = _sdata_with_weighted_graph()
    fig, ax = plt.subplots()
    (
        sdata.pl.render_shapes("my_shapes")
        .pl.render_graph("my_shapes", table_name="table", color=None, obsp_key="spatial_distances", cmap="viridis")
        .pl.show(ax=ax)
    )
    cbars = [c for c in fig.get_children() if isinstance(c, matplotlib.axes.Axes) and c is not ax]
    assert cbars
    plt.close(fig)


def test_render_graph_colorbar_can_be_disabled():
    sdata = _sdata_with_weighted_graph()
    fig, ax = plt.subplots()
    (
        sdata.pl.render_shapes("my_shapes")
        .pl.render_graph("my_shapes", table_name="table", color=None, obsp_key="spatial_distances", colorbar=False)
        .pl.show(ax=ax)
    )
    cbars = [
        c
        for c in fig.get_children()
        if isinstance(c, matplotlib.axes.Axes) and c is not ax and c.get_ylim() != (0.0, 1.0)
    ]
    assert not cbars
    plt.close(fig)


def test_render_graph_edge_width_by_weight_produces_normalised_array():
    sdata = _sdata_with_weighted_graph()
    fig, ax = plt.subplots()
    (
        sdata.pl.render_shapes("my_shapes")
        .pl.render_graph("my_shapes", table_name="table", edge_width="weight", weight_key="spatial_distances")
        .pl.show(ax=ax)
    )
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    widths = lc.get_linewidths()
    assert len(widths) > 1
    assert 0.5 - 1e-6 <= float(np.min(widths)) < float(np.max(widths)) <= 3.0 + 1e-6
    plt.close(fig)


@pytest.mark.parametrize("include_self_loops", [True, False])
def test_render_graph_include_self_loops(include_self_loops):
    sdata = _sdata_with_graph_on_shapes()
    adj = sdata["table"].obsp["spatial_connectivities"].tolil()
    for i in range(sdata["table"].n_obs):
        adj[i, i] = 1.0
    sdata["table"].obsp["spatial_connectivities"] = adj.tocsr()

    fig, ax = plt.subplots()
    (
        sdata.pl.render_shapes("my_shapes")
        .pl.render_graph("my_shapes", table_name="table", include_self_loops=include_self_loops)
        .pl.show(ax=ax)
    )
    has_circles = any(isinstance(c, CircleCollection) for c in ax.collections)
    assert has_circles is include_self_loops
    plt.close(fig)
