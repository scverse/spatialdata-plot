import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from anndata import AnnData
from scipy.sparse import lil_matrix
from scipy.spatial import KDTree
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.datasets import blobs
from spatialdata.models import ShapesModel, TableModel

import spatialdata_plot  # noqa: F401
from tests.conftest import DPI, PlotTester, PlotTesterMeta, get_standard_RNG

sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=DPI, color_map="viridis")
matplotlib.use("agg")
_ = spatialdata_plot


def _make_sdata_with_graph_on_shapes() -> SpatialData:
    """Create SpatialData with shapes, an annotating table, and a spatial connectivity graph in obsp."""
    rng = get_standard_RNG()
    n = 20

    # Shapes at reproducible positions
    coords = rng.uniform(10, 90, size=(n, 2))
    gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in coords],
        data={"radius": np.ones(n) * 2.5},
    )
    shapes = ShapesModel.parse(gdf)

    # Table annotating the shapes
    adata = AnnData(rng.normal(size=(n, 5)))
    adata.obs["instance_id"] = np.arange(n)
    adata.obs["region"] = "my_shapes"
    adata.obs["cell_type"] = pd.Categorical(rng.choice(["tumor", "immune", "stroma"], size=n))

    # Build KNN spatial graph (k=3 neighbors)
    tree = KDTree(coords)
    adj = lil_matrix((n, n))
    for i in range(n):
        _, indices = tree.query(coords[i], k=4)  # self + 3 neighbors
        for j in indices[1:]:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    adata.obsp["spatial_connectivities"] = adj.tocsr()

    table = TableModel.parse(adata, region="my_shapes", region_key="region", instance_key="instance_id")
    return SpatialData(shapes={"my_shapes": shapes}, tables={"table": table})


def _make_sdata_with_graph_on_labels() -> SpatialData:
    """Create SpatialData based on blobs with a spatial graph connecting label regions."""
    blob = blobs()
    table = blob["table"]
    n = table.n_obs

    # Compute label centroids to build a spatially meaningful graph
    centroids_df = sd.get_centroids(blob["blobs_labels"]).compute()
    instance_ids = table.obs["instance_id"].values.astype(int)

    # Align centroids to table instance order
    # centroids_df index corresponds to label IDs (excluding background 0)
    centroid_coords = np.column_stack([centroids_df["x"].values, centroids_df["y"].values])

    # Map table instance_ids to centroid positions
    # centroids_df is indexed 0..len-1, label IDs are in the index implicitly
    # We need to match table's instance_ids to the centroid rows
    if hasattr(centroids_df.index, "values"):
        label_ids_in_centroids = centroids_df.index.values
    else:
        label_ids_in_centroids = np.arange(len(centroids_df))

    # Build lookup: label_id -> row index in centroids
    id_to_row = {lid: row for row, lid in enumerate(label_ids_in_centroids)}

    # Only include table obs that have centroids
    valid_mask = np.array([iid in id_to_row for iid in instance_ids])
    valid_indices = np.where(valid_mask)[0]
    valid_coords = np.array([centroid_coords[id_to_row[instance_ids[i]]] for i in valid_indices])

    # Build KNN graph over valid observations
    adj = lil_matrix((n, n))
    if len(valid_coords) > 1:
        tree = KDTree(valid_coords)
        k = min(4, len(valid_coords))
        for idx_in_valid, i in enumerate(valid_indices):
            _, neighbors = tree.query(valid_coords[idx_in_valid], k=k)
            for nb in neighbors[1:]:
                j = valid_indices[nb]
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    table.obsp["spatial_connectivities"] = adj.tocsr()

    rng = get_standard_RNG()
    table.obs["cell_type"] = pd.Categorical(rng.choice(["tumor", "immune", "stroma"], size=n))

    return blob


class TestGraph(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_render_graph_on_shapes(self):
        """Basic graph rendering: edges overlaid on shapes."""
        sdata = _make_sdata_with_graph_on_shapes()
        (
            sdata.pl.render_graph(
                "my_shapes",
                connectivity_key="spatial",
                table_name="table",
            )
            .pl.render_shapes("my_shapes")
            .pl.show()
        )

    def test_plot_can_render_graph_on_labels(self):
        """Graph overlay on label segmentation with background image — most common real-world use case."""
        sdata = _make_sdata_with_graph_on_labels()
        (
            sdata.pl.render_images("blobs_image")
            .pl.render_graph(
                "blobs_labels",
                connectivity_key="spatial",
                table_name="table",
                edge_alpha=0.5,
            )
            .pl.render_labels("blobs_labels")
            .pl.show()
        )

    def test_plot_can_render_graph_with_groups_filter(self):
        """Graph filtered to show only edges between 'tumor' cells."""
        sdata = _make_sdata_with_graph_on_shapes()
        (
            sdata.pl.render_graph(
                "my_shapes",
                connectivity_key="spatial",
                table_name="table",
                group_key="cell_type",
                groups=["tumor"],
            )
            .pl.render_shapes("my_shapes", color="cell_type")
            .pl.show()
        )
