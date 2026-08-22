"""PROTOTYPE: extend the ``.pl`` accessor to plain AnnData objects.

Vision
------
``spatialdata_plot`` injects a ``pl`` accessor onto ``spatialdata.SpatialData`` by
hijacking xarray's private ``_register_accessor`` (see ``_accessor.py``). Because
``AnnData`` is an ordinary Python class with no ``__slots__``, the *exact same*
mechanism attaches a ``pl`` accessor to ``AnnData`` too.

An AnnData that carries per-cell coordinates in ``obsm["spatial"]`` (the
squidpy / scanpy convention, an ``(n_obs, 2)`` array) already holds everything a
points layer needs. So::

    import spatialdata_plot
    spatialdata_plot.enable_anndata_accessor()

    adata.pl.render_points(color="cell_type").pl.show()

works by adapting the AnnData into a one-element SpatialData (a *points* element
built from ``obsm["spatial"]`` plus a table that annotates it) and delegating to
the existing SpatialData plotting engine. Nothing in the renderer is duplicated:
the AnnData path is a thin front-door that reuses ``render_points`` and ``show``.

This is a prototype for design discussion, not a shipped API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from xarray.core.extensions import _register_accessor

# The AnnData attribute where the SpatialData adapter is memoized, so repeated
# ``adata.pl`` access and chained calls reuse one conversion.
_ADAPTER_CACHE_ATTR = "_spatialdata_plot_adapter"

# squidpy/scanpy convention for per-cell coordinates: obsm["spatial"], (n_obs, 2).
_DEFAULT_COORDS_KEY = "spatial"
_ELEMENT_NAME = "anndata_points"
_TABLE_NAME = "table"
_REGION_KEY = "region"
_INSTANCE_KEY = "instance_id"


def register_anndata_accessor(name: str) -> type:
    """Register an accessor on :class:`anndata.AnnData`.

    Mirror of :func:`spatialdata_plot._accessor.register_spatial_data_accessor`,
    pointing xarray's ``_register_accessor`` at ``AnnData`` instead of
    ``SpatialData``. Works because ``AnnData`` allows arbitrary instance
    attributes, so the cached-accessor descriptor can memoize on the instance.
    """
    return _register_accessor(name, AnnData)


def anndata_to_spatialdata(
    adata: AnnData,
    *,
    coords_key: str = _DEFAULT_COORDS_KEY,
    element_name: str = _ELEMENT_NAME,
) -> Any:
    """Adapt an AnnData with ``obsm[coords_key]`` into a one-points SpatialData.

    The coordinates become a ``PointsModel`` element and the AnnData itself
    becomes a ``TableModel`` that annotates that element, so coloring by
    ``obs`` columns *and* by genes (``var`` / ``X`` / layers) flows through the
    unmodified SpatialData color machinery.
    """
    # Imported lazily so importing this module never forces the (heavy)
    # spatialdata import unless the adapter is actually used.
    from spatialdata import SpatialData
    from spatialdata.models import PointsModel, TableModel

    if coords_key not in adata.obsm:
        raise KeyError(
            f"AnnData.obsm[{coords_key!r}] not found; cannot build a points layer. "
            f"Available obsm keys: {list(adata.obsm)}"
        )
    coords = np.asarray(adata.obsm[coords_key])
    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2:
        raise ValueError(
            f"AnnData.obsm[{coords_key!r}] must be an (n_obs, >=2) array; got shape {coords.shape}."
        )

    n = adata.n_obs
    instance_ids = np.arange(n, dtype=np.int64)
    points_df = pd.DataFrame(
        {
            "x": np.asarray(coords[:, 0], dtype=np.float64),
            "y": np.asarray(coords[:, 1], dtype=np.float64),
            _INSTANCE_KEY: instance_ids,
        }
    )
    points = PointsModel.parse(points_df, coordinates={"x": "x", "y": "y"})

    # Build a table that annotates the points element. Copy so we do not mutate
    # the caller's AnnData with region/instance bookkeeping columns.
    table = adata.copy()
    table.obs[_REGION_KEY] = pd.Categorical([element_name] * n)
    table.obs[_INSTANCE_KEY] = instance_ids
    table = TableModel.parse(
        table,
        region=element_name,
        region_key=_REGION_KEY,
        instance_key=_INSTANCE_KEY,
    )

    return SpatialData(points={element_name: points}, tables={_TABLE_NAME: table})


@register_anndata_accessor("pl")
class AnnDataPlotAccessor:
    """Prototype ``.pl`` front-door for plain AnnData objects.

    Holds the AnnData, lazily converts it to a SpatialData adapter (memoized on
    the AnnData instance), and forwards the plotting entry points to the real
    :class:`~spatialdata_plot.pl.basic.PlotAccessor`. Because ``render_*``
    returns a *SpatialData*, every subsequent ``.pl`` in a chain is already the
    native SpatialData accessor — this class only bootstraps the first hop.
    """

    def __init__(self, adata: AnnData) -> None:
        self._adata = adata

    # -- adapter plumbing --------------------------------------------------
    def _sdata(self, coords_key: str = _DEFAULT_COORDS_KEY) -> Any:
        cached = getattr(self._adata, _ADAPTER_CACHE_ATTR, None)
        if cached is None:
            cached = anndata_to_spatialdata(self._adata, coords_key=coords_key)
            setattr(self._adata, _ADAPTER_CACHE_ATTR, cached)
        return cached

    @property
    def element_name(self) -> str:
        return _ELEMENT_NAME

    def to_spatialdata(self, *, coords_key: str = _DEFAULT_COORDS_KEY) -> Any:
        """Return the SpatialData adapter (useful for escaping to the full API)."""
        return anndata_to_spatialdata(self._adata, coords_key=coords_key)

    # -- forwarded entry points -------------------------------------------
    def render_points(self, element: str | None = None, *args: Any, **kwargs: Any) -> Any:
        if element is None:
            element = _ELEMENT_NAME
        return self._sdata().pl.render_points(element, *args, **kwargs)

    def show(self, *args: Any, **kwargs: Any) -> Any:
        # Bare ``adata.pl.show()`` with no prior render: default to points.
        return self._sdata().pl.render_points(_ELEMENT_NAME).pl.show(*args, **kwargs)


def enable_anndata_accessor() -> None:
    """Opt-in switch that attaches ``.pl`` to :class:`anndata.AnnData`.

    Importing this module already runs the ``@register_anndata_accessor`` decorator,
    so this is a no-op marker kept for an explicit, discoverable call site::

        import spatialdata_plot
        spatialdata_plot.enable_anndata_accessor()
    """
    # Registration happens at import time via the class decorator above; calling
    # this function simply guarantees the module has been imported.
    return None
