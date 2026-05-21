"""Commit a ShapesModel into sdata.shapes (memory) and optionally to zarr."""
from __future__ import annotations

import datetime as _dt
from typing import Any

import spatialdata as sd


def commit_to_memory(
    sdata: sd.SpatialData,
    shapes_model: Any,
    name: str,
) -> str:
    """Add ``shapes_model`` to ``sdata.shapes`` under ``name``.

    On collision, the existing element is preserved and the new one is
    renamed to ``{name}_{utc-iso}``. Returns the final committed name.
    """
    target = name
    if target in sdata.shapes:
        ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        target = f"{name}_{ts}"
    sdata.shapes[target] = shapes_model
    return target


def persist_to_disk(sdata: sd.SpatialData, name: str) -> None:
    """Persist ``sdata.shapes[name]`` to the backing zarr store.

    Raises ValueError if ``sdata`` is not zarr-backed.
    """
    if sdata.path is None:
        raise ValueError(
            "SpatialData is not zarr-backed (sdata.path is None); cannot persist. "
            "Write the SpatialData object to a zarr store first, then re-open it."
        )
    sdata.write_element(name)
