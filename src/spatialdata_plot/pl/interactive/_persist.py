"""Commit a ShapesModel into sdata.shapes under a collision-safe name."""
from __future__ import annotations

import datetime as _dt
from typing import Any

import spatialdata as sd


def commit_to_memory(sdata: sd.SpatialData, shapes_model: Any, name: str) -> str:
    """Add ``shapes_model`` to ``sdata.shapes`` under ``name``.

    On collision, the existing element is preserved and the new one is
    renamed to ``{name}_{utc-iso}``. Returns the final committed name.
    """
    target = name
    if target in sdata.shapes:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = f"{name}_{ts}"
    sdata.shapes[target] = shapes_model
    return target
