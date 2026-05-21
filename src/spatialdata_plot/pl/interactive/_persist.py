"""Commit a ShapesModel into sdata.shapes."""

from __future__ import annotations

from typing import Any

import spatialdata as sd


def commit_to_memory(sdata: sd.SpatialData, shapes_model: Any, name: str) -> str:
    """Add ``shapes_model`` to ``sdata.shapes`` under ``name``, overwriting on collision."""
    sdata.shapes[name] = shapes_model
    return name
