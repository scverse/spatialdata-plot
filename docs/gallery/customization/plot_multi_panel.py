"""
Multi-panel layout
==================

Display multiple coordinate systems side by side.
"""

import numpy as np
import spatialdata as sd
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

import spatialdata_plot  # noqa: F401

rng = np.random.default_rng(0)
img_a = Image2DModel.parse(
    rng.random((3, 64, 64)),
    dims=("c", "y", "x"),
    transformations={"sample_a": Identity()},
)
img_b = Image2DModel.parse(
    rng.random((3, 64, 64)),
    dims=("c", "y", "x"),
    transformations={"sample_b": Identity()},
)
sdata = sd.SpatialData(images={"img_a": img_a, "img_b": img_b})

sdata.pl.render_images().pl.show(
    coordinate_systems=["sample_a", "sample_b"],
    figsize=(8, 4),
)
