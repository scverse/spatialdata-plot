"""
Filter by category groups
=========================

Show only selected categories using the ``groups`` parameter.
"""

from _helpers import load_visium_breast_cancer

import spatialdata_plot  # noqa: F401

sdata = load_visium_breast_cancer()

(
    sdata.pl.render_images("tissue")
    .pl.render_shapes("spots", color="in_tissue", groups=["1"], fill_alpha=0.7)
    .pl.show()
)
