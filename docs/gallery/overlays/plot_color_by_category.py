"""
Color spots by category
=======================

Overlay spots colored by a categorical annotation.
"""

from _helpers import load_visium_breast_cancer

import spatialdata_plot  # noqa: F401

sdata = load_visium_breast_cancer()

(sdata.pl.render_images("tissue").pl.render_shapes("spots", color="in_tissue", fill_alpha=0.7).pl.show())
