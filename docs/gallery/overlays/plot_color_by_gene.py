"""
Color spots by gene expression
===============================

Overlay spots colored by a gene on an H&E tissue image.
"""

from _helpers import load_visium_breast_cancer

import spatialdata_plot  # noqa: F401

sdata = load_visium_breast_cancer()

(
    sdata.pl
    .render_images("tissue")
    .pl.render_shapes("spots", color="ERBB2", fill_alpha=0.8)
    .pl.show()
)
