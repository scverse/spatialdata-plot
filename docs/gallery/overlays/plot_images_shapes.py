"""
Tissue with spots
=================

Overlay Visium spots on an H&E tissue image.
"""

from _helpers import load_visium_breast_cancer

import spatialdata_plot  # noqa: F401

sdata = load_visium_breast_cancer()

(
    sdata.pl
    .render_images("tissue")
    .pl.render_shapes("spots", fill_alpha=0.5)
    .pl.show()
)
