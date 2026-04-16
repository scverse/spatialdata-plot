"""
Gene expression with custom colormap
=====================================

Color spots by gene expression using a custom colormap.
"""

from _helpers import load_visium_breast_cancer

import spatialdata_plot  # noqa: F401

sdata = load_visium_breast_cancer()

(
    sdata.pl
    .render_images("tissue")
    .pl.render_shapes("spots", color="ERBB2", cmap="magma", fill_alpha=0.8)
    .pl.show()
)
