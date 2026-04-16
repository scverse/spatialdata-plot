"""
Spots with outlines
===================

Style spots with visible outlines and translucent fill.
"""

from _helpers import load_visium_breast_cancer

import spatialdata_plot  # noqa: F401

sdata = load_visium_breast_cancer()

(
    sdata.pl
    .render_images("tissue")
    .pl.render_shapes(
        "spots",
        fill_alpha=0.3,
        outline_width=1.5,
        outline_color="black",
        outline_alpha=1.0,
    )
    .pl.show()
)
