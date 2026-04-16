"""
Single channel with colormap
=============================

Render one image channel with a colormap.
"""

from _helpers import load_visium_breast_cancer

import spatialdata_plot  # noqa: F401

sdata = load_visium_breast_cancer()

sdata.pl.render_images("tissue", channel=0, cmap="viridis").pl.show()
