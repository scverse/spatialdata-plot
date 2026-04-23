"""
Render spots
============

Render Visium spot shapes on their own.
"""

from _helpers import load_visium_breast_cancer

import spatialdata_plot  # noqa: F401

sdata = load_visium_breast_cancer()

sdata.pl.render_shapes("spots").pl.show()
