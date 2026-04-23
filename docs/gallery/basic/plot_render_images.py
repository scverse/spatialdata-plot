"""
Render tissue image
===================

Render an H&E tissue image from a Visium experiment.
"""

from _helpers import load_visium_breast_cancer

import spatialdata_plot  # noqa: F401

sdata = load_visium_breast_cancer()

sdata.pl.render_images("tissue").pl.show()
