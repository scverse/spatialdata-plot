"""
Render labels
=============

Render a cell segmentation mask.
"""

import spatialdata as sd

import spatialdata_plot  # noqa: F401

sdata = sd.datasets.blobs()

sdata.pl.render_labels("blobs_labels").pl.show()
