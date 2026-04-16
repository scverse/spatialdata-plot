"""
Render label contours
=====================

Render segmentation boundaries using ``contour_px``.
"""

import spatialdata as sd

import spatialdata_plot  # noqa: F401

sdata = sd.datasets.blobs()

sdata.pl.render_labels("blobs_labels", contour_px=3).pl.show()
