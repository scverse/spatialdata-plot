"""
Render points
=============

Render transcript detections as points.
"""

import spatialdata as sd
import spatialdata_plot  # noqa: F401

sdata = sd.datasets.blobs()

sdata.pl.render_points("blobs_points", size=3).pl.show()
