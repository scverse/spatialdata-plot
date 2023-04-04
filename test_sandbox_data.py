#!/usr/bin/env python

# # Load real-world data and verify that functions run on it
# ### How to use this script
# - If you need it interactive, convert it to a jupyter notebook with `p2j` -> `p2j test_sandbox_data.py`
# - You can use `jupyter nbconvert --to python test_sandbox_data.ipynb` to convert it back
# - Otherwise run it from the CLI and verify that the plots are okay

# In[1]:


import matplotlib.pyplot as plt
import spatialdata as sd
from spatialdata.datasets import blobs

import spatialdata_plot

assert spatialdata_plot.__name__ == "spatialdata_plot"  # so mypy doesn't complain

DATA_DIR = "/Users/tim.treis/Documents/GitHub/spatialdata-sandbox/"


# In[2]:


(blobs().pl.render_images().pl.render_labels().pl.render_shapes().pl.render_points(color_key="genes").pl.show())


# In[4]:


# Mibi

mibitof = sd.read_zarr(DATA_DIR + "mibitof/data.zarr")

(mibitof.pl.render_images().pl.render_labels().pl.show())

plt.savefig("mibi.png")


# In[5]:


# Visium

visium = sd.read_zarr(DATA_DIR + "visium/data.zarr")

(visium.pl.render_images().pl.render_shapes().pl.show())

plt.savefig("visium.png")
