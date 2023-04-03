#!/usr/bin/env python

# # Load real-world data and verify that functions run on it
# ### How to use this script
# - If you need it interactive, convert it to a jupyter notebook with `p2j` -> `p2j test_sandbox_data.py`
# - You can use `jupyter nbconvert --to python test_sandbox_data.ipynb` to convert it back
# - Otherwise run it from the CLI and verify that the plots are okay

# In[4]:


import matplotlib.pyplot as plt
import spatialdata as sd

import spatialdata_plot

assert spatialdata_plot.__name__ == "spatialdata_plot"  # so mypy doesn't complain

DATA_DIR = "/Users/tim.treis/Documents/GitHub/spatialdata-sandbox/"


# ## Load data
# Adjust paths as neccecary

# In[15]:


# Mibi

sdata = sd.read_zarr(DATA_DIR + "mibitof/data.zarr")
sdata.pl.render_images().pl.render_labels().pl.show()

plt.savefig("mibi.png")


# In[16]:


# Visium

sdata = sd.read_zarr(DATA_DIR + "visium/data.zarr")
sdata.pl.render_images().pl.render_shapes().pl.show(width=12, height=12)

plt.savefig("visium.png")
