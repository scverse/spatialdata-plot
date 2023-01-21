import numpy as np
import anndata as ad
import spatialdata as sd

import pytest


@pytest.fixture
def test_sdata_single_image():
    """Creates a simple sdata object."""
    images = { 'data1': sd.Image2DModel.parse(np.zeros((1, 10, 10)), dims=('c', 'y', 'x')) }  
    sdata = sd.SpatialData(images=images)
    return sdata

@pytest.fixture
def test_sdata_multiple_images():
    """Creates an sdata object with multiple images."""
    images = { 
        'data1': sd.Image2DModel.parse(np.zeros((1, 10, 10)), dims=('c', 'y', 'x')),
        'data2': sd.Image2DModel.parse(np.zeros((1, 10, 10)), dims=('c', 'y', 'x')),
        'data3': sd.Image2DModel.parse(np.zeros((1, 10, 10)), dims=('c', 'y', 'x')),
    }  
    sdata = sd.SpatialData(images=images)
    return sdata

@pytest.fixture
def test_sdata_multiple_images_dims():
    """Creates an sdata object with multiple images."""
    images = { 
        'data1': sd.Image2DModel.parse(np.zeros((3, 10, 10)), dims=('c', 'y', 'x')),
        'data2': sd.Image2DModel.parse(np.zeros((3, 10, 10)), dims=('c', 'y', 'x')),
        'data3': sd.Image2DModel.parse(np.zeros((3, 10, 10)), dims=('c', 'y', 'x')),
    }  
    sdata = sd.SpatialData(images=images)
    return sdata

@pytest.fixture
def test_sdata_multiple_images_diverging_dims():
    """Creates an sdata object with multiple images."""
    images = { 
        'data1': sd.Image2DModel.parse(np.zeros((3, 10, 10)), dims=('c', 'y', 'x')),
        'data2': sd.Image2DModel.parse(np.zeros((6, 10, 10)), dims=('c', 'y', 'x')),
        'data3': sd.Image2DModel.parse(np.zeros((3, 10, 10)), dims=('c', 'y', 'x')),
    }  
    sdata = sd.SpatialData(images=images)
    return sdata

