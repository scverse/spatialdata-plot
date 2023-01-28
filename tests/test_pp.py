import pytest
import spatialdata_plot

import spatialdata as sd



@pytest.mark.parametrize(
    "sdata, keys",
    [
        ("test_sdata_single_image", "data1"),
        ("test_sdata_single_image", ["data1"]),
    ],
)   
def test_can_subset_to_images(sdata, keys, request):
    
    """Tests whether a subset of images can be selected from the sdata object."""
    
    sdata = request.getfixturevalue(sdata)
    
    clipped_sdata = sdata.pp.get_images(keys)
    
    assert list(clipped_sdata.images.keys()) == ([keys] if isinstance(keys, str) else keys)
    
    

@pytest.mark.parametrize(
    "sdata, bb_x, bb_y",
    [
        ("test_sdata_single_image", [0, 5], [5, 10]),
        ("test_sdata_single_image", (0, 5), (5, 10)),
        ("test_sdata_single_image", slice(0, 5), slice(5, 10)),
    ],
)   
def test_can_clip_a_single_img_to_bb(sdata, bb_x, bb_y, request):
    
    """Tests wether the images inside sdata can be clipped to a bounding box."""
    
    sdata = request.getfixturevalue(sdata)
    
    clipped_sdata = sdata.pp.get_bb(bb_x, bb_y)
    
    assert clipped_sdata.images['data1'].shape == (1, 5, 5)
    
    