import pytest
import spatialdata_plot

import spatialdata as sd

@pytest.mark.parametrize(
    "sdata, bb_x, bb_y",
    [
        ("test_sdata_single_image", [0, 5], [5, 10]),
        ("test_sdata_single_image", (0, 5), (5, 10)),
        ("test_sdata_single_image", slice(0, 5), slice(5, 10)),
    ],
)   
def test_sdata_fixture(sdata, bb_x, bb_y, request):
    """Tests the sdata fixture."""
    sdata = request.getfixturevalue(sdata)
    
    clipped_sdata = sdata.pp.get_bb(bb_x, bb_y)
    
    assert clipped_sdata.images['data1'].shape == (1, 5, 5)
    
    
# @pytest.mark.parametrize(
#     "sdata, query",
#     [
#         ("test_sdata_single_image", ["data1"]),
#         ("test_sdata_multiple_images", ["data1"]),
#         ("test_sdata_multiple_images", ["data1", "data2"]),
#         ("test_sdata_multiple_images", ["data1", "data3"]),
#         ("test_sdata_multiple_images_dims", ['data1']),
#         ("test_sdata_multiple_images_dims", ['data1', 'data2']),
#     ],
# )   
# def test_image_accessor_correct_image_key_list(sdata, query, request):
#     """Tests the image accessor with a correct image key list."""
#     sdata = request.getfixturevalue(sdata)
#     sdata.im[query]
#     assert sdata.im.i == query