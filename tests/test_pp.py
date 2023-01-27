import pytest
import spatialdata_plot



@pytest.mark.parametrize(
    "sdata",
    [
        "test_sdata_single_image",
        "test_sdata_multiple_images",
    ],
)
def test_sdata_fixture(sdata, request):
    """Tests the sdata fixture."""
    sdata = request.getfixturevalue(sdata)
    assert sdata.images['data1'].shape == (1, 10, 10)
    
    
    
@pytest.mark.parametrize(
    "sdata, query",
    [
        ("test_sdata_single_image", ["data1"]),
        ("test_sdata_multiple_images", ["data1"]),
        ("test_sdata_multiple_images", ["data1", "data2"]),
        ("test_sdata_multiple_images", ["data1", "data3"]),
        ("test_sdata_multiple_images_dims", ['data1']),
        ("test_sdata_multiple_images_dims", ['data1', 'data2']),
    ],
)   
def test_image_accessor_correct_image_key_list(sdata, query, request):
    """Tests the image accessor with a correct image key list."""
    sdata = request.getfixturevalue(sdata)
    sdata.im[query]
    assert sdata.im.i == query