import pytest
import spatialdata_plot



@pytest.mark.parametrize(
    "sdata",
    [
        "test_sdata_single_image",
        "test_sdata_multiple_images",
        # "test_sdata_multiple_images_dims"
    ],
)
def test_sdata_fixture(sdata, request):
    """Tests the sdata fixture."""
    sdata = request.getfixturevalue(sdata)
    assert sdata.images['data1'].shape == (1, 10, 10)
     
 
@pytest.mark.parametrize(
    "sdata",
    [
        "test_sdata_single_image",
        "test_sdata_multiple_images",
        "test_sdata_multiple_images_dims"
    ],
)   
def test_image_accessor_correct_image_key_string(sdata, request):
    """Tests the image accessor with a correct image key string."""
    sdata = request.getfixturevalue(sdata)
    sdata.im['data1']
    
    assert sdata.im.i == 'data1'


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


@pytest.mark.parametrize(
    "sdata",
    [
        "test_sdata_single_image",
        "test_sdata_multiple_images",
        "test_sdata_multiple_images_dims"
    ],
)   
def test_image_accessor_wrong_correct_image_key_string(sdata, request):
    """Tests the image accessor with a wrong image key string."""
    sdata = request.getfixturevalue(sdata)
    with pytest.raises(AssertionError):
        sdata.im['wrong']
    
    
@pytest.mark.parametrize(
    "sdata",
    [
        "test_sdata_single_image",
        "test_sdata_multiple_images",
        "test_sdata_multiple_images_dims"
    ],
)   
def test_image_accessor_correct_channel(sdata, request):
    """Tests the image accessor with a wrong image key string."""
    sdata = request.getfixturevalue(sdata)
    sdata.im[0]
    
    assert isinstance(sdata.im.i, list)
    assert sdata.im.c == 0
    
    

@pytest.mark.parametrize(
    "sdata",
    [
        "test_sdata_single_image",
        "test_sdata_multiple_images",
        "test_sdata_multiple_images_dims"
    ],
)   
def test_image_accessor_correct_image_key_and_channel(sdata, request):
    """Tests the image accessor with a wrong image key string."""
    sdata = request.getfixturevalue(sdata)
    sdata.im['data1', 0]
    
    assert isinstance(sdata.im.i, str)
    assert sdata.im.c == 0

    