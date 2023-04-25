import pytest


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
    assert sdata.images["data1_image"].shape == (1, 10, 10)
