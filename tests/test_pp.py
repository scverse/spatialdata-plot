import pytest


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
    "sdata, keys, nrows ",
    [
        ("test_sdata_multiple_images_with_table", "data1", 3),
        ("test_sdata_multiple_images_with_table", ["data1", "data3"], 23),
    ],
)
def test_table_gets_subset_when_images_are_subset(sdata, keys, nrows, request):
    """Tests wether the images inside sdata can be clipped to a bounding box."""

    sdata = request.getfixturevalue(sdata)

    assert sdata.table.n_obs == 30

    new_sdata = sdata.pp.get_images(keys)

    assert new_sdata.table.n_obs == nrows
