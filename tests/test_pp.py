import pytest


@pytest.mark.parametrize(
    "sdata, keys",
    [
        ("test_sdata_multiple_images", "data1"),
        ("test_sdata_multiple_images", ["data1"]),
        ("test_sdata_multiple_images", ["data1", "data2"]),
    ],
)
def test_can_subset_to_one_or_more_images(sdata, keys, request):
    """Tests whether a subset of images can be selected from the sdata object."""

    sdata = request.getfixturevalue(sdata)

    clipped_sdata = sdata.pp.get_elements(keys)

    assert list(clipped_sdata.images.keys()) == ([keys] if isinstance(keys, str) else keys)


# @pytest.mark.parametrize(
#     "sdata, keys, nrows ",
#     [
#         ("full_sdata", "data1", 3),
#         ("full_sdata", ["data1", "data3"], 23),
#     ],
# )
# def test_table_gets_subset_when_images_are_subset(sdata, keys, nrows, request):
#     """Tests wether the images inside sdata can be clipped to a bounding box."""

#     sdata = request.getfixturevalue(sdata)

#     assert sdata.table.n_obs == 30

#     new_sdata = sdata.pp.get_elements(keys)

#     print(new_sdata.table)

#     assert len(new_sdata.table.obs) == nrows
