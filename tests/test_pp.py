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


@pytest.mark.parametrize(
    "sdata",
    [
        "test_sdata_single_image",
        "test_sdata_multiple_images",
        "test_sdata_single_image_with_label",
        "test_sdata_multiple_images_with_table",
        # "full_sdata" that one is broken
    ],
)
def test_get_bb_correct_inputs(sdata, request):
    """Tests whether a subset of images can be selected from the sdata object."""
    sdata = request.getfixturevalue(sdata)

    sliced_slice = sdata.pp.get_bb(slice(0, 5), slice(0, 5))
    sliced_list = sdata.pp.get_bb([0, 5], [0, 5])
    sliced_tuple = sdata.pp.get_bb((0, 5), (0, 5))

    for sliced_object in [sliced_slice, sliced_list, sliced_tuple]:
        for _k, v in sliced_object.images.items():
            # test if images have the correct dimensionality
            assert v.shape[1] == 5
            assert v.shape[2] == 5

        if hasattr(sliced_object, "labels"):
            for _k, v in sliced_object.labels.items():
                # test if images have the correct dimensionality
                assert v.shape[0] == 5
                assert v.shape[1] == 5

        # check if the plotting tree was appended
        assert hasattr(sliced_object, "plotting_tree")


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
