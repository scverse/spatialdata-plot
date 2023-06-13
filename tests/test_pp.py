import pytest


@pytest.mark.parametrize(
    "sdata, keys",
    [
        ("get_sdata_with_multiple_images", "data1"),
        ("get_sdata_with_multiple_images", ["data1"]),
        ("get_sdata_with_multiple_images", ["data1", "data2"]),
    ],
)
def test_can_subset_to_one_or_more_images(sdata, keys, request):
    """Tests whether a subset of images can be selected from the sdata object."""

    sdata = request.getfixturevalue(sdata)(share_coordinate_system="all")

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

    # use all possible inputs
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
def test_get_bb_wrong_input_types(sdata, request):
    """Tests whether a subset of images can be selected from the sdata object."""
    sdata = request.getfixturevalue(sdata)

    with pytest.raises(TypeError, match="Parameter 'x' must be one "):
        sdata.pp.get_bb(4, 5)


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
def test_get_bb_wrong_input_dims(sdata, request):
    """Tests whether a subset of images can be selected from the sdata object."""
    sdata = request.getfixturevalue(sdata)

    # x values
    with pytest.raises(ValueError, match="The current choice of 'x' would result in an empty slice."):
        sdata.pp.get_bb(slice(5, 0), slice(0, 5))

    with pytest.raises(ValueError, match="The current choice of 'x' would result in an empty slice."):
        sdata.pp.get_bb(slice(5, 5), slice(0, 5))

    with pytest.raises(ValueError, match="The current choice of 'x' would result in an empty slice."):
        sdata.pp.get_bb([5, 0], [0, 5])

    with pytest.raises(ValueError, match="The current choice of 'x' would result in an empty slice."):
        sdata.pp.get_bb([5, 5], [0, 5])

    with pytest.raises(ValueError, match="The current choice of 'x' would result in an empty slice."):
        sdata.pp.get_bb((5, 0), (0, 5))

    with pytest.raises(ValueError, match="The current choice of 'x' would result in an empty slice."):
        sdata.pp.get_bb((5, 5), (0, 5))

    # y values
    with pytest.raises(ValueError, match="The current choice of 'x' would result in an empty slice."):
        sdata.pp.get_bb(slice(0, 5), slice(5, 0))

    with pytest.raises(ValueError, match="The current choice of 'x' would result in an empty slice."):
        sdata.pp.get_bb(slice(0, 5), slice(5, 5))

    with pytest.raises(ValueError, match="The current choice of 'y' would result in an empty slice."):
        sdata.pp.get_bb([0, 5], [5, 0])

    with pytest.raises(ValueError, match="The current choice of 'y' would result in an empty slice."):
        sdata.pp.get_bb([0, 5], [5, 5])

    with pytest.raises(ValueError, match="The current choice of 'y' would result in an empty slice."):
        sdata.pp.get_bb((0, 5), (5, 0))

    with pytest.raises(ValueError, match="The current choice of 'y' would result in an empty slice."):
        sdata.pp.get_bb((0, 5), (5, 5))


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
def test_get_bb_wrong_input_length(sdata, request):
    """Tests whether a subset of images can be selected from the sdata object."""
    sdata = request.getfixturevalue(sdata)

    with pytest.raises(ValueError, match="Parameter 'x' must be of length 2."):
        sdata.pp.get_bb([0, 5, 6], [0, 5])

    with pytest.raises(ValueError, match="Parameter 'x' must be of length 2."):
        sdata.pp.get_bb((0, 5, 1), (0, 5))

    with pytest.raises(ValueError, match="Parameter 'y' must be of length 2."):
        sdata.pp.get_bb([0, 5], [0, 5, 5])

    with pytest.raises(ValueError, match="Parameter 'y' must be of length 2."):
        sdata.pp.get_bb((0, 5), (0, 5, 2))
