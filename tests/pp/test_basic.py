import pytest


@pytest.mark.parametrize(
    "sdata, keys ",
    [
        ("get_sdata_with_multiple_images", 0),
        ("get_sdata_with_multiple_images", {"a": 0}),
        ("get_sdata_with_multiple_images", None),
        ("get_sdata_with_multiple_images", ["my_key", 0]),
    ],
)
def test_typerror_when_key_is_invalid(sdata, keys, request):
    """Tests wether the images inside sdata can be clipped to a bounding box."""
    sdata = request.getfixturevalue(sdata)(share_coordinate_system="all")

    with pytest.raises(TypeError):
        sdata.pp.get_elements(keys)


@pytest.mark.parametrize(
    "sdata, keys ",
    [
        ("get_sdata_with_multiple_images", "data4"),
        ("get_sdata_with_multiple_images", ["data1", "data4"]),
    ],
)
def test_valuerror_when_key_is_of_correct_type_but_not_in_sdata(sdata, keys, request):
    sdata = request.getfixturevalue(sdata)(share_coordinate_system="all")

    with pytest.raises(ValueError):
        sdata.pp.get_elements(keys)


def test_get_elements_correctly_filters_coordinate_systems(request):
    """Tests that get_elements correctly filters coordinate systems by their name."""

    fun = request.getfixturevalue("get_sdata_with_multiple_images")
    sdata_all_cs_shared = fun("all")
    sdata_no_cs_shared = fun("none")
    sdata_two_cs_shared = fun("two")
    sdata_two_cs_similar_name = fun("similar_name")

    assert len(sdata_all_cs_shared.images.keys()) == 3
    assert len(sdata_two_cs_shared.pp.get_elements("coord_sys1").images.keys()) == 2
    assert len(sdata_no_cs_shared.pp.get_elements("coord_sys1").images.keys()) == 1
    assert len(sdata_two_cs_similar_name.pp.get_elements("coord_sys1").images.keys()) == 1
