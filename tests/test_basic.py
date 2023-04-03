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
