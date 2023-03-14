import pytest


@pytest.mark.parametrize(
    "sdata, keys ",
    [
        ("test_sdata_multiple_images", 0),
        ("test_sdata_multiple_images", {"a": 0}),
        ("test_sdata_multiple_images", None),
        ("test_sdata_multiple_images", ["my_key", 0]),
    ],
)
def test_typerror_when_key_is_invalid(sdata, keys, request):
    """Tests wether the images inside sdata can be clipped to a bounding box."""

    sdata = request.getfixturevalue(sdata)

    with pytest.raises(TypeError):
        sdata.pp.get_elements(keys)


@pytest.mark.parametrize(
    "sdata, keys ",
    [
        ("test_sdata_multiple_images", "data4"),
        ("test_sdata_multiple_images", ["data1", "data4"]),
    ],
)
def test_valuerror_when_key_is_of_correct_type_but_not_in_sdata(sdata, keys, request):
    """Tests wether the images inside sdata can be clipped to a bounding box."""

    sdata = request.getfixturevalue(sdata)

    with pytest.raises(ValueError):
        sdata.pp.get_elements(keys)
