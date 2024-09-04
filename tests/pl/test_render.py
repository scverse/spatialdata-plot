import matplotlib.pyplot as plt
import pytest


def test_render_images_can_plot_one_cyx_image(request):
    sdata = request.getfixturevalue("test_sdata_single_image")

    _, ax = plt.subplots(1, 1)

    sdata.pl.render_images().pl.show(ax=ax)

    assert ax.get_title() == sdata.coordinate_systems[0]


@pytest.mark.parametrize(
    "share_coordinate_system",
    [
        "all",
        "two",
        "none",
    ],
)
def test_render_images_can_plot_multiple_cyx_images(share_coordinate_system: str, request):
    fun = request.getfixturevalue("get_sdata_with_multiple_images")
    sdata = fun(share_coordinate_system)
    sdata.pl.render_images().pl.show(
        colorbar=False,  # otherwise we'll get one cbar per image in the same cs
    )
    axs = plt.gcf().get_axes()

    if share_coordinate_system == "all":
        assert len(axs) == 1
    elif share_coordinate_system == "none":
        assert len(axs) == 3
    elif share_coordinate_system == "two":
        assert len(axs) == 2


def test_keyerror_when_image_element_does_not_exist(request):
    sdata = request.getfixturevalue("sdata_blobs")

    with pytest.raises(KeyError):
        sdata.pl.render_images(element="not_found").pl.show()


def test_keyerror_when_label_element_does_not_exist(request):
    sdata = request.getfixturevalue("sdata_blobs")

    with pytest.raises(KeyError):
        sdata.pl.render_labels(element="not_found").pl.show()


def test_keyerror_when_point_element_does_not_exist(request):
    sdata = request.getfixturevalue("sdata_blobs")

    with pytest.raises(KeyError):
        sdata.pl.render_points(element="not_found").pl.show()


def test_keyerror_when_shape_element_does_not_exist(request):
    sdata = request.getfixturevalue("sdata_blobs")

    with pytest.raises(KeyError):
        sdata.pl.render_shapes(element="not_found").pl.show()
