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
    sdata.pl.render_images().pl.show()
    axs = plt.gcf().get_axes()

    if share_coordinate_system == "all":
        assert len(axs) == 1

    if share_coordinate_system == "two":
        assert len(axs) == 2

    if share_coordinate_system == "none":
        assert len(axs) == 3
