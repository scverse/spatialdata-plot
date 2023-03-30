import pytest

import matplotlib.pyplot as plt


def test_render_images_can_plot_one_cyx_image(request):

    sdata = request.getfixturevalue("test_sdata_single_image")

    _, ax = plt.subplots(1,1)

    sdata.pl.render_images().pl.show(ax=ax)

    assert ax.get_title() == list(sdata.images.keys())[0]


@pytest.mark.parametrize(
    "sdata, share_coordinate_system",
    [
        ("test_sdata_multiple_images", True),
        ("test_sdata_multiple_images", False),
    ],
)
def test_render_images_can_plot_multiple_cyx_images(sdata, share_coordinate_system, request):

    sdata = request.getfixturevalue(sdata)

    axs = sdata.pl.render_images().pl.show()

    if share_coordinate_system:

        assert len(axs) == 1
        assert axs[0].get_title() == list(sdata.images.keys())[0]

    else:

        assert len(axs) == 3
        assert axs[0].get_title() == list(sdata.images.keys())[0]
        assert axs[1].get_title() == list(sdata.images.keys())[1]
        assert axs[2].get_title() == list(sdata.images.keys())[2]
