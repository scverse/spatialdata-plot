import timeit

import matplotlib.pyplot as plt
import numpy as np
import pytest
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, set_transformation

import spatialdata_plot  # noqa: F401
from spatialdata_plot.pl.utils import _get_cs_contents


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


# Regression tests for #176: plotting with user-supplied ax when elements
# have transformations to multiple coordinate systems.


def test_single_ax_after_filter_by_coordinate_system(sdata_multi_cs):
    """After filter_by_coordinate_system, single ax should work without specifying CS."""
    sdata_filt = sdata_multi_cs.filter_by_coordinate_system("aligned")

    _, ax = plt.subplots(1, 1)
    sdata_filt.pl.render_images("img").pl.render_shapes("shp").pl.show(ax=ax)
    assert ax.get_title() == "aligned"


def test_single_ax_with_explicit_cs(sdata_multi_cs):
    """Explicit coordinate_systems with single ax should work."""
    _, ax = plt.subplots(1, 1)
    sdata_multi_cs.pl.render_images("img").pl.render_shapes("shp").pl.show(ax=ax, coordinate_systems="aligned")
    assert ax.get_title() == "aligned"


def test_single_ax_explicit_multi_cs_raises(sdata_multi_cs):
    """Explicitly requesting more CS than axes should still raise."""
    _, ax = plt.subplots(1, 1)
    with pytest.raises(ValueError, match="Mismatch"):
        sdata_multi_cs.pl.render_shapes("shp").pl.show(ax=ax, coordinate_systems=["aligned", "global"])


def test_single_ax_auto_cs_unresolvable_raises(sdata_multi_cs):
    """When strict filtering can't resolve the mismatch, error includes hint."""
    _, ax = plt.subplots(1, 1)
    with pytest.raises(ValueError, match="coordinate_systems="):
        # Only render shapes (present in both CS), so strict filter can't narrow down
        sdata_multi_cs.pl.render_shapes("shp").pl.show(ax=ax)


def test_cs_name_with_apostrophe_does_not_crash():
    # Regression test for #602: .query(f"cs == '{cs}'") raised TokenError for names
    # containing single quotes (e.g. "patient's_sample").
    data = np.zeros((1, 10, 10), dtype=np.float64)
    img = Image2DModel.parse(data, dims=("c", "y", "x"))
    sdata = SpatialData(images={"img": img})
    set_transformation(sdata["img"], Identity(), to_coordinate_system="patient's_cs")
    _, ax = plt.subplots()
    sdata.pl.render_images("img").pl.show(ax=ax, coordinate_systems="patient's_cs")
    plt.close("all")


def test_get_cs_contents_is_linear():
    # Regression test for #602: pd.concat inside loop was O(n²).
    # Build two SpatialData objects: n=10 and n=50 coordinate systems.
    def build(n: int) -> SpatialData:
        data = np.zeros((1, 4, 4), dtype=np.float64)
        images = {}
        for i in range(n):
            img_i = Image2DModel.parse(data.copy(), dims=("c", "y", "x"))
            set_transformation(img_i, Identity(), to_coordinate_system=f"cs_{i}")
            images[f"img_{i}"] = img_i
        return SpatialData(images=images)

    sd10 = build(10)
    sd50 = build(50)
    t10 = timeit.timeit(lambda: _get_cs_contents(sd10), number=20) / 20
    t50 = timeit.timeit(lambda: _get_cs_contents(sd50), number=20) / 20
    ratio = t50 / t10
    # O(n) → ratio ≈ 5×; O(n²) → ratio ≈ 25×. Allow generous headroom for CI variance.
    assert ratio < 15, (
        f"_get_cs_contents appears quadratic: n=10 {t10 * 1e3:.1f}ms, n=50 {t50 * 1e3:.1f}ms, ratio={ratio:.1f}x"
    )
