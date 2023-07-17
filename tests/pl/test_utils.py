import pytest
from spatialdata.datasets import blobs


@pytest.mark.parametrize(
    "outline_color",
    [
        (0.0, 1.0, 0.0, 1.0),
        "#00ff00",
    ],
)
def test_set_outline_accepts_str_or_float_or_list_thereof(outline_color):
    sdata = blobs()
    sdata.pl.render_shapes(elements="blobs_polygons", outline=True, outline_color=outline_color).pl.show()
