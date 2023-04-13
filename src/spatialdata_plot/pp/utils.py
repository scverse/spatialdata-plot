from collections import OrderedDict

import matplotlib
import spatialdata as sd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from spatialdata.models import TableModel
from spatialdata.transformations import get_transformation


def _get_linear_colormap(colors: list[str], background: str) -> list[matplotlib.colors.LinearSegmentedColormap]:
    return [LinearSegmentedColormap.from_list(c, [background, c], N=256) for c in colors]


def _get_listed_colormap(color_dict: dict[str, str]) -> matplotlib.colors.ListedColormap:
    sorted_labels = sorted(color_dict.keys())
    colors = [color_dict[k] for k in sorted_labels]
    cmap = ListedColormap(["black"] + colors, N=len(colors) + 1)

    return cmap


def _get_region_key(sdata: sd.SpatialData) -> str:
    """Quick access to the data's region key."""
    region_key = str(sdata.table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY])

    return region_key


def _get_instance_key(sdata: sd.SpatialData) -> str:
    """Quick access to the data's instance key."""
    instance_key = str(sdata.table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY])

    return instance_key


def _verify_plotting_tree(sdata: sd.SpatialData) -> sd.SpatialData:
    """Verify that the plotting tree exists, and if not, create it."""
    if not hasattr(sdata, "plotting_tree"):
        sdata.plotting_tree = OrderedDict()

    return sdata


def _get_coordinate_system_mapping(sdata: sd.SpatialData) -> dict[str, list[str]]:
    has_images = True if sdata.images is not None else False
    has_labels = True if sdata.labels is not None else False
    has_shapes = True if sdata.shapes is not None else False
    has_points = True if sdata.points is not None else False

    coordsys_keys = sdata.coordinate_systems
    image_keys = sdata.images.keys() if has_images else []
    label_keys = sdata.labels.keys() if has_labels else []
    shape_keys = sdata.shapes.keys() if has_shapes else []
    point_keys = sdata.points.keys() if has_points else []

    mapping: dict[str, list[str]] = {}

    if len(coordsys_keys) < 1:
        raise ValueError("SpatialData object must have at least one coordinate system to generate a mapping.")

    for key in coordsys_keys:
        mapping[key] = []

        for image_key in image_keys:
            transformations = get_transformation(sdata.images[image_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(image_key)

        for label_key in label_keys:
            transformations = get_transformation(sdata.labels[label_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(label_key)

        for shape_key in shape_keys:
            transformations = get_transformation(sdata.shapes[shape_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(shape_key)

        for point_key in point_keys:
            transformations = get_transformation(sdata.points[point_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(point_key)

    return mapping
