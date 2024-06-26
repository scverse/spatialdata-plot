from collections import OrderedDict
from typing import Union

import spatialdata as sd
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from datatree import DataTree
from geopandas import GeoDataFrame
from spatialdata.models import get_table_keys
from xarray import DataArray

from spatialdata_plot._accessor import register_spatial_data_accessor
from spatialdata_plot.pp.utils import (
    _get_coordinate_system_mapping,
    _verify_plotting_tree,
)

# from .colorize import _colorize


@register_spatial_data_accessor("pp")
class PreprocessingAccessor:
    """
    Preprocessing functions for SpatialData objects.

    Parameters
    ----------
    sdata :
        A spatial data object.
    """

    @property
    def sdata(self) -> sd.SpatialData:
        """The `SpatialData` object to provide preprocessing functions for."""
        return self._sdata

    @sdata.setter
    def sdata(self, sdata: sd.SpatialData) -> None:
        self._sdata = sdata

    def __init__(self, sdata: sd.SpatialData) -> None:
        self._sdata = sdata

    def _copy(
        self,
        images: Union[None, dict[str, Union[DataArray, DataTree]]] = None,
        labels: Union[None, dict[str, Union[DataArray, DataTree]]] = None,
        points: Union[None, dict[str, DaskDataFrame]] = None,
        shapes: Union[None, dict[str, GeoDataFrame]] = None,
        tables: Union[None, dict[str, AnnData]] = None,
    ) -> sd.SpatialData:
        """Copy the references from the original to the new SpatialData object."""
        sdata = sd.SpatialData(
            images=self._sdata.images if images is None else images,
            labels=self._sdata.labels if labels is None else labels,
            points=self._sdata.points if points is None else points,
            shapes=self._sdata.shapes if shapes is None else shapes,
            tables=self._sdata.tables if tables is None else tables,
        )
        sdata.plotting_tree = self._sdata.plotting_tree if hasattr(self._sdata, "plotting_tree") else OrderedDict()

        return sdata

    def _verify_plotting_tree_exists(self) -> None:
        if not hasattr(self._sdata, "plotting_tree"):
            self._sdata.plotting_tree = OrderedDict()

    def get_elements(self, elements: Union[str, list[str]]) -> sd.SpatialData:
        """
        Get a subset of the spatial data object by specifying elements to keep.

        Parameters
        ----------
        elements :
            A string or a list of strings specifying the elements to keep.
            Valid element types are:

            - 'coordinate_systems'
            - 'images'
            - 'labels'
            - 'shapes'

        Returns
        -------
        sd.SpatialData
            A new spatial data object containing only the specified elements.

        Raises
        ------
        TypeError
            If `elements` is not a string or a list of strings.
            If `elements` is a list of strings but one or more of the strings
            are not valid element types.

        ValueError
            If any of the specified elements is not present in the original
            spatialdata object.

        AssertionError
            If `label_keys` is not an empty list but the spatial data object
            does not have a table or the table does not have 'uns' or 'obs'
            attributes.

        Notes
        -----
        If the original spatialdata object has a table, and `elements`
        includes label keys, the returned spatialdata object will have a
        subset of the original table with only the rows corresponding to the
        specified label keys. The `region` attribute of the returned spatial
        data object's table will be set to the list of specified label keys.

        If the original spatial data object has no table, or if `elements` does
        not include label keys, the returned spatialdata object will have no
        table.
        """
        if not isinstance(elements, (str, list)):
            raise TypeError("Parameter 'elements' must be a string or a list of strings.")

        if not all(isinstance(e, str) for e in elements):
            raise TypeError("When parameter 'elements' is a list, all elements must be strings.")

        if isinstance(elements, str):
            elements = [elements]

        coord_keys = []
        image_keys = []
        label_keys = []
        shape_keys = []
        point_keys = []

        # prepare list of valid keys to sort elements on
        valid_coord_keys = self._sdata.coordinate_systems if hasattr(self._sdata, "coordinate_systems") else None
        valid_image_keys = list(self._sdata.images.keys()) if hasattr(self._sdata, "images") else None
        valid_label_keys = list(self._sdata.labels.keys()) if hasattr(self._sdata, "labels") else None
        valid_shape_keys = list(self._sdata.shapes.keys()) if hasattr(self._sdata, "shapes") else None
        valid_point_keys = list(self._sdata.points.keys()) if hasattr(self._sdata, "points") else None

        # first, extract coordinate system keys becasuse they generate implicit keys
        mapping = _get_coordinate_system_mapping(self._sdata)
        implicit_keys = []
        for e in elements:
            for valid_coord_key in valid_coord_keys:
                if (valid_coord_keys is not None) and (e == valid_coord_key):
                    coord_keys.append(e)
                    implicit_keys += mapping[e]

        for e in elements + implicit_keys:
            found = False

            if valid_coord_keys is not None:
                for valid_coord_key in valid_coord_keys:
                    if e == valid_coord_key:
                        coord_keys.append(e)
                        found = True

            if valid_image_keys is not None:
                for valid_image_key in valid_image_keys:
                    if e == valid_image_key:
                        image_keys.append(e)
                        found = True

            if valid_label_keys is not None:
                for valid_label_key in valid_label_keys:
                    if e == valid_label_key:
                        label_keys.append(e)
                        found = True

            if valid_shape_keys is not None:
                for valid_shape_key in valid_shape_keys:
                    if e == valid_shape_key:
                        shape_keys.append(e)
                        found = True

            if valid_point_keys is not None:
                for valid_point_key in valid_point_keys:
                    if e == valid_point_key:
                        point_keys.append(e)
                        found = True

            if not found:
                msg = f"Element '{e}' not found. Valid choices are:"
                if valid_coord_keys is not None:
                    msg += "\n\ncoordinate_systems\n├ "
                    msg += "\n├ ".join(valid_coord_keys)
                if valid_image_keys is not None:
                    msg += "\n\nimages\n├ "
                    msg += "\n├ ".join(valid_image_keys)
                if valid_label_keys is not None:
                    msg += "\n\nlabels\n├ "
                    msg += "\n├ ".join(valid_label_keys)
                if valid_shape_keys is not None:
                    msg += "\n\nshapes\n├ "
                    msg += "\n├ ".join(valid_shape_keys)
                raise ValueError(msg)

        # copy that we hard-modify
        sdata = self._copy()

        if (valid_coord_keys is not None) and (len(coord_keys) > 0):
            sdata = sdata.filter_by_coordinate_system(coord_keys, filter_tables=False)

        elif len(coord_keys) == 0:
            if valid_image_keys is not None:
                if len(image_keys) == 0:
                    for valid_image_key in valid_image_keys:
                        del sdata.images[valid_image_key]
                elif len(image_keys) > 0:
                    for valid_image_key in valid_image_keys:
                        if valid_image_key not in image_keys:
                            del sdata.images[valid_image_key]

            if valid_label_keys is not None:
                if len(label_keys) == 0:
                    for valid_label_key in valid_label_keys:
                        del sdata.labels[valid_label_key]
                elif len(label_keys) > 0:
                    for valid_label_key in valid_label_keys:
                        if valid_label_key not in label_keys:
                            del sdata.labels[valid_label_key]

            if valid_shape_keys is not None:
                if len(shape_keys) == 0:
                    for valid_shape_key in valid_shape_keys:
                        del sdata.shapes[valid_shape_key]
                elif len(shape_keys) > 0:
                    for valid_shape_key in valid_shape_keys:
                        if valid_shape_key not in shape_keys:
                            del sdata.shapes[valid_shape_key]

            if valid_point_keys is not None:
                if len(point_keys) == 0:
                    for valid_point_key in valid_point_keys:
                        del sdata.points[valid_point_key]
                elif len(point_keys) > 0:
                    for valid_point_key in valid_point_keys:
                        if valid_point_key not in point_keys:
                            del sdata.points[valid_point_key]

        # subset table if it is present and the region key is a valid column
        if len(sdata.tables) != 0 and len(shape_keys + label_keys + point_keys) > 0:
            for name, table in sdata.tables.items():
                assert hasattr(table, "uns"), "Table in SpatialData object does not have 'uns'."
                assert hasattr(table, "obs"), "Table in SpatialData object does not have 'obs'."

                # create mask of used keys
                _, region_key, _ = get_table_keys(table)
                mask = table.obs[region_key]
                mask = list(mask.str.contains("|".join(shape_keys + label_keys)))

                # create copy and delete original so we can reuse slot
                old_table = table.copy()
                new_table = old_table[mask, :].copy()
                new_table.uns["spatialdata_attrs"]["region"] = list(set(new_table.obs[region_key]))
                sdata.tables[name] = new_table

        else:
            sdata.tables = {}

        return sdata

    def get_bb(
        self,
        x: Union[slice, list[int], tuple[int, int]] = (0, 0),
        y: Union[slice, list[int], tuple[int, int]] = (0, 0),
    ) -> sd.SpatialData:
        """Get bounding box around a point.

        Parameters
        ----------
        x :
            x range of the bounding box. Stepsize will be ignored if slice
        y :
            y range of the bounding box. Stepsize will be ignored if slice

        Returns
        -------
        sd.SpatialData
            subsetted SpatialData object
        """
        if not isinstance(x, (slice, list, tuple)):
            raise TypeError("Parameter 'x' must be one of 'slice', 'list', 'tuple'.")

        if isinstance(x, (list, tuple)) and len(x) == 2:
            if x[1] <= x[0]:
                raise ValueError("The current choice of 'x' would result in an empty slice.")

            x = slice(x[0], x[1])

        elif isinstance(x, slice):
            if x.stop <= x.start:
                raise ValueError("The current choice of 'x' would result in an empty slice.")
        else:
            raise ValueError("Parameter 'x' must be of length 2.")

        if not isinstance(y, (slice, list, tuple)):
            raise TypeError("Parameter 'y' must be one of 'slice', 'list', 'tuple'.")

        if isinstance(y, (list, tuple)):
            if len(y) != 2:
                raise ValueError("Parameter 'y' must be of length 2.")

            if y[1] <= y[0]:
                raise ValueError("The current choice of 'y' would result in an empty slice.")

            # y is clean
            y = slice(y[0], y[1])

        elif isinstance(y, slice) and y.stop <= y.start:
            raise ValueError("The current choice of 'x' would result in an empty slice.")

        selection = {"x": x, "y": y}  # makes use of xarray sel method

        # TODO: error handling if selection is out of bounds
        cropped_images = {key: img.sel(selection) for key, img in self._sdata.images.items()}
        cropped_labels = {key: img.sel(selection) for key, img in self._sdata.labels.items()}

        sdata = self._copy(
            images=cropped_images,
            labels=cropped_labels,
        )
        self._sdata = _verify_plotting_tree(self._sdata)

        # get current number of steps to create a unique key
        n_steps = len(self._sdata.plotting_tree.keys())
        sdata.plotting_tree[f"{n_steps+1}_get_bb"] = {
            "x": x,
            "y": y,
        }

        return sdata
