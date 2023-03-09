from typing import Callable, List, Union
from collections import OrderedDict
import spatialdata as sd
from anndata import AnnData
import pandas as pd


from ..accessor import register_spatial_data_accessor
from .colorize import _colorize
from .render import _render_label
from .utils import _get_listed_colormap
from spatialdata._core._spatialdata_ops import get_transformation


@register_spatial_data_accessor("pp")
class PreprocessingAccessor:
    def __init__(self, sdata):
        self._sdata = sdata

    def _copy(
        self,
        images: Union[None, dict] = None,
        labels: Union[None, dict] = None,
        points: Union[None, dict] = None,
        shapes: Union[None, dict] = None,
        table: Union[dict, AnnData] = None,
    ) -> sd.SpatialData:

        """
        Helper function to copies the references from the original SpatialData
        object to the subsetted SpatialData object.
        """

        return sd.SpatialData(
            images=self._sdata.images if images is None else images,
            labels=self._sdata.labels if labels is None else labels,
            points=self._sdata.points if points is None else points,
            shapes=self._sdata.shapes if shapes is None else shapes,
            table=self._sdata.table if table is None else table,
        )

    def _verify_plotting_tree_exists(self):

        if not hasattr(self._sdata, "plotting_tree"):
            self._sdata.plotting_tree = OrderedDict()

    def _get_coordinate_system_mapping(self) -> dict:

        has_images = hasattr(self._sdata, "images")
        has_labels = hasattr(self._sdata, "labels")
        has_polygons = hasattr(self._sdata, "polygons")

        coordsys_keys = self._sdata.coordinate_systems
        image_keys = self._sdata.images.keys() if has_images else []
        label_keys = self._sdata.labels.keys() if has_labels else []
        polygon_keys = self._sdata.images.keys() if has_polygons else []

        mapping = {}

        if len(coordsys_keys) < 1:

            raise ValueError("SpatialData object must have at least one coordinate system to generate a mapping.")

        for key in coordsys_keys:

            mapping[key] = []

            for image_key in image_keys:

                transformations = get_transformation(self._sdata.images[image_key], get_all=True)

                if key in list(transformations.keys()):

                    mapping[key].append(image_key)

            for label_key in label_keys:

                transformations = get_transformation(self._sdata.labels[label_key], get_all=True)

                if key in list(transformations.keys()):

                    mapping[key].append(label_key)

            for polygon_key in polygon_keys:

                transformations = get_transformation(self._sdata.polygons[polygon_key], get_all=True)

                if key in list(transformations.keys()):

                    mapping[key].append(polygon_key)

        return mapping

    def _get_region_key(self) -> str:

        "Quick access to the data's region key."

        return self._sdata.table.uns["spatialdata_attrs"]["region_key"]

    def get_elements(self, elements: Union[str, List[str]]) -> sd.SpatialData:

        if not isinstance(elements, (str, list)):
            raise TypeError("Parameter 'elements' must be a string or a list of strings.")

        if not all([isinstance(e, str) for e in elements]):
            raise TypeError("When parameter 'elements' is a list, all elements must be strings.")

        if isinstance(elements, str):

            elements = [elements]

        coord_keys = []
        image_keys = []
        label_keys = []
        polygon_keys = []

        # prepare list of valid keys to sort elements on
        valid_coord_keys = self._sdata.coordinate_systems if hasattr(self._sdata, "coordinate_systems") else None
        valid_image_keys = list(self._sdata.images.keys()) if hasattr(self._sdata, "images") else None
        valid_label_keys = list(self._sdata.labels.keys()) if hasattr(self._sdata, "labels") else None
        valid_polygon_keys = list(self._sdata.polygons.keys()) if hasattr(self._sdata, "polygons") else None

        # for key_dict in [coord_keys, image_keys, label_keys, polygon_keys]:
        #     key_dict = []
        #     key_dict["implicit"] = []

        # first, extract coordinate system keys becasuse they generate implicit keys
        mapping = self._get_coordinate_system_mapping()
        implicit_keys = []
        for e in elements:
            if (valid_coord_keys is not None) and (e in valid_coord_keys):
                coord_keys.append(e)
                implicit_keys += mapping[e]

        for e in elements + implicit_keys:
            if (valid_coord_keys is not None) and (e in valid_coord_keys):
                coord_keys.append(e)
            elif (valid_image_keys is not None) and (e in valid_image_keys):
                image_keys.append(e)
            elif (valid_label_keys is not None) and (e in valid_label_keys):
                label_keys.append(e)
            elif (valid_polygon_keys is not None) and (e in valid_polygon_keys):
                polygon_keys.append(e)
            else:
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
                if valid_polygon_keys is not None:
                    msg += "\n\npolygons\n├ "
                    msg += "\n├ ".join(valid_polygon_keys)
                raise ValueError(msg)

        # copy that we hard-modify
        sdata = self._copy()

        if (valid_coord_keys is not None) and (len(coord_keys) > 0):
            sdata = sdata.filter_by_coordinate_system(coord_keys)

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

            if valid_polygon_keys is not None:
                if len(polygon_keys) == 0:
                    for valid_polygon_key in valid_polygon_keys:
                        del sdata.polygons[valid_polygon_key]
                elif len(polygon_keys) > 0:
                    for valid_polygon_key in valid_polygon_keys:
                        if valid_polygon_key not in polygon_keys:
                            del sdata.polygons[valid_polygon_key]

        # subset table if label info is given
        if len(label_keys) > 0:

            assert hasattr(sdata, "table"), "SpatialData object does not have a table."
            assert hasattr(sdata.table, "uns"), "Table in SpatialData object does not have 'uns'."
            assert hasattr(sdata.table, "obs"), "Table in SpatialData object does not have 'obs'."

            # create mask of used keys
            mask = sdata.table.obs[sdata.pp._get_region_key()]
            mask = list(mask.str.contains("|".join(label_keys)))

            # create copy and delete original so we can reuse slot
            table = sdata.table.copy()
            table.uns["spatialdata_attrs"]["region"] = label_keys
            del sdata.table

            sdata.table = table[mask, :].copy()

        else:
            del sdata.table

        return sdata

    def get_bb(self, x: Union[slice, list, tuple], y: Union[slice, list, tuple]) -> sd.SpatialData:

        """Get bounding box around a point.

        Parameters
        ----------
        x : Union[slice, list, tuple]
            x range of the bounding box. Stepsize will be ignored if slice
        y : Union[slice, list, tuple]
            y range of the bounding box. Stepsize will be ignored if slice

        Returns
        -------
        sd.SpatialData
            subsetted SpatialData object
        """

        if not isinstance(x, (slice, list, tuple)):

            raise TypeError("Parameter 'x' must be one of 'slice', 'list', 'tuple'.")

        if isinstance(x, (list, tuple)):

            if len(x) != 2:

                raise ValueError("Parameter 'x' must be of length 2.")

            if x[1] <= x[0]:

                raise ValueError("The current choice of 'x' would result in an empty slice.")

            # x is clean
            x = slice(x[0], x[1])

        elif isinstance(x, slice):

            if x.stop <= x.start:

                raise ValueError("The current choice of 'x' would result in an empty slice.")

        if not isinstance(y, (slice, list, tuple)):

            raise TypeError("Parameter 'y' must be one of 'slice', 'list', 'tuple'.")

        if isinstance(y, (list, tuple)):

            if len(y) != 2:

                raise ValueError("Parameter 'y' must be of length 2.")

            if y[1] <= y[0]:

                raise ValueError("The current choice of 'y' would result in an empty slice.")

            # y is clean
            y = slice(y[0], y[1])

        elif isinstance(y, slice):

            if y.stop <= y.start:

                raise ValueError("The current choice of 'x' would result in an empty slice.")

        selection = dict(x=x, y=y)  # makes use of xarray sel method

        # TODO: error handling if selection is out of bounds
        cropped_images = {key: img.sel(selection) for key, img in self._sdata.images.items()}
        cropped_labels = {key: img.sel(selection) for key, img in self._sdata.labels.items()}

        sdata = self._copy(
            images=cropped_images,
            labels=cropped_labels,
        )
        self._verify_plotting_tree_exists()

        # get current number of steps to create a unique key
        n_steps = self._sdata.plotting_tree.keys()
        sdata.plotting_tree[f"{n_steps+1}_get_bb"] = {
            "x": x,
            "y": y,
        }

        return sdata

    # def get_images(self, keys: Union[list, str], label_func: Callable = lambda x: x) -> sd.SpatialData:
    #     """Get images from a list of keys.

    #     Parameters
    #     ----------
    #     keys : list
    #         list of keys to select

    #     Returns
    #     -------
    #     sd.SpatialData
    #         subsetted SpatialData object
    #     """
    #     # TODO: error handling if keys are not in images

    #     if not isinstance(keys, (list, str)):

    #         raise TypeError("Parameter 'keys' must either be of type 'str' or 'list'.")

    #     if isinstance(keys, list):

    #         if not all([isinstance(key, str) for key in keys]):

    #             raise TypeError("All elements in 'keys' must be of type 'str'.")

    #     self._verify_plotting_tree_exists()

    #     # get current number of steps to create a unique key
    #     table = self._sdata.table.copy()
    #     n_steps = self._sdata.plotting_tree.keys()
    #     sdata.plotting_tree[f"{n_steps+1}_get_images"] = {
    #         "keys": keys,
    #         "label_func": label_func,
    #     }

    #     if isinstance(keys, str):
    #         keys = [keys]

    #     assert all([isinstance(key, str) for key in keys])

    #     valid_keys = list(self._sdata.images.keys())

    #     for key in keys:

    #         if key not in valid_keys:

    #             raise ValueError(f"Key '{key}' is not a valid key. Valid choices are: " + ", ".join(valid_keys))

    #     selected_images = {key: img for key, img in self._sdata.images.items() if key in keys}
    #     # TODO: how to handle labels ? there might be multiple labels per image (e.g. nuclei and cell segmentation masks)
    #     selected_labels = {key: img for key, img in self._sdata.labels.items() if label_func(key) in keys}

    #     # initialise empty so that it is only overwritten if needed
    #     new_table = None
    #     # make sure that table exists
    #     if hasattr(self._sdata, "table"):

    #         if hasattr(self._sdata.table, "obs"):

    #             # create mask of used keys
    #             mask = self._sdata.table.obs[self._sdata.pp._get_region_key()]
    #             mask = list(mask.str.contains("|".join(keys)))
    #             # print(mask)

    #             new_table = table[mask, :]

    #     return self._copy(images=selected_images, labels=selected_labels, table=new_table)

    def get_channels(self, channels: Union[list, slice]) -> sd.SpatialData:
        """Subset a spatialdata object to the selected channels.

        Images that don't have the selected channels will be dropped.

        """

        if not isinstance(channels, (list, slice)):

            raise TypeError("Parameter 'channels' must either be of type 'list' or 'slice'.")

        if isinstance(channels, list):

            if not all([isinstance(channel, int) for channel in channels]):

                raise TypeError("All elements in 'channels' must be of type 'int'.")

        if isinstance(channels, list):

            if not len(channels) > 0:

                raise ValueError("The list of channels cannot be empty.")

        self._verify_plotting_tree_exists()

        # get current number of steps to create a unique key
        table = self._sdata.table.copy()
        n_steps = self._sdata.plotting_tree.keys()
        sdata.plotting_tree[f"{n_steps+1}_get_channels"] = {
            "channels": channels,
        }

        # validate that selection is within bounds
        # 1) parse slice into list, respecting stepsize
        if isinstance(channels, slice):

            channels = [x for x in range(start=channels.start, stop=channels.stop, step=channels.step or 1)]

        # 2) check which images have how many channels
        image_names = []
        n_channels = []
        for image_name, image in self._sdata.images.items():
            image_names.append(image_name)
            n_channels.append(image.shape[0])

        channels_in_image = pd.DataFrame({"image_name": image_names, "n_channels": n_channels})

        # 3) drop images that don't have enough channels for the selection
        channels_in_image = channels_in_image[channels_in_image.n_channels - 1 >= max(channels)]
        valid_images = channels_in_image.image_name.values.tolist()
        sdata_with_valid_images = self._sdata.pp.get_images(keys=valid_images)

        if len(sdata_with_valid_images.images.keys()) < 1:

            raise ValueError("The choice of channels results in an empty selection.")

        selected_channels = dict(c=channels)
        channels_images = {key: img.sel(selected_channels) for key, img in self._sdata.images.items()}

        return self._copy(images=channels_images, table=table)

    def colorize(
        self,
        colors: List[str] = ["C0", "C1", "C2", "C3"],
        background: str = "black",
        normalize: bool = True,
        merge=True,
    ) -> sd.SpatialData:
        """Colorizes a stack of images.

        Parameters
        ----------
        colors: List[str]
            A list of strings that denote the color of each channel.
        background: float
            Background color of the colorized image.
        normalize: bool
            Normalizes the image prior to colorizing it.
        merge: True
            Merge the channel dimension.


        Returns
        -------
        xr.Dataset
            The image container with the colorized image stored in Layers.PLOT.
        """
        rendered = {}

        for key, img in self._sdata.images.items():
            colored_image = _colorize(
                img,
                colors=colors,
                background=background,
                normalize=normalize,
            ).sum(0)
            rendered[key] = sd.Image2DModel.parse(colored_image.swapaxes(0, 2))

        return self._copy(images=rendered)

    def render_labels(
        self,
        border_colour: Union[str, None] = "#000000",
        border_alpha=1,
        fill_colour: Union[str, None] = None,
        fill_alpha=1,
        mode="inner",
        **kwargs,
    ):

        self._verify_plotting_tree_exists()

        # get current number of steps to create a unique key
        sdata = self._copy()
        n_steps = self._sdata.plotting_tree.keys()
        sdata.plotting_tree[f"{n_steps+1}_render_labels"] = {
            "border_colour": border_colour,
            "border_alpha": border_alpha,
            "fill_colour": fill_colour,
            "fill_alpha": fill_alpha,
            "mode": mode,
            "kwargs": kwargs,
        }

        # color_dict = {1: "white"}
        # cmap = _get_listed_colormap(color_dict)

        # # mask = _label_segmentation_mask(segmentation, cells_dict)
        # rendered = {}

        # for key, img in self._sdata.images.items():
        #     labels = self._sdata.labels[label_func(key)]
        #     rendered_image = _render_label(
        #         labels.values,
        #         cmap,
        #         img.values.T,
        #         alpha=alpha,
        #         alpha_boundary=alpha_boundary,
        #         mode=mode,
        #     )
        #     # print(rendered.swapaxes(0, 2).shape)
        #     rendered[key] = sd.Image2DModel.parse(rendered_image.swapaxes(0, 2))

        return sdata

    def render_images(self, **kwargs):

        self._verify_plotting_tree_exists()

        # get current number of steps to create a unique key
        sdata = self._copy()
        n_steps = self._sdata.plotting_tree.keys()
        sdata.plotting_tree[f"{n_steps+1}_render_images"] = {
            "kwargs": kwargs,
        }

        return sdata

    def render_shapes(self, **kwargs):

        self._verify_plotting_tree_exists()

        # get current number of steps to create a unique key
        sdata = self._copy()
        n_steps = self._sdata.plotting_tree.keys()
        sdata.plotting_tree[f"{n_steps+1}_render_shapes"] = {
            "kwargs": kwargs,
        }

        return sdata

    def render_points(self, **kwargs):

        self._verify_plotting_tree_exists()

        sdata = self._copy()
        n_steps = self._sdata.plotting_tree.keys()
        sdata.plotting_tree[f"{n_steps+1}_render_points"] = {
            "kwargs": kwargs,
        }

        return sdata
