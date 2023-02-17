from typing import Callable, List, Union

import pandas as pd
import spatialdata as sd
from anndata import AnnData

from ..accessor import register_spatial_data_accessor
from .colorize import _colorize
from .render import _render_label
from .utils import _get_listed_colormap


@register_spatial_data_accessor("pp")
class PreprocessingAccessor:
    def __init__(self, sdata):
        self._sdata = sdata

    def _copy(
        self,
        images: Union[None, dict] = None,
        labels: Union[None, dict] = None,
        points: Union[None, dict] = None,
        polygons: Union[None, dict] = None,
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
            polygons=self._sdata.polygons if polygons is None else polygons,
            shapes=self._sdata.shapes if shapes is None else shapes,
            table=self._sdata.table if table is None else table,
        )

    def get_region_key(self) -> str:
        "Quick access to the data's region key."

        # TODO: check if this intended behavior
        if self._sdata.table.uns["spatialdata_attrs"]["region_key"] is None:
            return self._sdata.table.uns["spatialdata_attrs"]["region"]

        return self._sdata.table.uns["spatialdata_attrs"]["region_key"]

    def get_instance_key(self) -> str:
        "Quick access to the data's instance key."

        return self._sdata.table.uns["spatialdata_attrs"]["instance_key"]

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

        # subset table
        if "label_coords" not in self._sdata.table.obsm:
            coordinates = self._sdata.tl.label_property("centroid", return_df=True)
        else:
            coordinates = self._sdata.table.obsm["label_coords"]

        query = (
            (coordinates["x"] > x.start)
            & (coordinates["x"] < x.stop)
            & (coordinates["y"] > y.start)
            & (coordinates["y"] < y.stop)
        )

        cropped_table = self._sdata.table.copy()[query]
        cropped_table.uns["bounding_box"] = [x.start, x.stop, y.start, y.stop]

        sdata = self._copy(images=cropped_images, labels=cropped_labels, table=cropped_table)

        return sdata

    def get_images(self, keys: Union[list, str], label_func: Callable = lambda x: x) -> sd.SpatialData:
        """Get images from a list of keys.

        Parameters
        ----------
        keys : list
            list of keys to select

        Returns
        -------
        sd.SpatialData
            subsetted SpatialData object
        """
        # TODO: error handling if keys are not in images

        if not isinstance(keys, (list, str)):

            raise TypeError("Parameter 'keys' must either be of type 'str' or 'list'.")

        if isinstance(keys, list):

            if not all([isinstance(key, str) for key in keys]):

                raise TypeError("All elements in 'keys' must be of type 'str'.")

        if isinstance(keys, str):
            keys = [keys]

        assert all([isinstance(key, str) for key in keys])

        valid_keys = list(self._sdata.images.keys())

        for key in keys:

            if key not in valid_keys:

                raise ValueError(f"Key '{key}' is not a valid key. Valid choices are: " + ", ".join(valid_keys))

        selected_images = {key: img for key, img in self._sdata.images.items() if key in keys}
        # TODO: how to handle labels ? there might be multiple labels per image (e.g. nuclei and cell segmentation masks)
        selected_labels = {key: img for key, img in self._sdata.labels.items() if label_func(key) in keys}

        # initialise empty so that it is only overwritten if needed
        new_table = None
        # make sure that table exists
        if hasattr(self._sdata, "table"):

            if hasattr(self._sdata.table, "obs"):

                # create mask of used keys
                mask = self._sdata.table.obs[self._sdata.pp.get_region_key()]
                mask = list(mask.str.contains("|".join(keys)))
                # print(mask)

                new_table = self._sdata.table[mask, :]

        return self._copy(images=selected_images, labels=selected_labels, table=new_table)

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

        return self._copy(images=channels_images)

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

    def render_labels(self, alpha=0, alpha_boundary=1, mode="inner", label_func=lambda x: x):
        color_dict = {1: "white"}
        cmap = _get_listed_colormap(color_dict)

        # mask = _label_segmentation_mask(segmentation, cells_dict)
        rendered = {}

        for key, img in self._sdata.images.items():
            labels = self._sdata.labels[label_func(key)]
            rendered_image = _render_label(
                labels.values,
                cmap,
                img.values.T,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
            )
            # print(rendered.swapaxes(0, 2).shape)
            rendered[key] = sd.Image2DModel.parse(rendered_image.swapaxes(0, 2))

        return self._copy(images=rendered)
