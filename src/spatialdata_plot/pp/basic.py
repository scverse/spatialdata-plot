from typing import Callable, List, Union

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

    def get_bb(self, x: Union[slice, list, tuple], y: Union[slice, list, tuple]) -> sd.SpatialData:
        """Get bounding box around a point.

        Parameters
        ----------
        x : Union[slice, list, tuple]
            x range of the bounding box
        y : Union[slice, list, tuple]
            y range of the bounding box

        Returns
        -------
        sd.SpatialData
            subsetted SpatialData object
        """
        # TODO: add support for list and tuple inputs ? (currently only slice is supported)
        selection = dict(x=x, y=y)  # makes use of xarray sel method

        # TODO: error handling if selection is out of bounds
        cropped_images = {key: img.sel(selection) for key, img in self._sdata.images.items()}
        cropped_labels = {key: img.sel(selection) for key, img in self._sdata.labels.items()}

        sdata = self._copy(
            images=cropped_images,
            labels=cropped_labels,
        )

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

        if isinstance(keys, str):
            keys = [keys]

        selected_images = {key: img for key, img in self._sdata.images.items() if key in keys}
        # TODO: how to handle labels ? there might be multiple labels per image (e.g. nuclei and cell segmentation masks)
        selected_labels = {key: img for key, img in self._sdata.labels.items() if label_func(key) in keys}

        return self._copy(images=selected_images, labels=selected_labels)

    def get_channels(self, keys: Union[list, slice]) -> sd.SpatialData:
        """Get channels from a list of keys.

        Parameters
        ----------
        keys : list
            list of keys to select

        Returns
        -------
        sd.SpatialData
            subsetted SpatialData object
        """
        selection = dict(c=keys)
        # TODO: error handling if selection is out of bounds
        channels_images = {key: img.sel(selection) for key, img in self._sdata.images.items()}

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
