from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import spatialdata as sd
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from pandas.api.types import is_categorical_dtype
from spatial_image import SpatialImage
from spatialdata._core.data_extent import get_extent
from spatialdata._core.query.relational_query import _locate_value
from spatialdata.transformations.operations import get_transformation

from spatialdata_plot._accessor import register_spatial_data_accessor
from spatialdata_plot._logging import logger
from spatialdata_plot.pl.render import (
    _render_images,
    _render_labels,
    _render_points,
    _render_shapes,
)
from spatialdata_plot.pl.render_params import (
    CmapParams,
    ImageRenderParams,
    LabelsRenderParams,
    LegendParams,
    PointsRenderParams,
    ShapesRenderParams,
    _FontSize,
    _FontWeight,
)
from spatialdata_plot.pl.utils import (
    _get_cs_contents,
    _get_valid_cs,
    _maybe_set_colors,
    _mpl_ax_contains_elements,
    _prepare_cmap_norm,
    _prepare_params_plot,
    _set_outline,
    save_fig,
)
from spatialdata_plot.pp.utils import _verify_plotting_tree

ColorLike = Union[tuple[float, ...], str]


@register_spatial_data_accessor("pl")
class PlotAccessor:
    """
    A class to provide plotting functions for `SpatialData` objects.

    Parameters
    ----------
    sdata :
        The `SpatialData` object to provide plotting functions for.

    Notes
    -----
    This class provides a number of methods that can be used to generate
    plots of the data stored in a `SpatialData` object. These methods are
    accessed via the `SpatialData.pl` accessor.


    """

    @property
    def sdata(self) -> sd.SpatialData:
        """The `SpatialData` object to provide plotting functions for."""
        return self._sdata

    @sdata.setter
    def sdata(self, sdata: sd.SpatialData) -> None:
        self._sdata = sdata

    def __init__(self, sdata: sd.SpatialData) -> None:
        self._sdata = sdata

    def _copy(
        self,
        images: dict[str, SpatialImage | MultiscaleSpatialImage] | None = None,
        labels: dict[str, SpatialImage | MultiscaleSpatialImage] | None = None,
        points: dict[str, DaskDataFrame] | None = None,
        shapes: dict[str, GeoDataFrame] | None = None,
        table: AnnData | None = None,
    ) -> sd.SpatialData:
        """Copy the current `SpatialData` object, optionally modifying some of its attributes.

        Parameters
        ----------
        images : dict[str, SpatialImage | MultiscaleSpatialImage] | None, optional
            A dictionary containing image data to replace the images in the
            original `SpatialData` object, or `None` to keep the original
            images. Defaults to `None`.
        labels : dict[str, SpatialImage | MultiscaleSpatialImage] | None, optional
            A dictionary containing label data to replace the labels in the
            original `SpatialData` object, or `None` to keep the original
            labels. Defaults to `None`.
        points : dict[str, DaskDataFrame] | None, optional
            A dictionary containing point data to replace the points in the
            original `SpatialData` object, or `None` to keep the original
            points. Defaults to `None`.
        shapes : dict[str, GeoDataFrame] | None, optional
            A dictionary containing shape data to replace the shapes in the
            original `SpatialData` object, or `None` to keep the original
            shapes. Defaults to `None`.
        table : AnnData | None, optional
            A dictionary or `AnnData` object containing table data to replace
            the table in the original `SpatialData` object, or `None` to keep
            the original table. Defaults to `None`.

        Returns
        -------
        sd.SpatialData
            A new `SpatialData` object that is a copy of the original
            `SpatialData` object, with any specified modifications.

        Notes
        -----
        This method creates a new `SpatialData` object with the same metadata
        and similar data as the original `SpatialData` object. The new object
        can be modified without affecting the original object.

        """
        sdata = sd.SpatialData(
            images=self._sdata.images if images is None else images,
            labels=self._sdata.labels if labels is None else labels,
            points=self._sdata.points if points is None else points,
            shapes=self._sdata.shapes if shapes is None else shapes,
            table=self._sdata.table if table is None else table,
        )
        sdata.plotting_tree = self._sdata.plotting_tree if hasattr(self._sdata, "plotting_tree") else OrderedDict()

        return sdata

    def render_shapes(
        self,
        elements: list[str] | str | None = None,
        color: str | None = None,
        fill_alpha: float | int = 1.0,
        groups: list[str] | str | None = None,
        palette: list[str] | str | None = None,
        na_color: ColorLike | None = "lightgrey",
        outline: bool = False,
        outline_width: float | int = 1.5,
        outline_color: str | list[float] = "#000000ff",
        outline_alpha: float | int = 1.0,
        layer: str | None = None,
        cmap: Colormap | str | None = None,
        norm: bool | Normalize = False,
        scale: float | int = 1.0,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render shapes elements in SpatialData.

        Parameters
        ----------
        elements : list[str] | str | None, optional
            The name(s) of the shapes element(s) to render. If `None`, all shapes
            elements in the `SpatialData` object will be used.
        color : Colorlike | str | None, optional
            Can either be a color-like or a key in :attr:`sdata.table.obs`. The latter
            can be used to color by categorical or continuous variables.
        fill_alpha : float | int, default 1.0
            Alpha value for the fill of shapes.
        groups : list[str] | str | None, optional
            When using `color` and the key represents discrete labels, `groups`
            can be used to show only a subset of them. Other values are set to NA.
        palette : list[str] | str | None, optional
            Palette for discrete annotations. List of valid color names that should be
            used for the categories. Must match the number of groups.
        na_color : str | list[float] | None, default "lightgrey"
            Color to be used for NAs values, if present. Can either be a named color
            ("red"), a hex representation ("#000000ff") or a list of floats that
            represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). When None, the values won't
            be shown.
        outline : bool, default False
            If `True`, a border around the shape elements is plotted.
        outline_width : float | int, default 1.5
            Width of the border.
        outline_color : str | list[float], default "#000000ff"
            Color of the border. Can either be a named color ("red"), a hex
            representation ("#000000ff") or a list of floats that represent RGB/RGBA
            values (1.0, 0.0, 0.0, 1.0).
        outline_alpha : float | int, default 1.0
            Alpha value for the outline of shapes.
        layer : str | None, optional
            Key in :attr:`anndata.AnnData.layers` or `None` for :attr:`anndata.AnnData.X`.
        cmap : Colormap | str | None, optional
            Colormap for discrete or continuous annotations using 'color', see :class:`matplotlib.colors.Colormap`.
        norm : bool | Normalize, default False
            Colormap normalization for continuous annotations.
        scale : float | int, default 1.0
            Value to scale circles, if present.
        **kwargs : Any
            Additional arguments to be passed to cmap and norm.

        Notes
        -----
        - Empty geometries will be removed at the time of plotting.
        - An `outline_width` of 0.0 leads to no border being plotted.
        - When passing a color-like to 'color', this has precendence over the potential existence as a column name.

        Returns
        -------
        sd.SpatialData
            The modified SpatialData object with the rendered shapes.
        """
        if elements is not None:
            if not isinstance(elements, (list, str)):
                raise TypeError("Parameter 'elements' must be a string or a list of strings.")

            elements = [elements] if isinstance(elements, str) else elements
            if any(e not in self._sdata.shapes for e in elements):
                raise ValueError(
                    "Not all specificed elements were found, available elements are: '"
                    + "', '".join(self._sdata.shapes.keys())
                    + "'"
                )

        if color is None:
            col_for_color = None

        elif colors.is_color_like(color):
            logger.info("Value for parameter 'color' appears to be a color, using it as such.")
            color = color
            col_for_color = None

        else:
            if not isinstance(color, str):
                raise TypeError(
                    "Parameter 'color' must be a string indicating which color "
                    + "in sdata.table to use for coloring the shapes."
                )
            col_for_color = color
            color = None

        # we're not enforcing the existence of 'color' here since it might
        # exist for one element in sdata.shapes, but not the others.
        # Gets validated in _set_color_source_vec()

        if not isinstance(fill_alpha, (float, int)):
            raise TypeError("Parameter 'fill_alpha' must be numeric.")

        if fill_alpha < 0:
            raise ValueError("Parameter 'fill_alpha' cannot be negative.")

        if groups is not None:
            if not isinstance(groups, (list, str)):
                raise TypeError("Parameter 'groups' must be a string or a list of strings.")
            groups = [groups] if isinstance(groups, str) else groups

        if palette is not None:
            if groups is None:
                raise ValueError("When specifying 'palette', 'groups' must also be specified.")

            if not isinstance(palette, (list, str)):
                raise TypeError("Parameter 'palette' must be a string or a list of strings.")

            palette = [palette] if isinstance(palette, str) else palette

            if len(groups) != len(palette):
                raise ValueError("The length of 'palette' and 'groups' must be the same.")

        if not colors.is_color_like(na_color):
            raise TypeError("Parameter 'na_color' must be color-like.")

        if not isinstance(outline, bool):
            raise TypeError("Parameter 'outline' must be a True or False.")

        if not isinstance(outline_width, (float, int)):
            raise TypeError("Parameter 'outline_width' must be numeric.")

        if outline_width < 0:
            raise ValueError("Parameter 'outline_width' cannot be negative.")

        if not colors.is_color_like(outline_color):
            raise TypeError("Parameter 'outline_color' must be color-like.")

        if not isinstance(outline_alpha, (float, int)):
            raise TypeError("Parameter 'outline_alpha' must be numeric.")

        if outline_alpha < 0:
            raise ValueError("Parameter 'outline_alpha' cannot be negative.")

        if layer is not None and not isinstance(layer, str):
            raise TypeError("Parameter 'layer' must be a string.")

        if layer is not None and layer not in self._sdata.table.layers:
            raise ValueError(
                f"Could not find layer '{layer}', available layers are: '"
                + "', '".join(self._sdata.table.layers.keys())
                + "'"
            )

        if cmap is not None and not isinstance(cmap, (str, Colormap)):
            raise TypeError("Parameter 'cmap' must be a mpl.Colormap or the name of one.")

        if norm is not None and not isinstance(norm, (bool, Normalize)):
            raise TypeError("Parameter 'norm' must be a boolean or a mpl.Normalize.")

        if not isinstance(scale, (float, int)):
            raise TypeError("Parameter 'scale' must be numeric.")

        if scale < 0:
            raise ValueError("Parameter 'scale' must be a positive number.")

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())
        cmap_params = _prepare_cmap_norm(
            cmap=cmap,
            norm=norm,
            na_color=na_color,  # type: ignore[arg-type]
            **kwargs,
        )
        if isinstance(elements, str):
            elements = [elements]
        outline_params = _set_outline(outline, outline_width, outline_color)
        sdata.plotting_tree[f"{n_steps+1}_render_shapes"] = ShapesRenderParams(
            elements=elements,
            color=color,
            col_for_color=col_for_color,
            groups=groups,
            scale=scale,
            outline_params=outline_params,
            layer=layer,
            cmap_params=cmap_params,
            palette=palette,
            outline_alpha=outline_alpha,
            fill_alpha=fill_alpha,
            transfunc=kwargs.get("transfunc", None),
        )

        return sdata

    def render_points(
        self,
        elements: list[str] | str | None = None,
        color: str | None = None,
        alpha: float | int = 1.0,
        groups: list[str] | str | None = None,
        palette: list[str] | str | None = None,
        na_color: ColorLike | None = "lightgrey",
        cmap: Colormap | str | None = None,
        norm: None | Normalize = None,
        size: float | int = 1.0,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render points elements in SpatialData.

        Parameters
        ----------
        elements : list[str] | str | None, optional
            The name(s) of the points element(s) to render. If `None`, all points
            elements in the `SpatialData` object will be used.
        color : Colorlike | str | None, optional
            Can either be a color-like or a key in :attr:`sdata.table.obs`. The latter
            can be used to color by categorical or continuous variables.
        alpha : float | int, default 1.0
            Alpha value for the points.
        groups : list[str] | str | None, optional
            When using `color` and the key represents discrete labels, `groups`
            can be used to show only a subset of them. Other values are set to NA.
        palette : list[str] | str | None, optional
            Palette for discrete annotations. List of valid color names that should be
            used for the categories. Must match the number of groups.
        na_color : str | list[float] | None, default "lightgrey"
            Color to be used for NAs values, if present. Can either be a named color
            ("red"), a hex representation ("#000000ff") or a list of floats that
            represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). When None, the values won't
            be shown.
        cmap : Colormap | str | None, optional
            Colormap for discrete or continuous annotations using 'color', see
            :class:`matplotlib.colors.Colormap`. If no palette is given and `color`
            refers to a categorical, the colors are sampled from this colormap.
        norm : bool | Normalize, default False
            Colormap normalization for continuous annotations.
        size : float | int, default 1.0
            Size of the points
        kwargs
            Additional arguments to be passed to cmap and norm.

        Returns
        -------
        sd.SpatialData
            The modified SpatialData object with the rendered shapes.
        """
        if elements is not None:
            if not isinstance(elements, (list, str)):
                raise TypeError("Parameter 'elements' must be a string or a list of strings.")

            elements = [elements] if isinstance(elements, str) else elements
            if any(e not in self._sdata.points for e in elements):
                raise ValueError(
                    "Not all specificed elements were found, available elements are: '"
                    + "', '".join(self._sdata.points.keys())
                    + "'"
                )

        if color is not None and not colors.is_color_like(color):
            tmp_e = self._sdata.points if elements is None else elements
            origins = [
                _locate_value(value_key=color, sdata=self._sdata, element_name=e) for e in tmp_e
            ]  # , element_name=element_name)
            if not any(origins):
                raise ValueError("The argument for 'color' is neither color-like nor in the data.")

        if color is None:
            col_for_color = None

        elif colors.is_color_like(color):
            logger.info("Value for parameter 'color' appears to be a color, using it as such.")
            color = color
            col_for_color = None

        else:
            if not isinstance(color, str):
                raise TypeError(
                    "Parameter 'color' must be a string indicating which color "
                    + "in sdata.table to use for coloring the shapes."
                )
            col_for_color = color
            color = None

        # we're not enforcing the existence of 'color' here since it might
        # exist for one element in sdata.shapes, but not the others.
        # Gets validated in _set_color_source_vec()

        if not isinstance(alpha, (int, float)):
            raise TypeError("Parameter 'alpha' must be numeric.")

        if alpha < 0:
            raise ValueError("Parameter 'alpha' cannot be negative.")

        if groups is not None:
            if not isinstance(groups, (list, str)):
                raise TypeError("Parameter 'groups' must be a string or a list of strings.")
            groups = [groups] if isinstance(groups, str) else groups

        if palette is not None:
            if groups is None:
                raise ValueError("When specifying 'palette', 'groups' must also be specified.")

            if not isinstance(palette, (list, str)):
                raise TypeError("Parameter 'palette' must be a string or a list of strings.")

            palette = [palette] if isinstance(palette, str) else palette

            if len(groups) != len(palette):
                raise ValueError("The length of 'palette' and 'groups' must be the same.")

        if not colors.is_color_like(na_color):
            raise TypeError("Parameter 'na_color' must be color-like.")

        if cmap is not None and not isinstance(cmap, (str, Colormap)):
            raise TypeError("Parameter 'cmap' must be a mpl.Colormap or the name of one.")

        if norm is not None and not isinstance(norm, (bool, Normalize)):
            raise TypeError("Parameter 'norm' must be a boolean or a mpl.Normalize.")

        if not isinstance(size, (float, int)):
            raise TypeError("Parameter 'size' must be numeric.")

        if size < 0:
            raise ValueError("Parameter 'size' must be a positive number.")

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())

        cmap_params = _prepare_cmap_norm(
            cmap=cmap,
            norm=norm,
            na_color=na_color,  # type: ignore[arg-type]
            **kwargs,
        )

        sdata.plotting_tree[f"{n_steps+1}_render_points"] = PointsRenderParams(
            elements=elements,
            color=color,
            col_for_color=col_for_color,
            groups=groups,
            cmap_params=cmap_params,
            palette=palette,
            alpha=alpha,
            transfunc=kwargs.get("transfunc", None),
            size=size,
        )

        return sdata

    def render_images(
        self,
        elements: list[str] | str | None = None,
        channel: list[str] | list[int] | str | int | None = None,
        cmap: list[Colormap] | Colormap | str | None = None,
        norm: Normalize | None = None,
        na_color: ColorLike | None = (0.0, 0.0, 0.0, 0.0),
        palette: list[str] | str | None = None,
        alpha: float | int = 1.0,
        quantiles_for_norm: tuple[float | None, float | None] | None = None,
        scale: list[str] | str | None = None,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render image elements in SpatialData.

        Parameters
        ----------
        elements : list[str] | str | None, optional
            The name(s) of the image element(s) to render. If `None`, all image
            elements in the `SpatialData` object will be used. If a string is provided,
            it is converted into a single-element list.
        channel : list[str] | list[int] | str | int | None, optional
            To select specific channels to plot. Can be a single channel name/int or a
            list of channel names/ints. If `None`, all channels will be used.
        cmap : list[Colormap] | Colormap | str | None, optional
            Colormap or list of colormaps for continuous annotations, see :class:`matplotlib.colors.Colormap`.
            Each colormap applies to a corresponding channel.
        norm : Normalize | None, optional
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
            Applies to all channels if set.
        na_color : ColorLike | None, default (0.0, 0.0, 0.0, 0.0)
            Color to be used for NA values. Accepts color-like values (string, hex, RGB(A)).
        alpha : float | int, default 1.0
            Alpha value for the images. Must be a numeric between 0 and 1.
        quantiles_for_norm : tuple[float | None, float | None] | None, optional
            Optional pair of floats (pmin < pmax, 0-100) which will be used for quantile normalization.
        scale : list[str] | str | None, optional
            Influences the resolution of the rendering. Possibilities include:
                1) `None` (default): The image is rasterized to fit the canvas size. For
                multiscale images, the best scale is selected before rasterization.
                2) A scale name: Renders the specified scale as-is (with adjustments for dpi
                in `show()`).
                3) "full": Renders the full image without rasterization. In the case of
                multiscale images, the highest resolution scale is selected. Note that
                this may result in long computing times for large images.
                4) A list matching the list of elements. Can contain `None`, scale names, or
                "full". Each scale applies to the corresponding element.
        kwargs
            Additional arguments to be passed to cmap, norm, and other rendering functions.

        Returns
        -------
        sd.SpatialData
            The SpatialData object with the rendered images.
        """
        if elements is not None:
            if not isinstance(elements, (list, str)):
                raise TypeError("Parameter 'elements' must be a string or a list of strings.")

            elements = [elements] if isinstance(elements, str) else elements
            if any(e not in self._sdata.images for e in elements):
                raise ValueError(
                    "Not all specificed elements were found, available elements are: '"
                    + "', '".join(self._sdata.images.keys())
                    + "'"
                )

        if channel is not None and not isinstance(channel, (list, str, int)):
            raise TypeError("Parameter 'channel' must be a string, an integer, or a list of strings or integers.")

        if isinstance(channel, list) and not all(isinstance(c, (str, int)) for c in channel):
            raise TypeError("Each item in 'channel' list must be a string or an integer.")

        if cmap is not None and not isinstance(cmap, (list, Colormap, str)):
            raise TypeError("Parameter 'cmap' must be a string, a Colormap, or a list of these types.")

        if isinstance(cmap, list) and not all(isinstance(c, (Colormap, str)) for c in cmap):
            raise TypeError("Each item in 'cmap' list must be a string or a Colormap.")

        if norm is not None and not isinstance(norm, Normalize):
            raise TypeError("Parameter 'norm' must be of type Normalize.")

        if na_color is not None and not colors.is_color_like(na_color):
            raise ValueError("Parameter 'na_color' must be color-like.")

        if palette is not None and not isinstance(palette, (list, str)):
            raise TypeError("Parameter 'palette' must be a string or a list of strings.")

        if not isinstance(alpha, (float, int)):
            raise TypeError("Parameter 'alpha' must be numeric.")

        if alpha < 0:
            raise ValueError("Parameter 'alpha' cannot be negative.")

        if quantiles_for_norm is None:
            quantiles_for_norm = (None, None)
        elif not isinstance(quantiles_for_norm, (list, tuple)):
            raise TypeError("Parameter 'quantiles_for_norm' must be a list or tuple of floats, or None.")
        elif len(quantiles_for_norm) != 2:
            raise ValueError("Parameter 'quantiles_for_norm' must contain exactly two elements.")
        else:
            if not all(
                isinstance(p, (float, int, type(None))) and (p is None or 0 <= p <= 100) for p in quantiles_for_norm
            ):
                raise TypeError("Each item in 'quantiles_for_norm' must be a float or int within [0, 100], or None.")

            pmin, pmax = quantiles_for_norm
            if pmin is not None and pmax is not None and pmin > pmax:
                raise ValueError("The first number in 'quantiles_for_norm' must not be smaller than the second.")

        if scale is not None and not isinstance(scale, (list, str)):
            raise TypeError("Parameter 'scale' must be a string or a list of strings.")

        if isinstance(scale, list) and not all(isinstance(s, str) for s in scale):
            raise TypeError("Each item in 'scale' list must be a string.")

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())

        cmap_params: list[CmapParams] | CmapParams
        if isinstance(cmap, list):
            cmap_params = [
                _prepare_cmap_norm(
                    cmap=c,
                    norm=norm,
                    na_color=na_color,  # type: ignore[arg-type]
                    **kwargs,
                )
                for c in cmap
            ]

        else:
            cmap_params = _prepare_cmap_norm(
                cmap=cmap,
                norm=norm,
                na_color=na_color,  # type: ignore[arg-type]
                **kwargs,
            )

        if isinstance(elements, str):
            elements = [elements]
        sdata.plotting_tree[f"{n_steps+1}_render_images"] = ImageRenderParams(
            elements=elements,
            channel=channel,
            cmap_params=cmap_params,
            palette=palette,
            alpha=alpha,
            quantiles_for_norm=quantiles_for_norm,
            scale=scale,
        )

        return sdata

    def render_labels(
        self,
        elements: list[str] | str | None = None,
        color: str | None = None,
        groups: list[str] | str | None = None,
        contour_px: int = 3,
        outline: bool = False,
        layer: str | None = None,
        palette: list[str] | str | None = None,
        cmap: Colormap | str | None = None,
        norm: Normalize | None = None,
        na_color: ColorLike | None = (0.0, 0.0, 0.0, 0.0),
        outline_alpha: float | int = 1.0,
        fill_alpha: float | int = 0.3,
        scale: list[str] | str | None = None,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render labels elements in SpatialData.

        Parameters
        ----------
        elements : list[str] | str | None, optional
            The name(s) of the label element(s) to render. If `None`, all label
            elements in the `SpatialData` object will be used.
        color : str | None, optional
            Key for annotations in :attr:`anndata.AnnData.obs` or variables/genes.
        groups : list[str] | str | None, optional
            When using `color` and the key represents discrete labels, `groups`
            can be used to show only a subset of them. Other values are set to NA.
        contour_px : int, default 3
            Draw contour of specified width for each segment. If `None`, fills
            entire segment, see :func:`skimage.morphology.erosion`.
        outline : bool, default False
            Whether to plot boundaries around segmentation masks.
        layer : str | None, optional
            Key in :attr:`anndata.AnnData.layers` or `None` for :attr:`anndata.AnnData.X`.
        palette : list[str] | str | None, optional
            Palette for discrete annotations. List of valid color names that should be
            used for the categories. Must match the number of groups.
        cmap : Colormap | str | None, optional
            Colormap for continuous annotations, see :class:`matplotlib.colors.Colormap`.
        norm : Normalize | None, optional
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
        na_color : ColorLike | None, optional
            Color to be used for NAs values, if present.
        outline_alpha : float | int, default 1.0
            Alpha value for the labels.
        fill_alpha : float | int, default 0.3
            Alpha value for the labels.
        scale : list[str] | str | None, optional
            Influences the resolution of the rendering. Possibilities for setting this parameter:
                1) None (default). The image is rasterized to fit the canvas size. For multiscale images, the best scale
                is selected before the rasterization step.
                2) Name of one of the scales in the multiscale image to be rendered. This scale is rendered as it is
                (exception: a dpi is specified in `show()`. Then the image is rasterized to fit the canvas and dpi).
                3) "full": render the full image without rasterization. In the case of a multiscale image, the scale
                with the highest resolution is selected. This can lead to long computing times for large images!
                4) List that is matched to the list of elements (can contain `None`, scale names or "full").
        kwargs
            Additional arguments to be passed to cmap and norm.

        Returns
        -------
        None
        """
        if elements is not None:
            if not isinstance(elements, (list, str)):
                raise TypeError("Parameter 'elements' must be a string or a list of strings.")

            elements = [elements] if isinstance(elements, str) else elements
            if any(e not in self._sdata.labels for e in elements):
                raise ValueError(
                    "Not all specificed elements were found, available elements are: '"
                    + "', '".join(self._sdata.labels.keys())
                    + "'"
                )

        if color is not None and not isinstance(color, str):
            raise TypeError("Parameter 'color' must be a string.")

        if groups is not None:
            if not isinstance(groups, (list, str)):
                raise TypeError("Parameter 'groups' must be a string or a list of strings.")
            groups = [groups] if isinstance(groups, str) else groups
            if not all(isinstance(g, str) for g in groups):
                raise TypeError("All items in 'groups' list must be strings.")

        if not isinstance(contour_px, int):
            raise TypeError("Parameter 'contour_px' must be an integer.")

        if not isinstance(outline, bool):
            raise TypeError("Parameter 'outline' must be a boolean.")

        if layer is not None and not isinstance(layer, str):
            raise TypeError("Parameter 'layer' must be a string.")

        if palette is not None:
            if not isinstance(palette, (list, str)):
                raise TypeError("Parameter 'palette' must be a string or a list of strings.")
            palette = [palette] if isinstance(palette, str) else palette
            if not all(isinstance(p, str) for p in palette):
                raise TypeError("All items in 'palette' list must be strings.")

        if cmap is not None and not isinstance(cmap, (Colormap, str)):
            raise TypeError("Parameter 'cmap' must be a Colormap or a string.")

        if norm is not None and not isinstance(norm, Normalize):
            raise TypeError("Parameter 'norm' must be of type Normalize.")

        if na_color is not None and not colors.is_color_like(na_color):
            raise ValueError("Parameter 'na_color' must be color-like.")

        if not isinstance(outline_alpha, float):
            raise TypeError("Parameter 'outline_alpha' must be a float.")

        if not isinstance(fill_alpha, float):
            raise TypeError("Parameter 'fill_alpha' must be a float.")

        if scale is not None:
            if not isinstance(scale, (list, str)):
                raise TypeError("Parameter 'scale' must be a string or a list of strings.")
            scale = [scale] if isinstance(scale, str) else scale
            if not all(isinstance(s, str) for s in scale):
                raise TypeError("All items in 'scale' list must be strings.")

        if (
            color is not None
            and color not in self._sdata.table.obs.columns
            and color not in self._sdata.table.var_names
        ):
            raise ValueError(f"'{color}' is not a valid table column.")

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())
        cmap_params = _prepare_cmap_norm(
            cmap=cmap,
            norm=norm,
            na_color=na_color,  # type: ignore[arg-type]
            **kwargs,
        )
        if isinstance(elements, str):
            elements = [elements]
        sdata.plotting_tree[f"{n_steps+1}_render_labels"] = LabelsRenderParams(
            elements=elements,
            color=color,
            groups=groups,
            contour_px=contour_px,
            outline=outline,
            layer=layer,
            cmap_params=cmap_params,
            palette=palette,
            outline_alpha=outline_alpha,
            fill_alpha=fill_alpha,
            transfunc=kwargs.get("transfunc", None),
            scale=scale,
        )

        return sdata

    def show(
        self,
        coordinate_systems: list[str] | str | None = None,
        legend_fontsize: int | float | _FontSize | None = None,
        legend_fontweight: int | _FontWeight = "bold",
        legend_loc: str | None = "right margin",
        legend_fontoutline: int | None = None,
        na_in_legend: bool = True,
        colorbar: bool = True,
        wspace: float | None = None,
        hspace: float = 0.25,
        ncols: int = 4,
        frameon: bool | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        fig: Figure | None = None,
        title: list[str] | str | None = None,
        share_extent: bool = True,
        pad_extent: int | float = 0,
        ax: list[Axes] | Axes | None = None,
        return_ax: bool = False,
        save: str | Path | None = None,
    ) -> sd.SpatialData:
        """
        Plot the images in the SpatialData object.

        Parameters
        ----------
        coordinate_systems :
            Name(s) of the coordinate system(s) to be plotted. If None, all coordinate systems are plotted.
            If a coordinate system doesn't contain any relevant elements (as specified in the render_* calls),
            it is automatically not plotted.
        figsize :
            Size of the figure (width, height) in inches. The size of the actual canvas may deviate from this,
            depending on the dpi! In matplotlib, the actual figure size (in pixels) is dpi * figsize.
            If None, the default of matlotlib is used (6.4, 4.8)
        dpi :
            Resolution of the plot in dots per inch (as in matplotlib).
            If None, the default of matplotlib is used (100.0).
        ax :
            Matplotlib axes object to plot on. If None, a new figure is created.
            Works only if there is one image in the SpatialData object.
        ncols :
            Number of columns in the figure. Default is 4.

        Returns
        -------
        sd.SpatialData
            A SpatialData object.
        """
        # copy the SpatialData object so we don't modify the original
        try:
            plotting_tree = self._sdata.plotting_tree
        except AttributeError as e:
            raise TypeError(
                "Please specify what to plot using the 'render_*' functions before calling 'show()`."
            ) from e

        if coordinate_systems is not None and not isinstance(coordinate_systems, (list, str)):
            raise TypeError("Parameter 'coordinate_systems' must be a string or a list of strings.")

        font_weights = ["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
        if legend_fontweight is not None and (
            not isinstance(legend_fontweight, (int, str))
            or (isinstance(legend_fontweight, str) and legend_fontweight not in font_weights)
        ):
            readable_font_weights = ", ".join(font_weights[:-1]) + ", or " + font_weights[-1]
            raise TypeError(
                "Parameter 'legend_fontweight' must be an integer or one of",
                f"the following strings: {readable_font_weights}.",
            )

        font_sizes = ["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]

        if legend_fontsize is not None and (
            not isinstance(legend_fontsize, (int, float, str))
            or (isinstance(legend_fontsize, str) and legend_fontsize not in font_sizes)
        ):
            readable_font_sizes = ", ".join(font_sizes[:-1]) + ", or " + font_sizes[-1]
            raise TypeError(
                "Parameter 'legend_fontsize' must be an integer, a float, or ",
                f"one of the following strings: {readable_font_sizes}.",
            )

        if legend_loc is not None and not isinstance(legend_loc, str):
            raise TypeError("Parameter 'legend_loc' must be a string.")

        if legend_fontoutline is not None and not isinstance(legend_fontoutline, int):
            raise TypeError("Parameter 'legend_fontoutline' must be an integer.")

        if not isinstance(na_in_legend, bool):
            raise TypeError("Parameter 'na_in_legend' must be a boolean.")

        if not isinstance(colorbar, bool):
            raise TypeError("Parameter 'colorbar' must be a boolean.")

        if wspace is not None and not isinstance(wspace, float):
            raise TypeError("Parameter 'wspace' must be a float.")

        if not isinstance(hspace, float):
            raise TypeError("Parameter 'hspace' must be a float.")

        if not isinstance(ncols, int):
            raise TypeError("Parameter 'ncols' must be an integer.")

        if frameon is not None and not isinstance(frameon, bool):
            raise TypeError("Parameter 'frameon' must be a boolean.")

        if figsize is not None and not isinstance(figsize, tuple):
            raise TypeError("Parameter 'figsize' must be a tuple of two floats.")

        if dpi is not None and not isinstance(dpi, int):
            raise TypeError("Parameter 'dpi' must be an integer.")

        if fig is not None and not isinstance(fig, Figure):
            raise TypeError("Parameter 'fig' must be a matplotlib.figure.Figure.")

        if title is not None and not isinstance(title, (list, str)):
            raise TypeError("Parameter 'title' must be a string or a list of strings.")

        if not isinstance(share_extent, bool):
            raise TypeError("Parameter 'share_extent' must be a boolean.")

        if not isinstance(pad_extent, (int, float)):
            raise TypeError("Parameter 'pad_extent' must be numeric.")

        if ax is not None and not isinstance(ax, (Axes, list)):
            raise TypeError("Parameter 'ax' must be a matplotlib.axes.Axes or a list of Axes.")

        if not isinstance(return_ax, bool):
            raise TypeError("Parameter 'return_ax' must be a boolean.")

        if save is not None and not isinstance(save, (str, Path)):
            raise TypeError("Parameter 'save' must be a string or a pathlib.Path.")

        sdata = self._copy()

        # Evaluate execution tree for plotting
        valid_commands = [
            "get_elements",
            "get_bb",
            "render_images",
            "render_shapes",
            "render_labels",
            "render_points",
        ]

        # prepare rendering params
        render_cmds = []
        for cmd, params in plotting_tree.items():
            # strip prefix from cmd and verify it's valid
            cmd = "_".join(cmd.split("_")[1:])

            if cmd not in valid_commands:
                raise ValueError(f"Command {cmd} is not valid.")

            if "render" in cmd:
                # verify that rendering commands have been called before
                render_cmds.append((cmd, params))

        if not render_cmds:
            raise TypeError("Please specify what to plot using the 'render_*' functions before calling 'imshow()'.")

        if title is not None:
            if isinstance(title, str):
                title = [title]

            if not all(isinstance(t, str) for t in title):
                raise TypeError("All titles must be strings.")

        # get original axis extent for later comparison
        ax_x_min, ax_x_max = (np.inf, -np.inf)
        ax_y_min, ax_y_max = (np.inf, -np.inf)

        if isinstance(ax, Axes) and _mpl_ax_contains_elements(ax):
            ax_x_min, ax_x_max = ax.get_xlim()
            ax_y_max, ax_y_min = ax.get_ylim()  # (0, 0) is top-left

        # handle coordinate system
        coordinate_systems = sdata.coordinate_systems if coordinate_systems is None else coordinate_systems
        if isinstance(coordinate_systems, str):
            coordinate_systems = [coordinate_systems]

        for cs in coordinate_systems:
            if cs not in sdata.coordinate_systems:
                raise ValueError(f"Unknown coordinate system '{cs}', valid choices are: {sdata.coordinate_systems}")

        # Check if user specified only certain elements to be plotted
        cs_contents = _get_cs_contents(sdata)
        elements_to_be_rendered = []
        for cmd, params in render_cmds:
            if cmd == "render_images" and cs_contents.query(f"cs == '{cs}'")["has_images"][0]:  # noqa: SIM114
                if params.elements is not None:
                    elements_to_be_rendered += (
                        [params.elements] if isinstance(params.elements, str) else params.elements
                    )
            elif cmd == "render_shapes" and cs_contents.query(f"cs == '{cs}'")["has_shapes"][0]:  # noqa: SIM114
                if params.elements is not None:
                    elements_to_be_rendered += (
                        [params.elements] if isinstance(params.elements, str) else params.elements
                    )
            elif cmd == "render_points" and cs_contents.query(f"cs == '{cs}'")["has_points"][0]:  # noqa: SIM114
                if params.elements is not None:
                    elements_to_be_rendered += (
                        [params.elements] if isinstance(params.elements, str) else params.elements
                    )
            elif cmd == "render_labels" and cs_contents.query(f"cs == '{cs}'")["has_labels"][0]:  # noqa: SIM102
                if params.elements is not None:
                    elements_to_be_rendered += (
                        [params.elements] if isinstance(params.elements, str) else params.elements
                    )

        # filter out cs without relevant elements
        cmds = [cmd for cmd, _ in render_cmds]
        coordinate_systems = _get_valid_cs(
            sdata=sdata,
            coordinate_systems=coordinate_systems,
            render_images="render_images" in cmds,
            render_labels="render_labels" in cmds,
            render_points="render_points" in cmds,
            render_shapes="render_shapes" in cmds,
            elements=elements_to_be_rendered,
        )

        # catch error in ruff-friendly way
        if ax is not None:  # we'll generate matching number then
            n_ax = 1 if isinstance(ax, Axes) else len(ax)
            if len(coordinate_systems) != n_ax:
                raise ValueError(
                    f"Mismatch between number of matplotlib axes objects ({n_ax}) "
                    f"and number of coordinate systems ({len(coordinate_systems)})."
                )

        # set up canvas
        fig_params, scalebar_params = _prepare_params_plot(
            num_panels=len(coordinate_systems),
            figsize=figsize,
            dpi=dpi,
            fig=fig,
            ax=ax,
            wspace=wspace,
            hspace=hspace,
            ncols=ncols,
            frameon=frameon,
        )
        legend_params = LegendParams(
            legend_fontsize=legend_fontsize,
            legend_fontweight=legend_fontweight,
            legend_loc=legend_loc,
            legend_fontoutline=legend_fontoutline,
            na_in_legend=na_in_legend,
            colorbar=colorbar,
        )

        cs_contents = _get_cs_contents(sdata)

        # go through tree

        for i, cs in enumerate(coordinate_systems):
            sdata = self._copy()
            _, has_images, has_labels, has_points, has_shapes = (
                cs_contents.query(f"cs == '{cs}'").iloc[0, :].values.tolist()
            )
            ax = fig_params.ax if fig_params.axs is None else fig_params.axs[i]

            wants_images = False
            wants_labels = False
            wants_points = False
            wants_shapes = False
            wanted_elements = []

            for cmd, params in render_cmds:
                if cmd == "render_images" and has_images:
                    wants_images = True
                    wanted_images = params.elements if params.elements is not None else list(sdata.images.keys())
                    wanted_images_on_this_cs = [
                        image
                        for image in wanted_images
                        if cs in set(get_transformation(sdata.images[image], get_all=True).keys())
                    ]
                    wanted_elements.extend(wanted_images_on_this_cs)
                    if wanted_images_on_this_cs:
                        rasterize = (params.scale is None) or (
                            isinstance(params.scale, str)
                            and params.scale != "full"
                            and (dpi is not None or figsize is not None)
                        )
                        _render_images(
                            sdata=sdata,
                            render_params=params,
                            coordinate_system=cs,
                            ax=ax,
                            fig_params=fig_params,
                            scalebar_params=scalebar_params,
                            legend_params=legend_params,
                            rasterize=rasterize,
                        )

                elif cmd == "render_shapes" and has_shapes:
                    wants_shapes = True
                    wanted_shapes = params.elements if params.elements is not None else list(sdata.shapes.keys())
                    wanted_shapes_on_this_cs = [
                        shape
                        for shape in wanted_shapes
                        if cs in set(get_transformation(sdata.shapes[shape], get_all=True).keys())
                    ]
                    wanted_elements.extend(wanted_shapes_on_this_cs)
                    if wanted_shapes_on_this_cs:
                        _render_shapes(
                            sdata=sdata,
                            render_params=params,
                            coordinate_system=cs,
                            ax=ax,
                            fig_params=fig_params,
                            scalebar_params=scalebar_params,
                            legend_params=legend_params,
                        )

                elif cmd == "render_points" and has_points:
                    wants_points = True
                    wanted_points = params.elements if params.elements is not None else list(sdata.points.keys())
                    wanted_points_on_this_cs = [
                        point
                        for point in wanted_points
                        if cs in set(get_transformation(sdata.points[point], get_all=True).keys())
                    ]
                    wanted_elements.extend(wanted_points_on_this_cs)
                    if wanted_points_on_this_cs:
                        _render_points(
                            sdata=sdata,
                            render_params=params,
                            coordinate_system=cs,
                            ax=ax,
                            fig_params=fig_params,
                            scalebar_params=scalebar_params,
                            legend_params=legend_params,
                        )

                elif cmd == "render_labels" and has_labels:
                    if sdata.table is not None and isinstance(params.color, str):
                        colors = sc.get.obs_df(sdata.table, params.color)
                        if is_categorical_dtype(colors):
                            _maybe_set_colors(
                                source=sdata.table,
                                target=sdata.table,
                                key=params.color,
                                palette=params.palette,
                            )
                    wants_labels = True
                    wanted_labels = params.elements if params.elements is not None else list(sdata.labels.keys())
                    wanted_labels_on_this_cs = [
                        label
                        for label in wanted_labels
                        if cs in set(get_transformation(sdata.labels[label], get_all=True).keys())
                    ]
                    wanted_elements.extend(wanted_labels_on_this_cs)
                    if wanted_labels_on_this_cs:
                        rasterize = (params.scale is None) or (
                            isinstance(params.scale, str)
                            and params.scale != "full"
                            and (dpi is not None or figsize is not None)
                        )
                        _render_labels(
                            sdata=sdata,
                            render_params=params,
                            coordinate_system=cs,
                            ax=ax,
                            fig_params=fig_params,
                            scalebar_params=scalebar_params,
                            legend_params=legend_params,
                            rasterize=rasterize,
                        )

                if title is None:
                    t = cs
                elif len(title) == 1:
                    t = title[0]
                else:
                    try:
                        t = title[i]
                    except IndexError as e:
                        raise IndexError("The number of titles must match the number of coordinate systems.") from e
                ax.set_title(t)
                ax.set_aspect("equal")

            extent = get_extent(
                sdata,
                coordinate_system=cs,
                has_images=has_images and wants_images,
                has_labels=has_labels and wants_labels,
                has_points=has_points and wants_points,
                has_shapes=has_shapes and wants_shapes,
                elements=wanted_elements,
            )
            cs_x_min, cs_x_max = extent["x"]
            cs_y_min, cs_y_max = extent["y"]

            if any([has_images, has_labels, has_points, has_shapes]):
                # If the axis already has limits, only expand them but not overwrite
                x_min = min(ax_x_min, cs_x_min) - pad_extent
                x_max = max(ax_x_max, cs_x_max) + pad_extent
                y_min = min(ax_y_min, cs_y_min) - pad_extent
                y_max = max(ax_y_max, cs_y_max) + pad_extent
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_max, y_min)  # (0, 0) is top-left

        if fig_params.fig is not None and save is not None:
            save_fig(fig_params.fig, path=save)

        # Manually show plot if we're not in interactive mode
        # https://stackoverflow.com/a/64523765
        if not hasattr(sys, "ps1"):
            plt.show()
        return (fig_params.ax if fig_params.axs is None else fig_params.axs) if return_ax else None  # shuts up ruff
