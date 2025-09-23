from __future__ import annotations

import sys
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from spatialdata import get_extent
from spatialdata._utils import _deprecation_alias
from xarray import DataArray, DataTree

from spatialdata_plot._accessor import register_spatial_data_accessor
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
    _get_elements_to_be_rendered,
    _get_valid_cs,
    _get_wanted_render_elements,
    _maybe_set_colors,
    _mpl_ax_contains_elements,
    _prepare_cmap_norm,
    _prepare_params_plot,
    _set_outline,
    _validate_image_render_params,
    _validate_label_render_params,
    _validate_points_render_params,
    _validate_shape_render_params,
    _validate_show_parameters,
    _verify_plotting_tree,
    save_fig,
)

# replace with
# from spatialdata._types import ColorLike
# once https://github.com/scverse/spatialdata/pull/689/ is in a release
ColorLike = tuple[float, ...] | str


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
        images: dict[str, DataArray | DataTree] | None = None,
        labels: dict[str, DataArray | DataTree] | None = None,
        points: dict[str, DaskDataFrame] | None = None,
        shapes: dict[str, GeoDataFrame] | None = None,
        tables: dict[str, AnnData] | None = None,
    ) -> sd.SpatialData:
        """Copy the current `SpatialData` object, optionally modifying some of its attributes.

        Parameters
        ----------
        images : dict[str, DataArray | DataTree] | None, optional
            A dictionary containing image data to replace the images in the
            original `SpatialData` object, or `None` to keep the original
            images. Defaults to `None`.
        labels : dict[str, DataArray | DataTree] | None, optional
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
        table : dict[str, AnnData] | None, optional
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
            tables=self._sdata.tables if tables is None else tables,
        )
        sdata.plotting_tree = self._sdata.plotting_tree if hasattr(self._sdata, "plotting_tree") else OrderedDict()

        return sdata

    @_deprecation_alias(elements="element", version="0.3.0")
    def render_shapes(
        self,
        element: str | None = None,
        color: str | None = None,
        fill_alpha: float | int = 1.0,
        groups: list[str] | str | None = None,
        palette: list[str] | str | None = None,
        na_color: ColorLike | None = "default",
        outline_width: float | int = 1.5,
        outline_color: str | list[float] = "#000000",
        outline_alpha: float | int = 0.0,
        cmap: Colormap | str | None = None,
        norm: Normalize | None = None,
        scale: float | int = 1.0,
        method: str | None = None,
        table_name: str | None = None,
        table_layer: str | None = None,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render shapes elements in SpatialData.

        In case of no elements specified, "broadcasting" of parameters is applied. This means that for any particular
        SpatialElement, we validate whether a given parameter is valid. If not valid for a particular SpatialElement the
        specific parameter for that particular SpatialElement will be ignored. If you want to set specific parameters
        for specific elements please chain the render functions: `pl.render_points(...).pl.render_points(...).pl.show()`
        .

        Parameters
        ----------
        element : str | None, optional
            The name of the shapes element to render. If `None`, all shapes elements in the `SpatialData` object will be
            used.
        color : str | None
            Can either be string representing a color-like or key in :attr:`sdata.table.obs`. The latter can be used to
            color by categorical or continuous variables. If `element` is `None`, if possible the color will be
            broadcasted to all elements. For this, the table in which the color key is found must annotate the
            respective element (region must be set to the specific element). If the color column is found in multiple
            locations, please provide the table_name to be used for the elements.
        fill_alpha : float | int, default 1.0
            Alpha value for the fill of shapes. If the alpha channel is present in a cmap passed by the user, this value
            will multiply the value present in the cmap.
        groups : list[str] | str | None
            When using `color` and the key represents discrete labels, `groups` can be used to show only a subset of
            them. Other values are set to NA. If elment is None, broadcasting behaviour is attempted (use the same
            values for all elements).
        palette :  list[str] | str | None
            Palette for discrete annotations. List of valid color names that should be used for the categories. Must
            match the number of groups. If element is None, broadcasting behaviour is attempted (use the same values for
            all elements). If groups is provided but not palette, palette is set to default "lightgray".
        na_color : ColorLike | None, default "default" (gets set to "lightgray")
            Color to be used for NAs values, if present. Can either be a named color ("red"), a hex representation
            ("#000000ff") or a list of floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). When None, the values
            won't be shown.
        outline_width : float | int, default 1.5
            Width of the border.
        outline_color : str | list[float], default "#000000"
            Color of the border. Can either be a named color ("red"), a hex representation ("#000000") or a list of
            floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). If the hex representation includes alpha, e.g.
            "#000000ff", the last two positions are ignored, since the alpha of the outlines is solely controlled by
            `outline_alpha`.
        outline_alpha : float | int, default 0.0
            Alpha value for the outline of shapes. Invisible by default.
        cmap : Colormap | str | None, optional
            Colormap for discrete or continuous annotations using 'color', see :class:`matplotlib.colors.Colormap`.
        norm : bool | Normalize, default False
            Colormap normalization for continuous annotations.
        scale : float | int, default 1.0
            Value to scale circles, if present.
        method : str | None, optional
            Whether to use 'matplotlib' and 'datashader'. When None, the method is
            chosen based on the size of the data.
        table_name: str | None
            Name of the table containing the color(s) columns. If one name is given than the table is used for each
            spatial element to be plotted if the table annotates it. If you want to use different tables for particular
            elements, as specified under element.
        table_layer: str | None
            Layer of the table to use for coloring if `color` is in :attr:`sdata.table.var_names`. If None, the data in
            :attr:`sdata.table.X` is used for coloring.

        **kwargs : Any
            Additional arguments for customization. This can include:

            datashader_reduction : Literal[
                "sum", "mean", "any", "count", "std", "var", "max", "min"
            ], default: "sum"
                Reduction method for datashader when coloring by continuous values. Defaults to 'sum'.


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
        # TODO add Normalize object in tutorial notebook and point to that notebook here
        if "vmin" in kwargs or "vmax" in kwargs:
            warnings.warn(
                "`vmin` and `vmax` are deprecated. Pass matplotlib `Normalize` object to norm instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        params_dict = _validate_shape_render_params(
            self._sdata,
            element=element,
            fill_alpha=fill_alpha,
            groups=groups,
            palette=palette,
            color=color,
            na_color=na_color,
            outline_alpha=outline_alpha,
            outline_color=outline_color,
            outline_width=outline_width,
            cmap=cmap,
            norm=norm,
            scale=scale,
            table_name=table_name,
            table_layer=table_layer,
            method=method,
            ds_reduction=kwargs.get("datashader_reduction"),
        )

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())
        outline_params = _set_outline(outline_alpha > 0, outline_width, outline_color)
        for element, param_values in params_dict.items():
            cmap_params = _prepare_cmap_norm(
                cmap=cmap,
                norm=norm,
                na_color=params_dict[element]["na_color"],  # type: ignore[arg-type]
            )
            sdata.plotting_tree[f"{n_steps + 1}_render_shapes"] = ShapesRenderParams(
                element=element,
                color=param_values["color"],
                col_for_color=param_values["col_for_color"],
                groups=param_values["groups"],
                scale=param_values["scale"],
                outline_params=outline_params,
                cmap_params=cmap_params,
                palette=param_values["palette"],
                outline_alpha=param_values["outline_alpha"],
                fill_alpha=param_values["fill_alpha"],
                transfunc=kwargs.get("transfunc"),
                table_name=param_values["table_name"],
                table_layer=param_values["table_layer"],
                zorder=n_steps,
                method=param_values["method"],
                ds_reduction=param_values["ds_reduction"],
            )
            n_steps += 1

        return sdata

    @_deprecation_alias(elements="element", version="0.3.0")
    def render_points(
        self,
        element: str | None = None,
        color: str | None = None,
        alpha: float | int = 1.0,
        groups: list[str] | str | None = None,
        palette: list[str] | str | None = None,
        na_color: ColorLike | None = "default",
        cmap: Colormap | str | None = None,
        norm: Normalize | None = None,
        size: float | int = 1.0,
        method: str | None = None,
        table_name: str | None = None,
        table_layer: str | None = None,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render points elements in SpatialData.

        In case of no elements specified, "broadcasting" of parameters is applied. This means that for any particular
        SpatialElement, we validate whether a given parameter is valid. If not valid for a particular SpatialElement the
        specific parameter for that particular SpatialElement will be ignored. If you want to set specific parameters
        for specific elements please chain the render functions: `pl.render_points(...).pl.render_points(...).pl.show()`
        .

        Parameters
        ----------
        element : str | None, optional
            The name of the points element to render. If `None`, all points elements in the `SpatialData` object will be
            used.
        color : str | None
            Can either be string representing a color-like or key in :attr:`sdata.table.obs`. The latter can be used to
            color by categorical or continuous variables. If `element` is `None`, if possible the color will be
            broadcasted to all elements. For this, the table in which the color key is found must annotate the
            respective element (region must be set to the specific element). If the color column is found in multiple
            locations, please provide the table_name to be used for the elements.
        alpha : float | int, default 1.0
            Alpha value for the points.
        groups : list[str] | str | None
            When using `color` and the key represents discrete labels, `groups` can be used to show only a subset of
            them. Other values are set to NA. If `element` is `None`, broadcasting behaviour is attempted (use the same
            values for all elements).
        palette : list[str] | str | None
            Palette for discrete annotations. List of valid color names that should be used for the categories. Must
            match the number of groups. If `element` is `None`, broadcasting behaviour is attempted (use the same values
            for all elements). If groups is provided but not palette, palette is set to default "lightgray".
        na_color : ColorLike | None, default "default" (gets set to "lightgray")
            Color to be used for NAs values, if present. Can either be a named color ("red"), a hex representation
            ("#000000ff") or a list of floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). When None, the values
            won't be shown.
        cmap : Colormap | str | None, optional
            Colormap for discrete or continuous annotations using 'color', see :class:`matplotlib.colors.Colormap`. If
            no palette is given and `color` refers to a categorical, the colors are sampled from this colormap.
        norm : bool | Normalize, default False
            Colormap normalization for continuous annotations.
        size : float | int, default 1.0
            Size of the points
        method : str | None, optional
            Whether to use 'matplotlib' and 'datashader'. When None, the method is
            chosen based on the size of the data.
        table_name: str | None
            Name of the table containing the color(s) columns. If one name is given than the table is used for each
            spatial element to be plotted if the table annotates it. If you want to use different tables for particular
            elements, as specified under element.
        table_layer: str | None
            Layer of the table to use for coloring if `color` is in :attr:`sdata.table.var_names`. If None, the data in
            :attr:`sdata.table.X` is used for coloring.

        **kwargs : Any
            Additional arguments for customization. This can include:

            datashader_reduction : Literal[
                "sum", "mean", "any", "count", "std", "var", "max", "min"
            ], default: "sum"
                Reduction method for datashader when coloring by continuous values. Defaults to 'sum'.

        Returns
        -------
        sd.SpatialData
            The modified SpatialData object with the rendered shapes.
        """
        # TODO add Normalize object in tutorial notebook and point to that notebook here
        if "vmin" in kwargs or "vmax" in kwargs:
            warnings.warn(
                "`vmin` and `vmax` are deprecated. Pass matplotlib `Normalize` object to norm instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        params_dict = _validate_points_render_params(
            self._sdata,
            element=element,
            alpha=alpha,
            color=color,
            groups=groups,
            palette=palette,
            na_color=na_color,
            cmap=cmap,
            norm=norm,
            size=size,
            table_name=table_name,
            table_layer=table_layer,
            ds_reduction=kwargs.get("datashader_reduction"),
        )

        if method is not None:
            if not isinstance(method, str):
                raise TypeError("Parameter 'method' must be a string.")
            if method not in ["matplotlib", "datashader"]:
                raise ValueError("Parameter 'method' must be either 'matplotlib' or 'datashader'.")

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())

        for element, param_values in params_dict.items():
            cmap_params = _prepare_cmap_norm(
                cmap=cmap,
                norm=norm,
                na_color=param_values["na_color"],  # type: ignore[arg-type]
            )
            sdata.plotting_tree[f"{n_steps + 1}_render_points"] = PointsRenderParams(
                element=element,
                color=param_values["color"],
                col_for_color=param_values["col_for_color"],
                groups=param_values["groups"],
                cmap_params=cmap_params,
                palette=param_values["palette"],
                alpha=param_values["alpha"],
                transfunc=kwargs.get("transfunc"),
                size=param_values["size"],
                table_name=param_values["table_name"],
                table_layer=param_values["table_layer"],
                zorder=n_steps,
                method=method,
                ds_reduction=param_values["ds_reduction"],
            )
            n_steps += 1

        return sdata

    @_deprecation_alias(elements="element", quantiles_for_norm="percentiles_for_norm", version="version 0.3.0")
    def render_images(
        self,
        element: str | None = None,
        channel: list[str] | list[int] | str | int | None = None,
        cmap: list[Colormap | str] | Colormap | str | None = None,
        norm: Normalize | None = None,
        na_color: ColorLike | None = "default",
        palette: list[str] | str | None = None,
        alpha: float | int = 1.0,
        scale: str | None = None,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render image elements in SpatialData.

        In case of no elements specified, "broadcasting" of parameters is applied. This means that for any particular
        SpatialElement, we validate whether a given parameter is valid. If not valid for a particular SpatialElement the
        specific parameter for that particular SpatialElement will be ignored. If you want to set specific parameters
        for specific elements please chain the render functions: `pl.render_images(...).pl.render_images(...).pl.show()`
        .

        Parameters
        ----------
        element : str | None
            The name of the image element to render. If `None`, all image
            elements in the `SpatialData` object will be used and all parameters will be broadcasted if possible.
        channel : list[str] | list[int] | str | int | None
            To select specific channels to plot. Can be a single channel name/int or a
            list of channel names/ints. If `None`, all channels will be used.
        cmap : list[Colormap | str] | Colormap | str | None
            Colormap or list of colormaps for continuous annotations, see :class:`matplotlib.colors.Colormap`.
            Each colormap applies to a corresponding channel.
        norm : Normalize | None, optional
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
            Applies to all channels if set.
        na_color : ColorLike | None, default "default" (gets set to "lightgray")
            Color to be used for NAs values, if present. Can either be a named color ("red"), a hex representation
            ("#000000ff") or a list of floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). When None, the values
            won't be shown.
        palette : list[str] | str | None
            Palette to color images. The number of palettes should be equal to the number of channels.
        alpha : float | int, default 1.0
            Alpha value for the images. Must be a numeric between 0 and 1.
        scale : str | None
            Influences the resolution of the rendering. Possibilities include:
                1) `None` (default): The image is rasterized to fit the canvas size. For
                multiscale images, the best scale is selected before rasterization.
                2) A scale name: Renders the specified scale ( of a multiscale image) as-is
                (with adjustments for dpi in `show()`).
                3) "full": Renders the full image without rasterization. In the case of
                multiscale images, the highest resolution scale is selected. Note that
                this may result in long computing times for large images.
        kwargs
            Additional arguments to be passed to cmap, norm, and other rendering functions.

        Returns
        -------
        sd.SpatialData
            The SpatialData object with the rendered images.
        """
        # TODO add Normalize object in tutorial notebook and point to that notebook here
        if "vmin" in kwargs or "vmax" in kwargs:
            warnings.warn(
                "`vmin` and `vmax` are deprecated. Pass matplotlib `Normalize` object to norm instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        params_dict = _validate_image_render_params(
            self._sdata,
            element=element,
            channel=channel,
            alpha=alpha,
            palette=palette,
            na_color=na_color,
            cmap=cmap,
            norm=norm,
            scale=scale,
        )

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())

        for element, param_values in params_dict.items():
            cmap_params: list[CmapParams] | CmapParams
            if isinstance(cmap, list):
                cmap_params = [
                    _prepare_cmap_norm(
                        cmap=c,
                        norm=norm,
                        na_color=param_values["na_color"],
                    )
                    for c in cmap
                ]

            else:
                cmap_params = _prepare_cmap_norm(
                    cmap=cmap,
                    norm=norm,
                    na_color=param_values["na_color"],
                    **kwargs,
                )
            sdata.plotting_tree[f"{n_steps + 1}_render_images"] = ImageRenderParams(
                element=element,
                channel=param_values["channel"],
                cmap_params=cmap_params,
                palette=param_values["palette"],
                alpha=param_values["alpha"],
                scale=param_values["scale"],
                zorder=n_steps,
            )
            n_steps += 1

        return sdata

    @_deprecation_alias(elements="element", version="0.3.0")
    def render_labels(
        self,
        element: str | None = None,
        color: str | None = None,
        groups: list[str] | str | None = None,
        contour_px: int | None = 3,
        palette: list[str] | str | None = None,
        cmap: Colormap | str | None = None,
        norm: Normalize | None = None,
        na_color: ColorLike | None = "default",
        outline_alpha: float | int = 0.0,
        fill_alpha: float | int = 0.4,
        scale: str | None = None,
        table_name: str | None = None,
        table_layer: str | None = None,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render labels elements in SpatialData.

        In case of no elements specified, "broadcasting" of parameters is applied. This means that for any particular
        SpatialElement, we validate whether a given parameter is valid. If not valid for a particular SpatialElement the
        specific parameter for that particular SpatialElement will be ignored. If you want to set specific parameters
        for specific elements please chain the render functions: `pl.render_images(...).pl.render_images(...).pl.show()`
        .

        Parameters
        ----------
        element : str | None
            The name of the labels element to render. If `None`, all label
            elements in the `SpatialData` object will be used and all parameters will be broadcasted if possible.
        color : list[str] | str | None
            Can either be string representing a color-like or key in :attr:`sdata.table.obs` or in the index of
            :attr:`sdata.table.var`. The latter can be used to color by categorical or continuous variables. If the
            color column is found in multiple locations, please provide the table_name to be used for the element if you
            would like a specific table to be used. By default one table will automatically be choosen.
        groups : list[str] | str | None
            When using `color` and the key represents discrete labels, `groups` can be used to show only a subset of
            them. Other values are set to NA. The list can contain multiple discrete labels to be visualized.
        palette : list[str] | str | None
            Palette for discrete annotations. List of valid color names that should be used for the categories. Must
            match the number of groups. The list can contain multiple palettes (one per group) to be visualized. If
            groups is provided but not palette, palette is set to default "lightgray".
        contour_px : int, default 3
            Draw contour of specified width for each segment. If `None`, fills entire segment, see:
            func:`skimage.morphology.erosion`.
        cmap : Colormap | str | None
            Colormap for continuous annotations, see :class:`matplotlib.colors.Colormap`.
        norm : Normalize | None
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
        na_color : ColorLike | None, default "default" (gets set to "lightgray")
            Color to be used for NAs values, if present. Can either be a named color ("red"), a hex representation
            ("#000000ff") or a list of floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). When None, the values
            won't be shown.
        outline_alpha : float | int, default 0.0
            Alpha value for the outline of the labels. Invisible by default.
        fill_alpha : float | int, default 0.3
            Alpha value for the fill of the labels.
        scale :  str | None
            Influences the resolution of the rendering. Possibilities for setting this parameter:
                1) None (default). The image is rasterized to fit the canvas size. For multiscale images, the best scale
                is selected before the rasterization step.
                2) Name of one of the scales in the multiscale image to be rendered. This scale is rendered as it is
                (exception: a dpi is specified in `show()`. Then the image is rasterized to fit the canvas and dpi).
                3) "full": render the full image without rasterization. In the case of a multiscale image, the scale
                with the highest resolution is selected. This can lead to long computing times for large images!
        table_name: str | None
            Name of the table containing the color columns.
        table_layer: str | None
            Layer of the AnnData table to use for coloring if `color` is in :attr:`sdata.table.var_names`. If None,
            :attr:`sdata.table.X` of the default table is used for coloring.
        kwargs
            Additional arguments to be passed to cmap and norm.

        Returns
        -------
        None
        """
        # TODO add Normalize object in tutorial notebook and point to that notebook here
        if "vmin" in kwargs or "vmax" in kwargs:
            warnings.warn(
                "`vmin` and `vmax` are deprecated. Pass matplotlib `Normalize` object to norm instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        params_dict = _validate_label_render_params(
            self._sdata,
            element=element,
            cmap=cmap,
            color=color,
            contour_px=contour_px,
            fill_alpha=fill_alpha,
            groups=groups,
            na_color=na_color,
            norm=norm,
            outline_alpha=outline_alpha,
            palette=palette,
            scale=scale,
            table_name=table_name,
            table_layer=table_layer,
        )

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())

        for element, param_values in params_dict.items():
            cmap_params = _prepare_cmap_norm(
                cmap=cmap,
                norm=norm,
                na_color=param_values["na_color"],  # type: ignore[arg-type]
            )
            sdata.plotting_tree[f"{n_steps + 1}_render_labels"] = LabelsRenderParams(
                element=element,
                color=param_values["color"],
                groups=param_values["groups"],
                contour_px=param_values["contour_px"],
                cmap_params=cmap_params,
                palette=param_values["palette"],
                outline_alpha=param_values["outline_alpha"],
                fill_alpha=param_values["fill_alpha"],
                transfunc=kwargs.get("transfunc"),
                scale=param_values["scale"],
                table_name=param_values["table_name"],
                table_layer=param_values["table_layer"],
                zorder=n_steps,
            )
            n_steps += 1
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
        return_ax :
            Whether to return the axes object created. False by default.
        colorbar :
            Whether to plot the colorbar. True by default.
        title :
            The title of the plot. If not provided the plot will have the name of the coordinate system as title.

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

        _validate_show_parameters(
            coordinate_systems,
            legend_fontsize,
            legend_fontweight,
            legend_loc,
            legend_fontoutline,
            na_in_legend,
            colorbar,
            wspace,
            hspace,
            ncols,
            frameon,
            figsize,
            dpi,
            fig,
            title,
            share_extent,
            pad_extent,
            ax,
            return_ax,
            save,
        )

        sdata = self._copy()

        # Evaluate execution tree for plotting
        valid_commands = [
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

        coordinate_systems = sdata.coordinate_systems if coordinate_systems is None else coordinate_systems
        if isinstance(coordinate_systems, str):
            coordinate_systems = [coordinate_systems]

        for cs in coordinate_systems:
            if cs not in sdata.coordinate_systems:
                raise ValueError(f"Unknown coordinate system '{cs}', valid choices are: {sdata.coordinate_systems}")

        # Check if user specified only certain elements to be plotted
        cs_contents = _get_cs_contents(sdata)

        elements_to_be_rendered = _get_elements_to_be_rendered(render_cmds, cs_contents, cs)

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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                sdata = self._copy()
            _, has_images, has_labels, has_points, has_shapes = (
                cs_contents.query(f"cs == '{cs}'").iloc[0, :].values.tolist()
            )
            ax = fig_params.ax if fig_params.axs is None else fig_params.axs[i]
            assert isinstance(ax, Axes)

            wants_images = False
            wants_labels = False
            wants_points = False
            wants_shapes = False
            wanted_elements: list[str] = []

            for cmd, params in render_cmds:
                # We create a copy here as the wanted elements can change from one cs to another.
                params_copy = deepcopy(params)
                if cmd == "render_images" and has_images:
                    wanted_elements, wanted_images_on_this_cs, wants_images = _get_wanted_render_elements(
                        sdata, wanted_elements, params_copy, cs, "images"
                    )

                    if wanted_images_on_this_cs:
                        rasterize = (params_copy.scale is None) or (
                            isinstance(params_copy.scale, str)
                            and params_copy.scale != "full"
                            and (dpi is not None or figsize is not None)
                        )
                        _render_images(
                            sdata=sdata,
                            render_params=params_copy,
                            coordinate_system=cs,
                            ax=ax,
                            fig_params=fig_params,
                            scalebar_params=scalebar_params,
                            legend_params=legend_params,
                            rasterize=rasterize,
                        )

                elif cmd == "render_shapes" and has_shapes:
                    wanted_elements, wanted_shapes_on_this_cs, wants_shapes = _get_wanted_render_elements(
                        sdata, wanted_elements, params_copy, cs, "shapes"
                    )

                    if wanted_shapes_on_this_cs:
                        _render_shapes(
                            sdata=sdata,
                            render_params=params_copy,
                            coordinate_system=cs,
                            ax=ax,
                            fig_params=fig_params,
                            scalebar_params=scalebar_params,
                            legend_params=legend_params,
                        )

                elif cmd == "render_points" and has_points:
                    wanted_elements, wanted_points_on_this_cs, wants_points = _get_wanted_render_elements(
                        sdata, wanted_elements, params_copy, cs, "points"
                    )

                    if wanted_points_on_this_cs:
                        _render_points(
                            sdata=sdata,
                            render_params=params_copy,
                            coordinate_system=cs,
                            ax=ax,
                            fig_params=fig_params,
                            scalebar_params=scalebar_params,
                            legend_params=legend_params,
                        )

                elif cmd == "render_labels" and has_labels:
                    wanted_elements, wanted_labels_on_this_cs, wants_labels = _get_wanted_render_elements(
                        sdata, wanted_elements, params_copy, cs, "labels"
                    )

                    if wanted_labels_on_this_cs:
                        if (table := params_copy.table_name) is not None:
                            assert isinstance(params_copy.color, str)
                            colors = sc.get.obs_df(sdata[table], [params_copy.color])
                            if isinstance(colors[params_copy.color].dtype, pd.CategoricalDtype):
                                _maybe_set_colors(
                                    source=sdata[table],
                                    target=sdata[table],
                                    key=params_copy.color,
                                    palette=params_copy.palette,
                                )

                        rasterize = (params_copy.scale is None) or (
                            isinstance(params_copy.scale, str)
                            and params_copy.scale != "full"
                            and (dpi is not None or figsize is not None)
                        )
                        _render_labels(
                            sdata=sdata,
                            render_params=params_copy,
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
