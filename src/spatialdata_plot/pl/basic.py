from __future__ import annotations

import contextlib
import sys
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from spatialdata import get_extent
from spatialdata._utils import _deprecation_alias
from xarray import DataArray, DataTree

from spatialdata_plot._accessor import register_spatial_data_accessor
from spatialdata_plot._logging import _log_context, logger
from spatialdata_plot.pl.render import (
    _draw_channel_legend,
    _render_graph,
    _render_images,
    _render_labels,
    _render_points,
    _render_shapes,
    _split_colorbar_params,
)
from spatialdata_plot.pl.render_params import (
    CBAR_DEFAULT_FRACTION,
    CBAR_DEFAULT_LOCATION,
    CBAR_DEFAULT_PAD,
    ChannelLegendEntry,
    CmapParams,
    ColorbarSpec,
    GraphRenderParams,
    ImageRenderParams,
    LabelsRenderParams,
    LegendParams,
    PointsRenderParams,
    ShapesRenderParams,
    _FontSize,
    _FontWeight,
)
from spatialdata_plot.pl.utils import (
    _RENDER_CMD_TO_CS_FLAG,
    _get_cs_contents,
    _get_elements_to_be_rendered,
    _get_valid_cs,
    _get_wanted_render_elements,
    _maybe_set_colors,
    _mpl_ax_contains_elements,
    _prepare_cmap_norm,
    _prepare_params_plot,
    _set_outline,
    _validate_graph_render_params,
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
ColorLike = tuple[float, ...] | list[float] | str


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
        color: ColorLike | None = None,
        *,
        fill_alpha: float | int | None = None,
        groups: list[str] | str | None = None,
        palette: dict[str, str] | list[str] | str | None = None,
        na_color: ColorLike | None = "default",
        outline_width: float | int | tuple[float | int, float | int] | None = None,
        outline_color: ColorLike | tuple[ColorLike] | None = None,
        outline_alpha: float | int | tuple[float | int, float | int] | None = None,
        cmap: Colormap | str | None = None,
        norm: Normalize | None = None,
        scale: float | int = 1.0,
        method: str | None = None,
        table_name: str | None = None,
        table_layer: str | None = None,
        gene_symbols: str | None = None,
        shape: Literal["circle", "hex", "visium_hex", "square"] | None = None,
        colorbar: bool | str | None = "auto",
        colorbar_params: dict[str, object] | None = None,
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
        color : ColorLike | None, optional
            Can either be color-like (name of a color as string, e.g. "red", hex representation, e.g. "#000000" or
            "#000000ff", or an RGB(A) array as a tuple or list containing 3-4 floats within [0, 1]. If an alpha value is
            indicated, the value of `fill_alpha` takes precedence if given) or a string representing a key in
            :attr:`sdata.table.obs`. The latter can be used to color by categorical or continuous variables. If
            `element` is `None`, if possible the color will be broadcasted to all elements. For this, the table in which
            the color key is found must annotate the respective element (region must be set to the specific element). If
            the color column is found in multiple locations, please provide the table_name to be used for the elements.
        fill_alpha : float | int | None, optional
            Alpha value for the fill of shapes. By default, it is set to 1.0 or, if a color is given that implies an
            alpha, that value is used for `fill_alpha`. If an alpha channel is present in a cmap passed by the user,
            `fill_alpha` will overwrite the value present in the cmap.
        groups : list[str] | str | None
            When using `color` and the key represents discrete labels, `groups` can be used to show only a subset of
            them. By default, non-matching elements are hidden. To show non-matching elements, set ``na_color``
            explicitly.
            If element is None, broadcasting behaviour is attempted (use the same values for all elements).
        palette : dict[str, str] | list[str] | str | None
            Palette for discrete annotations. Can be a dictionary mapping category names to colors, a list of valid
            color names (must match the number of groups), a single named palette or matplotlib colormap name, or
            ``None``. If element is None, broadcasting behaviour is attempted (use the same values for all elements).
        na_color : ColorLike | None, default "default" (gets set to "lightgray")
            Color to be used for NA values, if present. Can either be a named color ("red"), a hex representation
            ("#000000ff") or a list of floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). When None, the values
            won't be shown.
        outline_width : float | int | tuple[float | int, float | int], optional
            Width of the border. If 2 values are given (tuple), 2 borders are shown with these widths (outer & inner).
            If `outline_color` and/or `outline_alpha` are used to indicate that one/two outlines should be drawn, the
            default outline widths 1.5 and 0.5 are used for outer/only and inner outline respectively.
        outline_color : ColorLike | tuple[ColorLike], optional
            Color of the border. Can either be a named color ("red"), a hex representation ("#000000") or a list of
            floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). If the hex representation includes alpha, e.g.
            "#000000ff", and `outline_alpha` is not given, this value controls the opacity of the outline. If 2 values
            are given (tuple), 2 borders are shown with these colors (outer & inner). If `outline_width` and/or
            `outline_alpha` are used to indicate that one/two outlines should be drawn, the default outline colors
            "#000000" and "#ffffff are used for outer/only and inner outline respectively.
        outline_alpha : float | int | tuple[float | int, float | int] | None, optional
            Alpha value for the outline of shapes. Invisible by default, meaning outline_alpha=0.0 if both outline_color
            and outline_width are not specified. Else, outlines are rendered with the alpha implied by outline_color, or
            with outline_alpha=1.0 if outline_color does not imply an alpha. For two outlines, alpha values can be
            passed in a tuple of length 2.
        cmap : Colormap | str | None, optional
            Colormap for continuous annotations using 'color', see :class:`matplotlib.colors.Colormap`.
            For categorical data, use ``palette`` instead.
        norm : Normalize | None, optional
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
        scale : float | int, default 1.0
            Value to scale shapes (circles and polygons).
        method : str | None, optional
            Whether to use ``'matplotlib'`` or ``'datashader'``. When ``None``, the method is
            chosen automatically based on the size of the data (datashader for >10 000 elements).
        colorbar : bool | str | None, default "auto"
            Whether to request a colorbar for continuous colors. Use ``"auto"`` (default) for automatic selection.
        colorbar_params : dict[str, object] | None
            Parameters forwarded to Matplotlib's colorbar alongside layout hints such as ``loc``, ``width``, ``pad``,
            and ``label``.
        table_name: str | None
            Name of the table containing the color(s) columns. If one name is given than the table is used for each
            spatial element to be plotted if the table annotates it. If you want to use different tables for particular
            elements, as specified under element.
        table_layer: str | None
            Layer of the table to use for coloring if `color` is in :attr:`sdata.table.var_names`. If None, the data in
            :attr:`sdata.table.X` is used for coloring.
        gene_symbols: str | None
            Column name in :attr:`sdata.table.var` to use for looking up ``color``. Use this when
            ``var_names`` are e.g. ENSEMBL IDs but you want to refer to genes by their symbols stored
            in another column of ``var``. Mimics scanpy's ``gene_symbols`` parameter.
        shape: Literal["circle", "hex", "visium_hex", "square"] | None
            If None (default), the shapes are rendered as they are. Else, if either of "circle", "hex" or "square" is
            specified, the shapes are converted to a circle/hexagon/square before rendering. If "visium_hex" is
            specified, the shapes are assumed to be Visium spots and the size of the hexagons is adjusted to be adjacent
            to each other.

        **kwargs : Any
            Additional arguments for customization. This can include:

            datashader_reduction : Literal[
                "sum", "mean", "any", "count", "std", "var", "max", "min"
            ], default: "max"
                Reduction method for datashader when coloring by continuous values. Defaults to 'max'.


        Notes
        -----
        - Empty geometries will be removed at the time of plotting.
        - An `outline_width` of 0.0 leads to no border being plotted.
        - When passing a color-like to 'color', this has precedence over the potential existence as a column name.

        Returns
        -------
        sd.SpatialData
            A copy of the SpatialData object with the rendering parameters stored in its plotting tree.
        """
        if "vmin" in kwargs or "vmax" in kwargs:
            logger.warning("`vmin` and `vmax` are deprecated. Pass matplotlib `Normalize` object to norm instead.")
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
            shape=shape,
            method=method,
            ds_reduction=kwargs.get("datashader_reduction"),
            colorbar=colorbar,
            colorbar_params=colorbar_params,
            gene_symbols=gene_symbols,
        )

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())
        for element, param_values in params_dict.items():
            final_outline_alpha, outline_params = _set_outline(
                params_dict[element]["outline_alpha"],
                params_dict[element]["outline_width"],
                params_dict[element]["outline_color"],
            )
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
                outline_alpha=final_outline_alpha,
                fill_alpha=param_values["fill_alpha"],
                transfunc=kwargs.get("transfunc"),
                table_name=param_values["table_name"],
                table_layer=param_values["table_layer"],
                shape=param_values["shape"],
                zorder=n_steps,
                method=param_values["method"],
                ds_reduction=param_values["ds_reduction"],
                colorbar=param_values["colorbar"],
                colorbar_params=param_values["colorbar_params"],
            )
            n_steps += 1

        return sdata

    @_deprecation_alias(elements="element", version="0.3.0")
    def render_points(
        self,
        element: str | None = None,
        color: ColorLike | None = None,
        *,
        alpha: float | int | None = None,
        groups: list[str] | str | None = None,
        palette: dict[str, str] | list[str] | str | None = None,
        na_color: ColorLike | None = "default",
        cmap: Colormap | str | None = None,
        norm: Normalize | None = None,
        size: float | int = 1.0,
        method: str | None = None,
        table_name: str | None = None,
        table_layer: str | None = None,
        gene_symbols: str | None = None,
        colorbar: bool | str | None = "auto",
        colorbar_params: dict[str, object] | None = None,
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
        color : ColorLike | None, optional
            Can either be color-like (name of a color as string, e.g. "red", hex representation, e.g. "#000000" or
            "#000000ff", or an RGB(A) array as a tuple or list containing 3-4 floats within [0, 1]. If an alpha value is
            indicated, the value of ``alpha`` takes precedence if given) or a string representing a key in
            :attr:`sdata.table.obs`. The latter can be used to color by categorical or continuous variables. If
            `element` is `None`, if possible the color will be broadcasted to all elements. For this, the table in which
            the color key is found must annotate the respective element (region must be set to the specific element). If
            the color column is found in multiple locations, please provide the table_name to be used for the elements.
        alpha : float | int | None, optional
            Alpha value for the points. By default, it is set to 1.0 or, if a color is given that implies an alpha, that
            value is used instead.
        groups : list[str] | str | None
            When using `color` and the key represents discrete labels, `groups` can be used to show only a subset of
            them. By default, non-matching points are filtered out entirely. To show non-matching points, set
            ``na_color`` explicitly.
            If element is None, broadcasting behaviour is attempted (use the same values for all elements).
        palette : dict[str, str] | list[str] | str | None
            Palette for discrete annotations. Can be a dictionary mapping category names to colors, a list of valid
            color names (must match the number of groups), a single named palette or matplotlib colormap name, or
            ``None``. If `element` is `None`, broadcasting behaviour is attempted (use the same values for all
            elements).
        na_color : ColorLike | None, default "default" (gets set to "lightgray")
            Color to be used for NA values, if present. Can either be a named color ("red"), a hex representation
            ("#000000ff") or a list of floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). When None, the values
            won't be shown.
        cmap : Colormap | str | None, optional
            Colormap for continuous annotations using 'color', see :class:`matplotlib.colors.Colormap`. If
            no palette is given and `color` refers to a categorical, the colors are sampled from this colormap.
        norm : Normalize | None, optional
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
        size : float | int, default 1.0
            Size of the points.
        method : str | None, optional
            Whether to use ``'matplotlib'`` or ``'datashader'``. When ``None``, the method is
            chosen automatically based on the size of the data (datashader for >10 000 elements).
        colorbar : bool | str | None, default "auto"
            Whether to request a colorbar for continuous colors. Use ``"auto"`` (default) for automatic selection.
        colorbar_params : dict[str, object] | None
            Parameters forwarded to Matplotlib's colorbar alongside layout hints such as ``loc``, ``width``, ``pad``,
            and ``label``.
        table_name: str | None
            Name of the table containing the color(s) columns. If one name is given than the table is used for each
            spatial element to be plotted if the table annotates it. If you want to use different tables for particular
            elements, as specified under element.
        table_layer: str | None
            Layer of the table to use for coloring if `color` is in :attr:`sdata.table.var_names`. If None, the data in
            :attr:`sdata.table.X` is used for coloring.
        gene_symbols: str | None
            Column name in :attr:`sdata.table.var` to use for looking up ``color``. Use this when
            ``var_names`` are e.g. ENSEMBL IDs but you want to refer to genes by their symbols stored
            in another column of ``var``. Mimics scanpy's ``gene_symbols`` parameter.

        **kwargs : Any
            Additional arguments for customization. This can include:

            datashader_reduction : Literal[
                "sum", "mean", "any", "count", "std", "var", "max", "min"
            ], default: "sum"
                Reduction method for datashader when coloring by continuous values. Defaults to 'sum'.

        Returns
        -------
        sd.SpatialData
            A copy of the SpatialData object with the rendering parameters stored in its plotting tree.
        """
        if "vmin" in kwargs or "vmax" in kwargs:
            logger.warning("`vmin` and `vmax` are deprecated. Pass matplotlib `Normalize` object to norm instead.")
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
            colorbar=colorbar,
            colorbar_params=colorbar_params,
            gene_symbols=gene_symbols,
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
                colorbar=param_values["colorbar"],
                colorbar_params=param_values["colorbar_params"],
            )
            n_steps += 1

        return sdata

    @_deprecation_alias(elements="element", version="version 0.3.0")
    def render_images(
        self,
        element: str | None = None,
        *,
        channel: list[str] | list[int] | str | int | None = None,
        cmap: list[Colormap | str] | Colormap | str | None = None,
        norm: list[Normalize] | Normalize | None = None,
        palette: list[str] | str | None = None,
        alpha: float | int = 1.0,
        scale: str | None = None,
        grayscale: bool = False,
        transfunc: (Callable[[np.ndarray], np.ndarray] | list[Callable[[np.ndarray], np.ndarray]] | None) = None,
        colorbar: bool | str | None = "auto",
        colorbar_params: dict[str, object] | None = None,
        channels_as_legend: bool = False,
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
        norm : list[Normalize] | Normalize | None, optional
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
            A single :class:`~matplotlib.colors.Normalize` applies to all channels.
            A list of :class:`~matplotlib.colors.Normalize` objects applies per-channel
            (length must match the number of channels).
        palette : list[str] | str | None
            Palette to color images. Can be a single palette name (broadcast to all channels) or a list
            matching the number of channels.
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
        grayscale : bool, default False
            Convert the image to grayscale before rendering using luminance
            weights (Rec. 601: 0.2989 R + 0.5870 G + 0.1140 B). Requires
            exactly 3 channels at the point of conversion — if ``transfunc``
            is also provided, it runs first, and the result must have 3
            channels. The grayscale image is rendered as a single-channel
            image with ``cmap="gray"`` unless an explicit ``cmap`` is given.
            Useful for de-emphasising H&E tissue when overlaying colored
            annotations. Cannot be combined with ``palette``.
        transfunc : callable or list of callables, optional
            Transform(s) applied to the raw image array before normalization
            and rendering.

            **Single callable**: receives a numpy array of shape ``(c, y, x)``
            (channels first) and must return an array of the same layout.
            The number of channels may change (e.g., stain deconvolution).
            Elementwise functions like ``np.log1p`` broadcast naturally.
            Note that reductions like ``np.percentile`` will compute a
            *single* value across all channels.

            **List of callables**: one per channel (length must match the
            number of selected channels). Each receives a ``(y, x)`` array
            for its channel and must return a ``(y, x)`` array. Use this
            when each channel needs independent treatment (e.g., different
            gamma corrections for different fluorescence markers).

            When combined with ``grayscale=True``, ``transfunc`` runs first
            and ``grayscale`` is applied to the result.
        colorbar : bool | str | None, default "auto"
            Whether to request a colorbar for continuous colors. Use ``"auto"`` (default) for automatic selection.
        colorbar_params : dict[str, object] | None
            Parameters forwarded to Matplotlib's colorbar alongside layout hints such as ``loc``, ``width``, ``pad``,
            and ``label``.
        channels_as_legend : bool, default False
            When ``True`` and rendering multiple channels, show a categorical
            legend mapping each channel name to its compositing color.  The
            legend uses the ``legend_*`` parameters from :meth:`show`.
            Ignored for single-channel and RGB(A) images.  When multiple
            ``render_images`` calls use this flag on the same axes, all
            channel entries are combined into a single legend.

        Notes
        -----
        - **RGB(A) auto-detection**: when the channel names are exactly ``{r, g, b}`` or ``{r, g, b, a}``
          (case-insensitive) and no explicit ``cmap`` or ``palette`` is given, the image is rendered as
          true-color RGB(A) without colormaps.
        - **Multi-channel compositing**: when multiple channels are rendered with per-channel colormaps,
          they are additively blended. Colormaps that go from a color to white (rather than to transparent)
          will cause upper layers to occlude lower ones.
        - A single ``cmap`` is automatically broadcast to all selected channels.

        Returns
        -------
        sd.SpatialData
            A copy of the SpatialData object with the rendering parameters stored in its plotting tree.
        """
        if grayscale and palette is not None:
            raise ValueError("Cannot combine grayscale=True with palette.")
        params_dict = _validate_image_render_params(
            self._sdata,
            element=element,
            channel=channel,
            alpha=alpha,
            palette=palette,
            cmap=cmap,
            norm=norm,
            scale=scale,
            colorbar=colorbar,
            colorbar_params=colorbar_params,
        )

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())

        for element, param_values in params_dict.items():
            cmap_params: list[CmapParams] | CmapParams
            # Resolve which cmap to use for the norm-list path vs scalar path.
            effective_cmap = param_values.get("cmap") if isinstance(norm, list) else cmap

            # When the user passes per-channel norms without explicit cmaps,
            # generate a default cmap list so the per-channel path works.
            if isinstance(norm, list) and len(norm) > 1 and not isinstance(effective_cmap, list):
                effective_cmap = [None] * len(norm)

            if isinstance(effective_cmap, list) and len(effective_cmap) > 1:
                if isinstance(norm, list):
                    if len(norm) != len(effective_cmap):
                        raise ValueError(
                            f"Length of 'norm' list ({len(norm)}) must match "
                            f"the number of colormaps ({len(effective_cmap)})."
                        )
                    norms = norm
                else:
                    norms = [norm] * len(effective_cmap)
                cmap_params = [
                    _prepare_cmap_norm(
                        cmap=c,
                        norm=n,
                    )
                    for c, n in zip(effective_cmap, norms, strict=True)
                ]

            else:
                norm_scalar = norm[0] if isinstance(norm, list) else norm
                scalar_cmap = effective_cmap[0] if isinstance(effective_cmap, list) else cmap
                cmap_params = _prepare_cmap_norm(
                    cmap=scalar_cmap,
                    norm=norm_scalar,
                )
            sdata.plotting_tree[f"{n_steps + 1}_render_images"] = ImageRenderParams(
                element=element,
                channel=param_values["channel"],
                cmap_params=cmap_params,
                palette=param_values["palette"],
                alpha=param_values["alpha"],
                scale=param_values["scale"],
                zorder=n_steps,
                colorbar=param_values["colorbar"],
                colorbar_params=param_values["colorbar_params"],
                transfunc=transfunc,
                grayscale=grayscale,
                channels_as_legend=channels_as_legend,
            )
            n_steps += 1

        return sdata

    @_deprecation_alias(elements="element", version="0.3.0")
    def render_labels(
        self,
        element: str | None = None,
        color: ColorLike | None = None,
        *,
        groups: list[str] | str | None = None,
        contour_px: int | None = 3,
        palette: dict[str, str] | list[str] | str | None = None,
        cmap: Colormap | str | None = None,
        norm: Normalize | None = None,
        na_color: ColorLike | None = "default",
        outline_alpha: float | int = 0.0,
        fill_alpha: float | int | None = None,
        outline_color: ColorLike | None = None,
        scale: str | None = None,
        colorbar: bool | str | None = "auto",
        colorbar_params: dict[str, object] | None = None,
        table_name: str | None = None,
        table_layer: str | None = None,
        gene_symbols: str | None = None,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render labels elements in SpatialData.

        In case of no elements specified, "broadcasting" of parameters is applied. This means that for any particular
        SpatialElement, we validate whether a given parameter is valid. If not valid for a particular SpatialElement the
        specific parameter for that particular SpatialElement will be ignored. If you want to set specific parameters
        for specific elements please chain the render functions: `pl.render_labels(...).pl.render_labels(...).pl.show()`
        .

        Parameters
        ----------
        element : str | None
            The name of the labels element to render. If `None`, all label
            elements in the `SpatialData` object will be used and all parameters will be broadcasted if possible.
        color : ColorLike | None
            Can either be color-like (name of a color as string, e.g. "red", hex representation, e.g. "#000000" or
            "#000000ff", or an RGB(A) array as a tuple or list containing 3-4 floats within [0, 1]. If an alpha value
            is indicated, the value of `fill_alpha` takes precedence if given) or a string representing a key in
            :attr:`sdata.table.obs` or in the index of :attr:`sdata.table.var`. The latter can be used to color by
            categorical or continuous variables. If the color column is found in multiple locations, please provide the
            table_name to be used for the element if you would like a specific table to be used.
        groups : list[str] | str | None
            When using `color` and the key represents discrete labels, `groups` can be used to show only a subset of
            them. By default, non-matching labels are hidden. To show non-matching labels, set ``na_color`` explicitly.
        palette : dict[str, str] | list[str] | str | None
            Palette for discrete annotations. Can be a dictionary mapping category names to colors, a list of valid
            color names (must match the number of groups), a single named palette or matplotlib colormap name, or
            ``None``.
        contour_px : int, default 3
            Draw contour of specified width for each segment. If `None`, fills entire segment, see:
            func:`skimage.morphology.erosion`.
        cmap : Colormap | str | None, optional
            Colormap for continuous annotations using 'color', see :class:`matplotlib.colors.Colormap`.
            For categorical data, use ``palette`` instead.
        norm : Normalize | None, optional
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
        na_color : ColorLike | None, default "default" (gets set to "lightgray")
            Color to be used for NAs values, if present. Can either be a named color ("red"), a hex representation
            ("#000000ff") or a list of floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). When None, the values
            won't be shown.
        outline_alpha : float | int, default 0.0
            Alpha value for the outline of the labels. Invisible by default.
        fill_alpha : float | int | None, optional
            Alpha value for the fill of the labels. By default, it is set to 0.4 or, if a color is given that implies
            an alpha, that value is used for `fill_alpha`.
        outline_color : ColorLike | None
            Color of the outline of the labels. Can either be a named color ("red"), a hex representation
            ("#000000") or a list of floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). If ``None``,
            the outline inherits from the ``color`` parameter when it is a literal color, or uses data-driven
            per-label colors when ``color`` refers to a column.
        scale :  str | None
            Influences the resolution of the rendering. Possibilities for setting this parameter:
                1) None (default). The image is rasterized to fit the canvas size. For multiscale images, the best scale
                is selected before the rasterization step.
                2) Name of one of the scales in the multiscale image to be rendered. This scale is rendered as it is
                (exception: a dpi is specified in `show()`. Then the image is rasterized to fit the canvas and dpi).
                3) "full": render the full image without rasterization. In the case of a multiscale image, the scale
                with the highest resolution is selected. This can lead to long computing times for large images!
        colorbar : bool | str | None, default "auto"
            Whether to request a colorbar for continuous colors. Use ``"auto"`` (default) for automatic selection.
        colorbar_params : dict[str, object] | None
            Parameters forwarded to Matplotlib's colorbar alongside layout hints such as ``loc``, ``width``, ``pad``,
            and ``label``.
        table_name: str | None
            Name of the table containing the color columns.
        table_layer: str | None
            Layer of the AnnData table to use for coloring if `color` is in :attr:`sdata.table.var_names`. If None,
            :attr:`sdata.table.X` of the default table is used for coloring.
        gene_symbols: str | None
            Column name in :attr:`sdata.table.var` to use for looking up ``color``. Use this when
            ``var_names`` are e.g. ENSEMBL IDs but you want to refer to genes by their symbols stored
            in another column of ``var``. Mimics scanpy's ``gene_symbols`` parameter.

        Returns
        -------
        sd.SpatialData
            A copy of the SpatialData object with the rendering parameters stored in its plotting tree.
        """
        if "vmin" in kwargs or "vmax" in kwargs:
            logger.warning("`vmin` and `vmax` are deprecated. Pass matplotlib `Normalize` object to norm instead.")
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
            outline_color=outline_color,
            palette=palette,
            scale=scale,
            colorbar=colorbar,
            colorbar_params=colorbar_params,
            table_name=table_name,
            table_layer=table_layer,
            gene_symbols=gene_symbols,
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
                col_for_color=param_values["col_for_color"],
                groups=param_values["groups"],
                contour_px=param_values["contour_px"],
                cmap_params=cmap_params,
                palette=param_values["palette"],
                outline_alpha=param_values["outline_alpha"],
                outline_color=param_values["outline_color"],
                fill_alpha=param_values["fill_alpha"],
                scale=param_values["scale"],
                table_name=param_values["table_name"],
                table_layer=param_values["table_layer"],
                zorder=n_steps,
                colorbar=param_values["colorbar"],
                colorbar_params=param_values["colorbar_params"],
            )
            n_steps += 1
        return sdata

    def render_graph(
        self,
        element: str | None = None,
        color: ColorLike | None = "grey",
        *,
        connectivity_key: str = "spatial",
        groups: list[str] | str | None = None,
        group_key: str | None = None,
        edge_width: float = 1.0,
        edge_alpha: float = 1.0,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """Render spatial graph edges between observations.

        Draws edges from a connectivity matrix stored in a table's ``obsp``,
        using centroid coordinates of the linked spatial element.

        Parameters
        ----------
        element : str | None, optional
            Name of the spatial element (shapes, points, or labels) whose
            observations the graph connects. Auto-resolved from the table
            if not given.
        color : ColorLike | None, default "grey"
            Edge color as a color-like value (e.g. ``"red"``, ``"#aabbcc"``).
        connectivity_key : str, default "spatial"
            Key prefix in ``table.obsp``. Tries ``obsp[key]`` first, then
            ``obsp[f"{key}_connectivities"]``.
        groups : list[str] | str | None, optional
            Show only edges where **both** endpoints belong to the specified
            groups. Requires ``group_key``.
        group_key : str | None, optional
            Column in ``table.obs`` used for group filtering.
        edge_width : float, default 1.0
            Line width for edges.
        edge_alpha : float, default 1.0
            Transparency for edges (0 = invisible, 1 = opaque).
        table_name : str | None, optional
            Table containing the graph. Auto-discovered if not given.
        **kwargs
            Forwarded to :class:`matplotlib.collections.LineCollection`.

        Returns
        -------
        sd.SpatialData
            Copy with rendering parameters stored in the plotting tree.
        """
        params = _validate_graph_render_params(
            self._sdata,
            element=element,
            connectivity_key=connectivity_key,
            table_name=table_name,
            color=color,
            edge_width=edge_width,
            edge_alpha=edge_alpha,
            groups=groups,
            group_key=group_key,
        )

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())
        sdata.plotting_tree[f"{n_steps + 1}_render_graph"] = GraphRenderParams(
            element=params["element"],
            connectivity_key=params["obsp_key"],
            table_name=params["table_name"],
            color=params["color"],
            groups=params["groups"],
            group_key=params["group_key"],
            edge_width=params["edge_width"],
            edge_alpha=params["edge_alpha"],
            zorder=n_steps,
        )
        return sdata

    def show(
        self,
        coordinate_systems: list[str] | str | None = None,
        *,
        legend_fontsize: int | float | _FontSize | None = None,
        legend_fontweight: int | _FontWeight = "bold",
        legend_loc: str | None = "right margin",
        legend_fontoutline: int | None = None,
        na_in_legend: bool = True,
        colorbar: bool = True,
        colorbar_params: dict[str, object] | None = None,
        wspace: float | None = None,
        hspace: float = 0.25,
        ncols: int = 4,
        frameon: bool | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        fig: Figure | None = None,
        title: list[str] | str | None = None,
        pad_extent: int | float = 0,
        ax: list[Axes] | Axes | None = None,
        return_ax: bool = False,
        save: str | Path | None = None,
        show: bool | None = None,
    ) -> Axes | list[Axes] | None:
        """
        Execute the plotting tree and display the final figure.

        Parameters
        ----------
        coordinate_systems : list[str] | str | None
            Name(s) of the coordinate system(s) to be plotted. If ``None``, all coordinate systems that contain
            relevant elements (as specified in the ``render_*`` calls) are plotted automatically.
        legend_fontsize : int | float | str | None
            Font size for the legend text. Accepts numeric values or matplotlib font size strings
            (e.g. ``"small"``, ``"large"``).
        legend_fontweight : int | str, default "bold"
            Font weight for the legend text (e.g. ``"bold"``, ``"normal"``).
        legend_loc : str | None, default "right margin"
            Location of the legend. Standard matplotlib legend locations (e.g. ``"upper left"``) or
            ``"right margin"``, ``"left margin"``, ``"top margin"``, ``"bottom margin"`` to place
            the legend outside the axes.
        legend_fontoutline : int | None
            Stroke width for a white outline around legend text, improving readability on busy plots.
        na_in_legend : bool, default True
            Whether to include NA / unmapped categories in the legend.
        colorbar : bool, default True
            Global switch to enable/disable all colorbars. Per-layer settings are ignored when this is ``False``.
        colorbar_params : dict[str, object] | None
            Global overrides passed to colorbars for all axes. Accepts the same keys as per-layer ``colorbar_params``
            (e.g., ``loc``, ``width``, ``pad``, ``label``).
        wspace : float | None
            Horizontal spacing between panels (passed to :class:`matplotlib.gridspec.GridSpec`).
        hspace : float, default 0.25
            Vertical spacing between panels (passed to :class:`matplotlib.gridspec.GridSpec`).
        ncols : int, default 4
            Number of columns in the multi-panel grid.
        frameon : bool | None
            Whether to draw the axes frame. If ``None``, the frame is hidden automatically for multi-panel plots.
        figsize : tuple[float, float] | None
            Size of the figure ``(width, height)`` in inches. The actual canvas size in pixels is
            ``dpi * figsize``. If ``None``, the matplotlib default is used ``(6.4, 4.8)``.
        dpi : int | None
            Resolution of the plot in dots per inch. If ``None``, the matplotlib default is used ``(100.0)``.
        fig : Figure | None
            .. deprecated::
                Pass axes created from your figure via ``ax`` instead.
        title : list[str] | str | None
            Title(s) for the plot. A single string is applied to all panels; a list must match the number
            of coordinate systems. If ``None``, each panel is titled with its coordinate system name.
        pad_extent : int | float, default 0
            Padding added around the computed spatial extent on all sides.
        ax : list[Axes] | Axes | None
            Pre-existing matplotlib axes to plot on. Can be a single :class:`~matplotlib.axes.Axes` or a list
            matching the number of coordinate systems. If ``None``, a new figure and axes are created.
        return_ax : bool, default False
            Whether to return the axes object(s) instead of ``None``.
        save : str | Path | None
            Path to save the figure to. If ``None``, the figure is not saved.
        show : bool | None
            Whether to call ``plt.show()`` at the end. If ``None`` (default), the plot is shown
            automatically when running in non-interactive mode (scripts) and suppressed in
            interactive sessions (e.g. Jupyter). When ``ax`` is provided by the user, defaults
            to ``False`` to allow further modifications.

        Returns
        -------
        Axes | list[Axes] | None
            The axes object(s) if ``return_ax=True``, otherwise ``None``.
        """
        _log_context.set("show")
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
            colorbar_params,
            wspace,
            hspace,
            ncols,
            frameon,
            figsize,
            dpi,
            fig,
            title,
            pad_extent,
            ax,
            return_ax,
            save,
            show,
        )

        if fig is not None and not isinstance(ax, Sequence):
            warnings.warn(
                "`fig` is being deprecated as an argument to `PlotAccessor.show` in spatialdata-plot. "
                "To use a custom figure, create axes from it and pass them via `ax` instead: "
                "`ax = fig.add_subplot(111)`.",
                DeprecationWarning,
                stacklevel=2,
            )

        sdata = self._copy()

        # Evaluate execution tree for plotting
        valid_commands = [
            "render_images",
            "render_shapes",
            "render_labels",
            "render_points",
            "render_graph",
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

        # Track whether the caller supplied their own axes so we can skip
        # plt.show() later (ax is reassigned inside the rendering loop).
        user_supplied_ax = ax is not None

        # get original axis extent for later comparison
        ax_x_min, ax_x_max = (np.inf, -np.inf)
        ax_y_min, ax_y_max = (np.inf, -np.inf)

        if isinstance(ax, Axes) and _mpl_ax_contains_elements(ax):
            ax_x_min, ax_x_max = ax.get_xlim()
            ax_y_max, ax_y_min = ax.get_ylim()  # (0, 0) is top-left

        cs_was_auto = coordinate_systems is None
        coordinate_systems = list(sdata.coordinate_systems) if cs_was_auto else coordinate_systems
        if isinstance(coordinate_systems, str):
            coordinate_systems = [coordinate_systems]
        assert coordinate_systems is not None

        for cs in coordinate_systems:
            if cs not in sdata.coordinate_systems:
                raise ValueError(f"Unknown coordinate system '{cs}', valid choices are: {sdata.coordinate_systems}")

        # Check if user specified only certain elements to be plotted
        cs_contents = _get_cs_contents(sdata)
        pending_colorbars: list[tuple[Axes, list[ColorbarSpec]]] = []

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

        # When CS was auto-detected and ax is provided, keep only CS that have
        # element types for ALL render commands (workaround for upstream #176).
        if ax is not None:
            n_ax = 1 if isinstance(ax, Axes) else len(ax)
            if cs_was_auto and len(coordinate_systems) > n_ax:
                required_flags = [_RENDER_CMD_TO_CS_FLAG[cmd] for cmd in cmds if cmd in _RENDER_CMD_TO_CS_FLAG]
                strict_cs = [
                    cs_name
                    for cs_name in coordinate_systems
                    if all(cs_contents.query(f"cs == '{cs_name}'").iloc[0][flag] for flag in required_flags)
                ]
                if strict_cs:
                    coordinate_systems = strict_cs

            if len(coordinate_systems) != n_ax:
                msg = (
                    f"Mismatch between number of matplotlib axes objects ({n_ax}) "
                    f"and number of coordinate systems ({len(coordinate_systems)})."
                )
                if cs_was_auto:
                    msg += (
                        " This can happen when elements have transformations to multiple "
                        "coordinate systems (e.g. after filter_by_coordinate_system). "
                        "Pass `coordinate_systems=` explicitly to select which ones to plot."
                    )
                raise ValueError(msg)

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
        legend_colorbar = colorbar
        legend_params = LegendParams(
            legend_fontsize=legend_fontsize,
            legend_fontweight=legend_fontweight,
            legend_loc=legend_loc,
            legend_fontoutline=legend_fontoutline,
            na_in_legend=na_in_legend,
            colorbar=legend_colorbar,
        )

        def _draw_colorbar(
            spec: ColorbarSpec,
            fig: Figure,
            renderer: RendererBase,
            base_offsets_axes: dict[str, float],
            trackers_axes: dict[str, float],
        ) -> None:
            base_layout = {
                "location": CBAR_DEFAULT_LOCATION,
                "fraction": CBAR_DEFAULT_FRACTION,
                "pad": CBAR_DEFAULT_PAD,
            }
            layer_layout, layer_kwargs, layer_label_override = _split_colorbar_params(spec.params)
            global_layout, global_kwargs, global_label_override = _split_colorbar_params(colorbar_params)
            layout = {**base_layout, **layer_layout, **global_layout}
            cbar_kwargs = {**layer_kwargs, **global_kwargs}

            location = cast(str, layout.get("location", base_layout["location"]))
            if location not in {"left", "right", "top", "bottom"}:
                location = CBAR_DEFAULT_LOCATION
            default_orientation = "vertical" if location in {"right", "left"} else "horizontal"
            cbar_kwargs.setdefault("orientation", default_orientation)

            fraction = float(cast(float | int, layout.get("fraction", base_layout["fraction"])))
            pad = float(cast(float | int, layout.get("pad", base_layout["pad"])))

            if location in {"left", "right"}:
                pad_axes = pad + trackers_axes[location]
                x0 = -pad_axes - fraction if location == "left" else 1 + pad_axes
                bbox = (float(x0), 0.0, float(fraction), 1.0)
            else:
                pad_axes = pad + trackers_axes[location]
                y0 = -pad_axes - fraction if location == "bottom" else 1 + pad_axes
                bbox = (0.0, float(y0), 1.0, float(fraction))
            cax = inset_axes(
                spec.ax,
                width="100%",
                height="100%",
                loc="center",
                bbox_to_anchor=bbox,
                bbox_transform=spec.ax.transAxes,
                borderpad=0.0,
            )

            cb = fig.colorbar(spec.mappable, cax=cax, **cbar_kwargs)
            if location == "left":
                cb.ax.yaxis.set_ticks_position("left")
                cb.ax.yaxis.set_label_position("left")
                cb.ax.tick_params(labelleft=True, labelright=False)
            elif location == "top":
                cb.ax.xaxis.set_ticks_position("top")
                cb.ax.xaxis.set_label_position("top")
                cb.ax.tick_params(labeltop=True, labelbottom=False)
            elif location == "right":
                cb.ax.yaxis.set_ticks_position("right")
                cb.ax.yaxis.set_label_position("right")
                cb.ax.tick_params(labelright=True, labelleft=False)
            elif location == "bottom":
                cb.ax.xaxis.set_ticks_position("bottom")
                cb.ax.xaxis.set_label_position("bottom")
                cb.ax.tick_params(labelbottom=True, labeltop=False)

            final_label = global_label_override or layer_label_override or spec.label
            if final_label:
                cb.set_label(final_label)
            if spec.alpha is not None:
                with contextlib.suppress(Exception):
                    cb.solids.set_alpha(spec.alpha)
            bbox_axes = cb.ax.get_tightbbox(renderer).transformed(spec.ax.transAxes.inverted())
            if location == "left":
                trackers_axes["left"] = pad_axes + bbox_axes.width
            elif location == "right":
                trackers_axes["right"] = pad_axes + bbox_axes.width
            elif location == "bottom":
                trackers_axes["bottom"] = pad_axes + bbox_axes.height
            elif location == "top":
                trackers_axes["top"] = pad_axes + bbox_axes.height

        cs_contents = _get_cs_contents(sdata)

        # go through tree

        for i, cs in enumerate(coordinate_systems):
            sdata = self._copy()
            _, has_images, has_labels, has_points, has_shapes = (
                cs_contents.query(f"cs == '{cs}'").iloc[0, :].values.tolist()
            )
            ax = fig_params.ax if fig_params.axs is None else fig_params.axs[i]
            assert isinstance(ax, Axes)
            axis_colorbar_requests: list[ColorbarSpec] | None = [] if legend_params.colorbar else None
            axis_channel_legend_entries: list[ChannelLegendEntry] = []

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
                            colorbar_requests=axis_colorbar_requests,
                            channel_legend_entries=axis_channel_legend_entries,
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
                            colorbar_requests=axis_colorbar_requests,
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
                            colorbar_requests=axis_colorbar_requests,
                        )

                elif cmd == "render_labels" and has_labels:
                    wanted_elements, wanted_labels_on_this_cs, wants_labels = _get_wanted_render_elements(
                        sdata, wanted_elements, params_copy, cs, "labels"
                    )

                    if wanted_labels_on_this_cs:
                        table = params_copy.table_name
                        if table is not None and params_copy.col_for_color is not None:
                            colors = sc.get.obs_df(sdata[table], [params_copy.col_for_color])
                            if isinstance(
                                colors[params_copy.col_for_color].dtype,
                                pd.CategoricalDtype,
                            ):
                                _maybe_set_colors(
                                    source=sdata[table],
                                    target=sdata[table],
                                    key=params_copy.col_for_color,
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
                            colorbar_requests=axis_colorbar_requests,
                            rasterize=rasterize,
                        )

                elif cmd == "render_graph":
                    # Graph rendering: resolve which element the graph connects,
                    # check if that element exists in this CS.
                    graph_element = params_copy.element
                    element_in_cs = (
                        (graph_element in sdata.shapes and has_shapes)
                        or (graph_element in sdata.points and has_points)
                        or (graph_element in sdata.labels and has_labels)
                    )
                    if element_in_cs:
                        _render_graph(
                            sdata=sdata,
                            render_params=params_copy,
                            coordinate_system=cs,
                            ax=ax,
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
                if fig_params.frameon is False:
                    ax.axis("off")

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

            if legend_params.colorbar and axis_colorbar_requests:
                pending_colorbars.append((ax, axis_colorbar_requests))

            if axis_channel_legend_entries:
                _draw_channel_legend(ax, axis_channel_legend_entries, legend_params, fig_params)

        if pending_colorbars and fig_params.fig is not None:
            fig = fig_params.fig
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for axis, requests in pending_colorbars:
                unique_specs: list[ColorbarSpec] = []
                seen_mappables: set[int] = set()
                for spec in requests:
                    mappable_id = id(spec.mappable)
                    if mappable_id in seen_mappables:
                        continue
                    seen_mappables.add(mappable_id)
                    unique_specs.append(spec)
                tight_bbox = axis.get_tightbbox(renderer).transformed(axis.transAxes.inverted())
                base_offsets_axes = {
                    "left": max(0.0, -tight_bbox.x0),
                    "right": max(0.0, tight_bbox.x1 - 1),
                    "bottom": max(0.0, -tight_bbox.y0),
                    "top": max(0.0, tight_bbox.y1 - 1),
                }
                trackers_axes = {k: base_offsets_axes[k] for k in base_offsets_axes}
                for spec in unique_specs:
                    _draw_colorbar(spec, fig, renderer, base_offsets_axes, trackers_axes)

        if fig_params.fig is not None and save is not None:
            save_fig(fig_params.fig, path=save)

        # Show the plot unless the caller opted out.
        # Default (show=None): display in non-interactive mode (scripts), suppress in interactive
        # sessions. We check both sys.ps1 (standard REPL) and matplotlib.is_interactive()
        # (covers IPython, Jupyter, plt.ion(), and IDE consoles like PyCharm).
        # When the user supplies their own axes, they manage the figure lifecycle, so we
        # default to not calling plt.show(). This allows multiple .pl.show(ax=...) calls
        # to accumulate content on the same axes (see #362, #71).
        if show is None:
            show = False if user_supplied_ax else (not hasattr(sys, "ps1") and not matplotlib.is_interactive())
        if show:
            plt.show()
        return (fig_params.ax if fig_params.axs is None else fig_params.axs) if return_ax else None  # shuts up ruff
