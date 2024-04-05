from __future__ import annotations

import sys
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata._core.data_extent import get_extent

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
    _update_params,
    _validate_render_params,
    _validate_show_parameters,
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
        tables: dict[str, AnnData] | None = None,
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

    def render_shapes(
        self,
        elements: list[str] | str | None = None,
        color: list[str | None] | str | None = None,
        fill_alpha: float | int = 1.0,
        groups: list[list[str | None]] | list[str | None] | str | None = None,
        palette: list[list[str | None]] | list[str | None] | str | None = None,
        na_color: ColorLike | None = "lightgrey",
        outline: bool = False,
        outline_width: float | int = 1.5,
        outline_color: str | list[float] = "#000000ff",
        outline_alpha: float | int = 1.0,
        cmap: Colormap | str | None = None,
        norm: bool | Normalize = False,
        scale: float | int = 1.0,
        table_name: list[str] | str | None = None,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render shapes elements in SpatialData.

        Parameters
        ----------
        elements : list[str] | str | None, optional
            The name(s) of the shapes element(s) to render. If `None`, all shapes
            elements in the `SpatialData` object will be used.
        color : list[str | None] | str | None
            Can either be string(s) representing a color-like or key(s) in :attr:`sdata.table.obs`. The latter
            can be used to color by categorical or continuous variables. If provided as a list, the length of the list
            must match the number of elements that will be plotted. Otherwise, if possible the color will be broadcasted
            to all elements. For this, the table in which the color key is found must
            annotate the respective element (region must be set to the specific element). If the color column is found
            in multiple locations, please provide the table_name to be used for the element.
        fill_alpha : float | int, default 1.0
            Alpha value for the fill of shapes. If the alpha channel is present in a cmap passed by the
            user, this value will multiply the value present in the cmap.
        groups : list[list[str | None]] | list[str | None] | str | None
            When using `color` and the key represents discrete labels, `groups`
            can be used to show only a subset of them. Other values are set to NA. In general the case of a list of
            lists means that there is one list per element to be plotted in the list and this list can contain multiple
            discrete labels to be visualized. If not provided as list of lists, broadcasting behaviour is attempted
            (use the same values for all elements).
        palette : list[list[str | None]] | list[str | None] | str | None
            Palette for discrete annotations. List of valid color names that should be
            used for the categories. Must match the number of groups. Similarly to groups, in the case of a list of
            lists means that there is one list per element to be plotted in the list and this list can contain multiple
            palettes (one per group) to be visualized. If not provided as list of lists, broadcasting behaviour is
            attempted (use the same values for all elements). If groups is provided but not palette, palette is set to
            default "lightgray".
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
        cmap : Colormap | str | None, optional
            Colormap for discrete or continuous annotations using 'color', see :class:`matplotlib.colors.Colormap`.
        norm : bool | Normalize, default False
            Colormap normalization for continuous annotations.
        scale : float | int, default 1.0
            Value to scale circles, if present.
        table_name:
            Name of the table(s) containing the color(s) columns. If one name is given than the table is used for each
            spatial element to be plotted if the table annotates it. If multiple names are given in a list than the
            length must be equal to the number of spatial elements being plotted.
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
        params_dict = _validate_render_params(
            "shapes",
            self._sdata,
            elements=elements,
            fill_alpha=fill_alpha,
            groups=groups,
            palette=palette,
            color=color,
            na_color=na_color,
            outline=outline,
            outline_alpha=outline_alpha,
            outline_color=outline_color,
            outline_width=outline_width,
            cmap=cmap,
            norm=norm,
            scale=scale,
        )

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())
        cmap_params = _prepare_cmap_norm(
            cmap=cmap,
            norm=norm,
            na_color=na_color,  # type: ignore[arg-type]
            **kwargs,
        )

        outline_params = _set_outline(outline, outline_width, outline_color)
        sdata.plotting_tree[f"{n_steps+1}_render_shapes"] = ShapesRenderParams(
            elements=params_dict["elements"],
            color=params_dict["color"],
            col_for_color=params_dict["col_for_color"],
            groups=params_dict["groups"],
            scale=scale,
            outline_params=outline_params,
            cmap_params=cmap_params,
            palette=params_dict["palette"],
            outline_alpha=outline_alpha,
            fill_alpha=fill_alpha,
            transfunc=kwargs.get("transfunc", None),
            element_table_mapping=table_name,
        )

        return sdata

    def render_points(
        self,
        elements: list[str] | str | None = None,
        color: list[str | None] | str | None = None,
        alpha: float | int = 1.0,
        groups: list[list[str | None]] | list[str | None] | str | None = None,
        palette: list[list[str | None]] | list[str | None] | str | None = None,
        na_color: ColorLike | None = "lightgrey",
        cmap: Colormap | str | None = None,
        norm: None | Normalize = None,
        size: float | int = 1.0,
        table_name: list[str] | str | None = None,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render points elements in SpatialData.

        Parameters
        ----------
        elements : list[str] | str | None, optional
            The name(s) of the points element(s) to render. If `None`, all points
            elements in the `SpatialData` object will be used.
        color : list[str | None] | str | None
            Can either be string(s) representing a color-like or key(s) in :attr:`sdata.table.obs`. The latter
            can be used to color by categorical or continuous variables. If provided as a list, the length of the list
            must match the number of elements that will be plotted. Otherwise, if possible the color will be broadcasted
            to all elements. For this, the table in which the color key is found must
            annotate the respective element (region must be set to the specific element). If the color column is found
            in multiple locations, please provide the table_name to be used for the element.
        alpha : float | int, default 1.0
            Alpha value for the points.
        groups : list[list[str | None]] | list[str | None] | str | None
            When using `color` and the key represents discrete labels, `groups`
            can be used to show only a subset of them. Other values are set to NA. In general the case of a list of
            lists means that there is one list per element to be plotted in the list and this list can contain multiple
            discrete labels to be visualized. If not provided as list of lists, broadcasting behaviour is attempted
            (use the same values for all elements). If groups is provided but not palette, palette is set to
            default "lightgray".
        palette : list[list[str | None]] | list[str | None] | str | None
            Palette for discrete annotations. List of valid color names that should be
            used for the categories. Must match the number of groups. Similarly to groups, in the case of a list of
            lists means that there is one list per element to be plotted in the list and this list can contain multiple
            palettes (one per group) to be visualized. If not provided as list of lists, broadcasting behaviour is
            attempted (use the same values for all elements).
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
        table_name:
            Name of the table(s) containing the color(s) columns. If one name is given than the table is used for each
            spatial element to be plotted if the table annotates it. If multiple names are given in a list than the
            length must be equal to the number of spatial elements being plotted.
        kwargs
            Additional arguments to be passed to cmap and norm.

        Returns
        -------
        sd.SpatialData
            The modified SpatialData object with the rendered shapes.
        """
        params_dict = _validate_render_params(
            "points",
            self._sdata,
            elements=elements,
            alpha=alpha,
            color=color,
            groups=groups,
            palette=palette,
            na_color=na_color,
            cmap=cmap,
            norm=norm,
            size=size,
        )

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
            elements=params_dict["elements"],
            color=params_dict["color"],
            col_for_color=params_dict["col_for_color"],
            groups=params_dict["groups"],
            cmap_params=cmap_params,
            palette=params_dict["palette"],
            alpha=alpha,
            transfunc=kwargs.get("transfunc", None),
            size=size,
            element_table_mapping=table_name,
        )

        return sdata

    def render_images(
        self,
        elements: list[str] | str | None = None,
        channel: list[str] | list[int] | str | int | None = None,
        cmap: list[Colormap] | Colormap | str | None = None,
        norm: Normalize | None = None,
        na_color: ColorLike | None = (0.0, 0.0, 0.0, 0.0),
        palette: list[list[str | None]] | list[str | None] | str | None = None,
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
        palette : list[list[str | None]] | list[str | None] | str | None
            Palette to color images. In the case of a list of
            lists means that there is one list per element to be plotted in the list and this list contains the string
            indicating the palette to be used. If not provided as list of lists, broadcasting behaviour is
            attempted (use the same values for all elements).
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
        params_dict = _validate_render_params(
            "images",
            self._sdata,
            elements=elements,
            channel=channel,
            alpha=alpha,
            palette=palette,
            na_color=na_color,
            cmap=cmap,
            norm=norm,
            scale=scale,
            quantiles_for_norm=quantiles_for_norm,
        )
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

        sdata.plotting_tree[f"{n_steps+1}_render_images"] = ImageRenderParams(
            elements=params_dict["elements"],
            channel=channel,
            cmap_params=cmap_params,
            palette=params_dict["palette"],
            alpha=alpha,
            quantiles_for_norm=params_dict["quantiles_for_norm"],
            scale=params_dict["scale"],
        )

        return sdata

    def render_labels(
        self,
        elements: list[str] | str | None = None,
        color: list[str | None] | str | None = None,
        groups: list[list[str | None]] | list[str | None] | str | None = None,
        contour_px: int = 3,
        outline: bool = False,
        palette: list[list[str | None]] | list[str | None] | str | None = None,
        cmap: Colormap | str | None = None,
        norm: Normalize | None = None,
        na_color: ColorLike | None = (0.0, 0.0, 0.0, 0.0),
        outline_alpha: float | int = 1.0,
        fill_alpha: float | int = 0.3,
        scale: list[str] | str | None = None,
        table_name: list[str] | str | None = None,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render labels elements in SpatialData.

        Parameters
        ----------
        elements : list[str] | str | None, optional
            The name(s) of the label element(s) to render. If `None`, all label
            elements in the `SpatialData` object will be used.
        color : list[str | None] | str | None
            Can either be string(s) representing a color-like or key(s) in :attr:`sdata.table.obs`. The latter
            can be used to color by categorical or continuous variables. If provided as a list, the length of the list
            must match the number of elements that will be plotted. Otherwise, if possible the color will be broadcasted
            to all elements. For this, the table in which the color key is found must
            annotate the respective element (region must be set to the specific element). If the color column is found
            in multiple locations, please provide the table_name to be used for the element.
        groups : list[list[str | None]] | list[str | None] | str | None
            When using `color` and the key represents discrete labels, `groups`
            can be used to show only a subset of them. Other values are set to NA. In general the case of a list of
            lists means that there is one list per element to be plotted in the list and this list can contain multiple
            discrete labels to be visualized. If not provided as list of lists, broadcasting behaviour is attempted
            (use the same values for all elements).
        palette : list[list[str | None]] | list[str | None] | str | None
            Palette for discrete annotations. List of valid color names that should be
            used for the categories. Must match the number of groups. Similarly to groups, in the case of a list of
            lists means that there is one list per element to be plotted in the list and this list can contain multiple
            palettes (one per group) to be visualized. If not provided as list of lists, broadcasting behaviour is
            attempted (use the same values for all elements). If groups is provided but not palette, palette is set to
            default "lightgray".
        contour_px : int, default 3
            Draw contour of specified width for each segment. If `None`, fills
            entire segment, see :func:`skimage.morphology.erosion`.
        outline : bool, default False
            Whether to plot boundaries around segmentation masks.
        cmap : Colormap | str | None, optional
            Colormap for continuous annotations, see :class:`matplotlib.colors.Colormap`.
        norm : Normalize | None, optional
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
        na_color : ColorLike | None, optional
            Color to be used for NAs values, if present.
        outline_alpha : float | int, default 1.0
            Alpha value for the outline of the labels.
        fill_alpha : float | int, default 0.3
            Alpha value for the fill of the labels.
        scale : list[str] | str | None, optional
            Influences the resolution of the rendering. Possibilities for setting this parameter:
                1) None (default). The image is rasterized to fit the canvas size. For multiscale images, the best scale
                is selected before the rasterization step.
                2) Name of one of the scales in the multiscale image to be rendered. This scale is rendered as it is
                (exception: a dpi is specified in `show()`. Then the image is rasterized to fit the canvas and dpi).
                3) "full": render the full image without rasterization. In the case of a multiscale image, the scale
                with the highest resolution is selected. This can lead to long computing times for large images!
                4) List that is matched to the list of elements (can contain `None`, scale names or "full").
        table_name:
            Name of the table(s) containing the color(s) columns. If one name is given than the table is used for each
            spatial element to be plotted if the table annotates it. If multiple names are given in a list than the
            length must be equal to the number of spatial elements being plotted.
        kwargs
            Additional arguments to be passed to cmap and norm.

        Returns
        -------
        None
        """
        params_dict = _validate_render_params(
            "labels",
            self._sdata,
            elements=elements,
            cmap=cmap,
            color=color,
            contour_px=contour_px,
            fill_alpha=fill_alpha,
            groups=groups,
            na_color=na_color,
            norm=norm,
            outline=outline,
            outline_alpha=outline_alpha,
            palette=palette,
            scale=scale,
        )

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())
        cmap_params = _prepare_cmap_norm(
            cmap=cmap,
            norm=norm,
            na_color=na_color,  # type: ignore[arg-type]
            **kwargs,
        )

        sdata.plotting_tree[f"{n_steps+1}_render_labels"] = LabelsRenderParams(
            elements=params_dict["elements"],
            color=params_dict["color"],
            groups=params_dict["groups"],
            contour_px=contour_px,
            outline=outline,
            cmap_params=cmap_params,
            palette=params_dict["palette"],
            outline_alpha=outline_alpha,
            fill_alpha=fill_alpha,
            transfunc=kwargs.get("transfunc", None),
            scale=params_dict["scale"],
            element_table_mapping=table_name,
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
                        params_copy = _update_params(sdata, params_copy, wanted_images_on_this_cs, "images")
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
                        params_copy = _update_params(sdata, params_copy, wanted_shapes_on_this_cs, "shapes")
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
                        params_copy = _update_params(sdata, params_copy, wanted_points_on_this_cs, "points")
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
                        params_copy = _update_params(sdata, params_copy, wanted_labels_on_this_cs, "labels")

                        for index, table in enumerate(params_copy.element_table_mapping.values()):
                            if table is None:
                                continue
                            colors = sc.get.obs_df(sdata[table], params_copy.color[index])
                            if isinstance(colors.dtype, pd.CategoricalDtype):
                                _maybe_set_colors(
                                    source=sdata[table],
                                    target=sdata[table],
                                    key=params_copy.color[index],
                                    palette=params_copy.palette[index],
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
