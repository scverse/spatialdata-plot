from __future__ import annotations

import contextlib
import sys
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast, get_args

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
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from spatialdata import get_extent
from spatialdata._utils import _deprecation_alias
from spatialdata.transformations.operations import get_transformation
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
    ColorLike,
    GraphRenderParams,
    ImageRenderParams,
    LabelsRenderParams,
    LegendParams,
    PointsRenderParams,
    ShapesRenderParams,
    _DsReduction,
    _FontSize,
    _FontWeight,
    _ImageDsReduction,
)
from spatialdata_plot.pl.utils import (
    _RENDER_CMD_TO_CS_FLAG,
    _draw_scalebar,
    _expand_color_panels,
    _get_cs_contents,
    _get_elements_to_be_rendered,
    _get_extent_fast,
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
        # Shallow-copy the plotting tree so appending a render step to one chain does not mutate a
        # sibling chain branched off the same object (the RenderParams values stay shared; `show()`
        # deep-copies them before use).
        sdata.plotting_tree = (
            OrderedDict(self._sdata.plotting_tree) if hasattr(self._sdata, "plotting_tree") else OrderedDict()
        )
        sdata._source_sdata = getattr(self._sdata, "_source_sdata", self._sdata)

        return sdata

    def annotate(
        self,
        *,
        coordinate_systems: str | None = None,
        point_radius_frac: float = 0.005,
        figsize: tuple[float, float] = (7, 7),
        dpi: int = 120,
    ) -> Any:
        """Terminal step on a render chain: drop the plot into an interactive annotator.

        Renders the accumulated ``plotting_tree`` (so any ``render_images`` /
        ``render_shapes`` / ``render_points`` / ``render_labels`` overlays composed
        upstream of this call appear in the annotation canvas), then hands the
        rasterised figure to a ``BioImageViewer`` widget. The user draws
        rectangles, polygons, and points on the canvas, types a name, and clicks
        *Save* — the shapes are converted from canvas-pixel space to the chosen
        coordinate system and stored in ``sdata.shapes[<name>]`` with an
        ``Identity`` transformation in that CS. Points are stored as small
        circle polygons (radius = ``point_radius_frac`` of the rendered image's
        CS extent) so the resulting ``ShapesModel`` is uniform-type.

        Single coordinate system only. If the chain spans more than one CS, or
        none can be inferred, raises ``ValueError``.

        Requires the ``interactive`` extra: ``pip install 'spatialdata-plot[interactive]'``.

        Parameters
        ----------
        coordinate_systems :
            Coordinate system to render and resolve drawn shapes against.
            Drawn shapes are stored with an ``Identity`` transformation in this
            CS. If ``None`` and the SpatialData has exactly one CS, that one is
            used; otherwise this argument is required.
        point_radius_frac :
            Radius of the circle polygon used to store each point, expressed as
            a fraction of the rendered image's CS extent. Default 0.005 (0.5%).
        figsize :
            Matplotlib figure size used for the underlying rasterisation. The
            same value affects the canvas resolution alongside ``dpi``.
        dpi :
            DPI of the rasterised figure. Combined with ``figsize`` this sets
            the pixel resolution the annotator works in.

        Returns
        -------
        InteractiveSession
            The session object, with the widget already displayed. Holding the
            reference keeps the underlying ``BioImageViewer`` alive across cell
            re-runs; usually you can ignore the return value.

        Raises
        ------
        ValueError
            If no single coordinate system can be resolved.
        ImportError
            If the ``interactive`` extra is not installed.

        Examples
        --------
        >>> import spatialdata_plot  # noqa: F401  registers .pl
        >>> (
        ...     sdata.pl
        ...     .render_images(element="he")
        ...     .pl.render_shapes(element="cells", outline_color="red")
        ...     .pl.annotate()
        ... )
        >>> # ... user draws and clicks Save with name "tumor" ...
        >>> sdata.shapes["tumor"]
        """
        try:
            from spatialdata_plot.pl.interactive._session import _InteractiveSession
        except ImportError as exc:
            raise ImportError(
                "sdata.pl.annotate() requires the `interactive` extra. "
                "Install with: pip install 'spatialdata-plot[interactive]'"
            ) from exc

        import io as _io

        from PIL import Image as _Image

        available_cs = list(self._sdata.coordinate_systems)
        if coordinate_systems is None:
            if len(available_cs) != 1:
                raise ValueError(
                    "annotate() needs exactly one coordinate system. "
                    f"SpatialData has {len(available_cs)}: {available_cs!r}. "
                    "Pass coordinate_systems=<name> explicitly."
                )
            cs = available_cs[0]
        else:
            if isinstance(coordinate_systems, list):
                if len(coordinate_systems) != 1:
                    raise ValueError(f"annotate() supports a single coordinate system; got {coordinate_systems!r}.")
                cs = coordinate_systems[0]
            else:
                cs = coordinate_systems
            if cs not in available_cs:
                raise ValueError(f"Unknown coordinate system {cs!r}. Available: {available_cs!r}")

        fig = plt.figure(figsize=figsize, dpi=dpi)
        try:
            ax = fig.add_axes([0, 0, 1, 1])
            self.show(coordinate_systems=cs, ax=ax)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_axis_off()
            # set_aspect("equal") inside show() can shrink the axes box so the
            # figure has blank padding around the data. Crop the saved PNG to
            # the axes bbox so PNG pixels map 1:1 to (xlim, ylim) and the
            # px→cs transform in _commit.py stays correct.
            fig.canvas.draw()
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            buf = _io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox, pad_inches=0)
        finally:
            plt.close(fig)
        rgb = np.asarray(_Image.open(buf).convert("RGB"))

        target_sdata = getattr(self._sdata, "_source_sdata", self._sdata)
        session = _InteractiveSession(
            sdata=target_sdata,
            coordinate_system=cs,
            rgb=rgb,
            xlim=tuple(xlim),
            ylim=tuple(ylim),
            point_radius_frac=point_radius_frac,
        )
        session.show()
        return session

    @_deprecation_alias(elements="element", version="0.3.0")
    def render_shapes(
        self,
        element: str | None = None,
        color: ColorLike | list[str] | None = None,
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
        datashader_reduction: _DsReduction | None = None,
        transfunc: Callable[[float], float] | None = None,
        as_points: bool = False,
        size: float | int = 1.0,
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
        color : ColorLike | list[str] | None, optional
            Can either be color-like (name of a color as string, e.g. "red", hex representation, e.g. "#000000" or
            "#000000ff", or an RGB(A) array as a tuple or list containing 3-4 floats within [0, 1]. If an alpha value is
            indicated, the value of `fill_alpha` takes precedence if given) or a string representing a key in
            :attr:`sdata.table.obs`. The latter can be used to color by categorical or continuous variables. If
            `element` is `None`, if possible the color will be broadcasted to all elements. For this, the table in which
            the color key is found must annotate the respective element (region must be set to the specific element). If
            the color column is found in multiple locations, please provide the table_name to be used for the elements.
            A **list of column/key names** (e.g. ``["gene1", "gene2"]``) produces one panel per key, like
            ``scanpy``'s ``color=[...]``. ``palette``/``cmap``/``norm``/``groups`` are applied to every panel,
            each panel auto-scales independently, and ``show(ncols=...)`` controls the grid width. Multi-panel
            color requires a single coordinate system and only one ``render_*`` call in the chain may pass a list
            (other calls use a scalar color and are drawn into every panel as a shared background).
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
        na_color : ColorLike | None, optional
            Color for NA values and, when ``groups`` is set, for non-matching elements. When omitted, non-matching
            elements are hidden. Pass any explicit color (e.g. ``"lightgray"``) to show them in that color instead.
            Accepts a named color (``"red"``), a hex string (``"#000000ff"``), or an RGB/RGBA list
            (``[1.0, 0.0, 0.0, 1.0]``). Pass ``None`` to make NA values fully transparent.
        outline_width : float | int | tuple[float | int, float | int], optional
            Width of the border. If 2 values are given (tuple), 2 borders are shown with these widths (outer & inner).
            If `outline_color` and/or `outline_alpha` are used to indicate that one/two outlines should be drawn, the
            default outline widths 1.5 and 0.5 are used for outer/only and inner outline respectively.
        outline_color : ColorLike | tuple[ColorLike] | str, optional
            Color of the border. Can either be a named color ("red"), a hex representation ("#000000") or a list of
            floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). If the hex representation includes alpha, e.g.
            "#000000ff", and `outline_alpha` is not given, this value controls the opacity of the outline. If 2 values
            are given (tuple), 2 borders are shown with these colors (outer & inner). If `outline_width` and/or
            `outline_alpha` are used to indicate that one/two outlines should be drawn, the default outline colors
            "#000000" and "#ffffff are used for outer/only and inner outline respectively.
            A string that is not a recognized color is interpreted as a column key (in `obs` of the annotating table
            or in the element's own dataframe), mirroring how ``color`` is parsed. The outline is then colored
            per-shape using the same ``palette`` / ``cmap`` / ``na_color`` as the fill. When both ``color`` and
            ``outline_color`` resolve to columns, two stacked legends are drawn. Column-based outline coloring is
            only supported for a single outline (not the 2-tuple form).
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
        datashader_reduction : Literal["sum", "mean", "any", "count", "std", "var", "max", "min"] | None, optional
            Reduction method for datashader when coloring by continuous values. When ``None``, defaults to ``"max"``.
        transfunc : Callable[[float], float] | None, optional
            Optional transformation applied to the continuous color vector before normalization and colormap mapping.

        Notes
        -----
        - Empty geometries will be removed at the time of plotting.
        - An `outline_width` of 0.0 leads to no border being plotted.
        - If ``color`` is a string that is both a matplotlib color name and a column name in the
          element or an annotating table, a ``ValueError`` is raised. Disambiguate by passing
          a hex string (e.g. ``"#ffa500"``) or an RGB(A) tuple, or by renaming the column.

        Returns
        -------
        sd.SpatialData
            A copy of the SpatialData object with the rendering parameters stored in its plotting tree.
        """
        if as_points:
            if isinstance(size, bool) or not isinstance(size, (int, float)):
                raise TypeError("Parameter 'size' must be numeric.")
            if size <= 0:
                raise ValueError("Parameter 'size' must be a positive number.")
        panel_param_dicts = _expand_color_panels(
            self._sdata,
            color,
            "render_shapes",
            lambda color_value: _validate_shape_render_params(
                self._sdata,
                element=element,
                fill_alpha=fill_alpha,
                groups=groups,
                palette=palette,
                color=color_value,
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
                ds_reduction=datashader_reduction,
                colorbar=colorbar,
                colorbar_params=colorbar_params,
                gene_symbols=gene_symbols,
            ),
        )

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)

        n_steps = len(sdata.plotting_tree.keys())
        for panel_key, params_dict in panel_param_dicts:
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
                    col_for_outline_color=param_values["col_for_outline_color"],
                    outline_table_name=param_values["outline_table_name"],
                    groups=param_values["groups"],
                    scale=param_values["scale"],
                    outline_params=outline_params,
                    cmap_params=cmap_params,
                    palette=param_values["palette"],
                    outline_alpha=final_outline_alpha,
                    fill_alpha=param_values["fill_alpha"],
                    transfunc=transfunc,
                    table_name=param_values["table_name"],
                    table_layer=param_values["table_layer"],
                    shape=param_values["shape"],
                    zorder=n_steps,
                    method=param_values["method"],
                    ds_reduction=param_values["ds_reduction"],
                    colorbar=param_values["colorbar"],
                    colorbar_params=param_values["colorbar_params"],
                    as_points=as_points,
                    size=size,
                    panel_key=panel_key,
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
        datashader_reduction: _DsReduction | None = None,
        density: bool = False,
        density_how: Literal["linear", "log", "cbrt", "eq_hist"] = "linear",
        transfunc: Callable[[float], float] | None = None,
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
        na_color : ColorLike | None, optional
            Color for NA values and, when ``groups`` is set, for non-matching points. When omitted, non-matching
            points are hidden. Pass any explicit color (e.g. ``"lightgray"``) to show them in that color instead.
            Accepts a named color (``"red"``), a hex string (``"#000000ff"``), or an RGB/RGBA list
            (``[1.0, 0.0, 0.0, 1.0]``). Pass ``None`` to make NA values fully transparent.
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
        datashader_reduction : Literal["sum", "mean", "any", "count", "std", "var", "max", "min"] | None, optional
            Reduction method for datashader when coloring by continuous values. When ``None``, defaults to ``"sum"``.
        density : bool, default False
            Render the points as a 2-D count density via datashader instead of plotting individual markers.
            When ``True``, ``method`` is forced to ``"datashader"`` (passing ``method="matplotlib"`` raises).
            Density supports ``color=None`` (plain density) or a categorical ``color`` column (per-category
            density via :func:`datashader.by`). A continuous ``color`` column or a literal color value
            (e.g. ``"red"``) raises an error. Under ``density=True`` the following parameters are ignored
            (with a warning if explicitly set): ``size``, ``transfunc``, ``norm.vmin/vmax``, and
            ``datashader_reduction``.
        density_how : Literal["linear", "log", "cbrt", "eq_hist"], default "linear"
            How datashader maps aggregated counts to color intensity. ``"linear"`` (default) keeps the
            colorbar axis as a count; ``"log"`` and ``"cbrt"`` compress dynamic range; ``"eq_hist"``
            equalizes the histogram (rank-based, surfaces the most structure but the colorbar axis is
            no longer a count). Ignored when ``density=False``.
        transfunc : Callable[[float], float] | None, optional
            Optional transformation applied to the continuous color vector before normalization and colormap mapping.

        Returns
        -------
        sd.SpatialData
            A copy of the SpatialData object with the rendering parameters stored in its plotting tree.

        Examples
        --------
        Plain density of all transcripts:

        >>> sdata.pl.render_points("transcripts", density=True).pl.show()

        Per-gene density with a categorical palette:

        >>> sdata.pl.render_points(
        ...     "transcripts", color="gene", groups=["Gad1", "Slc17a7"], palette="tab20", density=True
        ... ).pl.show()
        """
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
            ds_reduction=datashader_reduction,
            colorbar=colorbar,
            colorbar_params=colorbar_params,
            gene_symbols=gene_symbols,
            density=density,
            density_how=density_how,
            transfunc=transfunc,
            method=method,
        )

        if method is not None:
            if not isinstance(method, str):
                raise TypeError("Parameter 'method' must be a string.")
            if method not in ["matplotlib", "datashader"]:
                raise ValueError("Parameter 'method' must be either 'matplotlib' or 'datashader'.")

        if density and method is None:
            method = "datashader"

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
                transfunc=transfunc,
                size=param_values["size"],
                table_name=param_values["table_name"],
                table_layer=param_values["table_layer"],
                zorder=n_steps,
                method=method,
                ds_reduction=param_values["ds_reduction"],
                colorbar=param_values["colorbar"],
                colorbar_params=param_values["colorbar_params"],
                density=density,
                density_how=density_how,
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
        method: Literal["matplotlib", "datashader"] | None = None,
        datashader_reduction: _ImageDsReduction | None = None,
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
        method : str | None, optional
            Whether to use ``'matplotlib'`` (default) or ``'datashader'`` for
            the downsampling step.  When ``'datashader'`` is selected, the
            rasterization-to-canvas step uses
            :meth:`datashader.Canvas.raster` with ``datashader_reduction`` as the
            downsample method (default ``'max'``), and ``imshow`` is rendered
            with ``interpolation='nearest'`` so the chosen reduction is not
            re-smoothed at display time.  Useful for very sparse images
            (mostly zeros) where mean aggregation collapses the signal —
            ``method='datashader'`` with ``datashader_reduction='max'`` preserves the
            rare non-zero pixels (``plt.spy``-style).
        datashader_reduction : {"max", "min", "mean", "mode", "first", "last", "var", "std"} | None, optional
            Downsample reduction used by the datashader path.  Defaults to
            ``'max'`` when ``method='datashader'``.  Ignored otherwise (a
            warning is emitted if set without ``method='datashader'``).

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

        if method is not None and not isinstance(method, str):
            raise TypeError("Parameter 'method' must be a string.")
        if method is not None and method not in ("matplotlib", "datashader"):
            raise ValueError("Parameter 'method' must be either 'matplotlib' or 'datashader'.")
        _valid_image_reductions = get_args(_ImageDsReduction)
        if datashader_reduction is not None and not isinstance(datashader_reduction, str):
            raise TypeError("Parameter 'datashader_reduction' must be a string.")
        if datashader_reduction is not None and datashader_reduction not in _valid_image_reductions:
            raise ValueError(
                f"Parameter 'datashader_reduction' must be one of {_valid_image_reductions}, "
                f"got {datashader_reduction!r}."
            )
        if datashader_reduction is not None and method != "datashader":
            logger.warning("Parameter 'datashader_reduction' has no effect unless method='datashader'; ignoring.")

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
                method=method,
                ds_reduction=datashader_reduction,
            )
            n_steps += 1

        return sdata

    @_deprecation_alias(elements="element", version="0.3.0")
    def render_labels(
        self,
        element: str | None = None,
        color: ColorLike | list[str] | None = None,
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
        transfunc: Callable[[float], float] | None = None,
        as_points: bool = False,
        size: float | int = 1.0,
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
        color : ColorLike | list[str] | None
            Can either be color-like (name of a color as string, e.g. "red", hex representation, e.g. "#000000" or
            "#000000ff", or an RGB(A) array as a tuple or list containing 3-4 floats within [0, 1]. If an alpha value
            is indicated, the value of `fill_alpha` takes precedence if given) or a string representing a key in
            :attr:`sdata.table.obs` or in the index of :attr:`sdata.table.var`. The latter can be used to color by
            categorical or continuous variables. If the color column is found in multiple locations, please provide the
            table_name to be used for the element if you would like a specific table to be used.
            A **list of column/key names** (e.g. ``["gene1", "gene2"]``) produces one panel per key, like
            ``scanpy``'s ``color=[...]``. ``palette``/``cmap``/``norm``/``groups`` are applied to every panel,
            each panel auto-scales independently, and ``show(ncols=...)`` controls the grid width. Multi-panel
            color requires a single coordinate system and only one ``render_*`` call in the chain may pass a list
            (other calls use a scalar color and are drawn into every panel as a shared background).
        groups : list[str] | str | None
            When using `color` and the key represents discrete labels, `groups` can be used to show only a subset of
            them. By default, non-matching labels are hidden. To show non-matching labels, set ``na_color`` explicitly.
        palette : dict[str, str] | list[str] | str | None
            Palette for discrete annotations. Can be a dictionary mapping category names to colors, a list of valid
            color names (must match the number of groups), a single named palette or matplotlib colormap name, or
            ``None``.
        contour_px : int, default 3
            Draw contour of specified width for each segment. Must be >= 2; ``contour_px=1`` is rejected
            because a 1x1 erosion is the identity transformation and produces no visible outline. If
            ``None``, fills entire segment, see :func:`skimage.morphology.erosion`.
        cmap : Colormap | str | None, optional
            Colormap for continuous annotations using 'color', see :class:`matplotlib.colors.Colormap`.
            For categorical data, use ``palette`` instead.
        norm : Normalize | None, optional
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
        na_color : ColorLike | None, optional
            Color for NA values and, when ``groups`` is set, for non-matching labels. When omitted, non-matching
            labels are hidden. Pass any explicit color (e.g. ``"lightgray"``) to show them in that color instead.
            Accepts a named color (``"red"``), a hex string (``"#000000ff"``), or an RGB/RGBA list
            (``[1.0, 0.0, 0.0, 1.0]``). Pass ``None`` to make NA values fully transparent.
        outline_alpha : float | int, default 0.0
            Alpha value for the outline of the labels. Invisible by default.
        fill_alpha : float | int | None, optional
            Alpha value for the fill of the labels. By default, it is set to 0.4 or, if a color is given that implies
            an alpha, that value is used for `fill_alpha`.
        outline_color : ColorLike | str | None
            Color of the outline of the labels. Can either be a named color ("red"), a hex representation
            ("#000000") or a list of floats that represent RGB/RGBA values (1.0, 0.0, 0.0, 1.0). If ``None``,
            the outline inherits from the ``color`` parameter when it is a literal color, or uses data-driven
            per-label colors when ``color`` refers to a column.
            A string that is not a recognized color is interpreted as a column key (in `obs` of the annotating
            table), mirroring how ``color`` is parsed. The outline is then colored per-label using the same
            ``palette`` / ``cmap`` / ``na_color`` as the fill.
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
        transfunc : Callable[[float], float] | None, optional
            Optional transformation applied to the continuous color vector before normalization and colormap mapping.

        Returns
        -------
        sd.SpatialData
            A copy of the SpatialData object with the rendering parameters stored in its plotting tree.
        """
        if as_points:
            if isinstance(size, bool) or not isinstance(size, (int, float)):
                raise TypeError("Parameter 'size' must be numeric.")
            if size <= 0:
                raise ValueError("Parameter 'size' must be a positive number.")
        panel_param_dicts = _expand_color_panels(
            self._sdata,
            color,
            "render_labels",
            lambda color_value: _validate_label_render_params(
                self._sdata,
                element=element,
                cmap=cmap,
                color=color_value,
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
            ),
        )

        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())

        for panel_key, params_dict in panel_param_dicts:
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
                    col_for_outline_color=param_values["col_for_outline_color"],
                    outline_table_name=param_values["outline_table_name"],
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
                    transfunc=transfunc,
                    zorder=n_steps,
                    colorbar=param_values["colorbar"],
                    colorbar_params=param_values["colorbar_params"],
                    as_points=as_points,
                    size=size,
                    panel_key=panel_key,
                )
                n_steps += 1
        return sdata

    def render_graph(
        self,
        element: str | None = None,
        color: ColorLike | None = None,
        *,
        connectivity_key: str = "spatial",
        obsp_key: str | None = None,
        palette: dict[str, str] | list[str] | str | None = None,
        na_color: ColorLike | None = "default",
        cmap: Colormap | str | None = None,
        norm: Normalize | None = None,
        groups: list[str] | str | None = None,
        group_key: str | None = None,
        edge_width: float | Literal["weight"] = 1.0,
        edge_alpha: float | Literal["weight"] = 1.0,
        weight_key: str | None = None,
        linestyle: str | Sequence[str] = "solid",
        rasterize: bool = True,
        include_self_loops: bool = False,
        colorbar: bool | str | None = "auto",
        colorbar_params: dict[str, object] | None = None,
        table_name: str | None = None,
    ) -> sd.SpatialData:
        """Render spatial graph edges between observations.

        Draws edges from a connectivity matrix in ``table.obsp`` using
        centroid coordinates of the linked spatial element.

        Parameters
        ----------
        element : str | None
            Name of the shapes/points/labels element the graph connects.
            Auto-resolved from the table if omitted.
        color : ColorLike | None
            A color-like value applied to every edge, or the name of a
            ``table.obs`` column. Categorical columns colour same-category
            edges by the shared value and cross-category edges by
            ``na_color``. Continuous columns colour edges by the mean of
            their endpoint values. Defaults to grey when unset.
        connectivity_key : str, default "spatial"
            ``table.obsp`` key. Tries ``key`` first, then ``f"{key}_connectivities"``.
        obsp_key : str | None
            ``table.obsp`` matrix used as per-edge scalar; coloured via
            ``cmap``/``norm``. Mutually exclusive with ``color``.
        palette : dict[str, str] | list[str] | str | None
            Palette for categorical obs coloring. Same as :meth:`render_shapes`.
        na_color : ColorLike | None, default "default"
            Colour for cross-category edges. ``None`` makes them transparent.
        cmap : Colormap | str | None
            Colormap for continuous edge coloring.
        norm : Normalize | None
            Pass ``Normalize(vmin=..., vmax=...)`` to clamp the colormap range.
        groups : list[str] | str | None
            Show only edges where **both** endpoints fall in these groups.
            Requires ``group_key``.
        group_key : str | None
            ``table.obs`` column used for group filtering.
        edge_width : float | Literal["weight"], default 1.0
            Line width. Pass ``"weight"`` to scale by ``weight_key`` values
            into ``[0.5, 3.0]``.
        edge_alpha : float | Literal["weight"], default 1.0
            Transparency. Pass ``"weight"`` to scale into ``[0.2, 1.0]``.
        weight_key : str | None
            ``table.obsp`` matrix providing per-edge weights. Defaults to
            ``connectivity_key`` when omitted.
        linestyle : str | Sequence[str], default "solid"
            ``LineCollection`` linestyle (scalar or per-edge).
        rasterize : bool, default True
            Rasterize the edge collection. Set ``False`` for vector output.
        include_self_loops : bool, default False
            Render diagonal entries of the connectivity matrix as circles.
        colorbar : bool | str | None, default "auto"
            Whether to draw a colorbar for continuous edge coloring
            (``obsp_key`` or a continuous obs column). ``"auto"`` draws it
            when a mappable is present; ``True``/``False`` force it on/off.
        colorbar_params : dict[str, object] | None
            Optional matplotlib colorbar kwargs and layout hints
            (e.g. ``{"loc": "right", "fraction": 0.05, "label": "..."}``).
        table_name : str | None
            Table containing the graph. Auto-discovered if omitted.

        Returns
        -------
        sd.SpatialData
            Copy with rendering parameters stored in the plotting tree.

        Notes
        -----
        Chaining with ``render_shapes``/``render_points`` on the same
        categorical column shares the legend; no dedicated edge legend is drawn.
        """
        params = _validate_graph_render_params(
            self._sdata,
            element=element,
            connectivity_key=connectivity_key,
            obsp_key=obsp_key,
            weight_key=weight_key,
            palette=palette,
            na_color=na_color,
            cmap=cmap,
            norm=norm,
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
            connectivity_obsp_key=params["connectivity_obsp_key"],
            table_name=params["table_name"],
            color=params["color"],
            obs_col=params["obs_col"],
            obsp_key=params["obsp_key"],
            cmap_params=params["cmap_params"],
            palette_map=params["palette_map"],
            na_color=params["na_color"],
            color_source=params["color_source"],
            groups=params["groups"],
            group_key=params["group_key"],
            edge_width=params["edge_width"],
            edge_alpha=params["edge_alpha"],
            weight_key=params["weight_key"],
            linestyle=linestyle,
            rasterize=rasterize,
            include_self_loops=include_self_loops,
            zorder=n_steps,
            colorbar=colorbar,
            colorbar_params=colorbar_params,
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
        legend_title: str | None = None,
        outline_legend_title: str | None = None,
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
        scalebar_dx: float | None = None,
        scalebar_units: str = "um",
        scalebar_params: dict[str, Any] | None = None,
        legend_params: dict[str, Any] | None = None,
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
        legend_title : str | None
            Title for the fill categorical legend. When both fill and outline are colored by an obs column, the
            two legends default to ``"fill"`` / ``"outline"`` to disambiguate; pass an explicit string to override
            the fill title. Set to ``None`` (default) to keep the auto-title behavior.
        outline_legend_title : str | None
            Title for the outline categorical legend. Mirrors ``legend_title`` for the outline channel.
        wspace : float | None
            Horizontal spacing between panels (passed to :class:`matplotlib.gridspec.GridSpec`).
        hspace : float, default 0.25
            Vertical spacing between panels (passed to :class:`matplotlib.gridspec.GridSpec`).
        ncols : int, default 4
            Number of columns in the multi-panel grid. Panels are created one per coordinate system, or,
            when a ``render_*`` call was given a list of color keys, one per key (scanpy-style ``color=[...]``).
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
            of panels. If ``None``, each panel is titled with its coordinate system name, or, in multi-panel
            color mode, with its color key.
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
        scalebar_dx : float | None
            Physical size of one axes-unit in ``scalebar_units``. If ``None``, no scalebar is drawn.
            SpatialData coordinate systems carry no unit metadata, so this value must be supplied
            explicitly (e.g. ``1.0`` when axes are already in micrometers; the microns-per-pixel
            value when axes are in image pixels).
        scalebar_units : str, default "um"
            Unit string for the scalebar (passed to :class:`matplotlib_scalebar.scalebar.ScaleBar`).
            Only takes effect when ``scalebar_dx`` is set.
        scalebar_params : dict[str, Any] | None
            Extra keyword arguments forwarded to :class:`matplotlib_scalebar.scalebar.ScaleBar`,
            e.g. ``{"location": "lower right", "color": "white", "length_fraction": 0.25}``.
            See the matplotlib-scalebar documentation for the full list of options.
        legend_params : dict[str, Any] | None
            Bundled legend options; overrides the matching ``legend_*`` flat kwargs. Accepted keys:
            ``location`` (or ``loc``), ``fontsize``, ``fontweight``, ``fontoutline``,
            ``na_in_legend``. Unknown keys raise ``ValueError``.

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
            scalebar_dx,
            scalebar_units,
            scalebar_params,
            legend_params,
        )

        if fig is not None:
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
        cs_index = cs_contents.set_index("cs")
        pending_colorbars: list[tuple[Axes, list[ColorbarSpec]]] = []

        elements_to_be_rendered = _get_elements_to_be_rendered(render_cmds, cs_index, cs)

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
        if ax is not None and cs_was_auto:
            n_ax = 1 if isinstance(ax, Axes) else len(ax)
            if len(coordinate_systems) > n_ax:
                required_flags = [_RENDER_CMD_TO_CS_FLAG[cmd] for cmd in cmds if cmd in _RENDER_CMD_TO_CS_FLAG]
                strict_cs = [
                    cs_name
                    for cs_name in coordinate_systems
                    if cs_name in cs_index.index and all(cs_index.loc[cs_name][flag] for flag in required_flags)
                ]
                if strict_cs:
                    coordinate_systems = strict_cs

        # Determine the panel layout. Panels are normally one per coordinate system, but when a
        # render_* call passed a list of color keys we instead lay out one panel per key within a
        # single coordinate system (scanpy-style `color=[...]`). Render entries tagged with a
        # `panel_key` belong to that key's panel; untagged entries are shared across all panels.
        panel_keys: list[str] = []
        for _cmd, _params in render_cmds:
            pkey = getattr(_params, "panel_key", None)
            if pkey is not None and pkey not in panel_keys:
                panel_keys.append(pkey)
        if panel_keys:
            if len(coordinate_systems) != 1:
                raise ValueError(
                    "A list of color keys (multi-panel plotting) requires exactly one coordinate system, "
                    f"but {len(coordinate_systems)} were selected: {coordinate_systems}. "
                    "Pass `coordinate_systems=` to choose a single one."
                )
            panels: list[tuple[str, str | None]] = [(coordinate_systems[0], key) for key in panel_keys]
        else:
            panels = [(cs, None) for cs in coordinate_systems]
        num_panels = len(panels)

        if ax is not None:
            n_ax = 1 if isinstance(ax, Axes) else len(ax)
            if num_panels != n_ax:
                msg = (
                    f"Mismatch between number of matplotlib axes objects ({n_ax}) and number of panels ({num_panels})."
                )
                if cs_was_auto:
                    msg += (
                        " This can happen when elements have transformations to multiple "
                        "coordinate systems (e.g. after filter_by_coordinate_system). "
                        "Pass `coordinate_systems=` explicitly to select which ones to plot."
                    )
                raise ValueError(msg)

        # set up canvas
        fig_params, scalebar_params_obj = _prepare_params_plot(
            num_panels=num_panels,
            figsize=figsize,
            dpi=dpi,
            fig=fig,
            ax=ax,
            wspace=wspace,
            hspace=hspace,
            ncols=ncols,
            frameon=frameon,
            scalebar_dx=scalebar_dx,
            scalebar_units=scalebar_units,
            scalebar_kwargs=scalebar_params,
        )
        if legend_params:
            legend_fontsize = legend_params.get("fontsize", legend_fontsize)
            legend_fontweight = legend_params.get("fontweight", legend_fontweight)
            # `loc` is matplotlib.Legend's native key; `location` aligns with colorbar/scalebar.
            legend_loc = legend_params.get("location", legend_params.get("loc", legend_loc))
            legend_fontoutline = legend_params.get("fontoutline", legend_fontoutline)
            na_in_legend = legend_params.get("na_in_legend", na_in_legend)

        if legend_loc == "on data":
            raise ValueError("legend_loc='on data' is not supported in spatialdata-plot.")

        legend_params_obj = LegendParams(
            legend_fontsize=legend_fontsize,
            legend_fontweight=legend_fontweight,
            legend_loc=legend_loc,
            legend_fontoutline=legend_fontoutline,
            na_in_legend=na_in_legend,
            colorbar=colorbar,
            legend_title=legend_title,
            outline_legend_title=outline_legend_title,
        )

        def _draw_colorbar(
            spec: ColorbarSpec,
            fig: Figure,
            renderer: RendererBase,
            base_offsets_axes: dict[str, float],
            trackers_axes: dict[str, float],
        ) -> None:
            norm = spec.mappable.norm
            if isinstance(norm, LogNorm):
                vmin, vmax = norm.vmin, norm.vmax
                if vmin is None or vmax is None or vmin <= 0 or vmin >= vmax:
                    warnings.warn(
                        "Data contains zeros or non-positive values; colorbar suppressed for `LogNorm`. "
                        "Pass `colorbar=False` to silence this warning, or clip the data to positive values.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return

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

        # go through tree

        for i, (cs, panel_key) in enumerate(panels):
            sdata = self._copy()
            cs_row = cs_index.loc[cs]
            has_images = cs_row["has_images"]
            has_labels = cs_row["has_labels"]
            has_points = cs_row["has_points"]
            has_shapes = cs_row["has_shapes"]
            ax = fig_params.ax if fig_params.axs is None else fig_params.axs[i]
            assert isinstance(ax, Axes)
            axis_colorbar_requests: list[ColorbarSpec] | None = [] if legend_params_obj.colorbar else None
            axis_channel_legend_entries: list[ChannelLegendEntry] = []

            wants_images = False
            wants_labels = False
            wants_points = False
            wants_shapes = False
            wanted_elements: list[str] = []

            for cmd, params in render_cmds:
                # Skip render entries that belong to a different color panel. Entries with no
                # `panel_key` (None) are shared and drawn into every panel (e.g. a background image).
                cmd_panel_key = getattr(params, "panel_key", None)
                if panel_key is not None and cmd_panel_key is not None and cmd_panel_key != panel_key:
                    continue
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
                            legend_params=legend_params_obj,
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
                            legend_params=legend_params_obj,
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
                            legend_params=legend_params_obj,
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
                            legend_params=legend_params_obj,
                            colorbar_requests=axis_colorbar_requests,
                            rasterize=rasterize,
                        )

                elif cmd == "render_graph":
                    graph_element = params_copy.element
                    element_in_cs = graph_element in sdata and cs in set(
                        get_transformation(sdata[graph_element], get_all=True).keys()
                    )
                    if element_in_cs:
                        _render_graph(
                            sdata=sdata,
                            render_params=params_copy,
                            coordinate_system=cs,
                            ax=ax,
                            legend_params=legend_params_obj,
                            colorbar_requests=axis_colorbar_requests,
                        )

                if title is None:
                    t = panel_key if panel_key is not None else cs
                elif len(title) == 1:
                    t = title[0]
                else:
                    try:
                        t = title[i]
                    except IndexError as e:
                        raise IndexError("The number of titles must match the number of panels.") from e
                ax.set_title(t)
                ax.set_aspect("equal")
                if fig_params.frameon is False:
                    ax.axis("off")

            if has_shapes and wants_shapes:
                empty_shape_elements = [
                    name
                    for name in wanted_elements
                    if name in sdata.shapes and not sdata.shapes[name]["geometry"].apply(lambda g: not g.is_empty).any()
                ]
                if empty_shape_elements:
                    raise ValueError(
                        f"Cannot render shape element(s) {empty_shape_elements} in coordinate system {cs!r}: "
                        "all geometries are empty. Drop the element or restore at least one non-empty geometry."
                    )

            # `_get_extent_fast` skips transforming every shapes/points geometry when the element's
            # transform is axis-aligned (the common scale+translation case); identical result, but
            # avoids the O(N-geometries) bottleneck for large shape collections.
            extent = _get_extent_fast(
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

            if legend_params_obj.colorbar and axis_colorbar_requests:
                pending_colorbars.append((ax, axis_colorbar_requests))

            if axis_channel_legend_entries:
                _draw_channel_legend(ax, axis_channel_legend_entries, legend_params_obj, fig_params)

            _draw_scalebar(ax, scalebar_params_obj, panel_idx=i)

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
