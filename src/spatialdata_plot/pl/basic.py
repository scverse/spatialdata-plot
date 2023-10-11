from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import spatialdata as sd
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from pandas.api.types import is_categorical_dtype
from spatial_image import SpatialImage
from spatialdata._core.data_extent import get_extent
from spatialdata.transformations.operations import get_transformation

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
    _maybe_set_colors,
    _mpl_ax_contains_elements,
    _prepare_cmap_norm,
    _prepare_params_plot,
    _set_outline,
    save_fig,
)
from spatialdata_plot.pp.utils import _verify_plotting_tree


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
        images: None | dict[str, SpatialImage | MultiscaleSpatialImage] = None,
        labels: None | dict[str, SpatialImage | MultiscaleSpatialImage] = None,
        points: None | dict[str, DaskDataFrame] = None,
        shapes: None | dict[str, GeoDataFrame] = None,
        table: None | AnnData = None,
    ) -> sd.SpatialData:
        """Copy the current `SpatialData` object, optionally modifying some of its attributes.

        Parameters
        ----------
        images :
            A dictionary containing image data to replace the images in the
            original `SpatialData` object, or `None` to keep the original
            images. Defaults to `None`.
        labels :
            A dictionary containing label data to replace the labels in the
            original `SpatialData` object, or `None` to keep the original
            labels. Defaults to `None`.
        points :
            A dictionary containing point data to replace the points in the
            original `SpatialData` object, or `None` to keep the original
            points. Defaults to `None`.
        shapes :
            A dictionary containing shape data to replace the shapes in the
            original `SpatialData` object, or `None` to keep the original
            shapes. Defaults to `None`.
        table :
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
        elements: str | list[str] | None = None,
        color: str | None = None,
        groups: str | Sequence[str] | None = None,
        scale: float = 1.0,
        outline: bool = False,
        outline_width: float = 1.5,
        outline_color: str | list[float] = "#000000ff",
        layer: str | None = None,
        palette: str | list[str] | None = None,
        cmap: Colormap | str | None = None,
        norm: bool | Normalize = False,
        na_color: str | tuple[float, ...] | None = "lightgrey",
        outline_alpha: float = 1.0,
        fill_alpha: float = 1.0,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render shapes elements in SpatialData.

        Parameters
        ----------
        elements
            The name of the shapes element(s) to render. If `None`, all
            shapes element in the `SpatialData` object will be used.
        color
            Key for annotations in :attr:`anndata.AnnData.obs` or variables/genes.
        groups
            For discrete annotation in ``color``, select which values
            to plot (other values are set to NAs).
        scale
            Value to scale circles, if present.
        outline
            If `True`, a thin border around points/shapes is plotted.
        outline_width
            Width of the border.
        outline_color
            Color of the border.
        layer
            Key in :attr:`anndata.AnnData.layers` or `None` for :attr:`anndata.AnnData.X`.
        palette
            Palette for discrete annotations. List of valid color names that should be used
            for the categories (all or as specified by `groups`). For a single category,
            a valid color name can be given as string.
        cmap
            Colormap for continuous annotations, see :class:`matplotlib.colors.Colormap`.
            If no palette is given and `color` refers to a categorical, the colors are
            sampled from this colormap.
        norm
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
        na_color
            Color to be used for NAs values, if present.
        alpha
            Alpha value for the shapes.
        kwargs
            Additional arguments to be passed to cmap and norm.

        Notes
        -----
            Empty geometries will be removed at the time of plotting.
            An ``outline_width`` of 0.0 leads to no border being plotted.

        Returns
        -------
        None
        """
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
        elements: str | list[str] | None = None,
        color: str | None = None,
        groups: str | Sequence[str] | None = None,
        size: float = 1.0,
        palette: str | list[str] | None = None,
        cmap: Colormap | str | None = None,
        norm: None | Normalize = None,
        na_color: str | tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0),
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render points elements in SpatialData.

        Parameters
        ----------
        elements
            The name of the points element(s) to render. If `None`, all
            shapes element in the `SpatialData` object will be used.
        color
            Key for annotations in :attr:`anndata.AnnData.obs` or variables/genes.
        groups
            For discrete annotation in ``color``, select which values
            to plot (other values are set to NAs).
        size
            Value to scale points.
        palette
            Palette for discrete annotations. List of valid color names that should be used
            for the categories (all or as specified by `groups`). For a single category,
            a valid color name can be given as string.
        cmap
            Colormap for continuous annotations, see :class:`matplotlib.colors.Colormap`.
            If no palette is given and `color` refers to a categorical, the colors are
            sampled from this colormap.
        norm
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
        na_color
            Color to be used for NAs values, if present.
        alpha
            Alpha value for the shapes.
        kwargs
            Additional arguments to be passed to cmap and norm.

        Returns
        -------
        None
        """
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
        sdata.plotting_tree[f"{n_steps+1}_render_points"] = PointsRenderParams(
            elements=elements,
            color=color,
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
        elements: str | list[str] | None = None,
        channel: list[str] | list[int] | int | str | None = None,
        cmap: list[Colormap] | list[str] | Colormap | str | None = None,
        norm: None | Normalize = None,
        na_color: str | tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0),
        palette: str | list[str] | None = None,
        alpha: float = 1.0,
        quantiles_for_norm: tuple[float | None, float | None] = (None, None),
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render image elements in SpatialData.

        Parameters
        ----------
        elements
            The name of the image element(s) to render. If `None`, all
            shapes elements in the `SpatialData` object will be used.
        channel
            To select which channel to plot (all by default).
        cmap
            Colormap for continuous annotations, see :class:`matplotlib.colors.Colormap`.
        norm
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
        na_color
            Color to be used for NAs values, if present.
        alpha
            Alpha value for the shapes.
        quantiles_for_norm
            Tuple of (pmin, pmax) which will be used for quantile normalization.
        kwargs
            Additional arguments to be passed to cmap and norm.

        Returns
        -------
        None
        """
        sdata = self._copy()
        sdata = _verify_plotting_tree(sdata)
        n_steps = len(sdata.plotting_tree.keys())

        if channel is None and cmap is None:
            cmap = "brg"

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
        )

        return sdata

    def render_labels(
        self,
        elements: str | list[str] | None = None,
        color: str | None = None,
        groups: str | Sequence[str] | None = None,
        contour_px: int = 3,
        outline: bool = False,
        layer: str | None = None,
        palette: str | list[str] | None = None,
        cmap: Colormap | str | None = None,
        norm: None | Normalize = None,
        na_color: str | tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0),
        outline_alpha: float = 1.0,
        fill_alpha: float = 0.3,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render labels elements in SpatialData.

        Parameters
        ----------
        elements
            The name of the labels element(s) to render. If `None`, all
            labels elements in the `SpatialData` object will be used.
        color
            Key for annotations in :attr:`anndata.AnnData.obs` or variables/genes.
        groups
            For discrete annotation in ``color``, select which values
            to plot (other values are set to NAs).
        contour_px
            Draw contour of specified width for each segment. If `None`, fills
            entire segment, see :func:`skimage.morphology.erosion`.
        outline
            Whether to plot boundaries around segmentation masks.
        layer
            Key in :attr:`anndata.AnnData.layers` or `None` for :attr:`anndata.AnnData.X`.
        palette
            Palette for discrete annotations, see :class:`matplotlib.colors.Colormap`.
        cmap
            Colormap for continuous annotations, see :class:`matplotlib.colors.Colormap`.
        norm
            Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
        na_color
            Color to be used for NAs values, if present.
        alpha
            Alpha value for the labels.
        kwargs
            Additional arguments to be passed to cmap and norm.

        Returns
        -------
        None
        """
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
        )

        return sdata

    def show(
        self,
        coordinate_systems: str | Sequence[str] | None = None,
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
        title: None | str | Sequence[str] = None,
        share_extent: bool = True,
        pad_extent: int = 0,
        ax: Axes | Sequence[Axes] | None = None,
        return_ax: bool = False,
        save: None | str | Path = None,
    ) -> sd.SpatialData:
        """
        Plot the images in the SpatialData object.

        Parameters
        ----------
        ax :
            Matplotlib axes object to plot on. If None, a new figure is created.
            Works only if there is one image in the SpatialData object.
        ncols :
            Number of columns in the figure. Default is 4.
        width :
            Width of each subplot. Default is 4.
        height :
            Height of each subplot. Default is 3.

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
        render_cmds = OrderedDict()
        for cmd, params in plotting_tree.items():
            # strip prefix from cmd and verify it's valid
            cmd = "_".join(cmd.split("_")[1:])

            if cmd not in valid_commands:
                raise ValueError(f"Command {cmd} is not valid.")

            if "render" in cmd:
                # verify that rendering commands have been called before
                render_cmds[cmd] = params

        if len(render_cmds.keys()) == 0:
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

            for cmd, params in render_cmds.items():
                if cmd == "render_images" and has_images:
                    _render_images(
                        sdata=sdata,
                        render_params=params,
                        coordinate_system=cs,
                        ax=ax,
                        fig_params=fig_params,
                        scalebar_params=scalebar_params,
                        legend_params=legend_params,
                    )
                    wants_images = True
                    wanted_images = params.elements if params.elements is not None else list(sdata.images.keys())
                    wanted_elements.extend(
                        [
                            image
                            for image in wanted_images
                            if cs in set(get_transformation(sdata.images[image], get_all=True).keys())
                        ]
                    )

                elif cmd == "render_shapes" and has_shapes:
                    _render_shapes(
                        sdata=sdata,
                        render_params=params,
                        coordinate_system=cs,
                        ax=ax,
                        fig_params=fig_params,
                        scalebar_params=scalebar_params,
                        legend_params=legend_params,
                    )
                    wants_shapes = True
                    wanted_shapes = params.elements if params.elements is not None else list(sdata.shapes.keys())
                    wanted_elements.extend(
                        [
                            shape
                            for shape in wanted_shapes
                            if cs in set(get_transformation(sdata.shapes[shape], get_all=True).keys())
                        ]
                    )

                elif cmd == "render_points" and has_points:
                    _render_points(
                        sdata=sdata,
                        render_params=params,
                        coordinate_system=cs,
                        ax=ax,
                        fig_params=fig_params,
                        scalebar_params=scalebar_params,
                        legend_params=legend_params,
                    )
                    wants_points = True
                    wanted_points = params.elements if params.elements is not None else list(sdata.points.keys())
                    wanted_elements.extend(
                        [
                            point
                            for point in wanted_points
                            if cs in set(get_transformation(sdata.points[point], get_all=True).keys())
                        ]
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
                    _render_labels(
                        sdata=sdata,
                        render_params=params,
                        coordinate_system=cs,
                        ax=ax,
                        fig_params=fig_params,
                        scalebar_params=scalebar_params,
                        legend_params=legend_params,
                    )
                    wants_labels = True
                    wanted_labels = params.elements if params.elements is not None else list(sdata.labels.keys())
                    wanted_elements.extend(
                        [
                            label
                            for label in wanted_labels
                            if cs in set(get_transformation(sdata.labels[label], get_all=True).keys())
                        ]
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
