from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

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
from spatialdata import transform
from spatialdata._logging import logger as logg
from spatialdata.transformations import get_transformation

from spatialdata_plot._accessor import register_spatial_data_accessor
from spatialdata_plot.pl.render import (
    ImageRenderParams,
    LabelsRenderParams,
    PointsRenderParams,
    ShapesRenderParams,
    _render_images,
    _render_labels,
    _render_points,
    _render_shapes,
)
from spatialdata_plot.pl.utils import (
    LegendParams,
    Palette_t,
    _FontSize,
    _FontWeight,
    _get_cs_contents,
    _get_extent,
    _maybe_set_colors,
    _multiscale_to_image,
    _prepare_cmap_norm,
    _prepare_params_plot,
    _set_outline,
    _translate_image,
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
        images: Union[None, dict[str, Union[SpatialImage, MultiscaleSpatialImage]]] = None,
        labels: Union[None, dict[str, Union[SpatialImage, MultiscaleSpatialImage]]] = None,
        points: Union[None, dict[str, DaskDataFrame]] = None,
        shapes: Union[None, dict[str, GeoDataFrame]] = None,
        table: Union[None, AnnData] = None,
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
        element: str | None = None,
        color: str | None = None,
        groups: str | Sequence[str] | None = None,
        size: float = 1.0,
        outline: bool = False,
        outline_width: tuple[float, float] = (0.3, 0.05),
        outline_color: tuple[str, str] = ("black", "white"),
        alt_var: str | None = None,
        layer: str | None = None,
        palette: Palette_t = None,
        cmap: Colormap | str | None = None,
        norm: Optional[Normalize] = None,
        na_color: str | tuple[float, ...] | None = "lightgrey",
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render shapes elements in SpatialData.

        Parameters
        ----------
        element
            The name of the shapes element to render. If `None`, the first
            shapes element in the `SpatialData` object will be used.
        color
            Key for annotations in :attr:`anndata.AnnData.obs` or variables/genes.
        groups
            For discrete annotation in ``color``, select which values
            to plot (other values are set to NAs).
        size
            Value to scale circles, if present.
        outline
            If `True`, a thin border around points/shapes is plotted.
        outline_width
            Width of the border.
        outline_color
            Color of the border.
        alt_var
            Which column to use in :attr:`anndata.AnnData.var` to select alternative ``var_name``.
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
        outline_params = _set_outline(size, outline, outline_width, outline_color)
        sdata.plotting_tree[f"{n_steps+1}_render_shapes"] = ShapesRenderParams(
            element=element,
            color=color,
            groups=groups,
            outline_params=outline_params,
            alt_var=alt_var,
            layer=layer,
            cmap_params=cmap_params,
            palette=palette,
            alpha=alpha,
        )

        return sdata

    def render_points(
        self,
        element: str | None = None,
        color: str | None = None,
        groups: str | Sequence[str] | None = None,
        size: float = 1.0,
        palette: Palette_t = None,
        cmap: Colormap | str | None = None,
        norm: Optional[Normalize] = None,
        na_color: str | tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0),
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render points elements in SpatialData.

        Parameters
        ----------
        element
            The name of the points element to render. If `None`, the first
            shapes element in the `SpatialData` object will be used.
        color
            Key for annotations in :attr:`anndata.AnnData.obs` or variables/genes.
        groups
            For discrete annotation in ``color``, select which values
            to plot (other values are set to NAs).
        size
            Value to scale points.
        palette
            Palette for discrete annotations, see :class:`matplotlib.colors.Colormap`.
        cmap
            Colormap for continuous annotations, see :class:`matplotlib.colors.Colormap`.
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
        sdata.plotting_tree[f"{n_steps+1}_render_points"] = PointsRenderParams(
            element=element,
            color=color,
            groups=groups,
            cmap_params=cmap_params,
            palette=palette,
            alpha=alpha,
        )

        return sdata

    def render_images(
        self,
        element: str | None = None,
        channel: list[str] | list[int] | int | str | None = None,
        cmap: Colormap | str | None = None,
        norm: Optional[Normalize] = None,
        na_color: str | tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0),
        palette: Palette_t = None,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render image elements in SpatialData.

        Parameters
        ----------
        element
            The name of the image element to render. If `None`, the first
            shapes element in the `SpatialData` object will be used.
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
        sdata.plotting_tree[f"{n_steps+1}_render_images"] = ImageRenderParams(
            element=element,
            channel=channel,
            cmap_params=cmap_params,
            palette=palette,
            alpha=alpha,
        )

        return sdata

    def render_labels(
        self,
        element: str | None = None,
        color: str | None = None,
        groups: str | Sequence[str] | None = None,
        contour_px: int = 3,
        outline: bool = False,
        alt_var: str | None = None,
        layer: str | None = None,
        palette: Palette_t = None,
        cmap: Colormap | str | None = None,
        norm: Optional[Normalize] = None,
        na_color: str | tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0),
        outline_alpha: float = 1.0,
        fill_alpha: float = 0.3,
        **kwargs: Any,
    ) -> sd.SpatialData:
        """
        Render labels elements in SpatialData.

        Parameters
        ----------
        element
            The name of the labels element to render. If `None`, the first
            labels element in the `SpatialData` object will be used.
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
        alt_var
            Which column to use in :attr:`anndata.AnnData.var` to select alternative ``var_name``.
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
        sdata.plotting_tree[f"{n_steps+1}_render_labels"] = LabelsRenderParams(
            element=element,
            color=color,
            groups=groups,
            contour_px=contour_px,
            outline=outline,
            alt_var=alt_var,
            layer=layer,
            cmap_params=cmap_params,
            palette=palette,
            outline_alpha=outline_alpha,
            fill_alpha=fill_alpha,
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
        ax: Axes | Sequence[Axes] | None = None,
        return_ax: bool = False,
        save: Optional[Union[str, Path]] = None,
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
            raise TypeError("Please specify what to plot using the 'render_*' functions before calling 'imshow().")

        # Simplicstic solution: If the images are multiscale, just use the first
        sdata = _multiscale_to_image(sdata)

        img_transformations = {}
        # transform all elements
        for cmd, _ in render_cmds.items():
            if cmd == "render_images" or cmd == "render_channels":
                translations = {}

                for key in sdata.images:
                    img_transformations[key] = {}
                    all_transformations = get_transformation(sdata.images[key], get_all=True)

                    for cs, transformation in all_transformations.items():
                        img_transformations[key][cs] = transformation

                        translations[key] = []
                        if isinstance(transformation, sd.transformations.transformations.Translation):
                            sdata.images[key] = _translate_image(image=sdata.images[key], translation=transformation)

                        elif isinstance(transformation, sd.transformations.transformations.Sequence):
                            # we have more than one transformation, let's find the translation(s)
                            for t in list(transformation.transformations):
                                if isinstance(t, sd.transformations.transformations.Translation):
                                    sdata.images[key] = _translate_image(image=sdata.images[key], translation=t)

                                else:
                                    sdata.images[key] = transform(sdata.images[key], t)

            elif cmd == "render_shapes":
                for key in sdata.shapes:
                    shape_transformation = get_transformation(sdata.shapes[key], get_all=True)
                    shape_transformation = list(shape_transformation.values())[0]
                    sdata.shapes[key] = transform(sdata.shapes[key], shape_transformation)

            elif cmd == "render_labels":
                for key in sdata.labels:
                    label_transformation = get_transformation(sdata.labels[key], get_all=True)
                    label_transformation = list(label_transformation.values())[0]
                    sdata.labels[key] = transform(sdata.labels[key], label_transformation)

        extent = _get_extent(
            sdata=sdata,
            images="render_images" in render_cmds,
            labels="render_labels" in render_cmds,
            points="render_points" in render_cmds,
            shapes="render_shapes" in render_cmds,
            img_transformations=img_transformations if len(img_transformations) > 0 else None,
        )

        # handle coordinate system
        coordinate_systems = sdata.coordinate_systems if coordinate_systems is None else coordinate_systems
        if isinstance(coordinate_systems, str):
            coordinate_systems = [coordinate_systems]

        # Use extent to filter out coordinate system without the relevant elements
        valid_cs = []
        for cs in coordinate_systems:
            if cs in extent:
                valid_cs.append(cs)
            else:
                logg.info(f"Dropping coordinate system '{cs}' since it doesn't have relevant elements.")
        coordinate_systems = valid_cs

        # check that coordinate system and elements to be rendered match
        for cmd, params in render_cmds.items():
            if params.element is not None and len([params.element]) != len(coordinate_systems):
                raise ValueError(
                    f"Number of coordinate systems ({len(coordinate_systems)}) does not match number of elements "
                    f"({len(params.element)}) in command {cmd}."
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

        # go through tree
        cs_contents = _get_cs_contents(sdata)
        for i, cs in enumerate(coordinate_systems):
            ax = fig_params.ax if fig_params.axs is None else fig_params.axs[i]
            for cmd, params in render_cmds.items():
                if cmd == "render_images" and cs_contents.query(f"cs == '{cs}'")["has_images"][0]:
                    _render_images(
                        sdata=sdata,
                        render_params=params,
                        coordinate_system=cs,
                        ax=ax,
                        fig_params=fig_params,
                        scalebar_params=scalebar_params,
                        legend_params=legend_params,
                    )
                elif cmd == "render_shapes" and cs_contents.query(f"cs == '{cs}'")["has_shapes"][0]:
                    if sdata.table is not None and isinstance(params.color, str):
                        colors = sc.get.obs_df(sdata.table, params.color)
                        if is_categorical_dtype(colors):
                            _maybe_set_colors(
                                source=sdata.table,
                                target=sdata.table,
                                key=params.color,
                                palette=params.palette,
                            )
                    _render_shapes(
                        sdata=sdata,
                        render_params=params,
                        coordinate_system=cs,
                        ax=ax,
                        fig_params=fig_params,
                        scalebar_params=scalebar_params,
                        legend_params=legend_params,
                    )

                elif cmd == "render_points" and cs_contents.query(f"cs == '{cs}'")["has_points"][0]:
                    _render_points(
                        sdata=sdata,
                        render_params=params,
                        coordinate_system=cs,
                        ax=ax,
                        fig_params=fig_params,
                        scalebar_params=scalebar_params,
                        legend_params=legend_params,
                    )

                elif cmd == "render_labels" and cs_contents.query(f"cs == '{cs}'")["has_labels"][0]:
                    if (
                        sdata.table is not None
                        # and isinstance(params["instance_key"], str)
                        and isinstance(params.color, str)
                    ):
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

                ax.set_title(cs)
                if any(
                    [
                        cs_contents.query(f"cs == '{cs}'")["has_images"][0],
                        cs_contents.query(f"cs == '{cs}'")["has_labels"][0],
                        cs_contents.query(f"cs == '{cs}'")["has_points"][0],
                        cs_contents.query(f"cs == '{cs}'")["has_shapes"][0],
                    ]
                ):
                    ax.set_xlim(extent[cs][0], extent[cs][1])
                    ax.set_ylim(extent[cs][3], extent[cs][2])  # (0, 0) is top-left

        if fig_params.fig is not None and save is not None:
            save_fig(fig_params.fig, path=save)

        return (fig_params.ax if fig_params.axs is None else fig_params.axs) if return_ax else None  # shuts up ruff
