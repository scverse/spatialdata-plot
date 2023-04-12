from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

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
from spatialdata import transform
from spatialdata.models import Image2DModel
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
    _get_extent,
    _maybe_set_colors,
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
        na_color: str | tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0),
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
        channel: str | None = None,
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
        contour_px: int | None = None,
        outline: bool = False,
        alt_var: str | None = None,
        layer: str | None = None,
        palette: Palette_t = None,
        cmap: Colormap | str | None = None,
        norm: Optional[Normalize] = None,
        na_color: str | tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0),
        alpha: float = 1.0,
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
            alpha=alpha,
        )

        return sdata

    def show(
        self,
        coordinate_system: str | Sequence[str] | None = None,
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

        # handle coordinate system
        coordinate_system = sdata.coordinate_systems if coordinate_system is None else coordinate_system
        if isinstance(coordinate_system, str):
            coordinate_system = [coordinate_system]

        # Evaluate execution tree for plotting
        valid_commands = [
            "get_elements",
            "get_bb",
            "render_images",
            "render_shapes",
            "render_labels",
            "render_points",
            "render_channels",
        ]

        # prepare rendering params
        render_cmds = OrderedDict()
        for cmd, params in plotting_tree.items():
            # strip prefix from cmd and verify it's valid
            cmd = "_".join(cmd.split("_")[1:])

            if cmd not in valid_commands:
                raise ValueError(f"Command {cmd} is not valid.")

            elif "render" in cmd:
                # verify that rendering commands have been called before
                render_cmds[cmd] = params

        if len(render_cmds.keys()) == 0:
            raise TypeError("Please specify what to plot using the 'render_*' functions before calling 'imshow().")

        # check that coordinate system and elements to be rendered match
        for cmd, params in render_cmds.items():
            if params.element is not None and len(params.element) != len(coordinate_system):
                raise ValueError(
                    f"Number of coordinate systems ({len(coordinate_system)}) does not match number of elements "
                    f"({len(params.element)}) in command {cmd}."
                )

        # set up canvas
        fig_params, scalebar_params = _prepare_params_plot(
            num_panels=len(coordinate_system),
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

        # transform all elements
        for cmd, _ in render_cmds.items():
            if cmd == "render_images" or cmd == "render_channels":
                for key in sdata.images.keys():
                    img_transformation = get_transformation(sdata.images[key], get_all=True)
                    img_transformation = list(img_transformation.values())[0]

                    if isinstance(img_transformation, sd.transformations.transformations.Translation):
                        shifts: dict[str, int] = {}
                        for idx, axis in enumerate(img_transformation.axes):
                            shifts[axis] = int(img_transformation.translation[idx])

                        img = sdata.images[key].values.copy()
                        shifted_channels = []

                        # split channels, shift axes individually, them recombine
                        if len(sdata.images[key].shape) == 3:
                            for c in range(sdata.images[key].shape[0]):
                                channel = img[c, :, :]

                                # iterates over [x, y]
                                for axis, shift in shifts.items():
                                    pad_x, pad_y = (0, 0), (0, 0)
                                    if axis == "x" and shift > 0:
                                        pad_x = (abs(shift), 0)
                                    elif axis == "x" and shift < 0:
                                        pad_x = (0, abs(shift))

                                    if axis == "y" and shift > 0:
                                        pad_y = (abs(shift), 0)
                                    elif axis == "y" and shift < 0:
                                        pad_y = (0, abs(shift))

                                    channel = np.pad(channel, (pad_y, pad_x), mode="constant")

                                shifted_channels.append(channel)

                        sdata.images[key] = Image2DModel.parse(np.array(shifted_channels), dims=["c", "y", "x"])

                    else:
                        sdata.images[key] = transform(sdata.images[key], img_transformation)

            elif cmd == "render_shapes":
                for key in sdata.shapes.keys():
                    shape_transformation = get_transformation(sdata.shapes[key], get_all=True)
                    shape_transformation = list(shape_transformation.values())[0]
                    sdata.shapes[key] = transform(sdata.shapes[key], shape_transformation)

            elif cmd == "render_labels":
                for key in sdata.labels.keys():
                    label_transformation = get_transformation(sdata.labels[key], get_all=True)
                    label_transformation = list(label_transformation.values())[0]
                    sdata.labels[key] = transform(sdata.labels[key], label_transformation)

        # Get extent of data to be plotted
        want_images_extent: bool = False
        want_labels_extent: bool = False
        want_points_extent: bool = False
        want_shapes_extent: bool = False

        for cmd, _ in render_cmds.items():
            if cmd == "render_images":
                want_images_extent = True
            elif cmd == "render_labels":
                want_labels_extent = True
            elif cmd == "render_points":
                want_points_extent = True
            elif cmd == "render_shapes":
                want_shapes_extent = True

        extent = _get_extent(
            sdata=sdata,
            coordinate_systems=coordinate_system,
            images=want_images_extent,
            labels=want_labels_extent,
            points=want_points_extent,
            shapes=want_shapes_extent,
        )

        print(extent)

        # get biggest image after transformations to set ax size
        # x_dims = []
        # y_dims = []

        # for cmd, _ in render_cmds.items():
        #     if cmd == "render_images" or cmd == "render_channels":
        #         y_dims += [(0, x.shape[1]) for x in sdata.images.values()]
        #         x_dims += [(0, x.shape[2]) for x in sdata.images.values()]

        #     elif cmd == "render_shapes":
        #         for key in sdata.shapes.keys():
        #             points = []
        #             polygons = []
        #             # TODO: improve getting extent of polygons
        #             for _, row in sdata.shapes[key].iterrows():
        #                 if row["geometry"].geom_type == "Point":
        #                     points.append(row)
        #                 elif row["geometry"].geom_type == "Polygon":
        #                     polygons.append(row)
        #                 else:
        #                     raise NotImplementedError(
        #                         "Only shapes of type 'Point' and 'Polygon' are supported right now."
        #                     )

        #             if len(points) > 0:
        #                 points_df = gpd.GeoDataFrame(data=points)
        #                 x_dims += [(min(points_df.geometry.x), max(points_df.geometry.x))]
        #                 y_dims += [(min(points_df.geometry.y), max(points_df.geometry.y))]

        #             if len(polygons) > 0:
        #                 for p in polygons:
        #                     minx, miny, maxx, maxy = p.geometry.bounds
        #                     x_dims += [(minx, maxx)]
        #                     y_dims += [(miny, maxy)]

        #     elif cmd == "render_labels":
        #         y_dims += [(0, x.shape[0]) for x in sdata.labels.values()]
        #         x_dims += [(0, x.shape[1]) for x in sdata.labels.values()]

        # extent = {"x": [min_x[0], max_x[1]], "y": [max_y[1], min_y[0]]}

        # go through tree
        for i, cs in enumerate(coordinate_system):
            ax = fig_params.ax if fig_params.axs is None else fig_params.axs[i]
            for cmd, params in render_cmds.items():
                if cmd == "render_images":
                    _render_images(
                        sdata=sdata,
                        render_params=params,
                        coordinate_system=cs,
                        ax=ax,
                        fig_params=fig_params,
                        scalebar_params=scalebar_params,
                        legend_params=legend_params,
                    )
                elif cmd == "render_shapes":
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

                elif cmd == "render_points":
                    _render_points(
                        sdata=sdata,
                        render_params=params,
                        coordinate_system=cs,
                        ax=ax,
                        fig_params=fig_params,
                        scalebar_params=scalebar_params,
                        legend_params=legend_params,
                    )

                elif cmd == "render_labels":
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

        if fig_params.fig is not None and save is not None:
            save_fig(fig_params.fig, path=save)

        if return_ax:
            return fig_params.ax if fig_params.axs is None else fig_params.axs
