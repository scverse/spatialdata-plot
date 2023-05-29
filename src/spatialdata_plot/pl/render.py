from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Union

import dask.dataframe as dd
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import xarray as xr
from anndata import AnnData
from geopandas import GeoDataFrame
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.colors import ColorConverter, ListedColormap, Normalize
from matplotlib.patches import Circle, Polygon
from pandas.api.types import is_categorical_dtype
from scanpy._settings import settings as sc_settings

from spatialdata_plot.pl.utils import (
    CmapParams,
    FigParams,
    LegendParams,
    OutlineParams,
    ScalebarParams,
    _decorate_axs,
    _get_colors_for_categorical_obs,
    _get_linear_colormap,
    _map_color_seg,
    _maybe_set_colors,
    _normalize,
    _set_color_source_vec,
)
from spatialdata_plot.pp.utils import _get_instance_key, _get_region_key

Palette_t = Optional[Union[str, ListedColormap]]
_Normalize = Union[Normalize, Sequence[Normalize]]
to_hex = partial(colors.to_hex, keep_alpha=True)


@dataclass
class ShapesRenderParams:
    """Labels render parameters.."""

    cmap_params: CmapParams
    outline_params: OutlineParams
    element: str | None = None
    color: str | None = None
    groups: str | Sequence[str] | None = None
    contour_px: int | None = None
    alt_var: str | None = None
    layer: str | None = None
    palette: Palette_t = None
    outline_alpha: float = 1.0
    fill_alpha: float = 0.3
    size: float = 1.0


def _render_shapes(
    sdata: sd.SpatialData,
    render_params: ShapesRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
) -> None:
    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_table=sdata.table is not None,
    )
    if render_params.element is not None:
        shapes = sdata_filt.shapes[render_params.element]
        shapes_key = render_params.element
    else:
        shapes_key = list(sdata_filt.shapes.keys())[0]
        shapes = sdata_filt.shapes[shapes_key]

    if sdata.table is None:
        table = AnnData(None, obs=pd.DataFrame(index=np.arange(len(shapes))))
    else:
        table = sdata.table[sdata.table.obs[_get_region_key(sdata)].isin([shapes_key])]

    # refactor plz, squidpy leftovers
    render_params.outline_params.bg_color = (0.83, 0.83, 0.83, render_params.fill_alpha)

    # get color vector (categorical or continuous)
    color_source_vector, color_vector, _ = _set_color_source_vec(
        adata=table,
        value_to_plot=render_params.color,
        alt_var=render_params.alt_var,
        layer=render_params.layer,
        groups=render_params.groups,
        palette=render_params.palette,
        na_color=render_params.cmap_params.na_color,
        alpha=render_params.fill_alpha,
    )

    def _get_collection_shape(
        shapes: GeoDataFrame,
        c: Any,
        s: float,
        norm: Any,
        fill_alpha: Optional[float] = None,
        outline_alpha: Optional[float] = None,
        **kwargs: Any,
    ) -> PatchCollection:
        """Get collection of shapes."""
        if shapes["geometry"].iloc[0].geom_type == "Polygon":
            patches = [Polygon(p.exterior.coords, closed=True) for p in shapes["geometry"]]
        elif shapes["geometry"].iloc[0].geom_type == "Point":
            patches = [Circle((circ.x, circ.y), radius=r * s) for circ, r in zip(shapes["geometry"], shapes["radius"])]

        cmap = kwargs["cmap"]
        norm = colors.Normalize(vmin=min(c), vmax=max(c))

        try:
            # fails when numeric
            fill_c = ColorConverter().to_rgba_array(c)
        except ValueError:
            c = cmap(norm(c))
            
        fill_c = ColorConverter().to_rgba_array(c)
        fill_c[..., -1] = render_params.fill_alpha
        outline_c = ColorConverter().to_rgba_array(c)
        outline_c[..., -1] = render_params.outline_alpha

        return PatchCollection(
            patches,
            snap=False,
            # zorder=4,
            lw=1.5,
            facecolor=fill_c,
            edgecolor=outline_c, 
            **kwargs
        )

    norm = copy(render_params.cmap_params.norm)

    if len(color_vector) == 0:
        color_vector = [(0.83, 0.83, 0.83, 1.0)]  # grey

    _cax = _get_collection_shape(
        shapes=shapes,
        s=render_params.size,
        c=color_vector,
        rasterized=sc_settings._vector_friendly,
        cmap=render_params.cmap_params.cmap,
        norm=norm,
        fill_alpha=render_params.fill_alpha,
        outline_alpha=render_params.outline_alpha
        # **kwargs,
    )
    cax = ax.add_collection(_cax)

    _ = _decorate_axs(
        ax=ax,
        cax=cax,
        fig_params=fig_params,
        adata=table,
        value_to_plot=render_params.color,
        color_source_vector=color_source_vector,
        palette=render_params.palette,
        alpha=render_params.fill_alpha,
        na_color=render_params.cmap_params.na_color,
        legend_fontsize=legend_params.legend_fontsize,
        legend_fontweight=legend_params.legend_fontweight,
        legend_loc=legend_params.legend_loc,
        legend_fontoutline=legend_params.legend_fontoutline,
        na_in_legend=legend_params.na_in_legend,
        colorbar=legend_params.colorbar,
        scalebar_dx=scalebar_params.scalebar_dx,
        scalebar_units=scalebar_params.scalebar_units,
        # scalebar_kwargs=scalebar_params.scalebar_kwargs,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()


@dataclass
class PointsRenderParams:
    """Points render parameters.."""

    cmap_params: CmapParams
    element: str | None = None
    color: str | None = None
    groups: str | Sequence[str] | None = None
    palette: Palette_t = None
    alpha: float = 1.0
    size: float = 1.0


def _render_points(
    sdata: sd.SpatialData,
    render_params: PointsRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
) -> None:
    # make colors a list
    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_table=sdata.table is not None,
    )
    if render_params.element is not None:
        points = sdata_filt.points[render_params.element]
    else:
        points_key = list(sdata_filt.points.keys())[0]
        points = sdata_filt.points[points_key]

    coords = ["x", "y"]
    if render_params.color is not None:
        color = [render_params.color] if isinstance(render_params.color, str) else render_params.color
        coords.extend(color)

    # get points
    if isinstance(points, dd.DataFrame):
        points = points[coords].compute()

    # we construct an anndata to hack the plotting functions
    adata = AnnData(
        X=points[["x", "y"]].values, obs=points[coords].reset_index(), dtype=points[["x", "y"]].values.dtype
    )
    if render_params.color is not None:
        cols = sc.get.obs_df(adata, render_params.color)
        # maybe set color based on type
        if is_categorical_dtype(cols):
            _maybe_set_colors(
                source=adata,
                target=adata,
                key=render_params.color,
                palette=render_params.palette,
            )
    color_source_vector, color_vector, _ = _set_color_source_vec(
        adata=adata,
        value_to_plot=render_params.color,
        groups=render_params.groups,
        palette=render_params.palette,
        na_color=render_params.cmap_params.na_color,
        alpha=render_params.alpha,
    )

    norm = copy(render_params.cmap_params.norm)
    _cax = ax.scatter(
        adata[:, 0].X.flatten(),
        adata[:, 1].X.flatten(),
        s=render_params.size,
        c=color_vector,
        rasterized=sc_settings._vector_friendly,
        cmap=render_params.cmap_params.cmap,
        norm=norm,
        alpha=render_params.alpha,
        # **kwargs,
    )
    cax = ax.add_collection(_cax)
    _ = _decorate_axs(
        ax=ax,
        cax=cax,
        fig_params=fig_params,
        adata=adata,
        value_to_plot=render_params.color,
        color_source_vector=color_source_vector,
        palette=render_params.palette,
        alpha=render_params.alpha,
        na_color=render_params.cmap_params.na_color,
        legend_fontsize=legend_params.legend_fontsize,
        legend_fontweight=legend_params.legend_fontweight,
        legend_loc=legend_params.legend_loc,
        legend_fontoutline=legend_params.legend_fontoutline,
        na_in_legend=legend_params.na_in_legend,
        colorbar=legend_params.colorbar,
        scalebar_dx=scalebar_params.scalebar_dx,
        scalebar_units=scalebar_params.scalebar_units,
        # scalebar_kwargs=scalebar_params.scalebar_kwargs,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()


@dataclass
class ImageRenderParams:
    """Labels render parameters.."""

    cmap_params: CmapParams
    element: str | None = None
    channel: list[str] | list[int] | int | str | None = None
    palette: Palette_t = None
    alpha: float = 1.0


def _render_images(
    sdata: sd.SpatialData,
    render_params: ImageRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
) -> None:
    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_table=sdata.table is not None,
    )
    if render_params.element is not None:
        img = sdata_filt.images[render_params.element]
    else:
        img_key = list(sdata_filt.images.keys())[0]
        img = sdata_filt.images[img_key]

    if (len(img.c) > 3 or len(img.c) == 2) and render_params.channel is None:
        raise NotImplementedError("Only 1 or 3 channels are supported at the moment.")
    if render_params.channel is None and len(img.c) == 1:
        render_params.channel = 0
    if render_params.channel is not None:
        channels = [render_params.channel] if isinstance(render_params.channel, (str, int)) else render_params.channel
        img = img.sel(c=channels)
        num_channels = img.sizes["c"]

        if render_params.palette is not None:
            if num_channels > len(render_params.palette):
                raise ValueError("If palette is provided, it must match the number of channels.")

            color = render_params.palette

        else:
            color = _get_colors_for_categorical_obs(img.coords["c"].values.tolist())

        cmaps = _get_linear_colormap([str(c) for c in color[:num_channels]], "k")
        img = _normalize(img, clip=True)
        colored = np.stack([cmaps[i](img.values[i]) for i in range(num_channels)], 0).sum(0)
        img = xr.DataArray(
            data=colored,
            coords=[
                img.coords["y"],
                img.coords["x"],
                ["R", "G", "B", "A"],
            ],
            dims=["y", "x", "c"],
        )

    img = img.transpose("y", "x", "c")  # for plotting

    ax.imshow(
        img.data,
        cmap=render_params.cmap_params.cmap,
        alpha=render_params.alpha,
    )


@dataclass
class LabelsRenderParams:
    """Labels render parameters.."""

    cmap_params: CmapParams
    element: str | None = None
    color: str | None = None
    groups: str | Sequence[str] | None = None
    contour_px: int | None = None
    outline: bool = False
    alt_var: str | None = None
    layer: str | None = None
    palette: Palette_t = None
    outline_alpha: float = 1.0
    fill_alpha: float = 0.4


def _render_labels(
    sdata: sd.SpatialData,
    render_params: LabelsRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
) -> None:
    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_table=sdata.table is not None,
    )
    if render_params.element is not None:
        labels = sdata_filt.labels[render_params.element].values
        labels_key = render_params.element
    else:
        labels_key = list(sdata_filt.labels.keys())[0]
        labels = sdata_filt.labels[labels_key].values

    if sdata.table is None:
        instance_id = np.unique(labels)
        table = AnnData(None, obs=pd.DataFrame(index=np.arange(len(instance_id))))
    else:
        instance_key = _get_instance_key(sdata)
        region_key = _get_region_key(sdata)

        table = sdata.table[sdata.table.obs[region_key].isin([labels_key])]

        # get isntance id based on subsetted table
        instance_id = table.obs[instance_key].values

    # get color vector (categorical or continuous)
    color_source_vector, color_vector, categorical = _set_color_source_vec(
        adata=table,
        value_to_plot=render_params.color,
        alt_var=render_params.alt_var,
        layer=render_params.layer,
        groups=render_params.groups,
        palette=render_params.palette,
        na_color=render_params.cmap_params.na_color,
        alpha=render_params.fill_alpha,
    )

    if (render_params.fill_alpha != render_params.outline_alpha) and render_params.contour_px is not None:
        # First get the labels infill and plot them
        labels_infill = _map_color_seg(
            seg=labels,
            cell_id=instance_id,
            color_vector=color_vector,
            color_source_vector=color_source_vector,
            cmap_params=render_params.cmap_params,
            seg_erosionpx=None,
            seg_boundaries=render_params.outline,
            na_color=render_params.cmap_params.na_color,
        )

        _cax = ax.imshow(
            labels_infill,
            rasterized=True,
            cmap=render_params.cmap_params.cmap if not categorical else None,
            norm=render_params.cmap_params.norm if not categorical else None,
            alpha=render_params.fill_alpha,
            origin="lower",
            # zorder=3,
        )
        cax = ax.add_image(_cax)

        # Then overlay the contour
        labels_contour = _map_color_seg(
            seg=labels,
            cell_id=instance_id,
            color_vector=color_vector,
            color_source_vector=color_source_vector,
            cmap_params=render_params.cmap_params,
            seg_erosionpx=render_params.contour_px,
            seg_boundaries=render_params.outline,
            na_color=render_params.cmap_params.na_color,
        )

        _cax = ax.imshow(
            labels_contour,
            rasterized=True,
            cmap=render_params.cmap_params.cmap if not categorical else None,
            norm=render_params.cmap_params.norm if not categorical else None,
            alpha=render_params.outline_alpha,
            origin="lower",
            # zorder=4,
        )
        cax = ax.add_image(_cax)

    else:
        # Default: no alpha, contour = infill
        labels = _map_color_seg(
            seg=labels,
            cell_id=instance_id,
            color_vector=color_vector,
            color_source_vector=color_source_vector,
            cmap_params=render_params.cmap_params,
            seg_erosionpx=render_params.contour_px,
            seg_boundaries=render_params.outline,
            na_color=render_params.cmap_params.na_color,
        )

        _cax = ax.imshow(
            labels,
            rasterized=True,
            cmap=render_params.cmap_params.cmap if not categorical else None,
            norm=render_params.cmap_params.norm if not categorical else None,
            alpha=render_params.fill_alpha,
            origin="lower",
            # zorder=4,
        )
        cax = ax.add_image(_cax)

    _ = _decorate_axs(
        ax=ax,
        cax=cax,
        fig_params=fig_params,
        adata=table,
        value_to_plot=render_params.color,
        color_source_vector=color_source_vector,
        palette=render_params.palette,
        alpha=render_params.fill_alpha,
        na_color=render_params.cmap_params.na_color,
        legend_fontsize=legend_params.legend_fontsize,
        legend_fontweight=legend_params.legend_fontweight,
        legend_loc=legend_params.legend_loc,
        legend_fontoutline=legend_params.legend_fontoutline,
        na_in_legend=legend_params.na_in_legend,
        colorbar=legend_params.colorbar,
        scalebar_dx=scalebar_params.scalebar_dx,
        scalebar_units=scalebar_params.scalebar_units,
        # scalebar_kwargs=scalebar_params.scalebar_kwargs,
    )
