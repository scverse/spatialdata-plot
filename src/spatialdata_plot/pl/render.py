from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Union

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from anndata import AnnData
from geopandas import GeoDataFrame
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.colors import ColorConverter, ListedColormap, Normalize
from matplotlib.patches import Circle, Polygon
from pandas.api.types import is_categorical_dtype
from scanpy._settings import settings as sc_settings

from spatialdata_plot._logging import logger
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

_Normalize = Union[Normalize, Sequence[Normalize]]
to_hex = partial(colors.to_hex, keep_alpha=True)


@dataclass
class ShapesRenderParams:
    """Labels render parameters.."""

    cmap_params: CmapParams
    outline_params: OutlineParams
    elements: str | Sequence[str] | None = None
    color: str | None = None
    groups: str | Sequence[str] | None = None
    contour_px: int | None = None
    alt_var: str | None = None
    layer: str | None = None
    palette: ListedColormap | str | None = None
    outline_alpha: float = 1.0
    fill_alpha: float = 0.3
    size: float = 1.0
    transfunc: Callable[[float], float] | None = None


def _render_shapes(
    sdata: sd.SpatialData,
    render_params: ShapesRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
) -> None:
    elements = render_params.elements

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_table=sdata.table is not None,
    )
    if isinstance(elements, str):
        elements = [elements]

    if elements is None:
        elements = list(sdata_filt.shapes.keys())

    shapes = [sdata.shapes[e] for e in elements]
    n_shapes = sum([len(s) for s in shapes])

    if sdata.table is None:
        table = AnnData(None, obs=pd.DataFrame(index=pd.Index(np.arange(n_shapes), dtype=str)))
    else:
        table = sdata.table[sdata.table.obs[_get_region_key(sdata)].isin(elements)]

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

    # color_source_vector is None when the values aren't categorical
    if color_source_vector is None and render_params.transfunc is not None:
        color_vector = render_params.transfunc(color_vector)

    def _get_collection_shape(
        shapes: list[GeoDataFrame],
        c: Any,
        s: float,
        norm: Any,
        fill_alpha: None | float = None,
        outline_alpha: None | float = None,
        **kwargs: Any,
    ) -> PatchCollection:
        patches = []
        for shape in shapes:
            # We assume that all elements in one collection are of the same type
            if shape["geometry"].iloc[0].geom_type == "Polygon":
                patches += [Polygon(p.exterior.coords, closed=True) for p in shape["geometry"]]
            elif shape["geometry"].iloc[0].geom_type == "Point":
                patches += [
                    Circle((circ.x, circ.y), radius=r * s) for circ, r in zip(shape["geometry"], shape["radius"])
                ]

        cmap = kwargs["cmap"]

        try:
            # fails when numeric
            fill_c = ColorConverter().to_rgba_array(c)
        except ValueError:
            if norm is None:
                c = cmap(c)
            else:
                norm = colors.Normalize(vmin=min(c), vmax=max(c))
                c = cmap(norm(c))

        fill_c = ColorConverter().to_rgba_array(c)
        fill_c[..., -1] = render_params.fill_alpha

        if render_params.outline_params.outline:
            outline_c = ColorConverter().to_rgba_array(render_params.outline_params.outline_color)
            outline_c[..., -1] = render_params.outline_alpha
        else:
            outline_c = None

        return PatchCollection(
            patches,
            snap=False,
            # zorder=4,
            lw=1.5,
            facecolor=fill_c,
            edgecolor=outline_c,
            **kwargs,
        )

    norm = copy(render_params.cmap_params.norm)

    if len(color_vector) == 0:
        color_vector = [render_params.cmap_params.na_color]

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

    palette = ListedColormap(set(color_vector)) if render_params.palette is None else render_params.palette

    _ = _decorate_axs(
        ax=ax,
        cax=cax,
        fig_params=fig_params,
        adata=table,
        value_to_plot=render_params.color,
        color_source_vector=color_source_vector,
        palette=palette,
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
    elements: str | Sequence[str] | None = None
    color: str | None = None
    groups: str | Sequence[str] | None = None
    palette: ListedColormap | str | None = None
    alpha: float = 1.0
    size: float = 1.0
    transfunc: Callable[[float], float] | None = None


def _render_points(
    sdata: sd.SpatialData,
    render_params: PointsRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
) -> None:
    elements = render_params.elements

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_table=sdata.table is not None,
    )
    if isinstance(elements, str):
        elements = [elements]

    if elements is None:
        elements = list(sdata_filt.points.keys())

    points = [sdata.points[e] for e in elements]

    coords = ["x", "y"]
    if render_params.color is not None:
        color = [render_params.color] if isinstance(render_params.color, str) else render_params.color
        coords.extend(color)

    point_df = pd.concat([point[coords].compute() for point in points], axis=0)

    # we construct an anndata to hack the plotting functions
    adata = AnnData(
        X=point_df[["x", "y"]].values, obs=point_df[coords].reset_index(), dtype=point_df[["x", "y"]].values.dtype
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

    # color_source_vector is None when the values aren't categorical
    if color_source_vector is None and render_params.transfunc is not None:
        color_vector = render_params.transfunc(color_vector)

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

    cmap_params: list[CmapParams] | CmapParams
    elements: str | Sequence[str] | None = None
    channel: list[str] | list[int] | int | str | None = None
    palette: ListedColormap | str | None = None
    alpha: float = 1.0
    quantiles_for_norm: tuple[float | None, float | None] = (3.0, 99.8)  # defaults from CSBDeep


def _render_images(
    sdata: sd.SpatialData,
    render_params: ImageRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
) -> None:
    elements = render_params.elements

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_table=sdata.table is not None,
    )

    if isinstance(elements, str):
        elements = [elements]

    if elements is None:
        elements = list(sdata_filt.images.keys())

    images = [sdata.images[e] for e in elements]

    for img in images:
        if render_params.channel is None:
            channels = img.coords["c"].values
        else:
            channels = (
                [render_params.channel] if isinstance(render_params.channel, (str, int)) else render_params.channel
            )

        n_channels = len(channels)

        # True if user gave n cmaps for n channels
        got_multiple_cmaps = isinstance(render_params.cmap_params, list)
        if got_multiple_cmaps:
            logger.warning(
                "You're blending multiple cmaps. "
                "If the plot doesn't look like you expect, it might be because your "
                "cmaps go from a given color to 'white', and not to 'transparent'. "
                "Therefore, the 'white' of higher layers will overlay the lower layers. "
                "Consider using 'palette' instead."
            )

        # not using got_multiple_cmaps here because of ruff :(
        if isinstance(render_params.cmap_params, list) and len(render_params.cmap_params) != n_channels:
            raise ValueError("If 'cmap' is provided, its length must match the number of channels.")

        # 1) Image has only 1 channel
        if n_channels == 1 and not isinstance(render_params.cmap_params, list):
            layer = img.sel(c=channels).squeeze()

            if render_params.quantiles_for_norm != (None, None):
                layer = _normalize(
                    layer, pmin=render_params.quantiles_for_norm[0], pmax=render_params.quantiles_for_norm[1], clip=True
                )

            if render_params.cmap_params.norm is not None:  # type: ignore[attr-defined]
                layer = render_params.cmap_params.norm(layer)  # type: ignore[attr-defined]

            if render_params.palette is None:
                cmap = render_params.cmap_params.cmap  # type: ignore[attr-defined]
            else:
                cmap = _get_linear_colormap([render_params.palette], "k")[0]

            ax.imshow(
                layer,  # get rid of the channel dimension
                cmap=cmap,
                alpha=render_params.alpha,
            )

        # 2) Image has any number of channels but 1
        else:
            layers = {}
            for i, c in enumerate(channels):
                layers[c] = img.sel(c=c).copy(deep=True).squeeze()

                if render_params.quantiles_for_norm != (None, None):
                    layers[c] = _normalize(
                        layers[c],
                        pmin=render_params.quantiles_for_norm[0],
                        pmax=render_params.quantiles_for_norm[1],
                        clip=True,
                    )

                if not isinstance(render_params.cmap_params, list):
                    if render_params.cmap_params.norm is not None:
                        layers[c] = render_params.cmap_params.norm(layers[c])
                else:
                    if render_params.cmap_params[i].norm is not None:
                        layers[c] = render_params.cmap_params[i].norm(layers[c])

            # 2A) Image has 3 channels, no palette/cmap info -> use RGB
            if n_channels == 3 and render_params.palette is None and not got_multiple_cmaps:
                ax.imshow(np.stack([layers[c] for c in channels], axis=-1), alpha=render_params.alpha)

            # 2B) Image has n channels, no palette/cmap info -> sample n categorical colors
            elif render_params.palette is None and not got_multiple_cmaps:
                # overwrite if n_channels == 2 for intuitive result
                if n_channels == 2:
                    seed_colors = ["#ff0000ff", "#00ff00ff"]
                else:
                    seed_colors = _get_colors_for_categorical_obs(list(range(n_channels)))

                channel_cmaps = [_get_linear_colormap([c], "k")[0] for c in seed_colors]

                # Apply cmaps to each channel and add up
                colored = np.stack([channel_cmaps[i](layers[c]) for i, c in enumerate(channels)], 0).sum(0)

                # Remove alpha channel so we can overwrite it from render_params.alpha
                colored = colored[:, :, :3]

                ax.imshow(
                    colored,
                    alpha=render_params.alpha,
                )

            # 2C) Image has n channels and palette info
            elif render_params.palette is not None and not got_multiple_cmaps:
                if len(render_params.palette) != n_channels:
                    raise ValueError("If 'palette' is provided, its length must match the number of channels.")

                channel_cmaps = [_get_linear_colormap([c], "k")[0] for c in render_params.palette]

                # Apply cmaps to each channel and add up
                colored = np.stack([channel_cmaps[i](layers[c]) for i, c in enumerate(channels)], 0).sum(0)

                # Remove alpha channel so we can overwrite it from render_params.alpha
                colored = colored[:, :, :3]

                ax.imshow(
                    colored,
                    alpha=render_params.alpha,
                )

            elif render_params.palette is None and got_multiple_cmaps:
                channel_cmaps = [cp.cmap for cp in render_params.cmap_params]  # type: ignore[union-attr]

                # Apply cmaps to each channel, add up and normalize to [0, 1]
                colored = np.stack([channel_cmaps[i](layers[c]) for i, c in enumerate(channels)], 0).sum(0) / n_channels

                # Remove alpha channel so we can overwrite it from render_params.alpha
                colored = colored[:, :, :3]

                ax.imshow(
                    colored,
                    alpha=render_params.alpha,
                )

            elif render_params.palette is not None and got_multiple_cmaps:
                raise ValueError("If 'palette' is provided, 'cmap' must be None.")


@dataclass
class LabelsRenderParams:
    """Labels render parameters.."""

    cmap_params: CmapParams
    elements: str | Sequence[str] | None = None
    color: str | None = None
    groups: str | Sequence[str] | None = None
    contour_px: int | None = None
    outline: bool = False
    alt_var: str | None = None
    layer: str | None = None
    palette: ListedColormap | str | None = None
    outline_alpha: float = 1.0
    fill_alpha: float = 0.4
    transfunc: Callable[[float], float] | None = None


def _render_labels(
    sdata: sd.SpatialData,
    render_params: LabelsRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
) -> None:
    elements = render_params.elements

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_table=sdata.table is not None,
    )
    if isinstance(elements, str):
        elements = [elements]

    if elements is None:
        elements = list(sdata_filt.labels.keys())

    labels = [sdata.labels[e].values for e in elements]

    for label, labels_key in zip(labels, elements):
        if sdata.table is None:
            instance_id = np.unique(label)
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
                seg=label,
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
                seg=label,
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
            label = _map_color_seg(
                seg=label,
                cell_id=instance_id,
                color_vector=color_vector,
                color_source_vector=color_source_vector,
                cmap_params=render_params.cmap_params,
                seg_erosionpx=render_params.contour_px,
                seg_boundaries=render_params.outline,
                na_color=render_params.cmap_params.na_color,
            )

            _cax = ax.imshow(
                label,
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
