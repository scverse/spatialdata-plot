from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import copy
from functools import partial
from typing import Callable, Optional, Union, Any
from matplotlib.colors import Colormap
import matplotlib
from spatialdata.models import TableModel
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd
from scanpy._settings import settings as sc_settings
import xarray as xr
from matplotlib import colors
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, Normalize, to_rgb, TwoSlopeNorm, ColorConverter
from matplotlib.collections import Collection, PatchCollection
from matplotlib.patches import Circle, Polygon
from pandas.api.types import is_categorical_dtype
from geopandas import GeoDataFrame
from sklearn.decomposition import PCA
from dataclasses import dataclass
from spatialdata_plot.pl.utils import (
    CmapParams,
    LegendParams,
    FigParams,
    ScalebarParams,
    OutlineParams,
    _map_color_seg,
    _normalize,
    _set_color_source_vec,
    _get_palette,
    _decorate_axs,
)
from spatialdata_plot.pp.utils import _get_linear_colormap, _get_region_key

Palette_t = Optional[Union[str, ListedColormap]]
_Normalize = Union[Normalize, Sequence[Normalize]]
_SeqStr = Union[str, Sequence[str]]
to_hex = partial(colors.to_hex, keep_alpha=True)


@dataclass
class ShapesRenderParams:
    """Labels render parameters.."""

    cmap_params: CmapParams
    outline_params: OutlineParams
    region: str | None = None
    color: str | None = None
    groups: str | Sequence[str] | None = None
    contour_px: int | None = None
    alt_var: str | None = None
    layer: str | None = None
    palette: Palette_t = None
    alpha: float = 1.0
    size: float = 1.0


def _render_shapes(
    sdata: sd.SpatialData,
    render_params: ShapesRenderParams,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
    # ax: matplotlib.axes.SubplotBase,
    # extent: dict[str, list[int]],
) -> None:
    # get instance and region keys
    # instance_key = str(sdata.table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY])
    region_key = str(sdata.table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY])

    # subset table based on region
    if render_params.region is not None:
        table = sdata.table[sdata.table.obs[region_key].isin([render_params.region])]
        region = render_params.region
    else:
        region = sdata.table.obs[region_key].unique()[0]  # TODO: handle multiple regions
        table = sdata.table

    # get instance id based on subsetted table
    # instance_id = table.obs[instance_key].values
    # TODO: use it for subsetting shapes

    # get labels
    shapes = sdata.shapes[region]

    # get color vector (categorical or continuous)
    color_source_vector, color_vector, _ = _set_color_source_vec(
        adata=table,
        value_to_plot=render_params.color,
        alt_var=render_params.alt_var,
        layer=render_params.layer,
        groups=render_params.groups,
        palette=render_params.palette,
        na_color=render_params.cmap_params.na_color,
        alpha=render_params.alpha,
    )

    def _get_collection_shape(
        shapes: GeoDataFrame,
        c: Any,
        s: float,
        norm: Any,
        **kwargs: Any,
    ) -> PatchCollection:
        """Get collection of shapes."""
        if shapes["geometry"][0].geom_type == "Polygon":
            patches = [Polygon(p.exterior.coords, closed=False) for p in shapes["geometry"]]
        elif shapes["geometry"][0].geom_type == "Point":
            patches = [Circle(circ, radius=r * s) for circ, r in zip(shapes["geometry"], shapes["radius"])]

        collection = PatchCollection(patches, snap=False, **kwargs)
        print(c)
        if isinstance(c, np.ndarray) and np.issubdtype(c.dtype, np.number):
            collection.set_array(np.ma.masked_invalid(c))
            collection.set_norm(norm)
        else:
            alpha = ColorConverter().to_rgba_array(c)[..., -1]
            collection.set_facecolor(c)
            collection.set_alpha(alpha)
        return collection

    norm = copy(render_params.cmap_params.norm)
    ax = fig_params.ax
    if render_params.outline_params.outline:
        _cax = _get_collection_shape(
            shapes=shapes,
            s=render_params.outline_params.bg_size,
            c=render_params.outline_params.bg_color,
            rasterized=sc_settings._vector_friendly,
            cmap=render_params.cmap_params.cmap,
            norm=norm,
            # **kwargs,
        )
        ax.add_collection(_cax)
        _cax = _get_collection_shape(
            shapes=shapes,
            s=render_params.outline_params.gap_size,
            c=render_params.outline_params.gap_color,
            rasterized=sc_settings._vector_friendly,
            cmap=render_params.cmap_params.cmap,
            norm=norm,
            # **kwargs,
        )
        ax.add_collection(_cax)
    _cax = _get_collection_shape(
        shapes=shapes,
        s=render_params.size,
        c=color_vector,
        rasterized=sc_settings._vector_friendly,
        cmap=render_params.cmap_params.cmap,
        norm=norm,
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


def _render_points(
    sdata: sd.SpatialData,
    params: dict[str, Union[str, int, float, Iterable[str]]],
    key: str,
    ax: matplotlib.axes.SubplotBase,
    extent: dict[str, list[int]],
) -> None:
    ax.set_xlim(extent["x"][0], extent["x"][1])
    ax.set_ylim(extent["y"][0], extent["y"][1])

    if isinstance(params["color_key"], str):
        colors = sdata.points[key][params["color_key"]].compute()

        if is_categorical_dtype(colors):
            category_colors = _get_palette(categories=colors.cat.categories)

            for i, cat in enumerate(colors.cat.categories):
                ax.scatter(
                    x=sdata.points[key]["x"].compute()[colors == cat],
                    y=sdata.points[key]["y"].compute()[colors == cat],
                    color=category_colors[i],
                    label=cat,
                )

        else:
            ax.scatter(
                x=sdata.points[key]["x"].compute(),
                y=sdata.points[key]["y"].compute(),
                c=colors,
            )
    else:
        ax.scatter(
            x=sdata.points[key]["x"].compute(),
            y=sdata.points[key]["y"].compute(),
        )

    ax.set_title(key)


@dataclass
class ImageRenderParams:
    """Labels render parameters.."""

    image: str | None = None
    channel: Sequence[str] | None = None
    cmap_params: CmapParams = None
    palette: Palette_t = None
    alpha: float = 1.0


def _render_images(
    sdata: sd.SpatialData,
    render_params: ImageRenderParams,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
    # ax: matplotlib.axes.SubplotBase,
    # extent: dict[str, list[int]],
) -> None:
    if render_params.image is not None:
        img = sdata.images[render_params.image]
    else:
        image_key = list(sdata.images.keys())[0]  # TODO: handle multiple images
        img = sdata.images[image_key]

    if (len(img.c) > 3 or len(img.c) == 2) and render_params.channel is None:
        raise NotImplementedError("Only 1 or 3 channels are supported at the moment.")

    img = _normalize(img, clip=True)

    # If channel colors are not specified, use default colors
    # colors = render_params.palette
    # if colors is None and render_params.channels is not None and len(render_params.channels) > 3:
    #     flattened_img = np.reshape(img, (n_channels, -1))
    #     pca = PCA(n_components=3)
    #     transformed_image = pca.fit_transform(flattened_img.T)
    #     img = xr.DataArray(transformed_image.T.reshape(3, y_dim, x_dim), dims=("c", "y", "x"))

    if render_params.channel is not None:
        img = img.sel(c=[render_params.channel])

    img = img.transpose("y", "x", "c")  # for plotting

    ax = fig_params.ax

    # ax.set_xlim(extent["x"][0], extent["x"][1])
    # ax.set_ylim(extent["y"][0], extent["y"][1])

    ax.imshow(
        img.data,
        cmap=render_params.cmap_params.cmap,
        alpha=render_params.alpha,
    )

    # ax.set_title(key)


@dataclass
class LabelsRenderParams:
    """Labels render parameters.."""

    region: str | None = None
    color: str | None = None
    groups: str | Sequence[str] | None = None
    contour_px: int | None = None
    outline: bool = False
    alt_var: str | None = None
    layer: str | None = None
    cmap_params: CmapParams = None
    palette: Palette_t = None
    alpha: float = 1.0


def _render_labels(
    sdata: sd.SpatialData,
    render_params: LabelsRenderParams,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
    # ax: matplotlib.axes.SubplotBase,
    # extent: dict[str, list[int]],
) -> None:

    # get instance and region keys
    instance_key = str(sdata.table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY])
    region_key = str(sdata.table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY])

    # subset table based on region
    if render_params.region is not None:
        table = sdata.table[sdata.table.obs[region_key].isin([render_params.region])]
        region = render_params.region
    else:
        region = sdata.table.obs[region_key].unique()[0]  # TODO: handle multiple regions
        table = sdata.table

    # get isntance id based on subsetted table
    instance_id = table.obs[instance_key].values

    # get labels
    segmentation = sdata.labels[region].values

    # get color vector (categorical or continuous)
    color_source_vector, color_vector, categorical = _set_color_source_vec(
        adata=table,
        value_to_plot=render_params.color,
        alt_var=render_params.alt_var,
        layer=render_params.layer,
        groups=render_params.groups,
        palette=render_params.palette,
        na_color=render_params.cmap_params.na_color,
        alpha=render_params.alpha,
    )

    # map color vector to segmentation
    labels = _map_color_seg(
        seg=segmentation,
        cell_id=instance_id,
        color_vector=color_vector,
        color_source_vector=color_source_vector,
        cmap_params=render_params.cmap_params,
        seg_erosionpx=render_params.contour_px,
        seg_boundaries=render_params.outline,
        na_color=render_params.cmap_params.na_color,
    )

    ax = fig_params.ax
    _cax = ax.imshow(
        labels,
        rasterized=True,
        cmap=render_params.cmap_params.cmap if not categorical else None,
        norm=render_params.cmap_params.norm if not categorical else None,
        alpha=render_params.alpha,
        origin="lower",
        zorder=3,
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
