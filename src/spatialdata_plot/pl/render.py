from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import copy
from functools import partial
from typing import Callable, Optional, Union
from matplotlib.colors import Colormap
import matplotlib
from spatialdata.models import TableModel
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd
import xarray as xr
from matplotlib import colors
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, Normalize, to_rgb, TwoSlopeNorm
from pandas.api.types import is_categorical_dtype
from sklearn.decomposition import PCA
from dataclasses import dataclass
from spatialdata_plot.pl.basic import LabelsRenderParams, 
from spatialdata_plot.pl.utils import (
    CmapParams,
    LegendParams,
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


def _render_channels(
    sdata: sd.SpatialData,
    channels: list[Union[str, int]],
    colors: list[str],
    clip: bool,
    normalize: bool,
    background: str,
    pmin: float,
    pmax: float,
    key: str,
    ax: matplotlib.axes.SubplotBase,
) -> None:
    selection = sdata.images[key].sel({"c": channels})
    n_channels, y_dim, x_dim = selection.shape  # (c, y, x)
    img = selection.values.copy()
    img = img.astype("float")

    if normalize:
        img = _normalize(img, pmin, pmax, clip)

    cmaps = _get_linear_colormap(colors[:n_channels], background)
    colored = np.stack([cmaps[i](img[i]) for i in range(n_channels)], 0).sum(0)

    if clip:
        colored = np.clip(colored, 0, 1)

    ax.imshow(colored)
    ax.set_title(key)
    ax.set_xlabel("spatial1")
    ax.set_ylabel("spatial2")
    ax.set_xticks([])
    ax.set_yticks([])


def _render_shapes(
    sdata: sd.SpatialData,
    params: dict[str, Optional[Union[str, int, float, Iterable[str]]]],
    key: str,
    ax: matplotlib.axes.SubplotBase,
    extent: dict[str, list[int]],
) -> None:
    colors: Optional[Union[str, int, float, Iterable[str]]] = None  # to shut up mypy
    if sdata.table is not None and isinstance(params["instance_key"], str) and isinstance(params["color_key"], str):
        colors = [to_rgb(c) for c in sdata.table.uns[f"{params['color_key']}_colors"]]
    elif isinstance(params["palette"], str):
        colors = [params["palette"]]
    elif isinstance(params["palette"], Iterable):
        colors = [to_rgb(c) for c in list(params["palette"])]
    else:
        colors = params["palette"]

    ax.set_xlim(extent["x"][0], extent["x"][1])
    ax.set_ylim(extent["y"][0], extent["y"][1])

    points = []
    polygons = []

    for _, row in sdata.shapes[key].iterrows():
        if row["geometry"].geom_type == "Point":
            points.append((row[0], row[1]))  # (point, radius)
        elif row["geometry"].geom_type == "Polygon":
            polygons.append(row[0])  # just polygon
        else:
            raise NotImplementedError(f"Geometry type {row['geometry'].type} not supported.")

    if len(polygons) > 0:
        for polygon in polygons:
            ax.add_patch(
                mpatches.Polygon(
                    polygon.exterior.coords,
                    color=colors,
                )
            )

    if len(points) > 0:
        for idx, (point, radius) in enumerate(points):
            ax.add_patch(
                mpatches.Circle(
                    (point.x, point.y),
                    radius=radius,
                    color=colors[idx],  # type: ignore
                )
            )

    ax.set_title(key)


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


def _render_images(
    sdata: sd.SpatialData,
    params: dict[str, Union[str, int, float]],
    key: str,
    ax: matplotlib.axes.SubplotBase,
    extent: dict[str, list[int]],
) -> None:
    n_channels, y_dim, x_dim = sdata.images[key].shape  # (c, y, x)
    img = sdata.images[key].values.copy()
    img = img.astype("float")

    if params["trans_fun"] is not None:
        trans_fun: Callable[[xr.DataArray], xr.DataArray] = params["trans_fun"]  # type: ignore
        img = trans_fun(img)

    img = _normalize(img, clip=True)

    # If channel colors are not specified, use default colors
    colors: Union[matplotlib.colors.ListedColormap, list[matplotlib.colors.ListedColormap]] = params["palette"]
    if params["palette"] is None:
        if n_channels == 1:
            colors = ListedColormap(["gray"])
        elif n_channels == 2:
            colors = ListedColormap(["#d30cb8", "#6df1d8"])
        elif n_channels == 3:
            colors = ListedColormap(["red", "blue", "green"])
        else:
            # we do PCA to reduce to 3 channels
            flattened_img = np.reshape(img, (n_channels, -1))
            pca = PCA(n_components=3)
            pca.fit(flattened_img.T)
            transformed_image = pca.transform(flattened_img.T)
            img = xr.DataArray(transformed_image.T.reshape(3, y_dim, x_dim), dims=("c", "y", "x"))

    img = xr.DataArray(img, dims=("c", "y", "x")).transpose("y", "x", "c")  # for plotting

    ax.set_xlim(extent["x"][0], extent["x"][1])
    ax.set_ylim(extent["y"][0], extent["y"][1])
    ax.imshow(
        img.transpose("y", "x", "c").data,
        cmap=colors,
        interpolation="nearest",
    )

    ax.set_title(key)


def _render_labels(
    sdata: sd.SpatialData,
    render_params: LabelsRenderParams,
    legend_params: LegendParams,
    ax: matplotlib.axes.SubplotBase,
    extent: dict[str, list[int]],
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
        legend_fontsize=legend_prams.legend_fontsize,
        legend_fontweight=legend_params.legend_fontweight,
        legend_loc=legend_params.legend_loc,
        legend_fontoutline=legend_Params.legend_fontoutline,
        na_in_legend=legend_params.legend_na,
        colorbar=legend_params.colorbar,
        scalebar_dx=scalebar_params.scalebar_dx,
        scalebar_units=scalebar_params.scalebar_units,
        scalebar_kwargs=scalebar_params.scalebar_kwargs,
    )
