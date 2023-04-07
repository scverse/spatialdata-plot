from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import copy
from functools import partial
from typing import Callable, Optional, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd
import xarray as xr
from matplotlib import colors
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, Normalize, to_rgb
from pandas.api.types import is_categorical_dtype
from sklearn.decomposition import PCA

from spatialdata_plot.pl._categorical_utils import (
    _get_colors_for_categorical_obs,
    _get_palette,
)
from spatialdata_plot.pl.utils import (
    CmapParams,
    _map_color_seg,
    _normalize,
    _set_color_source_vec,
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
            category_colors = _get_colors_for_categorical_obs(colors.cat.categories)

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
    params: dict[str, Union[str, int, float]],
    key: str,
    ax: matplotlib.axes.SubplotBase,
    extent: dict[str, list[int]],
) -> None:
    region_key = _get_region_key(sdata)

    # subset table to only the entires specified by 'key'
    table = sdata.table[sdata.table.obs[region_key] == key]
    segmentation = sdata.labels[key].values

    norm = Normalize(vmin=None, vmax=None)
    cmap = copy(get_cmap(None))
    # cmap.set_bad("lightgray" if na_color is None else na_color)
    cmap_params = CmapParams(cmap, cmap, norm)

    color_source_vector, color_vector, categorical = _set_color_source_vec(table, params["color_key"])  # type: ignore[arg-type]
    segmentation = _map_color_seg(
        seg=segmentation,
        cell_id=table.obs[params["instance_key"]].values,
        color_vector=color_vector,
        color_source_vector=color_source_vector,
        cmap_params=cmap_params,
    )

    ax.set_xlim(extent["x"][0], extent["x"][1])
    ax.set_ylim(extent["y"][0], extent["y"][1])

    cax = ax.imshow(
        segmentation,
        rasterized=True,
        cmap=cmap_params.cmap if not categorical else None,
        norm=cmap_params.norm if not categorical else None,
        # alpha=color_params.alpha,
        origin="lower",
        zorder=3,
    )

    # if params["add_legend"]:
    #     patches = []
    #     for group, color in group_to_color.values:
    #         patches.append(mpatches.Patch(color=color, label=group))

    #     ax.legend(handles=patches, bbox_to_anchor=(0.9, 0.9), loc="upper left", frameon=False)
    # ax.colorbar(pad=0.01, fraction=0.08, aspect=30)
    if is_categorical_dtype(color_source_vector):
        clusters = color_source_vector.categories  # type: ignore[union-attr]
        palette = _get_palette(table, cluster_key=params["color_key"], categories=clusters)
        for label in clusters:
            ax.scatter([], [], c=palette[label], label=label)
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if len(clusters) <= 14 else 2 if len(clusters) <= 30 else 3),
            fontsize=None,
        )
    else:
        plt.colorbar(cax, ax=ax, pad=0.01, fraction=0.08, aspect=30)
    ax.set_title(key)
