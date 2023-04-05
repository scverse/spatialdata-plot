from __future__ import annotations
from collections.abc import Iterable
from typing import Callable, Optional, Union, Sequence, NamedTuple
from matplotlib.cm import get_cmap
from copy import copy
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from matplotlib.colors import ListedColormap, to_rgb
from pandas.api.types import is_categorical_dtype
from skimage.segmentation import find_boundaries
from sklearn.decomposition import PCA
import scanpy as sc
from pandas.api.types import is_categorical_dtype
from spatialdata._types import ArrayLike
from anndata import AnnData
from pandas.api.types import CategoricalDtype
from matplotlib import colors, patheffects, rcParams
from matplotlib.colors import (
    ColorConverter,
    Colormap,
    Normalize,
    TwoSlopeNorm,
)
from skimage.color import label2rgb
from skimage.morphology import erosion, square
from skimage.segmentation import find_boundaries
from skimage.util import map_array
from functools import partial
from ..pl._categorical_utils import _get_colors_for_categorical_obs, _get_palette, _maybe_set_colors
from ..pl.utils import _normalize
from ..pp.utils import _get_linear_colormap, _get_region_key


Palette_t = Optional[Union[str, ListedColormap]]
_Normalize = Union[Normalize, Sequence[Normalize]]
_SeqStr = Union[str, Sequence[str]]
to_hex = partial(colors.to_hex, keep_alpha=True)


class CmapParams(NamedTuple):
    """Cmap params."""

    cmap: Colormap
    img_cmap: Colormap
    norm: Normalize


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


import matplotlib.pyplot as plt


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

    color_source_vector, color_vector, categorical = _set_color_source_vec(table, params["color_key"])
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
        clusters = color_source_vector.categories
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


def _set_color_source_vec(
    adata: AnnData,
    value_to_plot: str | None,
    use_raw: bool | None = None,
    alt_var: str | None = None,
    layer: str | None = None,
    groups: _SeqStr | None = None,
    palette: Palette_t = None,
    na_color: str | tuple[float, ...] | None = None,
    alpha: float = 1.0,
) -> tuple[ArrayLike | pd.Series | None, ArrayLike, bool]:
    if value_to_plot is None:
        color = np.full(adata.n_obs, to_hex(na_color))
        return color, color, False

    if alt_var is not None and value_to_plot not in adata.obs and value_to_plot not in adata.var_names:
        value_to_plot = adata.var_names[adata.var[alt_var] == value_to_plot][0]
    if use_raw and value_to_plot not in adata.obs:
        color_source_vector = adata.raw.obs_vector(value_to_plot)
    else:
        color_source_vector = adata.obs_vector(value_to_plot, layer=layer)

    if not is_categorical_dtype(color_source_vector):
        return None, color_source_vector, False

    color_source_vector = pd.Categorical(color_source_vector)  # convert, e.g., `pd.Series`
    categories = color_source_vector.categories
    if groups is not None:
        color_source_vector = color_source_vector.remove_categories(categories.difference(groups))

    color_map = _get_palette(adata, cluster_key=value_to_plot, categories=categories, palette=palette, alpha=alpha)
    if color_map is None:
        raise ValueError("Unable to create color palette.")
    # do not rename categories, as colors need not be unique
    color_vector = color_source_vector.map(color_map)
    if color_vector.isna().any():
        color_vector = color_vector.add_categories([to_hex(na_color)])
        color_vector = color_vector.fillna(to_hex(na_color))

    return color_source_vector, color_vector, True


def _map_color_seg(
    seg: ArrayLike,
    cell_id: ArrayLike,
    color_vector: ArrayLike | pd.Series[CategoricalDtype],
    color_source_vector: pd.Series[CategoricalDtype],
    cmap_params: CmapParams,
    seg_erosionpx: int | None = None,
    seg_boundaries: bool = False,
    na_color: str | tuple[float, ...] = (0, 0, 0, 0),
) -> ArrayLike:
    cell_id = np.array(cell_id)

    if is_categorical_dtype(color_vector):
        if isinstance(na_color, tuple) and len(na_color) == 4 and np.any(color_source_vector.isna()):
            cell_id[color_source_vector.isna()] = 0
        val_im: ArrayLike = map_array(seg, cell_id, color_vector.codes + 1)  # type: ignore
        cols = colors.to_rgba_array(color_vector.categories)  # type: ignore
    else:
        val_im = map_array(seg, cell_id, cell_id)  # replace with same seg id to remove missing segs
        try:
            cols = cmap_params.cmap(cmap_params.norm(color_vector))
        except TypeError:
            assert all(colors.is_color_like(c) for c in color_vector), "Not all values are color-like."
            cols = colors.to_rgba_array(color_vector)

    if seg_erosionpx is not None:
        val_im[val_im == erosion(val_im, square(seg_erosionpx))] = 0

    seg_im: ArrayLike = label2rgb(
        label=val_im,
        colors=cols,
        bg_label=0,
        bg_color=(1, 1, 1),  # transparency doesn't really work
    )

    if seg_boundaries:
        seg_bound: ArrayLike = np.clip(seg_im - find_boundaries(seg)[:, :, None], 0, 1)
        seg_bound = np.dstack((seg_bound, np.where(val_im > 0, 1, 0)))  # add transparency here
        return seg_bound
    seg_im = np.dstack((seg_im, np.where(val_im > 0, 1, 0)))  # add transparency here
    return seg_im
