from collections.abc import Iterable
from typing import Callable, Union

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from matplotlib.colors import ListedColormap, to_rgb
from skimage.segmentation import find_boundaries
from sklearn.decomposition import PCA

from ..pl.utils import _normalize
from ..pp.utils import _get_linear_colormap, _get_region_key


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
    params: dict[str, Union[str, int, float, Iterable[str]]],
    key: str,
    ax: matplotlib.axes.SubplotBase,
    extent: dict[str, list[int]],
) -> None:
    if sdata.table is not None and isinstance(params["instance_key"], str) and isinstance(params["color_key"], str):
        colors = [to_rgb(c) for c in sdata.table.uns[f"{params['color_key']}_colors"]]

    elif isinstance(params["palette"], str):
        colors = [params["palette"]]
    elif isinstance(params["palette"], Iterable):
        colors = [to_rgb(c) for c in list(params["palette"])]
    else:
        # assert isinstance(params["palette"], Iterable)
        colors = [params["palette"]]

    ax.set_xlim(extent["x"][0], extent["x"][1])
    ax.set_ylim(extent["y"][0], extent["y"][1])

    shape = sdata.shapes[key]

    ax.scatter(
        x=shape.geometry.x,
        y=shape.geometry.y,
        s=shape.radius,
        color=colors,
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
            # bg = [(1, 1, 1, 1)]
            # cmap_red = ListedColormap([(1, 0, 0, i) for i in reversed(range(0, 256, 1))] + bg)
            # cmap_green = ListedColormap([(0, 1, 0, i) for i in reversed(range(0, 256, 1))] + bg)
            # cmap_blue = ListedColormap([(0, 0, 1, i) for i in reversed(range(0, 256, 1))] + bg)
            # colors = [cmap_red, cmap_green, cmap_blue]
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
    # ax.set_xlabel("spatial1")
    # ax.set_ylabel("spatial2")
    # ax.set_xticks([])
    # ax.set_yticks([])


def _render_labels(
    sdata: sd.SpatialData,
    params: dict[str, Union[str, int, float]],
    key: str,
    ax: matplotlib.axes.SubplotBase,
    extent: dict[str, list[int]],
) -> None:
    region_key = _get_region_key(sdata)

    # subset table to only the entires specified by 'key'
    table = sdata.table.obs
    table = table[table[region_key] == key]

    # If palette is not None, table.uns contains the relevant vector
    if f"{params['instance_key']}_colors" in sdata.table.uns.keys():
        colors = [to_rgb(c) for c in sdata.table.uns[f"{params['instance_key']}_colors"]]
        colors = [tuple(list(c) + [1]) for c in colors]

    groups = sdata.table.obs[params["color_key"]].unique()
    group_to_color = pd.DataFrame({params["color_key"]: groups, "color": colors})

    segmentation = sdata.labels[key].values

    ax.set_xlim(extent["x"][0], extent["x"][1])
    ax.set_ylim(extent["y"][0], extent["y"][1])

    for group in groups:
        vaid_cell_ids = table[table[params["color_key"]] == group][params["instance_key"]].values

        # define all out-of-group cells as background
        in_group_mask = segmentation.copy()
        in_group_mask[~np.isin(segmentation, vaid_cell_ids)] = 0

        # get correct color for the group
        group_color = list(group_to_color[group_to_color[params["color_key"]] == group].color.values[0])

        if params["fill_alpha"] != 0:
            infill_mask = in_group_mask > 0

            fill_color = group_color.copy()
            fill_color[-1] = params["fill_alpha"]
            colors = [[0, 0, 0, 0], fill_color]  # add transparent for bg

            ax.imshow(
                infill_mask,
                cmap=ListedColormap(colors),
                interpolation="nearest",
            )

        if params["border_alpha"] != 0:
            border_mask = find_boundaries(in_group_mask, mode=params["mode"])
            border_mask = np.ma.masked_array(in_group_mask, ~border_mask)

            border_color = group_color.copy()
            border_color[-1] = params["border_alpha"]

            ax.imshow(
                border_mask,
                cmap=ListedColormap([border_color]),
                interpolation="nearest",
            )

    if params["add_legend"]:
        patches = []
        for group, color in group_to_color.values:
            patches.append(mpatches.Patch(color=color, label=group))

        ax.legend(handles=patches, bbox_to_anchor=(0.9, 0.9), loc="upper left", frameon=False)

    ax.set_title(key)
    # ax.set_xlabel("spatial1")
    # ax.set_ylabel("spatial2")
    # ax.set_xticks([])
    # ax.set_yticks([])
