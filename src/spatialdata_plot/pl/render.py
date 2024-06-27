from __future__ import annotations

import warnings
from collections import abc
from copy import copy
from typing import Union

import dask
import geopandas as gpd
import matplotlib
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from anndata import AnnData
from datatree import DataTree
from matplotlib.colors import ListedColormap, Normalize
from scanpy._settings import settings as sc_settings
from spatialdata import get_extent
from spatialdata.models import PointsModel, get_table_keys
from spatialdata.transformations import (
    get_transformation,
)

from spatialdata_plot._logging import logger
from spatialdata_plot.pl.render_params import (
    FigParams,
    ImageRenderParams,
    LabelsRenderParams,
    LegendParams,
    PointsRenderParams,
    ScalebarParams,
    ShapesRenderParams,
)
from spatialdata_plot.pl.utils import (
    _ax_show_and_transform,
    _decorate_axs,
    _get_collection_shape,
    _get_colors_for_categorical_obs,
    _get_linear_colormap,
    _is_coercable_to_float,
    _map_color_seg,
    _maybe_set_colors,
    _multiscale_to_spatial_image,
    _normalize,
    _rasterize_if_necessary,
    _set_color_source_vec,
    to_hex,
)

_Normalize = Union[Normalize, abc.Sequence[Normalize]]


def _render_shapes(
    sdata: sd.SpatialData,
    render_params: ShapesRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
) -> None:
    element = render_params.element
    col_for_color = render_params.col_for_color
    groups = render_params.groups

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_tables=bool(render_params.table_name),
    )

    # for index, e in enumerate(elements):
    shapes = sdata[element]

    if (table_name := render_params.table_name) is None:
        table = None
    else:
        _, region_key, _ = get_table_keys(sdata[table_name])
        table = sdata[table_name][sdata[table_name].obs[region_key].isin([element])]

    if (
        col_for_color is not None
        and table_name is not None
        and col_for_color in sdata_filt[table_name].obs.columns
        and (color_col := sdata_filt[table_name].obs[col_for_color]).dtype == "O"
        and not _is_coercable_to_float(color_col)
    ):
        warnings.warn(
            f"Converting copy of '{col_for_color}' column to categorical dtype for categorical plotting. "
            f"Consider converting before plotting.",
            UserWarning,
            stacklevel=2,
        )
        sdata_filt[table_name].obs[col_for_color] = sdata_filt[table_name].obs[col_for_color].astype("category")

    # get color vector (categorical or continuous)
    color_source_vector, color_vector, _ = _set_color_source_vec(
        sdata=sdata_filt,
        element=sdata_filt[element],
        element_name=element,
        value_to_plot=col_for_color,
        groups=groups,
        palette=render_params.palette,
        na_color=render_params.color or render_params.cmap_params.na_color,
        cmap_params=render_params.cmap_params,
        table_name=table_name,
    )

    values_are_categorical = color_source_vector is not None

    # color_source_vector is None when the values aren't categorical
    if values_are_categorical and render_params.transfunc is not None:
        color_vector = render_params.transfunc(color_vector)

    norm = copy(render_params.cmap_params.norm)

    if len(color_vector) == 0:
        color_vector = [render_params.cmap_params.na_color]

    # filter by `groups`

    if isinstance(groups, list) and color_source_vector is not None:
        mask = color_source_vector.isin(groups)
        shapes = shapes[mask]
        shapes = shapes.reset_index()
        color_source_vector = color_source_vector[mask]
        color_vector = color_vector[mask]
    shapes = gpd.GeoDataFrame(shapes, geometry="geometry")

    _cax = _get_collection_shape(
        shapes=shapes,
        s=render_params.scale,
        c=color_vector,
        render_params=render_params,
        rasterized=sc_settings._vector_friendly,
        cmap=render_params.cmap_params.cmap,
        norm=norm,
        fill_alpha=render_params.fill_alpha,
        outline_alpha=render_params.outline_alpha,
        # **kwargs,
    )

    # Sets the limits of the colorbar to the values instead of [0, 1]
    if not norm and not values_are_categorical:
        _cax.set_clim(min(color_vector), max(color_vector))

    cax = ax.add_collection(_cax)

    # Apply the transformation to the PatchCollection's paths
    trans = get_transformation(sdata_filt[element], get_all=True)[coordinate_system]
    affine_trans = trans.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    trans = mtransforms.Affine2D(matrix=affine_trans)

    for path in _cax.get_paths():
        path.vertices = trans.transform(path.vertices)

    # Using dict.fromkeys here since set returns in arbitrary order
    # remove the color of NaN values, else it might be assigned to a category
    # order of color in the palette should agree to order of occurence
    if color_source_vector is None:
        palette = ListedColormap(dict.fromkeys(color_vector))
    else:
        palette = ListedColormap(dict.fromkeys(color_vector[~pd.Categorical(color_source_vector).isnull()]))

    if not (len(set(color_vector)) == 1 and list(set(color_vector))[0] == to_hex(render_params.cmap_params.na_color)):
        # necessary in case different shapes elements are annotated with one table
        if color_source_vector is not None and col_for_color is not None:
            color_source_vector = color_source_vector.remove_unused_categories()

        # False if user specified color-like with 'color' parameter
        colorbar = False if col_for_color is None else legend_params.colorbar

        _ = _decorate_axs(
            ax=ax,
            cax=cax,
            fig_params=fig_params,
            adata=table,
            value_to_plot=col_for_color,
            color_source_vector=color_source_vector,
            palette=palette,
            alpha=render_params.fill_alpha,
            na_color=render_params.cmap_params.na_color,
            legend_fontsize=legend_params.legend_fontsize,
            legend_fontweight=legend_params.legend_fontweight,
            legend_loc=legend_params.legend_loc,
            legend_fontoutline=legend_params.legend_fontoutline,
            na_in_legend=legend_params.na_in_legend,
            colorbar=colorbar,
            scalebar_dx=scalebar_params.scalebar_dx,
            scalebar_units=scalebar_params.scalebar_units,
        )


def _render_points(
    sdata: sd.SpatialData,
    render_params: PointsRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
) -> None:
    element = render_params.element
    col_for_color = render_params.col_for_color
    table_name = render_params.table_name
    color = render_params.color
    groups = render_params.groups
    palette = render_params.palette

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_tables=bool(table_name),
    )

    points = sdata.points[element]
    coords = ["x", "y"]

    if col_for_color is None or (table_name is not None and col_for_color in sdata_filt[table_name].obs.columns):
        points = points[coords].compute()
        if (
            col_for_color
            and (color_col := sdata_filt[table_name].obs[col_for_color]).dtype == "O"
            and not _is_coercable_to_float(color_col)
        ):
            warnings.warn(
                f"Converting copy of '{col_for_color}' column to categorical dtype for categorical "
                f"plotting. Consider converting before plotting.",
                UserWarning,
                stacklevel=2,
            )
            sdata_filt[table_name].obs[col_for_color] = sdata_filt[table_name].obs[col_for_color].astype("category")
    else:
        coords += [col_for_color]
        points = points[coords].compute()

    if groups is not None and col_for_color is not None:
        points = points[points[col_for_color].isin(groups)]

    # we construct an anndata to hack the plotting functions
    if table_name is None:
        adata = AnnData(
            X=points[["x", "y"]].values, obs=points[coords].reset_index(), dtype=points[["x", "y"]].values.dtype
        )
    else:
        adata = AnnData(
            X=points[["x", "y"]].values,
            obs=sdata_filt[table_name].obs,
            dtype=points[["x", "y"]].values.dtype,
            uns=sdata_filt[table_name].uns,
        )
        sdata_filt[table_name] = adata

    # we can do this because of dealing with a copy

    # Convert back to dask dataframe to modify sdata
    points = dask.dataframe.from_pandas(points, npartitions=1)
    sdata_filt.points[element] = PointsModel.parse(points, coordinates={"x": "x", "y": "y"})

    if col_for_color is not None:
        cols = sc.get.obs_df(adata, col_for_color)
        # maybe set color based on type
        if isinstance(cols.dtype, pd.CategoricalDtype):
            _maybe_set_colors(
                source=adata,
                target=adata,
                key=col_for_color,
                palette=palette,
            )

    # when user specified a single color, we overwrite na with it
    default_color = color if col_for_color is None and color is not None else render_params.cmap_params.na_color

    color_source_vector, color_vector, _ = _set_color_source_vec(
        sdata=sdata_filt,
        element=points,
        element_name=element,
        value_to_plot=col_for_color,
        groups=groups,
        palette=palette,
        na_color=default_color,
        cmap_params=render_params.cmap_params,
        table_name=table_name,
    )

    # color_source_vector is None when the values aren't categorical
    if color_source_vector is None and render_params.transfunc is not None:
        color_vector = render_params.transfunc(color_vector)

    trans = get_transformation(sdata.points[element], get_all=True)[coordinate_system]
    affine_trans = trans.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    trans = mtransforms.Affine2D(matrix=affine_trans) + ax.transData

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
        transform=trans,
    )
    cax = ax.add_collection(_cax)

    if len(set(color_vector)) != 1 or list(set(color_vector))[0] != to_hex(render_params.cmap_params.na_color):
        if color_source_vector is None:
            palette = ListedColormap(dict.fromkeys(color_vector))
        else:
            palette = ListedColormap(dict.fromkeys(color_vector[~pd.Categorical(color_source_vector).isnull()]))

        _ = _decorate_axs(
            ax=ax,
            cax=cax,
            fig_params=fig_params,
            adata=adata,
            value_to_plot=col_for_color,
            color_source_vector=color_source_vector,
            palette=palette,
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
        )


def _render_images(
    sdata: sd.SpatialData,
    render_params: ImageRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
    rasterize: bool,
) -> None:

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_tables=False,
    )

    palette = render_params.palette
    img = sdata_filt[render_params.element]
    extent = get_extent(img, coordinate_system=coordinate_system)
    scale = render_params.scale

    # get best scale out of multiscale image
    if isinstance(img, DataTree):
        img = _multiscale_to_spatial_image(
            multiscale_image=img,
            dpi=fig_params.fig.dpi,
            width=fig_params.fig.get_size_inches()[0],
            height=fig_params.fig.get_size_inches()[1],
            scale=scale,
        )
    # rasterize spatial image if necessary to speed up performance
    if rasterize:
        img = _rasterize_if_necessary(
            image=img,
            dpi=fig_params.fig.dpi,
            width=fig_params.fig.get_size_inches()[0],
            height=fig_params.fig.get_size_inches()[1],
            coordinate_system=coordinate_system,
            extent=extent,
        )

    channels = img.coords["c"].values if render_params.channel is None else render_params.channel

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

    # prepare transformations
    trans = get_transformation(img, get_all=True)[coordinate_system]
    affine_trans = trans.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    trans = mtransforms.Affine2D(matrix=affine_trans)
    trans_data = trans + ax.transData

    # 1) Image has only 1 channel
    if n_channels == 1 and not isinstance(render_params.cmap_params, list):
        layer = img.sel(c=channels[0]).squeeze() if isinstance(channels[0], str) else img.isel(c=channels[0]).squeeze()

        if render_params.percentiles_for_norm != (None, None):
            layer = _normalize(
                layer, pmin=render_params.percentiles_for_norm[0], pmax=render_params.percentiles_for_norm[1], clip=True
            )

        if render_params.cmap_params.norm:  # type: ignore[attr-defined]
            layer = render_params.cmap_params.norm(layer)  # type: ignore[attr-defined]

        cmap = (
            _get_linear_colormap(palette, "k")[0]
            if isinstance(palette, list) and all(isinstance(p, str) for p in palette)
            else render_params.cmap_params.cmap
            # render_params.cmap_params.cmap if render_params.palette is None else _get_linear_colormap(palette, "k")[0]
        )

        # Overwrite alpha in cmap: https://stackoverflow.com/a/10127675
        cmap._init()
        cmap._lut[:, -1] = render_params.alpha

        _ax_show_and_transform(layer, trans_data, ax, cmap=cmap)

    # 2) Image has any number of channels but 1
    else:
        layers = {}
        for ch_index, c in enumerate(channels):
            layers[c] = img.sel(c=c).copy(deep=True).squeeze()

            if render_params.percentiles_for_norm != (None, None):
                layers[c] = _normalize(
                    layers[c],
                    pmin=render_params.percentiles_for_norm[0],
                    pmax=render_params.percentiles_for_norm[1],
                    clip=True,
                )

            if not isinstance(render_params.cmap_params, list) and render_params.cmap_params.norm:
                layers[c] = render_params.cmap_params.norm(layers[c])
            elif isinstance(render_params.cmap_params, list) and render_params.cmap_params[ch_index].norm:
                layers[c] = render_params.cmap_params[ch_index].norm(layers[c])

        # 2A) Image has 3 channels, no palette info, and no/only one cmap was given
        if palette is None and n_channels == 3 and not isinstance(render_params.cmap_params, list):
            if render_params.cmap_params.is_default:  # -> use RGB
                stacked = np.stack([layers[c] for c in channels], axis=-1)
            else:  # -> use given cmap for each channel
                channel_cmaps = [render_params.cmap_params.cmap] * n_channels
                # Apply cmaps to each channel, add up and normalize to [0, 1]
                stacked = (
                    np.stack([channel_cmaps[ind](layers[ch]) for ind, ch in enumerate(channels)], 0).sum(0) / n_channels
                )
                # Remove alpha channel so we can overwrite it from render_params.alpha
                stacked = stacked[:, :, :3]
                logger.warning(
                    "One cmap was given for multiple channels and is now used for each channel. "
                    "You're blending multiple cmaps. "
                    "If the plot doesn't look like you expect, it might be because your "
                    "cmaps go from a given color to 'white', and not to 'transparent'. "
                    "Therefore, the 'white' of higher layers will overlay the lower layers. "
                    "Consider using 'palette' instead."
                )

            _ax_show_and_transform(stacked, trans_data, ax, render_params.alpha)

        # 2B) Image has n channels, no palette/cmap info -> sample n categorical colors
        elif palette is None and not got_multiple_cmaps:
            # overwrite if n_channels == 2 for intuitive result
            if n_channels == 2:
                seed_colors = ["#ff0000ff", "#00ff00ff"]
            else:
                seed_colors = _get_colors_for_categorical_obs(list(range(n_channels)))

            channel_cmaps = [_get_linear_colormap([c], "k")[0] for c in seed_colors]

            # Apply cmaps to each channel and add up
            colored = np.stack([channel_cmaps[ind](layers[ch]) for ind, ch in enumerate(channels)], 0).sum(0)

            # Remove alpha channel so we can overwrite it from render_params.alpha
            colored = colored[:, :, :3]

            _ax_show_and_transform(colored, trans_data, ax, render_params.alpha)

        # 2C) Image has n channels and palette info
        elif palette is not None and not got_multiple_cmaps:
            if len(palette) != n_channels:
                raise ValueError("If 'palette' is provided, its length must match the number of channels.")

            channel_cmaps = [_get_linear_colormap([c], "k")[0] for c in palette if isinstance(c, str)]

            # Apply cmaps to each channel and add up
            colored = np.stack([channel_cmaps[i](layers[c]) for i, c in enumerate(channels)], 0).sum(0)

            # Remove alpha channel so we can overwrite it from render_params.alpha
            colored = colored[:, :, :3]

            _ax_show_and_transform(colored, trans_data, ax, render_params.alpha)

        elif palette is None and got_multiple_cmaps:
            channel_cmaps = [cp.cmap for cp in render_params.cmap_params]  # type: ignore[union-attr]

            # Apply cmaps to each channel, add up and normalize to [0, 1]
            colored = (
                np.stack([channel_cmaps[ind](layers[ch]) for ind, ch in enumerate(channels)], 0).sum(0) / n_channels
            )

            # Remove alpha channel so we can overwrite it from render_params.alpha
            colored = colored[:, :, :3]

            _ax_show_and_transform(colored, trans_data, ax, render_params.alpha)

        elif palette is not None and got_multiple_cmaps:
            raise ValueError("If 'palette' is provided, 'cmap' must be None.")


def _render_labels(
    sdata: sd.SpatialData,
    render_params: LabelsRenderParams,
    coordinate_system: str,
    ax: matplotlib.axes.SubplotBase,
    fig_params: FigParams,
    scalebar_params: ScalebarParams,
    legend_params: LegendParams,
    rasterize: bool,
) -> None:
    element = render_params.element
    table_name = render_params.table_name
    palette = render_params.palette
    color = render_params.color
    groups = render_params.groups
    scale = render_params.scale

    if render_params.outline is False:
        render_params.outline_alpha = 0

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_tables=bool(table_name),
    )

    label = sdata_filt.labels[element]
    extent = get_extent(label, coordinate_system=coordinate_system)

    # get best scale out of multiscale label
    if isinstance(label, DataTree):
        label = _multiscale_to_spatial_image(
            multiscale_image=label,
            dpi=fig_params.fig.dpi,
            width=fig_params.fig.get_size_inches()[0],
            height=fig_params.fig.get_size_inches()[1],
            scale=scale,
            is_label=True,
        )
    # rasterize spatial image if necessary to speed up performance
    if rasterize:
        label = _rasterize_if_necessary(
            image=label,
            dpi=fig_params.fig.dpi,
            width=fig_params.fig.get_size_inches()[0],
            height=fig_params.fig.get_size_inches()[1],
            coordinate_system=coordinate_system,
            extent=extent,
        )

    if table_name is None:
        instance_id = np.unique(label)
        table = None
    else:
        regions, region_key, instance_key = get_table_keys(sdata[table_name])
        table = sdata[table_name][sdata[table_name].obs[region_key].isin([element])]

        # get instance id based on subsetted table
        instance_id = table.obs[instance_key].values

    trans = get_transformation(label, get_all=True)[coordinate_system]
    affine_trans = trans.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    trans = mtransforms.Affine2D(matrix=affine_trans)
    trans_data = trans + ax.transData

    color_source_vector, color_vector, categorical = _set_color_source_vec(
        sdata=sdata_filt,
        element=label,
        element_name=element,
        value_to_plot=color,
        groups=groups,
        palette=palette,
        na_color=render_params.cmap_params.na_color,
        cmap_params=render_params.cmap_params,
        table_name=table_name,
    )

    if (render_params.fill_alpha != render_params.outline_alpha) and render_params.contour_px is not None:
        # First get the labels infill and plot them
        labels_infill = _map_color_seg(
            seg=label.values,
            cell_id=instance_id,
            color_vector=color_vector,
            color_source_vector=color_source_vector,
            cmap_params=render_params.cmap_params,
            seg_erosionpx=None,
            seg_boundaries=render_params.outline,
            na_color=render_params.cmap_params.na_color,
        )

        # Then overlay the contour
        labels_contour = _map_color_seg(
            seg=label.values,
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
            cmap=None if categorical else render_params.cmap_params.cmap,
            norm=None if categorical else render_params.cmap_params.norm,
            alpha=render_params.outline_alpha,
            origin="lower",
        )
        _cax = ax.imshow(
            labels_infill,
            rasterized=True,
            cmap=None if categorical else render_params.cmap_params.cmap,
            norm=None if categorical else render_params.cmap_params.norm,
            alpha=render_params.fill_alpha,
            origin="lower",
        )
        _cax.set_transform(trans_data)
        cax = ax.add_image(_cax)
    else:
        # Default: no alpha, contour = infill
        label = _map_color_seg(
            seg=label.values,
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
            cmap=None if categorical else render_params.cmap_params.cmap,
            norm=None if categorical else render_params.cmap_params.norm,
            alpha=render_params.fill_alpha,
            origin="lower",
        )
    _cax.set_transform(trans_data)
    cax = ax.add_image(_cax)

    _ = _decorate_axs(
        ax=ax,
        cax=cax,
        fig_params=fig_params,
        adata=table,
        value_to_plot=color,
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
