from __future__ import annotations

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
import spatial_image
import spatialdata as sd
from anndata import AnnData
from matplotlib.colors import ListedColormap, Normalize
from pandas.api.types import is_categorical_dtype
from scanpy._settings import settings as sc_settings
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
)
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
    _decorate_axs,
    _get_collection_shape,
    _get_colors_for_categorical_obs,
    _get_linear_colormap,
    _map_color_seg,
    _maybe_set_colors,
    _normalize,
    _set_color_source_vec,
    to_hex,
)
from spatialdata_plot.pp.utils import _get_instance_key, _get_region_key

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
    elements = render_params.elements

    if render_params.groups is not None:
        if isinstance(render_params.groups, str):
            render_params.groups = [render_params.groups]
        if not all(isinstance(g, str) for g in render_params.groups):
            raise TypeError("All groups must be strings.")

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_table=sdata.table is not None,
    )
    if isinstance(elements, str):
        elements = [elements]

    if elements is None:
        elements = list(sdata_filt.shapes.keys())

    for e in elements:
        shapes = sdata.shapes[e]
        n_shapes = sum([len(s) for s in shapes])

        if sdata.table is None:
            table = AnnData(None, obs=pd.DataFrame(index=pd.Index(np.arange(n_shapes), dtype=str)))
        else:
            table = sdata.table[sdata.table.obs[_get_region_key(sdata)].isin([e])]

        # get color vector (categorical or continuous)
        color_source_vector, color_vector, _ = _set_color_source_vec(
            sdata=sdata_filt,
            element=sdata_filt.shapes[e],
            element_name=e,
            value_to_plot=render_params.color,
            layer=render_params.layer,
            groups=render_params.groups,
            palette=render_params.palette,
            na_color=render_params.cmap_params.na_color,
            alpha=render_params.fill_alpha,
            cmap_params=render_params.cmap_params,
        )

        values_are_categorical = color_source_vector is not None

        # color_source_vector is None when the values aren't categorical
        if values_are_categorical and render_params.transfunc is not None:
            color_vector = render_params.transfunc(color_vector)

        norm = copy(render_params.cmap_params.norm)

        if len(color_vector) == 0:
            color_vector = [render_params.cmap_params.na_color]

        # filter by `groups`
        if render_params.groups is not None and color_source_vector is not None:
            mask = color_source_vector.isin(render_params.groups)
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
            outline_alpha=render_params.outline_alpha
            # **kwargs,
        )

        # Sets the limits of the colorbar to the values instead of [0, 1]
        if not norm and not values_are_categorical:
            _cax.set_clim(min(color_vector), max(color_vector))

        cax = ax.add_collection(_cax)

        # Apply the transformation to the PatchCollection's paths
        trans = get_transformation(sdata_filt.shapes[e], get_all=True)[coordinate_system]
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

        if not (
            len(set(color_vector)) == 1 and list(set(color_vector))[0] == to_hex(render_params.cmap_params.na_color)
        ):
            # necessary in case different shapes elements are annotated with one table
            if color_source_vector is not None:
                color_source_vector = color_source_vector.remove_unused_categories()
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
    if render_params.groups is not None:
        if isinstance(render_params.groups, str):
            render_params.groups = [render_params.groups]
        if not all(isinstance(g, str) for g in render_params.groups):
            raise TypeError("All groups must be strings.")

    elements = render_params.elements

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_table=sdata.table is not None,
    )
    if isinstance(elements, str):
        elements = [elements]

    if elements is None:
        elements = list(sdata_filt.points.keys())

    for e in elements:
        points = sdata.points[e]
        coords = ["x", "y"]
        if render_params.color is not None:
            color = [render_params.color] if isinstance(render_params.color, str) else render_params.color
            coords.extend(color)

        points = points[coords].compute()
        # points[color[0]].cat.set_categories(render_params.groups, inplace=True)
        if render_params.groups is not None:
            points = points[points[color].isin(render_params.groups).values]
            points[color[0]] = points[color[0]].cat.set_categories(render_params.groups)
        points = dask.dataframe.from_pandas(points, npartitions=1)
        sdata_filt.points[e] = PointsModel.parse(points, coordinates={"x": "x", "y": "y"})

        point_df = points[coords].compute()

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
            sdata=sdata_filt,
            element=points,
            element_name=e,
            value_to_plot=render_params.color,
            groups=render_params.groups,
            palette=render_params.palette,
            na_color=render_params.cmap_params.na_color,
            alpha=render_params.alpha,
            cmap_params=render_params.cmap_params,
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

        trans = get_transformation(sdata.points[e], get_all=True)[coordinate_system]
        affine_trans = trans.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        trans = mtransforms.Affine2D(matrix=affine_trans)

        for path in _cax.get_paths():
            path.vertices = trans.transform(path.vertices)

        if not (
            len(set(color_vector)) == 1 and list(set(color_vector))[0] == to_hex(render_params.cmap_params.na_color)
        ):
            if color_source_vector is None:
                palette = ListedColormap(dict.fromkeys(color_vector))
            else:
                palette = ListedColormap(dict.fromkeys(color_vector[~pd.Categorical(color_source_vector).isnull()]))

            _ = _decorate_axs(
                ax=ax,
                cax=cax,
                fig_params=fig_params,
                adata=adata,
                value_to_plot=render_params.color,
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

    for e in elements:
        img = sdata.images[e]
        if not isinstance(img, spatial_image.SpatialImage):
            img = Image2DModel.parse(img["scale0"].ds.to_array().squeeze(axis=0))
            logger.warning(f"Multi-scale images not yet supported, using scale0 of multi-scale image '{e}'.")

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

        # prepare transformations
        trans = get_transformation(sdata.images[e], get_all=True)[coordinate_system]
        affine_trans = trans.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        trans = mtransforms.Affine2D(matrix=affine_trans)
        trans_data = trans + ax.transData

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

            im = ax.imshow(
                layer,  # get rid of the channel dimension
                cmap=cmap,
                alpha=render_params.alpha,
            )
            im.set_transform(trans_data)

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
                im = ax.imshow(
                    np.stack([layers[c] for c in channels], axis=-1),
                    alpha=render_params.alpha,
                )
                im.set_transform(trans_data)

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

                im = ax.imshow(
                    colored,
                    alpha=render_params.alpha,
                )
                im.set_transform(trans_data)

            # 2C) Image has n channels and palette info
            elif render_params.palette is not None and not got_multiple_cmaps:
                if len(render_params.palette) != n_channels:
                    raise ValueError("If 'palette' is provided, its length must match the number of channels.")

                channel_cmaps = [_get_linear_colormap([c], "k")[0] for c in render_params.palette]

                # Apply cmaps to each channel and add up
                colored = np.stack([channel_cmaps[i](layers[c]) for i, c in enumerate(channels)], 0).sum(0)

                # Remove alpha channel so we can overwrite it from render_params.alpha
                colored = colored[:, :, :3]

                im = ax.imshow(
                    colored,
                    alpha=render_params.alpha,
                )
                im.set_transform(trans_data)

            elif render_params.palette is None and got_multiple_cmaps:
                channel_cmaps = [cp.cmap for cp in render_params.cmap_params]  # type: ignore[union-attr]

                # Apply cmaps to each channel, add up and normalize to [0, 1]
                colored = np.stack([channel_cmaps[i](layers[c]) for i, c in enumerate(channels)], 0).sum(0) / n_channels

                # Remove alpha channel so we can overwrite it from render_params.alpha
                colored = colored[:, :, :3]

                im = ax.imshow(
                    colored,
                    alpha=render_params.alpha,
                )
                im.set_transform(trans_data)

            elif render_params.palette is not None and got_multiple_cmaps:
                raise ValueError("If 'palette' is provided, 'cmap' must be None.")


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

    if render_params.groups is not None:
        if isinstance(render_params.groups, str):
            render_params.groups = [render_params.groups]
        if not all(isinstance(g, str) for g in render_params.groups):
            raise TypeError("All groups must be strings.")

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_table=sdata.table is not None,
    )
    if isinstance(elements, str):
        elements = [elements]

    if elements is None:
        elements = list(sdata_filt.labels.keys())

    for e in elements:
        label = sdata_filt.labels[e]
        if not isinstance(label, spatial_image.SpatialImage):
            label = Labels2DModel.parse(label["scale0"].ds.to_array().squeeze(axis=0))
            logger.warning(f"Multi-scale labels not yet supported, using scale0 of multi-scale label '{e}'.")

        if sdata.table is None:
            instance_id = np.unique(label)
            table = AnnData(None, obs=pd.DataFrame(index=np.arange(len(instance_id))))
        else:
            instance_key = _get_instance_key(sdata)
            region_key = _get_region_key(sdata)

            table = sdata.table[sdata.table.obs[region_key].isin([e])]

            # get instance id based on subsetted table
            instance_id = table.obs[instance_key].values

        trans = get_transformation(sdata.labels[e], get_all=True)[coordinate_system]
        affine_trans = trans.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        trans = mtransforms.Affine2D(matrix=affine_trans)
        trans_data = trans + ax.transData

        # get color vector (categorical or continuous)
        color_source_vector, color_vector, categorical = _set_color_source_vec(
            sdata=sdata_filt,
            element=sdata_filt.labels[e],
            element_name=e,
            value_to_plot=render_params.color,
            layer=render_params.layer,
            groups=render_params.groups,
            palette=render_params.palette,
            na_color=render_params.cmap_params.na_color,
            alpha=render_params.fill_alpha,
            cmap_params=render_params.cmap_params,
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

            _cax = ax.imshow(
                labels_infill,
                rasterized=True,
                cmap=render_params.cmap_params.cmap if not categorical else None,
                norm=render_params.cmap_params.norm if not categorical else None,
                alpha=render_params.fill_alpha,
                origin="lower",
            )
            _cax.set_transform(trans_data)
            cax = ax.add_image(_cax)

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
                cmap=render_params.cmap_params.cmap if not categorical else None,
                norm=render_params.cmap_params.norm if not categorical else None,
                alpha=render_params.outline_alpha,
                origin="lower",
            )
            _cax.set_transform(trans_data)
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
            )
            _cax.set_transform(trans_data)
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
