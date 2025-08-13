from __future__ import annotations

import warnings
from collections import abc
from copy import copy

import dask
import datashader as ds
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from anndata import AnnData
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from scanpy._settings import settings as sc_settings
from spatialdata import get_extent, get_values, join_spatialelement_table
from spatialdata.models import PointsModel, ShapesModel, get_table_keys
from spatialdata.transformations import get_transformation, set_transformation
from spatialdata.transformations.transformations import Identity
from xarray import DataTree

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
    _create_image_from_datashader_result,
    _datashader_aggregate_with_function,
    _datashader_map_aggregate_to_color,
    _datshader_get_how_kw_for_spread,
    _decorate_axs,
    _get_collection_shape,
    _get_colors_for_categorical_obs,
    _get_extent_and_range_for_datashader_canvas,
    _get_linear_colormap,
    _get_transformation_matrix_for_datashader,
    _hex_no_alpha,
    _is_coercable_to_float,
    _map_color_seg,
    _maybe_set_colors,
    _mpl_ax_contains_elements,
    _multiscale_to_spatial_image,
    _prepare_transformation,
    _rasterize_if_necessary,
    _set_color_source_vec,
    to_hex,
)

_Normalize = Normalize | abc.Sequence[Normalize]


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
    table_layer = render_params.table_layer

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_tables=bool(render_params.table_name),
    )

    if (table_name := render_params.table_name) is None:
        table = None
        shapes = sdata_filt[element]
    else:
        element_dict, joined_table = join_spatialelement_table(
            sdata, spatial_element_names=element, table_name=table_name, how="inner"
        )
        sdata_filt[element] = shapes = element_dict[element]
        joined_table.uns["spatialdata_attrs"]["region"] = (
            joined_table.obs[joined_table.uns["spatialdata_attrs"]["region_key"]].unique().tolist()
        )
        sdata_filt[table_name] = table = joined_table

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
        table_layer=table_layer,
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

    # Using dict.fromkeys here since set returns in arbitrary order
    # remove the color of NaN values, else it might be assigned to a category
    # order of color in the palette should agree to order of occurence
    if color_source_vector is None:
        palette = ListedColormap(dict.fromkeys(color_vector))
    else:
        palette = ListedColormap(dict.fromkeys(color_vector[~pd.Categorical(color_source_vector).isnull()]))

    if len(set(color_vector)) != 1 or list(set(color_vector))[0] != to_hex(render_params.cmap_params.na_color):
        # necessary in case different shapes elements are annotated with one table
        if color_source_vector is not None and col_for_color is not None:
            color_source_vector = color_source_vector.remove_unused_categories()

        # False if user specified color-like with 'color' parameter
        colorbar = False if col_for_color is None else legend_params.colorbar

    # Apply the transformation to the PatchCollection's paths
    trans, trans_data = _prepare_transformation(sdata_filt.shapes[element], coordinate_system)

    shapes = gpd.GeoDataFrame(shapes, geometry="geometry")

    # Determine which method to use for rendering
    method = render_params.method

    if method is None:
        method = "datashader" if len(shapes) > 10000 else "matplotlib"

    if method != "matplotlib":
        # we only notify the user when we switched away from matplotlib
        logger.info(
            f"Using '{method}' backend with '{render_params.ds_reduction}' as reduction"
            " method to speed up plotting. Depending on the reduction method, the value"
            " range of the plot might change. Set method to 'matplotlib' to disable"
            " this behaviour."
        )

    if method == "datashader":
        _geometry = shapes["geometry"]
        is_point = _geometry.type == "Point"

        # Handle circles encoded as points with radius
        if is_point.any():
            scale = shapes[is_point]["radius"] * render_params.scale
            sdata_filt.shapes[element].loc[is_point, "geometry"] = _geometry[is_point].buffer(scale.to_numpy())

        # apply transformations to the individual points
        element_trans = get_transformation(sdata_filt.shapes[element], to_coordinate_system=coordinate_system)
        tm = _get_transformation_matrix_for_datashader(element_trans)
        transformed_element = sdata_filt.shapes[element].transform(
            lambda x: (np.hstack([x, np.ones((x.shape[0], 1))]) @ tm)[:, :2]
        )
        transformed_element = ShapesModel.parse(
            gpd.GeoDataFrame(
                data=sdata_filt.shapes[element].drop("geometry", axis=1),
                geometry=transformed_element,
            )
        )

        plot_width, plot_height, x_ext, y_ext, factor = _get_extent_and_range_for_datashader_canvas(
            transformed_element, "global", ax, fig_params
        )

        cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_ext, y_range=y_ext)

        # in case we are coloring by a column in table
        if col_for_color is not None and col_for_color not in transformed_element.columns:
            transformed_element[col_for_color] = color_vector if color_source_vector is None else color_source_vector
        # Render shapes with datashader
        color_by_categorical = col_for_color is not None and color_source_vector is not None
        aggregate_with_reduction = None
        if col_for_color is not None and (render_params.groups is None or len(render_params.groups) > 1):
            if color_by_categorical:
                agg = cvs.polygons(
                    transformed_element,
                    geometry="geometry",
                    agg=ds.by(col_for_color, ds.count()),
                )
            else:
                reduction_name = render_params.ds_reduction if render_params.ds_reduction is not None else "mean"
                logger.info(
                    f'Using the datashader reduction "{reduction_name}". "max" will give an output very close '
                    "to the matplotlib result."
                )
                agg = _datashader_aggregate_with_function(
                    render_params.ds_reduction,
                    cvs,
                    transformed_element,
                    col_for_color,
                    "shapes",
                )
                # save min and max values for drawing the colorbar
                aggregate_with_reduction = (agg.min(), agg.max())
        else:
            agg = cvs.polygons(transformed_element, geometry="geometry", agg=ds.count())
        # render outlines if needed
        if (render_outlines := render_params.outline_alpha) > 0:
            agg_outlines = cvs.line(
                transformed_element,
                geometry="geometry",
                line_width=render_params.outline_params.linewidth,
            )

        ds_span = None
        if norm.vmin is not None or norm.vmax is not None:
            norm.vmin = np.min(agg) if norm.vmin is None else norm.vmin
            norm.vmax = np.max(agg) if norm.vmax is None else norm.vmax
            ds_span = [norm.vmin, norm.vmax]
            if norm.vmin == norm.vmax:
                # edge case, value vmin is rendered as the middle of the cmap
                ds_span = [0, 1]
                if norm.clip:
                    agg = (agg - agg) + 0.5
                else:
                    agg = agg.where((agg >= norm.vmin) | (np.isnan(agg)), other=-1)
                    agg = agg.where((agg <= norm.vmin) | (np.isnan(agg)), other=2)
                    agg = agg.where((agg != norm.vmin) | (np.isnan(agg)), other=0.5)

        color_key = (
            [_hex_no_alpha(x) for x in color_vector.categories.values]
            if (type(color_vector) is pd.core.arrays.categorical.Categorical)
            and (len(color_vector.categories.values) > 1)
            else None
        )

        if color_by_categorical or col_for_color is None:
            ds_cmap = None
            if color_vector is not None:
                ds_cmap = color_vector[0]
                if isinstance(ds_cmap, str) and ds_cmap[0] == "#":
                    ds_cmap = _hex_no_alpha(ds_cmap)

            ds_result = _datashader_map_aggregate_to_color(
                agg,
                cmap=ds_cmap,
                color_key=color_key,
                min_alpha=np.min([254, render_params.fill_alpha * 255]),
            )  # prevent min_alpha == 255, bc that led to fully colored test plots instead of just colored points/shapes
        elif aggregate_with_reduction is not None:  # to shut up mypy
            ds_cmap = render_params.cmap_params.cmap
            # in case all elements have the same value X: we render them using cmap(0.0),
            # using an artificial "span" of [X, X + 1] for the color bar
            # else: all elements would get alpha=0 and the color bar would have a weird range
            if aggregate_with_reduction[0] == aggregate_with_reduction[1]:
                ds_cmap = matplotlib.colors.to_hex(render_params.cmap_params.cmap(0.0), keep_alpha=False)
                aggregate_with_reduction = (
                    aggregate_with_reduction[0],
                    aggregate_with_reduction[0] + 1,
                )

            ds_result = _datashader_map_aggregate_to_color(
                agg,
                cmap=ds_cmap,
                min_alpha=np.min([254, render_params.fill_alpha * 255]),
                span=ds_span,
                clip=norm.clip,
            )  # prevent min_alpha == 255, bc that led to fully colored test plots instead of just colored points/shapes

        # shade outlines if needed
        outline_color = render_params.outline_params.outline_color
        if isinstance(outline_color, str) and outline_color.startswith("#") and len(outline_color) == 9:
            logger.info(
                "alpha component of given RGBA value for outline color is discarded, because outline_alpha"
                " takes precedent."
            )
            outline_color = outline_color[:-2]

        if render_outlines:
            ds_outlines = ds.tf.shade(
                agg_outlines,
                cmap=outline_color,
                min_alpha=np.min([254, render_params.outline_alpha * 255]),
                how="linear",
            )  # prevent min_alpha == 255, bc that led to fully colored test plots instead of just colored points/shapes

        rgba_image, trans_data = _create_image_from_datashader_result(ds_result, factor, ax)
        _cax = _ax_show_and_transform(
            rgba_image,
            trans_data,
            ax,
            zorder=render_params.zorder,
            alpha=render_params.fill_alpha,
            extent=x_ext + y_ext,
        )
        # render outline image if needed
        if render_outlines:
            rgba_image, trans_data = _create_image_from_datashader_result(ds_outlines, factor, ax)
            _ax_show_and_transform(
                rgba_image,
                trans_data,
                ax,
                zorder=render_params.zorder,
                alpha=render_params.outline_alpha,
                extent=x_ext + y_ext,
            )

        cax = None
        if aggregate_with_reduction is not None:
            vmin = aggregate_with_reduction[0].values if norm.vmin is None else norm.vmin
            vmax = aggregate_with_reduction[1].values if norm.vmin is None else norm.vmax
            if (norm.vmin is not None or norm.vmax is not None) and norm.vmin == norm.vmax:
                assert norm.vmin is not None
                assert norm.vmax is not None
                # value (vmin=vmax) is placed in the middle of the colorbar so that we can distinguish it from over and
                # under values in case clip=True or clip=False with cmap(under)=cmap(0) & cmap(over)=cmap(1)
                vmin = norm.vmin - 0.5
                vmax = norm.vmin + 0.5
            cax = ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
                cmap=render_params.cmap_params.cmap,
            )

    elif method == "matplotlib":
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
            zorder=render_params.zorder,
            # **kwargs,
        )
        cax = ax.add_collection(_cax)

        # Transform the paths in PatchCollection
        for path in _cax.get_paths():
            path.vertices = trans.transform(path.vertices)

    if not values_are_categorical:
        # If the user passed a Normalize object with vmin/vmax we'll use those,
        # if not we'll use the min/max of the color_vector
        _cax.set_clim(
            vmin=render_params.cmap_params.norm.vmin or min(color_vector),
            vmax=render_params.cmap_params.norm.vmax or max(color_vector),
        )

    if len(set(color_vector)) != 1 or list(set(color_vector))[0] != to_hex(render_params.cmap_params.na_color):
        # necessary in case different shapes elements are annotated with one table
        if color_source_vector is not None and render_params.col_for_color is not None:
            color_source_vector = color_source_vector.remove_unused_categories()

        # False if user specified color-like with 'color' parameter
        colorbar = False if render_params.col_for_color is None else legend_params.colorbar

        _ = _decorate_axs(
            ax=ax,
            cax=cax,
            fig_params=fig_params,
            adata=table,
            value_to_plot=col_for_color,
            color_source_vector=color_source_vector,
            color_vector=color_vector,
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
    table_layer = render_params.table_layer
    color = render_params.color
    groups = render_params.groups
    palette = render_params.palette

    sdata_filt = sdata.filter_by_coordinate_system(
        coordinate_system=coordinate_system,
        filter_tables=bool(table_name),
    )

    points = sdata.points[element]
    coords = ["x", "y"]

    if table_name is not None and col_for_color not in points.columns:
        warnings.warn(
            f"Annotating points with {col_for_color} which is stored in the table `{table_name}`. "
            f"To improve performance, it is advisable to store point annotations directly in the .parquet file.",
            UserWarning,
            stacklevel=2,
        )

    if col_for_color is None or (
        table_name is not None
        and (col_for_color in sdata_filt[table_name].obs.columns or col_for_color in sdata_filt[table_name].var_names)
    ):
        points = points[coords].compute()
        if (
            col_for_color
            and col_for_color in sdata_filt[table_name].obs.columns
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
        if col_for_color in points.columns:
            points_color_values = points[col_for_color]
        else:
            points_color_values = get_values(
                value_key=col_for_color,
                sdata=sdata_filt,
                element_name=element,
                table_name=table_name,
                table_layer=table_layer,
            )
            points_color_values = points.merge(points_color_values, how="left", left_index=True, right_index=True)[
                col_for_color
            ]
        points = points[points_color_values.isin(groups)]
        if len(points) <= 0:
            raise ValueError(f"None of the groups {groups} could be found in the column '{col_for_color}'.")

    # we construct an anndata to hack the plotting functions
    if table_name is None:
        adata = AnnData(
            X=points[["x", "y"]].values,
            obs=points[coords].reset_index(),
            dtype=points[["x", "y"]].values.dtype,
        )
    else:
        adata_obs = sdata_filt[table_name].obs
        # if the points are colored by values in X (or a different layer), add the values to obs
        if col_for_color in sdata_filt[table_name].var_names:
            if table_layer is None:
                adata_obs[col_for_color] = sdata_filt[table_name][:, col_for_color].X.flatten().copy()
            else:
                adata_obs[col_for_color] = sdata_filt[table_name][:, col_for_color].layers[table_layer].flatten().copy()
        if groups is not None:
            adata_obs = adata_obs[adata_obs[col_for_color].isin(groups)]
        adata = AnnData(
            X=points[["x", "y"]].values,
            obs=adata_obs,
            dtype=points[["x", "y"]].values.dtype,
            uns=sdata_filt[table_name].uns,
        )
        sdata_filt[table_name] = adata

    # we can modify the sdata because of dealing with a copy

    # Convert back to dask dataframe to modify sdata
    transformation_in_cs = sdata_filt.points[element].attrs["transform"][coordinate_system]
    points = dask.dataframe.from_pandas(points, npartitions=1)
    sdata_filt.points[element] = PointsModel.parse(points, coordinates={"x": "x", "y": "y"})
    # restore transformation in coordinate system of interest
    set_transformation(
        element=sdata_filt.points[element],
        transformation=transformation_in_cs,
        to_coordinate_system=coordinate_system,
    )

    if col_for_color is not None:
        assert isinstance(col_for_color, str)
        cols = sc.get.obs_df(adata, [col_for_color])
        # maybe set color based on type
        if isinstance(cols[col_for_color].dtype, pd.CategoricalDtype):
            _maybe_set_colors(
                source=adata,
                target=adata,
                key=col_for_color,
                palette=palette,
            )

    # when user specified a single color, we emulate the form of `na_color` and use it
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
        alpha=render_params.alpha,
        table_name=table_name,
        render_type="points",
    )

    # color_source_vector is None when the values aren't categorical
    if color_source_vector is None and render_params.transfunc is not None:
        color_vector = render_params.transfunc(color_vector)

    trans, trans_data = _prepare_transformation(sdata.points[element], coordinate_system, ax)

    norm = copy(render_params.cmap_params.norm)

    method = render_params.method

    if method is None:
        method = "datashader" if len(points) > 10000 else "matplotlib"

    if method != "matplotlib":
        # we only notify the user when we switched away from matplotlib
        logger.info(
            f"Using '{method}' backend with '{render_params.ds_reduction}' as reduction"
            " method to speed up plotting. Depending on the reduction method, the value"
            " range of the plot might change. Set method to 'matplotlib' do disable"
            " this behaviour."
        )

    if method == "datashader":
        # NOTE: s in matplotlib is in units of points**2
        # use dpi/100 as a factor for cases where dpi!=100
        px = int(np.round(np.sqrt(render_params.size) * (fig_params.fig.dpi / 100)))

        # apply transformations
        transformed_element = PointsModel.parse(
            trans.transform(sdata_filt.points[element][["x", "y"]]),
            annotation=sdata_filt.points[element][sdata_filt.points[element].columns.drop(["x", "y"])],
            transformations={coordinate_system: Identity()},
        )

        plot_width, plot_height, x_ext, y_ext, factor = _get_extent_and_range_for_datashader_canvas(
            transformed_element, coordinate_system, ax, fig_params
        )

        # use datashader for the visualization of points
        cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_ext, y_range=y_ext)

        color_by_categorical = col_for_color is not None and transformed_element[col_for_color].values.dtype in (
            object,
            "categorical",
        )
        if color_by_categorical and transformed_element[col_for_color].values.dtype == object:
            transformed_element[col_for_color] = transformed_element[col_for_color].astype("category")
        aggregate_with_reduction = None
        if col_for_color is not None and (render_params.groups is None or len(render_params.groups) > 1):
            if color_by_categorical:
                agg = cvs.points(transformed_element, "x", "y", agg=ds.by(col_for_color, ds.count()))
            else:
                reduction_name = render_params.ds_reduction if render_params.ds_reduction is not None else "sum"
                logger.info(
                    f'Using the datashader reduction "{reduction_name}". "max" will give an output very close '
                    "to the matplotlib result."
                )
                agg = _datashader_aggregate_with_function(
                    render_params.ds_reduction,
                    cvs,
                    transformed_element,
                    col_for_color,
                    "points",
                )
                # save min and max values for drawing the colorbar
                aggregate_with_reduction = (agg.min(), agg.max())
        else:
            agg = cvs.points(transformed_element, "x", "y", agg=ds.count())

        ds_span = None
        if norm.vmin is not None or norm.vmax is not None:
            norm.vmin = np.min(agg) if norm.vmin is None else norm.vmin
            norm.vmax = np.max(agg) if norm.vmax is None else norm.vmax
            ds_span = [norm.vmin, norm.vmax]
            if norm.vmin == norm.vmax:
                ds_span = [0, 1]
                if norm.clip:
                    # all data is mapped to 0.5
                    agg = (agg - agg) + 0.5
                else:
                    # values equal to norm.vmin are mapped to 0.5, the rest to -1 or 2
                    agg = agg.where((agg >= norm.vmin) | (np.isnan(agg)), other=-1)
                    agg = agg.where((agg <= norm.vmin) | (np.isnan(agg)), other=2)
                    agg = agg.where((agg != norm.vmin) | (np.isnan(agg)), other=0.5)

        color_key = (
            list(color_vector.categories.values)
            if (type(color_vector) is pd.core.arrays.categorical.Categorical)
            and (len(color_vector.categories.values) > 1)
            else None
        )

        # remove alpha from color if it's hex
        if color_key is not None and all(len(x) == 9 for x in color_key) and color_key[0][0] == "#":
            color_key = [x[:-2] for x in color_key]
        if isinstance(color_vector[0], str) and (
            color_vector is not None and all(len(x) == 9 for x in color_vector) and color_vector[0][0] == "#"
        ):
            color_vector = np.asarray([x[:-2] for x in color_vector])

        if color_by_categorical or col_for_color is None:
            ds_result = _datashader_map_aggregate_to_color(
                ds.tf.spread(agg, px=px),
                cmap=color_vector[0],
                color_key=color_key,
                min_alpha=np.min([254, render_params.alpha * 255]),
            )  # prevent min_alpha == 255, bc that led to fully colored test plots instead of just colored points/shapes
        else:
            spread_how = _datshader_get_how_kw_for_spread(render_params.ds_reduction)
            agg = ds.tf.spread(agg, px=px, how=spread_how)
            aggregate_with_reduction = (agg.min(), agg.max())

            ds_cmap = render_params.cmap_params.cmap
            # in case all elements have the same value X: we render them using cmap(0.0),
            # using an artificial "span" of [X, X + 1] for the color bar
            # else: all elements would get alpha=0 and the color bar would have a weird range
            if aggregate_with_reduction[0] == aggregate_with_reduction[1] and (ds_span is None or ds_span != [0, 1]):
                ds_cmap = matplotlib.colors.to_hex(render_params.cmap_params.cmap(0.0), keep_alpha=False)
                aggregate_with_reduction = (
                    aggregate_with_reduction[0],
                    aggregate_with_reduction[0] + 1,
                )

            ds_result = _datashader_map_aggregate_to_color(
                agg,
                cmap=ds_cmap,
                span=ds_span,
                clip=norm.clip,
                min_alpha=np.min([254, render_params.alpha * 255]),
            )  # prevent min_alpha == 255, bc that led to fully colored test plots instead of just colored points/shapes

        rgba_image, trans_data = _create_image_from_datashader_result(ds_result, factor, ax)
        _ax_show_and_transform(
            rgba_image,
            trans_data,
            ax,
            zorder=render_params.zorder,
            alpha=render_params.alpha,
            extent=x_ext + y_ext,
        )

        cax = None
        if aggregate_with_reduction is not None:
            vmin = aggregate_with_reduction[0].values if norm.vmin is None else norm.vmin
            vmax = aggregate_with_reduction[1].values if norm.vmax is None else norm.vmax
            if (norm.vmin is not None or norm.vmax is not None) and norm.vmin == norm.vmax:
                assert norm.vmin is not None
                assert norm.vmax is not None
                # value (vmin=vmax) is placed in the middle of the colorbar so that we can distinguish it from over and
                # under values in case clip=True or clip=False with cmap(under)=cmap(0) & cmap(over)=cmap(1)
                vmin = norm.vmin - 0.5
                vmax = norm.vmin + 0.5
            cax = ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
                cmap=render_params.cmap_params.cmap,
            )

    elif method == "matplotlib":
        # update axis limits if plot was empty before (necessary if datashader comes after)
        update_parameters = not _mpl_ax_contains_elements(ax)
        _cax = ax.scatter(
            adata[:, 0].X.flatten(),
            adata[:, 1].X.flatten(),
            s=render_params.size,
            c=color_vector,
            rasterized=sc_settings._vector_friendly,
            cmap=render_params.cmap_params.cmap,
            norm=norm,
            alpha=render_params.alpha,
            transform=trans_data,
            zorder=render_params.zorder,
        )
        cax = ax.add_collection(_cax)
        if update_parameters:
            # necessary if points are plotted with mpl first and then with datashader
            extent = get_extent(sdata_filt.points[element], coordinate_system=coordinate_system)
            ax.set_xbound(extent["x"])
            ax.set_ybound(extent["y"])

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
            color_vector=color_vector,
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

    channels = img.coords["c"].values.tolist() if render_params.channel is None else render_params.channel

    # the channel parameter has been previously validated, so when not None, render_params.channel is a list
    assert isinstance(channels, list)
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

    _, trans_data = _prepare_transformation(img, coordinate_system, ax)

    # 1) Image has only 1 channel
    if n_channels == 1 and not isinstance(render_params.cmap_params, list):
        layer = img.sel(c=channels[0]).squeeze() if isinstance(channels[0], str) else img.isel(c=channels[0]).squeeze()

        cmap = (
            _get_linear_colormap(palette, "k")[0]
            if isinstance(palette, list) and all(isinstance(p, str) for p in palette)
            else render_params.cmap_params.cmap
        )

        # Overwrite alpha in cmap: https://stackoverflow.com/a/10127675
        cmap._init()
        cmap._lut[:, -1] = render_params.alpha

        # norm needs to be passed directly to ax.imshow(). If we normalize before, that method would always clip.
        _ax_show_and_transform(
            layer,
            trans_data,
            ax,
            cmap=cmap,
            zorder=render_params.zorder,
            norm=render_params.cmap_params.norm,
        )

        if legend_params.colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=render_params.cmap_params.norm)
            fig_params.fig.colorbar(sm, ax=ax)

    # 2) Image has any number of channels but 1
    else:
        layers = {}
        for ch_index, c in enumerate(channels):
            layers[c] = img.sel(c=c).copy(deep=True).squeeze()

            if not isinstance(render_params.cmap_params, list):
                if render_params.cmap_params.norm is not None:
                    layers[c] = render_params.cmap_params.norm(layers[c])
            else:
                if render_params.cmap_params[ch_index].norm is not None:
                    layers[c] = render_params.cmap_params[ch_index].norm(layers[c])

        # 2A) Image has 3 channels, no palette info, and no/only one cmap was given
        if palette is None and n_channels == 3 and not isinstance(render_params.cmap_params, list):
            if render_params.cmap_params.cmap_is_default:  # -> use RGB
                stacked = np.stack([layers[c] for c in channels], axis=-1)
            else:  # -> use given cmap for each channel
                channel_cmaps = [render_params.cmap_params.cmap] * n_channels
                stacked = (
                    np.stack(
                        [channel_cmaps[ind](layers[ch]) for ind, ch in enumerate(channels)],
                        0,
                    ).sum(0)
                    / n_channels
                )
                stacked = stacked[:, :, :3]
                logger.warning(
                    "One cmap was given for multiple channels and is now used for each channel. "
                    "You're blending multiple cmaps. "
                    "If the plot doesn't look like you expect, it might be because your "
                    "cmaps go from a given color to 'white', and not to 'transparent'. "
                    "Therefore, the 'white' of higher layers will overlay the lower layers. "
                    "Consider using 'palette' instead."
                )

            _ax_show_and_transform(
                stacked,
                trans_data,
                ax,
                render_params.alpha,
                zorder=render_params.zorder,
            )

        # 2B) Image has n channels, no palette/cmap info -> sample n categorical colors
        elif palette is None and not got_multiple_cmaps:
            # overwrite if n_channels == 2 for intuitive result
            if n_channels == 2:
                seed_colors = ["#ff0000ff", "#00ff00ff"]
            else:
                seed_colors = _get_colors_for_categorical_obs(list(range(n_channels)))

            channel_cmaps = [_get_linear_colormap([c], "k")[0] for c in seed_colors]
            colored = np.stack([channel_cmaps[ind](layers[ch]) for ind, ch in enumerate(channels)], 0).sum(0)
            colored = colored[:, :, :3]

            _ax_show_and_transform(
                colored,
                trans_data,
                ax,
                render_params.alpha,
                zorder=render_params.zorder,
            )

        # 2C) Image has n channels and palette info
        elif palette is not None and not got_multiple_cmaps:
            if len(palette) != n_channels:
                raise ValueError("If 'palette' is provided, its length must match the number of channels.")

            channel_cmaps = [_get_linear_colormap([c], "k")[0] for c in palette if isinstance(c, str)]
            colored = np.stack([channel_cmaps[i](layers[c]) for i, c in enumerate(channels)], 0).sum(0)
            colored = colored[:, :, :3]

            _ax_show_and_transform(
                colored,
                trans_data,
                ax,
                render_params.alpha,
                zorder=render_params.zorder,
            )

        elif palette is None and got_multiple_cmaps:
            channel_cmaps = [cp.cmap for cp in render_params.cmap_params]  # type: ignore[union-attr]
            colored = (
                np.stack(
                    [channel_cmaps[ind](layers[ch]) for ind, ch in enumerate(channels)],
                    0,
                ).sum(0)
                / n_channels
            )
            colored = colored[:, :, :3]

            _ax_show_and_transform(
                colored,
                trans_data,
                ax,
                render_params.alpha,
                zorder=render_params.zorder,
            )

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
    table_layer = render_params.table_layer
    palette = render_params.palette
    color = render_params.color
    groups = render_params.groups
    scale = render_params.scale

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

        # the above adds a useless c dimension of 1 (y, x) -> (1, y, x)
        label = label.squeeze()

    if table_name is None:
        instance_id = np.unique(label)
        table = None
    else:
        _, region_key, instance_key = get_table_keys(sdata[table_name])
        table = sdata[table_name][sdata[table_name].obs[region_key].isin([element])]

        # get instance id based on subsetted table
        instance_id = np.unique(table.obs[instance_key].values)

    _, trans_data = _prepare_transformation(label, coordinate_system, ax)

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
        table_layer=table_layer,
    )

    # rasterize could have removed labels from label
    # only problematic if color is specified
    if rasterize and color is not None:
        labels_in_rasterized_image = np.unique(label.values)
        mask = np.isin(instance_id, labels_in_rasterized_image)
        instance_id = instance_id[mask]
        color_vector = color_vector[mask]
        if isinstance(color_vector.dtype, pd.CategoricalDtype):
            color_vector = color_vector.remove_unused_categories()
            assert color_source_vector is not None
            color_source_vector = color_source_vector[mask]
        else:
            assert color_source_vector is None

    def _draw_labels(seg_erosionpx: int | None, seg_boundaries: bool, alpha: float) -> matplotlib.image.AxesImage:
        labels = _map_color_seg(
            seg=label.values,
            cell_id=instance_id,
            color_vector=color_vector,
            color_source_vector=color_source_vector,
            cmap_params=render_params.cmap_params,
            seg_erosionpx=seg_erosionpx,
            seg_boundaries=seg_boundaries,
            na_color=render_params.cmap_params.na_color,
            na_color_modified_by_user=render_params.cmap_params.na_color_modified_by_user,
        )

        _cax = ax.imshow(
            labels,
            rasterized=True,
            cmap=None if categorical else render_params.cmap_params.cmap,
            norm=None if categorical else render_params.cmap_params.norm,
            alpha=alpha,
            origin="lower",
            zorder=render_params.zorder,
        )
        _cax.set_transform(trans_data)
        cax = ax.add_image(_cax)
        return cax  # noqa: RET504

    # default case: no contour, just fill
    # since contour_px is passed to skimage.morphology.erosion to create the contour,
    # any border thickness is only within the label, not outside. Therefore, the case
    # of fill_alpha == outline_alpha is equivalent to fill-only
    if (render_params.fill_alpha > 0.0 and render_params.outline_alpha == 0.0) or (
        render_params.fill_alpha == render_params.outline_alpha
    ):
        cax = _draw_labels(seg_erosionpx=None, seg_boundaries=False, alpha=render_params.fill_alpha)
        alpha_to_decorate_ax = render_params.fill_alpha

    # outline-only case
    elif render_params.fill_alpha == 0.0 and render_params.outline_alpha > 0.0:
        cax = _draw_labels(
            seg_erosionpx=render_params.contour_px,
            seg_boundaries=True,
            alpha=render_params.outline_alpha,
        )
        alpha_to_decorate_ax = render_params.outline_alpha

    # pretty case: both outline and infill
    elif render_params.fill_alpha > 0.0 and render_params.outline_alpha > 0.0:
        # first plot the infill ...
        cax_infill = _draw_labels(seg_erosionpx=None, seg_boundaries=False, alpha=render_params.fill_alpha)

        # ... then overlay the contour
        cax_contour = _draw_labels(
            seg_erosionpx=render_params.contour_px,
            seg_boundaries=True,
            alpha=render_params.outline_alpha,
        )

        # pass the less-transparent _cax for the legend
        cax = cax_infill if render_params.fill_alpha > render_params.outline_alpha else cax_contour
        alpha_to_decorate_ax = max(render_params.fill_alpha, render_params.outline_alpha)

    else:
        raise ValueError("Parameters 'fill_alpha' and 'outline_alpha' cannot both be 0.")

    _ = _decorate_axs(
        ax=ax,
        cax=cax,
        fig_params=fig_params,
        adata=table,
        value_to_plot=color,
        color_source_vector=color_source_vector,
        color_vector=color_vector,
        palette=palette,
        alpha=alpha_to_decorate_ax,
        na_color=render_params.cmap_params.na_color,
        legend_fontsize=legend_params.legend_fontsize,
        legend_fontweight=legend_params.legend_fontweight,
        legend_loc=legend_params.legend_loc,
        legend_fontoutline=legend_params.legend_fontoutline,
        na_in_legend=(legend_params.na_in_legend if groups is None else len(groups) == len(set(color_vector))),
        colorbar=legend_params.colorbar,
        scalebar_dx=scalebar_params.scalebar_dx,
        scalebar_units=scalebar_params.scalebar_units,
        # scalebar_kwargs=scalebar_params.scalebar_kwargs,
    )
