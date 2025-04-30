from typing import Any
from uuid import uuid4

import spatialdata
from spatialdata import SpatialData
from spatialdata._io.format import CurrentPointsFormat, CurrentRasterFormat, CurrentShapesFormat
from spatialdata.models import get_table_keys

from spatialdata_plot.pl.render_params import (
    CmapParams,
    ImageRenderParams,
    LabelsRenderParams,
    PointsRenderParams,
    ShapesRenderParams,
)

Params = ImageRenderParams | LabelsRenderParams | PointsRenderParams | ShapesRenderParams


def _add_datashade_transform(
    params: PointsRenderParams | ShapesRenderParams, data_object: dict[str, Any]
) -> dict[str, Any]:
    """Add a datashade transform to a vega like data object.

    The datashade transform is specifically added in case a SpatialData points or shapes element was
    visualized using datashader.

    Parameters
    ----------
    params: PointsRenderParams | ShapesRenderParams
        The parameters used to visualize the particular SpatialData points or shapes element.
    data_object:
        The vega like data object pertaining to a particular SpatialData points or shapes element.

    Returns
    -------
    data_object: dict[str, Any]
        The data object with the added datashade transform.
    """
    if params.ds_reduction is None:
        return data_object

    reduction_map = {"std": "stdev", "var": "variance"}
    ds_reduction = reduction_map.get(params.ds_reduction, params.ds_reduction)

    last_transform = data_object["transform"][-1]

    if last_transform["type"] == "formula":
        field = as_field = data_object["transform"][-1]["as"]
    elif params.col_for_color:
        field = as_field = params.col_for_color or "*"
        if field == "*":
            as_field = "count"

    data_object["transform"].append({"type": "aggregate", "field": [field], "ops": [ds_reduction], "as": [as_field]})
    data_object = add_norm_transform_to_data_object(params.cmap_params, data_object)
    # if data_object["transform"][-1]["type"] == "formula":
    #     field = data_object["transform"][-1]["as"]
    if isinstance(params, PointsRenderParams):
        data_object["transform"].append(
            {"type": "spread", "field": [as_field], "px": params.ds_pixel_spread, "as": [as_field]}
        )

    return data_object


def add_norm_transform_to_data_object(
    cmap_params: CmapParams | list[CmapParams], data_object: dict[str, Any]
) -> dict[str, Any]:
    """Add a normalization transform to a vega like derived data object.

    Parameters
    ----------
    cmap_params: CmapParams | list[CmapParams]
        The render parameters used to plot the particular spatialdata element.
    data_object: dict[str, Any]
        The vega like derived data object.

    Returns
    -------
    The vega like derived data object with an added normalization transform if normalization was defined
    in the render parameters.
    """
    norm = cmap_params.norm if not isinstance(cmap_params, list) else cmap_params[0].norm
    last_transform = data_object["transform"][-1]
    field = last_transform["as"][0] if last_transform["type"] == "aggregate" else "value"

    if norm.vmin is None and norm.vmax is None:
        return data_object

    norm_expr = f"(datum.{field} - {norm.vmin}) / ({norm.vmax} - {norm.vmin})"
    if norm.clip:
        norm_expr = f"clamp({norm_expr}, 0, 1)"

    data_object["transform"].append({"type": "formula", "expr": norm_expr, "as": str(uuid4())})
    return data_object


def create_base_level_sdata_object(url: str) -> dict[str, Any]:
    """Create the vega json object for the SpatialData zarr store.

    Parameters
    ----------
    url : Path
        The location of the SpatialData zarr store.

    This config is to be added to the vega data field block.
    """
    return {
        "name": str(uuid4()),
        "url": url,
        "format": {"type": "SpatialData", "version": spatialdata.__version__},
    }


def create_table_data_object(table_name: str, base_uuid: str, table_layer: str | None) -> dict[str, Any]:
    """Create the vega like data object for a spatialdata table.

    Parameters
    ----------
    table_name : str
        Name of the table in the SpatialData object.
    base_uuid : str
        The ID or name of the vega like data object pertaining to the SpatialData zarr store containing
        the table to be added.
    table_layer: str | None
        The layer of the anndata table to be used.

    Returns
    -------
    The vega like data object for the SpatialData table.
    """
    table_object = {
        "name": str(uuid4()),
        "format": {"type": "spatialdata_table", "version": 0.1},
        "source": base_uuid,
        "transform": [{"type": "filter_element", "expr": table_name}],
    }
    if table_layer is not None:
        table_object["transform"].append({"type": "filter_layer", "expr": table_layer})  # type: ignore[attr-defined]
    return table_object


def _add_table_lookup(
    sdata: SpatialData, params: Params, data_object: dict[str, Any], table_id: str | None
) -> dict[str, Any]:
    """Add a lookup transform to a vega like derived data object.

    Parameters
    ----------
    sdata : SpatialData
        The spatialdata object containing the table.
    params: params
        The render parameters used to plot the particular spatialdata element.
    data_object: dict[str, Any]
        The vega like derived data object.
    table_id: str
        The ID of the vega data object pertaining to the spatialdata table.

    Returns
    -------
    The vega like derived data object with the added lookup transform.
    """
    if table_id and not isinstance(params, ImageRenderParams):
        _, _, instance_key = get_table_keys(sdata[params.table_name])
        if isinstance(params, LabelsRenderParams):
            color = params.color
        else:
            color = params.color if params.color else params.col_for_color
        data_object["transform"].append(
            {
                "type": "lookup",
                "from": table_id,
                "key": instance_key,
                "fields": ["instance_ids"],
                "values": [color],
                "as": [color],
                "default": None,
            }
        )
    return data_object


def _create_base_derived_data_object(element_name: str, call: str, cs: str, base_uuid: str) -> dict[str, Any]:
    """Create the base vega like object of derived SpatialData elements.

    The object returned by this function contains the fields shared by all the vega like data objects, no matter
    what type of SpatialData element it pertains to.

    Parameters
    ----------
    element_name: str
        The name of the SpatialData element
    call: str
        The render call from spatialdata plot, either render_images, render_labels, render_points
        or render_shapes, prefixed by n_ where n is the index of the render call starting from 0.
    cs: str
        The name of the coordinate system in which the SpatialData element was plotted.
    base_uuid: str
        Unique identifier used to refer to the base level SpatialData zarr store in the vega
        like view configuration.

    Returns
    -------
    A base vega like data object for derived SpatialData elements.
    """
    format_types = {
        "render_images": (CurrentRasterFormat.__name__, CurrentRasterFormat().spatialdata_format_version),
        "render_labels": (CurrentRasterFormat.__name__, CurrentRasterFormat().spatialdata_format_version),
        "render_points": (CurrentPointsFormat.__name__, CurrentPointsFormat().spatialdata_format_version),
        "render_shapes": (CurrentShapesFormat.__name__, CurrentShapesFormat().spatialdata_format_version),
    }

    for key, fmt in format_types.items():
        if key in call:
            format_object = {"type": fmt[0], "version": fmt[1]}
            break
    else:
        raise ValueError(f"Unknown call: {call}")

    return {
        "name": element_name + "_" + str(uuid4()),
        "format": format_object,
        "source": base_uuid,
        "transform": [
            {"type": "filter_element", "expr": element_name},
            {"type": "filter_cs", "expr": cs},
        ],
    }


def create_derived_data_object(
    sdata: SpatialData, call: str, params: Params, base_uuid: str, cs: str, table_id: str | None = None
) -> dict[str, Any]:
    """Create the base data object for a SpatialData element.

    Parameters
    ----------
    sdata: SpatialData
        The SpatialData object of which elements have been plotted.
    call: str
        The render call from spatialdata plot, either render_images, render_labels, render_points
        or render_shapes, prefixed by n_ where n is the index of the render call starting from 0.
    params: Params
        The render parameters used in spatialdata-plot for the particular type of SpatialData
        element.
    base_uuid: str
        Unique identifier used to refer to the base level SpatialData zarr store in the vega
        like view configuration.
    cs: str
        The name of the coordinate system in which the SpatialData element was plotted.
    table_id: str | None
        The value of the `name` key in the vega like data object pertaining to the table used
        for plotting a SpatialData element.

    Returns
    -------
    A vega like data object for derived SpatialData elements.
    """
    data_object = _create_base_derived_data_object(params.element, call, cs, base_uuid)

    if "render_images" in call and isinstance(params, ImageRenderParams):
        selected_scale = "full" if params.scale is None else params.scale
        data_object["transform"].extend(
            [
                {"type": "filter_scale", "expr": selected_scale},
                {"type": "filter_channel", "expr": params.channel},
            ]
        )
        data_object = add_norm_transform_to_data_object(params.cmap_params, data_object)

    elif "render_labels" in call and isinstance(params, LabelsRenderParams):
        selected_scale = "full" if params.scale is None else params.scale
        data_object["transform"].append({"type": "filter_scale", "expr": selected_scale})
        data_object = _add_table_lookup(sdata, params, data_object, table_id)
        data_object = add_norm_transform_to_data_object(params.cmap_params, data_object)

    elif ("render_points" in call or "render_shapes" in call) and isinstance(
        params, PointsRenderParams | ShapesRenderParams
    ):
        data_object = _add_table_lookup(sdata, params, data_object, table_id)

        if params.ds_reduction is not None:
            data_object = _add_datashade_transform(params, data_object)
        else:
            data_object = add_norm_transform_to_data_object(params.cmap_params, data_object)

    return data_object
