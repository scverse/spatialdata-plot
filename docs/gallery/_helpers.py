"""Shared data loaders for gallery examples."""

from __future__ import annotations

import warnings

import numpy as np
import scanpy as sc
import spatialdata as sd
from spatialdata.models import Image2DModel, ShapesModel, TableModel


def load_visium_breast_cancer() -> sd.SpatialData:
    """Load Visium breast cancer tissue as a SpatialData object.

    Uses ``scanpy.datasets.visium_sge`` which caches the download
    via :mod:`pooch`, so the data is only fetched once.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.datasets.visium_sge(
            sample_id="V1_Breast_Cancer_Block_A_Section_1",
        )

    sample = list(adata.uns["spatial"].keys())[0]
    meta = adata.uns["spatial"][sample]
    sf = meta["scalefactors"]["tissue_hires_scalef"]

    image = Image2DModel.parse(
        np.moveaxis(meta["images"]["hires"], -1, 0),
        dims=("c", "y", "x"),
    )

    radius = meta["scalefactors"]["spot_diameter_fullres"] * sf / 2
    circles = ShapesModel.parse(
        adata.obsm["spatial"] * sf,
        geometry=0,
        radius=radius,
        index=adata.obs_names,
    )

    adata.obs["region"] = "spots"
    adata.obs["region"] = adata.obs["region"].astype("category")
    adata.obs["instance_key"] = adata.obs_names
    table = TableModel.parse(
        adata,
        region="spots",
        region_key="region",
        instance_key="instance_key",
    )

    table.var_names_make_unique()
    sc.pp.normalize_total(table)
    sc.pp.log1p(table)

    return sd.SpatialData(
        images={"tissue": image},
        shapes={"spots": circles},
        tables={"table": table},
    )
