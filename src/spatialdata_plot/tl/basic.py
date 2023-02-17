from typing import Union

import pandas as pd
from skimage.measure import regionprops_table

from ..accessor import register_spatial_data_accessor


@register_spatial_data_accessor("tl")
class ToolsAccessor:
    def __init__(self, sdata):
        self._sdata = sdata

    def label_property(self, properties: Union[str, list], obsm_key_added="label_props", return_df=False, **kwargs):
        """Extract properties from the label images.
        """
        if isinstance(properties, str):
            properties = [properties]

        if "label" not in properties:
            properties = ["label"] + properties
            
        # unpack region and instance keys
        region_key = self._sdata.pp.get_region_key()
        instance_key = self._sdata.pp.get_instance_key()
        # create dictionry that maps each label to the region key
        regions = self._sdata.table.obs[region_key].unique().tolist()
        region_key_dict = {key.split("/")[-1]: key for key in regions}

        properties_list = []
        for label in self._sdata.labels:
            # calculate properties on all labels
            props = regionprops_table(self._sdata.labels[label].values, properties=properties, **kwargs)

            df = pd.DataFrame(props).assign(region_key=region_key_dict[label])
            properties_list.append(df)
            
        # concat and rename
        property_table = pd.concat(properties_list).rename(
            columns={"label": instance_key, "region_key": region_key, "centroid-0": "y", "centroid-1": "x"}
        )
        
        # align with obs
        property_table = (
            self._sdata.table.obs[[region_key, instance_key]]
            .reset_index()
            .merge(property_table, on=[region_key, instance_key], how="left")
            .set_index("index")
        )
        
        if return_df:
            # for internal use
            return property_table            
        
        adata = self._sdata.table.copy()
        adata.obsm[obsm_key_added] = property_table
        return self._sdata.pp._copy(table=adata)