from typing import Callable, List, Union

from anndata import AnnData
import spatialdata as sd


from ..accessor import register_spatial_data_accessor


@register_spatial_data_accessor("tl")
class ToolAccessor:
    def __init__(self, sdata):
        self._sdata = sdata

    def _copy(
        self,
        images: Union[None, dict] = None,
        labels: Union[None, dict] = None,
        points: Union[None, dict] = None,
        polygons: Union[None, dict] = None,
        shapes: Union[None, dict] = None,
        table: Union[dict, AnnData] = None,
    ) -> sd.SpatialData:

        """
        Helper function to copies the references from the original SpatialData
        object to the subsetted SpatialData object.
        """

        return sd.SpatialData(
            images=self._sdata.images if images is None else images,
            labels=self._sdata.labels if labels is None else labels,
            points=self._sdata.points if points is None else points,
            polygons=self._sdata.polygons if polygons is None else polygons,
            shapes=self._sdata.shapes if shapes is None else shapes,
            table=self._sdata.table if table is None else table,
        )

    def filter_polygon(self, polygon_key: str) -> sd.SpatialData:
        """Subsets the SpatialData object to the polygon with the given key.

        Parameters
        ----------
        polygon_key
            Key of the polygon to subset to.

        Returns
        -------
        sd.SpatialData
        """
        
        if polygon_key not in self._sdata.polygons:
            raise ValueError(f"Polygon with key {polygon_key} not found in SpatialData object.")
        
        polygons = {polygon_key: self._sdata.polygons[polygon_key]}
        
        return self._copy(polygons=polygons)