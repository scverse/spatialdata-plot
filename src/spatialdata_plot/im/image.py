from ..accessor import register_spatial_data_accessor


@register_spatial_data_accessor("im")
class ImageAccessor:
    
    def __init__(self, spatialdata_obj):
        self._obj = spatialdata_obj
        

    def __getitem__(self, indices):
        """Indexing method for spatial data images.
        """
        
        if isinstance(indices, tuple):
            image_key = indices[0]
            
            if len(indices) == 2:
                c_slice = slice(None)
                y_slice = indices[1]
                x_slice = slice(None)
            
            if len(indices) == 3:
                c_slice = slice(None)
                y_slice = indices[1]
                x_slice = indices[2]
                
            if len(indices) == 4:
                c_slice = indices[1]
                y_slice = indices[2]
                x_slice = indices[3]
                
        self._obj.table.uns['sel'] = dict(
            image_key=image_key,
            c_slice=c_slice,
            y_slice=y_slice,
            x_slice=x_slice
        )
        
        # argument handling
        # if type(indices) is str:
        #     c_slice = [indices]
        #     x_slice = slice(None)
        #     y_slice = slice(None)
        # elif type(indices) is slice:
        #     c_slice = slice(None)
        #     x_slice = indices
        #     y_slice = slice(None)
        # elif type(indices) is list:
        #     all_str = all([type(s) is str for s in indices])

        #     if all_str:
        #         c_slice = indices
        #         x_slice = slice(None)
        #         y_slice = slice(None)
        # elif type(indices) is tuple:
        #     all_str = all([type(s) is str for s in indices])

        #     if all_str:
        #         c_slice = [*indices]
        #         x_slice = slice(None)
        #         y_slice = slice(None)

        #     if len(indices) == 2:
        #         if (type(indices[0]) is slice) & (type(indices[1]) is slice):
        #             c_slice = slice(None)
        #             x_slice = indices[0]
        #             y_slice = indices[1]
        #         elif (type(indices[0]) is str) & (type(indices[1]) is slice):
        #             # Handles arguments in form of im['Hoechst', 500:1000]
        #             c_slice = [indices[0]]
        #             x_slice = indices[1]
        #             y_slice = slice(None)
        #         elif (type(indices[0]) is list) & (type(indices[1]) is slice):
        #             c_slice = indices[0]
        #             x_slice = indices[1]
        #             y_slice = slice(None)
        #         else:
        #             raise AssertionError("Some error in handling the input arguments")

        #     elif len(indices) == 3:
        #         if type(indices[0]) is str:
        #             c_slice = [indices[0]]
        #         elif type(indices[0]) is list:
        #             c_slice = indices[0]
        #         else:
        #             raise AssertionError("First index must index channel coordinates.")

        #         if (type(indices[1]) is slice) & (type(indices[2]) is slice):
        #             x_slice = indices[1]
        #             y_slice = indices[2]

        # xdim = self._obj.coords[Dims.X]
        # ydim = self._obj.coords[Dims.Y]

        # x_start = xdim[0] if x_slice.start is None else x_slice.start
        # y_start = ydim[0] if y_slice.start is None else y_slice.start
        # x_stop = xdim[-1] if x_slice.stop is None else x_slice.stop
        # y_stop = ydim[-1] if y_slice.stop is None else y_slice.stop

        # selection = {
        #     Dims.CHANNELS: c_slice,
        #     Dims.X: x_slice,
        #     Dims.Y: y_slice,
        # }

        # if Dims.CELLS in self._obj.dims:
        #     coords = self._obj[Layers.OBS]
        #     cells = (
        #         (coords.loc[:, Features.X] >= x_start)
        #         & (coords.loc[:, Features.X] <= x_stop)
        #         & (coords.loc[:, Features.Y] >= y_start)
        #         & (coords.loc[:, Features.Y] <= y_stop)
        #     ).values

        #     selection[Dims.CELLS] = cells

        # ds = self._obj.sel(selection)

        # if Dims.CELLS in self._obj.dims:
        #     lost_cells = num_cells - ds.dims[Dims.CELLS]

        # if Dims.CELLS in self._obj.dims and lost_cells > 0:
        #     logger.warning(f"Dropped {lost_cells} cells.")

        return self._obj