from typing import List
from ..accessor import register_spatial_data_accessor
from .colorize import _colorize, _normalize

@register_spatial_data_accessor("im")
class ImageAccessor:
    
    def __init__(self, sdata):
        self._sdata = sdata
        
        # pull information from the AnnData object
        self._images = self._get_images()
        self._channels = self._get_image_dims('c')
        self._xdims = self._get_image_dims('x')
        self._ydims = self._get_image_dims('y')
        
        # select all by default
        self._i = self._images[0]
        self._x = slice(None)
        self._y = slice(None)
        self._c = slice(None)
        
        # self._selection = self._get_selection()        

    def _get_images(self) -> list:
        """Get the image keys from the spatial data object."""
        return list(self._sdata.images.keys())
    
    def _get_image_dims(self, dim) -> dict:
        """Get the image dimensions."""
        return { k:v.coords[dim].values for k, v in self._sdata.images.items() }
    
    def is_image_key(self, value: str) -> bool:
        return value in self._images
    
    def is_channel(self, image_key: str, value: int) -> int:
        return value in self._channels[image_key]
    
    def is_list_of_channels(self, value: list) -> bool:
        return all([ isinstance(v, int) for v in value ]) 
    
    def is_list_of_strings(self, value: list) -> bool:
        return all([ isinstance(v, str) for v in value ])
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = value
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value
    
    @property
    def i(self):
        return self._i
    
    @i.setter
    def i(self, value):
        if isinstance(value, str):
            assert(self.is_image_key(value)), "Image key not found."
            self._i = value
        elif isinstance(value, list):
            # do we want to refactor i.e. is_image_key(self, val)
            assert(all([ v in self._images for v in value])), "At least one image key was not found."
            self._i = value
        else:
            raise ValueError("Image key must be a string or list of strings.")
        
        
    @property
    def c(self):
        return self._c
    
    @c.setter
    def c(self, value):
        # TODO: image_key may also be a list
        if isinstance(self.i, str):
            if isinstance(value, int):
                assert(value in self._sdata.images[self.i].coords['c']), f"Channel not found in {self.i}."
                self._c = value
        
        elif isinstance(self.i, list):
            if isinstance(value, int):
                for image_key in self.i:
                    assert(value in self._sdata.images[image_key].coords['c']), f"Channel not found in {image_key}."
                self._c = value
                
            if isinstance(value, list):
                for val in value:
                    for image_key in self.i:
                        assert(self.is_channel(image_key, val)), f'Channel {val} not found in {image_key}.'
                self._c = value

    def get_selection(self):
        """Gets the current selection of the image accessor.
        """
        sel =  {}
        if isinstance(self.i, str):
            sel[self.i] = self._sdata.images[self.i][self.c, self.y, self.x]
        if isinstance(self.i, list):
            for image_key in self.i:
                sel[image_key] = self._sdata.images[image_key][self.c, self.y, self.x]
                
        return sel


    def __getitem__(self, indices):
        """Indexing method for spatial data images. Accepts queries in the form of:
        
        sdata.im['image_key', channel, y_slice, x_slice]
        
        or 
        
        sdata.im[channel, y_slice, x_slice]
        """
        # If a tuple is passed to the image accesor one needs to carefully parse the arguments     
        if isinstance(indices, tuple):
            
            if len(indices) == 1:
                c_slice = slice(None)
                y_slice = indices[1]
                x_slice = slice(None)

            if len(indices) == 2:
                ix1, ix2 = indices
                
                # if the first index is a string, this is interpreted as the image key
                if isinstance(ix1, str) and isinstance(ix2, int):
                    self.i = ix1
                    self.c = ix2
                    self.x = slice(None)
                    self.y = slice(None)
                    
                    
                if isinstance(ix1, list) and isinstance(ix2, int):
                    self.i = ix1
                    self.c = ix2
                    self.x = slice(None)
                    self.y = slice(None)
                    
                if isinstance(ix1, str) and isinstance(ix2, list):
                    self.i = ix1
                    self.c = ix2
                    self.x = slice(None)
                    self.y = slice(None)
                    
                if isinstance(ix1, list) and isinstance(ix2, list):
                    self.i = ix1
                    self.c = ix2
                    self.x = slice(None)
                    self.y = slice(None)
                    
                    
                if isinstance(ix1, slice) and isinstance(ix2, slice):
                    self.i = self._get_images()
                    self.c = slice(None)
                    self.x = ix1
                    self.y = ix2
                
            
            if len(indices) == 3:
                ix1, ix2, ix3 = indices
                
                if self.is_list_of_strings(ix1):
                    self.i = ix1
                    self.c = slice(None)
                    self.x = ix2
                    self.y = ix3
                    
                if self.is_list_of_channels(ix1):
                    self.i = self._get_images()
                    self.c = ix1
                    self.x = ix2
                    self.y = ix3
                    
            if len(indices) == 4:
                ix1, ix2, ix3, ix4 = indices
                self.i = ix1
                self.c = ix2
                self.x = ix3
                self.y = ix4
            
            
        
        # If just a string is passed to the image accesor, this is interpreted as the image key
        if isinstance(indices, str):
            image_key = indices
            self.i = image_key
        
        # If just an integer is passed to the image accesor, this is interpreted as the channel index
        if isinstance(indices, int):
            channel = indices
            self.i = self._get_images()
            self.c = channel
            
        # If a list is passed to the image accesor, this is interpreted as a list of image keys
        if isinstance(indices, list):
            self.i = indices
            
            
            
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
        # self._selection = self._get_selection()
        return self._sdata    
    
    def colorize(
        self,
        colors: List[str] = ["C0", "C1", "C2", "C3"],
        background: str = "black",
        normalize: bool = True,
        merge=True,
    ) :
        """Colorizes a stack of images.

        Parameters
        ----------
        colors: List[str]
            A list of strings that denote the color of each channel.
        background: float
            Background color of the colorized image.
        normalize: bool
            Normalizes the image prior to colorizing it.
        merge: True
            Merge the channel dimension.


        Returns
        -------
        xr.Dataset
            The image container with the colorized image stored in Layers.PLOT.
        """
        images = self.get_selection()
        
        self._rendered = {}
        
        for k, v in images.items():
            self._selection[k] = _colorize(
                v,
                colors=colors,
                background=background,
                normalize=normalize,
            ).sum(0)
            
        return self._sdata
            
