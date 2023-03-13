from typing import Tuple

import numpy as np

from .utils import _get_linear_colormap


def _normalize(
    img: np.ndarray,
    pmin: float = 3.0,
    pmax: float = 99.8,
    eps: float = 1e-20,
    clip: bool = False,
    name: str = "normed",
) -> np.ndarray:
    """Performs a min max normalisation.

    This function was adapted from the csbdeep package.

    Parameters
    ----------
    dataarray: xr.DataArray
        A xarray DataArray with an image field.
    pmin: float
        Lower quantile (min value) used to perform qunatile normalization.
    pmax: float
        Upper quantile (max value) used to perform qunatile normalization.
    eps: float
        Epsilon float added to prevent 0 division.
    clip: bool
        Ensures that normed image array contains no values greater than 1.

    Returns
    -------
    xr.DataArray
        A min-max normalized image.
    """
    perc = np.percentile(img, [pmin, pmax], axis=(1, 2)).T

    norm = (img - np.expand_dims(perc[:, 0], (1, 2))) / (np.expand_dims(perc[:, 1] - perc[:, 0], (1, 2)) + eps)

    if clip:
        norm = np.clip(norm, 0, 1)

    return norm


def _colorize(
    img: np.ndarray,
    colors: Tuple[str] = ("C1", "C2", "C3", "C4", "C5"),
    background: str = "black",
    normalize: bool = True,
    name: str = "colored",
) -> np.ndarray:
    """Colorizes a stack of images

    Parameters
    ----------
    dataarray: xr.DataArray
        A xarray DataArray with an image field.
    clors: List[str]
        A list of strings that denote the color of each channel
    background: float
        Background color of the colorized image.
    normalize: bool
        Normalizes the image prior to colorizing it.

    Returns
    -------
    np.ndarray
        A colorized image
    """
    num_channels = img.shape[0]

    assert (
        len(colors) >= num_channels
    ), "Length of colors must at least be greater or equal the number of channels of the image."

    cmaps = _get_linear_colormap(colors[:num_channels], background)

    if normalize:
        img = _normalize(img)

    colored = np.stack([cmaps[i](img[i]) for i in range(num_channels)], 0)

    return colored
