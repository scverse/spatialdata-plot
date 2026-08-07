"""Version-tolerant access to scanpy internals used by spatialdata-plot.

scanpy 1.13 relocated the default palettes and several private plotting helpers from
``scanpy.plotting.{palettes,_tools,_utils}`` to ``scanpy.plotting.legacy.*``, and dropped the
``settings._vector_friendly`` flag. The values and behaviour are unchanged, so we import from
whichever path the installed scanpy exposes and re-export from a single place. This keeps
spatialdata-plot working on scanpy both < and >= 1.13 and confines the reliance on scanpy
internals to one module.

The fallbacks catch ``ModuleNotFoundError`` (not the broader ``ImportError``) so that a legacy
module which exists but fails to import for an unrelated reason surfaces instead of being masked
by a silent fall-back to the old path.
"""

from scanpy import settings as _sc_settings

try:  # scanpy >= 1.13
    from scanpy.plotting.legacy.palettes import default_20, default_28, default_102
except ModuleNotFoundError:  # scanpy < 1.13
    from scanpy.plotting.palettes import default_20, default_28, default_102

try:  # scanpy >= 1.13
    from scanpy.plotting.legacy._tools.scatterplots import _add_categorical_legend
except ModuleNotFoundError:  # scanpy < 1.13
    from scanpy.plotting._tools.scatterplots import _add_categorical_legend

try:  # scanpy >= 1.13
    from scanpy.plotting.legacy._utils import add_colors_for_categorical_sample_annotation
except ModuleNotFoundError:  # scanpy < 1.13
    from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation


def vector_friendly() -> bool:
    """Scanpy's rasterize-for-vector-output flag, read dynamically.

    Controls whether scatter/image artists are rasterized (so vector output stays small). scanpy
    1.13 removed the ``settings._vector_friendly`` attribute, so on scanpy >= 1.13 this always
    returns ``False`` (scanpy's unconfigured default): users who had enabled it via
    ``sc.set_figure_params(vector_friendly=True)`` lose rasterization there — unavoidable, as the
    flag no longer exists. On older scanpy the configured value is still honoured.
    """
    return bool(getattr(_sc_settings, "_vector_friendly", False))


__all__ = [
    "_add_categorical_legend",
    "add_colors_for_categorical_sample_annotation",
    "default_20",
    "default_28",
    "default_102",
    "vector_friendly",
]
