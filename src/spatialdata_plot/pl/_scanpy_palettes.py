"""Access scanpy's built-in categorical palettes across scanpy versions.

scanpy relocated the ``default_20`` / ``default_28`` / ``default_102`` palettes from
``scanpy.plotting.palettes`` to ``scanpy.plotting.legacy.palettes`` in 1.13. The values are
frozen (identical across versions), so we import from whichever path the installed scanpy
exposes and re-export them from a single place for the rest of the package.
"""

try:  # scanpy >= 1.13
    from scanpy.plotting.legacy.palettes import default_20, default_28, default_102
except ImportError:  # scanpy < 1.13
    from scanpy.plotting.palettes import default_20, default_28, default_102

__all__ = ["default_20", "default_28", "default_102"]
