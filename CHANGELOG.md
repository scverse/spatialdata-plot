# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.2.8] - 2024-11-26

- Support for `xarray.DataTree` (which moved from `datatree.DataTree`) 

## [0.2.7] - 2024-10-24

### Added

-   The user can now specify `datashader_reduction` to control the rendering behaviour (#309)
-   Rendering outlines of shapes with datashader works now (#309)

### Fixed

-   datashader now uses canvas size = image size which speeds up the rendering (#309)
-   datashader now uses the `linear` as interpolation method for colormaps instead of the default `eq_hist` to make it equivalent to matplotlib (#309)
-   point sizes of datashader now agree with matplotlib also when dpi != 100 (#309)
-   Giving a custom colormap when rendering a multiscale image now works (#586)

## [0.2.6] - 2024-09-04

### Changed

-   Lowered RMSE-threshold for plot-based tests from 45 to 15 (#344)
-   When subsetting to `groups`, `NA` isn't automatically added to legend (#344)
-   When rendering a single image channel, a colorbar is now shown (#346)
-   Removed `percentiles_for_norm` parameter (#346)
-   Changed `norm` to no longer accept bools, only `mpl.colors.Normalise` or `None` (#346)

### Fixed

-   Filtering with `groups` now preserves original cmap (#344)
-   Non-selected `groups` are now not shown in `na_color` (#344)
-   Several issues associated with `norm` and `colorbar` (#346)

## [0.2.5] - 2024-08-23

### Added

-

### Changed

-   Replaced `outline` parameter in `render_labels` with alpha-based logic (#323)
-   Lowered RMSE-threshold for plot-based tests from 60 to 45 (#323)
-   Removed `preprocessing` (.pp) accessor (#329)

### Fixed

-   Minor fixes for several tests as a result of the threshold change (#323)

## [0.2.4] - 2024-08-07

### Added

-   Added utils function for 0-transparent cmaps (#302)

### Fixed

-   Took RNG out of categorical label test (#306)
-   Performance bug when plotting shapes (#298)
-   scale parameter was ignored for single-scale images (#301)
-   Changes to support for dask-expr (#283)
-   Added error handling for non-existent elements (#305)
-   Specifying vmin and vmax properly clips image data (#307)
-   import bug `get_cmap()` (8fd969c)

## [0.2.3] - 2024-07-03

### Added

-   Datashader support for points and shapes (#244)

### Changed

-   All parameters are now provided for a single element (#272)

### Fixed

-   Fix color assignment for NaN values (#257)
-   Zorder of rendering now strictly follows the order of the render_x calls (#244)

## [0.2.2] - 2024-05-02

### Fixed

-   Fixed `fill_alpha` ignoring `alpha` channel from custom cmap (#236)
-   Fix channel str support (#221)

## [0.2.1] - 2024-03-26

### Minor

-   Adjusted GitHub worklows

## [0.2.0] - 2024-03-24

### Added

-   Support for plotting multiple tables @melonora

### Fixed

-   Several bugfixes, especially for colors and palettes @melonora

## [0.1.0] - 2024-01-17

### Added

-   Multiscale image handling: user can specify a scale, else the best scale is selected automatically given the figure size and dpi (#164)
-   Large images are automatically rasterized to speed up performance (#164)
-   Added better error message for mismatch in cs and ax number (#185)
-   Beter test coverage for correct plotting of elements after transformation (#198)
-   Can now stack render commands (#190, #192)
-   The `color` argument in render_shapes/points now accepts actual colors as well (#199)
-   Input arguments are now evaulated for their types in basic.py (#199)

### Fixed

-   Now dropping index when plotting shapes after spatial query (#177)
-   Points are now being correctly rotated (#198)
-   User can now pass Colormap objects to the cmap argument in render_images. When only one cmap is given for 3 channels, it is now applied to each channel (#188, #194)

## [0.0.6] - 2023-11-06

### Added

-   Pushed `get_extent` functionality upstream to `spatialdata` (#162)

## [0.0.5] - 2023-10-02

### Added

-   Can now scale shapes (#152)
-   Can now plot columns from GeoDataFrame (#149)

### Fixed

-   Multipolygons are now handled correctly (#93)
-   Legend order is now deterministic (#143)
-   Images no longer normalised by default (#150)
-   Filtering of shapes and points using the `groups` argument is now possible, coloring by palette and cmap arguments works for shapes and points (#153)
-   Colorbar no longer autoscales to [0, 1] (#155)
-   Plotting shapes after a spatial query is now possible (#163)

## [0.0.4] - 2023-08-11

### Fixed

-   Multi-scale images/labels are now correctly substituted and the action is logged (#131).
-   Empty geometries among the shapes can be handeled (#133).
-   `outline_width` parameter in render_shapes is now a float that actually determines the line width (#139).

## [0.0.2] - 2023-06-25

### Fixed

-   Multiple bugfixes of which I didn't keep track of.

## [0.0.1] - 2023-04-04

### Added

-   Initial release of `spatialdata-plot` with support for `images`, `labels`, `points` and `shapes`.
