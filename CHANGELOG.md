# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.0] - tbd

## [0.0.5] - 2023-10-02

### Added

-   Can now scale shapes (#152)
-   Can now plot columns from GeoDataFrame (#149)

### Fixed

-   Multipolygons are now handled correctly (#93)
-   Legend order is now deterministic (#143)
-   Images no longer normalised by default (#150)
-   Colorbar no longer autoscales to [0, 1] (#155)

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
