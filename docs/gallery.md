# Gallery

Runnable examples demonstrating `spatialdata-plot` on real spatial-omics
datasets and on the lightweight `blobs` dataset.

Sources live in
[`scverse/spatialdata-plot-notebooks`](https://github.com/scverse/spatialdata-plot-notebooks);
every notebook is executable end-to-end and re-executed on a weekly schedule
against the latest `spatialdata-plot` release.

## Tutorials

Entry-point material for learning the API on synthetic data.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: notebooks/tutorials/getting_started
:link-type: doc
:img-top: _static/gallery/getting_started.png

The fluent `.pl` API, layering, and styling on the in-memory `blobs`
dataset. Ideal first read.
:::

:::{grid-item-card} Colour and palettes
:link: notebooks/tutorials/color_and_palette
:link-type: doc
:img-top: _static/gallery/color_and_palette.png

How `color=` resolves, the v0.3.0 `groups` behaviour, and building
perceptually well-spaced or colourblind-safe palettes with
`make_palette` and `make_palette_from_data`.
:::

:::{grid-item-card} Speeding up rendering
:link: notebooks/tutorials/performance
:link-type: doc
:img-top: _static/gallery/performance.png

Keep rendering fast on large data: automatic rasterization and scale
selection for images, and the `datashader` backend for large collections
of shapes and points.
:::

:::{grid-item-card} Normalization and contrast
:link: notebooks/tutorials/normalization_and_contrast
:link-type: doc
:img-top: _static/gallery/normalization_and_contrast.png

How `norm=` maps data to colours: fixed contrast limits, clipping,
logarithmic and percentile scaling with `PercentileNormalize`, and
per-channel norms for images.
:::

::::

## Examples

Worked examples on real datasets you'd actually analyse.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Visium mouse brain
:link: notebooks/examples/visium_mouse_brain
:link-type: doc
:img-top: _static/gallery/visium_mouse_brain.png

Render H&E tissue, overlay spots, color by gene expression and by cluster,
and finish with a publication-style figure.
:::

:::{grid-item-card} Interactive region annotation
:link: notebooks/examples/interactive_annotate
:link-type: doc
:img-top: _static/gallery/interactive_annotate.png

Draw regions of interest directly on a `spatialdata-plot` canvas with
`sdata.pl.annotate(...)` and persist them as a `ShapesModel` element.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

notebooks/tutorials/index
notebooks/examples/index
```
