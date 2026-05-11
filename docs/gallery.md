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

::::

```{toctree}
:hidden:
:maxdepth: 2

notebooks/tutorials/index
notebooks/examples/index
```
