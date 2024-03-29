![SpatialData banner](https://github.com/scverse/spatialdata/blob/main/docs/_static/img/spatialdata_horizontal.png?raw=true)

# spatialdata-plot: rich static plotting from SpatialData objects

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]
[![Codecov][badge-codecov]][link-codecov]
[![Documentation][badge-pypi]][link-pypi]
[![DOI](https://zenodo.org/badge/588223127.svg)](https://zenodo.org/badge/latestdoi/588223127)

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/spatialdata-plot/test.yaml?branch=main
[link-tests]: https://github.com/scverse/spatialdata-plot/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/spatialdata-plot
[badge-codecov]: https://codecov.io/gh/scverse/spatialdata-plot/branch/main/graph/badge.svg?token=C45F3ATSVI
[link-codecov]: https://app.codecov.io/gh/scverse/spatialdata-plot
[badge-pypi]: https://badge.fury.io/py/spatialdata_plot.svg
[link-pypi]: https://pypi.org/project/spatialdata-plot/

The `spatialdata-plot` package extends `spatialdata` with a declarative plotting API that enables to quickly visualize `spatialdata` objects and their respective elements (i.e. `images`, `labels`, `points` and `shapes`).

SpatialData’s plotting capabilities allow to quickly visualise all contained modalities. The user can specify which elements should be rendered (images, labels, points, shapes) and specify certain parameters for each layer, such as for example the intent to color shapes by a gene’s expression profile or which color to use for which image channel. When the plot is then eventually displayed, all transformations, alignments and coordinate systems are internally processed to form the final visualisation. In concordance with the general SpatialData philosophy, all modalities of the major spatial technologies are supported out of the box.

<img src='https://raw.githubusercontent.com/scverse/spatialdata-plot/main/docs/spatialdata-plot.png'/>

## Getting started

For more information on the `spatialdata-plot` library, please refer to the [documentation](https://spatialdata.scverse.org/projects/plot/en/latest/index.html). In particular, the

-   [API documentation][link-api].
-   [Example notebooks][link-notebooks] (section "Visiualizations")

## Installation

You need to have Python 3.9 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install spatialdata-plot:

<!--
1) Install the latest release of `spatialdata-plot` from `PyPI <https://pypi.org/project/spatialdata-plot/>`_:

```bash
pip install spatialdata-plot
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/scverse/spatialdata-plot.git@main
```

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

Marconato, L., Palla, G., Yamauchi, K.A. et al. SpatialData: an open and universal data framework for spatial omics. Nat Methods (2024). https://doi.org/10.1038/s41592-024-02212-x

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/spatialdata-plot/issues
[changelog]: https://spatialdata-plot.readthedocs.io/latest/changelog.html
[link-docs]: https://spatialdata-plot.readthedocs.io
[link-api]: https://spatialdata.scverse.org/projects/plot/en/latest/api.html
[link-design-doc]: https://spatialdata.scverse.org/en/latest/design_doc.html
[link-notebooks]: https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks.html
