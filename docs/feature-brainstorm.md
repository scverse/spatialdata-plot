# 100 ideas for `spatialdata-plot`

A survey of features inspired by ~100 real spatial-omics, geospatial, and
imaging projects, with honest critique of each. Density-mode rendering
landed easily because the architecture (declarative chain → datashader →
matplotlib) was a good fit. This document tries to find the next set of
features that have similar leverage, and to be loud about the ones that
don't.

Conventions:
- **Idea** — what could be built and the one-line API sketch.
- **Inspiration** — the real project(s) where the technique appears.
- **Critique** — why it might be a bad idea, redundant, or out of scope.

The list is organised by category. Categories are not equally important;
A, C, G, J, K probably contain the highest-leverage additions. Categories
H and I are more "ecosystem bridge" work than rendering work.

---

## A. Spatial statistics overlays (1–10)

These project per-element scalar statistics that users currently compute
in Squidpy/Voyager/PySAL but visualise with raw matplotlib. The natural
shape is `pl.render_*(color=<obs_key_with_stat>)` already, but the
following go further than a single column.

**1. LISA cluster maps (Local Indicators of Spatial Association).**
Render shapes/labels coloured by a four-class palette (HH, LL, HL, LH)
plus "not significant". Inspiration: `Voyager::plotLocalResult`, `pysal`
`esda.Moran_Local`, GeoDa. Critique: the *statistic* belongs upstream
(squidpy/voyager); we'd only ship the dedicated categorical palette +
significance mask, which is arguably just a documented recipe rather than
a feature. The actual win is a `pl.render_lisa(...)` helper that enforces
the right palette ordering and a non-significance NA colour — small but
loud about intent.

**2. Getis-Ord G* hotspot overlay.** Diverging colormap centred on z=0,
with significance contour. Inspiration: PySAL `G_Local`, ArcGIS hotspot
analysis, Voyager. Critique: same upstream-vs-here split as LISA. Both
#1 and #2 could collapse into a single `render_local_stat(kind=...)`.

**3. Per-cell Ripley deviation map.** Colour each cell by its deviation
from CSR (K(r)−πr²) within a local window. Inspiration: spatstat,
`squidpy.gr.ripley`. Critique: scientifically dubious at the single-cell
scale (high variance, multiple-testing nightmare) and not how Ripley is
normally used. **Skip.**

**4. Empirical variogram panel.** Marginal axis next to the spatial plot
showing semivariance γ(h) for a chosen feature. Inspiration: gstat,
sdmTMB, scikit-gstat. Critique: a marginal-axis system doesn't exist
yet; once it does, this drops in cheaply. Useful for genuinely
quantitative work but niche.

**5. Concentric co-occurrence ring inset.** Take Squidpy's
`co_occurrence(interval=...)`, render the curve as a ring colormap at the
queried anchor cell. Inspiration: Squidpy figure 3-style panels.
Critique: probably better as a marginal subplot than as an overlay; rings
clutter the main canvas.

**6. Neighborhood enrichment "atlas grid".** Compute `nhood_enrichment`
per ROI and tile the n×n matrices in a small-multiples grid. Inspiration:
Squidpy, CODEX neighborhood papers (Schürch, Goltsev). Critique: this is
a matplotlib helper, not a `render_*` extension. Lives more naturally in
`squidpy.pl` but a thin convenience wrapper is fine.

**7. MULTISPATI / spatial PCA loading overlay.** Colour cells by PC
scores from spatially constrained PCA. Inspiration: `Voyager`,
`adegenet::multispati`, GraphST. Critique: trivially expressible as
`color="MULTISPATI1"` if the column exists. The novelty is documentation
+ a single-call demo, not a new render fn.

**8. Top-Moran-I small multiples.** Auto-pick the N genes with highest
global Moran's I and render them as a panel. Inspiration: Voyager
vignettes. Critique: this is a 6-line snippet against the existing API
— shipping it as a function locks a recipe that users will want to
parameterise differently every time. Risk of API bloat.

**9. Spillover-corrected channel display (IMC/CyTOF).** Apply CATALYST-
style spillover matrix to channels before display. Inspiration:
`CATALYST::compCytof`. Critique: out of scope — spillover correction is
a data transformation that belongs in the upstream pipeline (image
processing/QC). Doing it inside a plot function would silently change
displayed numbers. **Skip.**

**10. Marginal K/L/F-function curves.** Add a margin plot of pair
correlation / L / F functions alongside the spatial canvas. Inspiration:
spatstat, Squidpy `ripley`. Critique: again, depends on a marginal-axis
system; without one this is a `subplots(1,2)` recipe.

---

## B. Cell–cell communication overlays (11–17)

CCC tools (CellChat, COMMOT, NICHES, SpatialDM, LIANA) currently dump
results to AnnData and users render with seaborn/matplotlib. None of
this output has a first-class spatial visual primitive.

**11. Directed sender→receiver arrow layer.** New `render_ccc(...)` that
consumes a sparse cell×cell matrix (or per-pair table) and draws curved
arrows weighted by signal strength. Inspiration: CellChat circle plot,
COMMOT `pl.cell_communication`, SpatialDM. Critique: arrow rendering at
>1000 pairs is unreadable and matplotlib-slow; we'd need built-in
top-k/percentile pruning and bundling. Likely worth doing **if** we
commit to edge bundling — without it we ship "noise garnish". The
existing `render_graph` already covers undirected edges; this is the
directed/weighted sibling.

**12. Ligand–receptor co-localisation 2D KDE.** Smooth ligand expression
× receptor expression as two channels and render their product. Inspiration:
SpatialDM, MISTy. Critique: scientifically suspect for spot data (you
get a strong product wherever both are expressed, not where signalling
actually happens). For high-resolution Xenium-like data it's more
defensible.

**13. Sender/receiver role pseudocolour.** One categorical colour per
cell: pure-sender, pure-receiver, both, neither. Inspiration: NICHES.
Critique: useful, trivially expressible by users; ship as a docs recipe.

**14. Inferred signalling flux streamlines.** Use COMMOT's vector-field
output (it produces per-cell flux vectors) and stream-plot it.
Inspiration: COMMOT, RNA-velocity-style streamlines. Critique: this is
the genuinely novel one in this block. Same plumbing serves #90 (RNA
velocity) — invest once, reuse. Caveat: COMMOT is the only common
producer of these vectors today, so it's a single-source feature.

**15. Chord/circular inset for CCC counts.** Embed a holoviews/matplotlib
chord diagram in a marginal axes. Inspiration: CellChat `netVisual_chord`.
Critique: niche; users will want the chord standalone, not inset.

**16. Distance-decay marginal curve.** Plot average ligand-receptor score
as a function of cell-pair distance next to the spatial canvas.
Inspiration: SpatialDM, MISTy paper figures. Critique: marginal-axis
dependency again. Cheap once that exists.

**17. Receiver-weighted ligand expression.** Colour each cell by `ligand
× mean(receptor over k-NN)`. Inspiration: NICHES "signalling neighborhood
profile". Critique: this is a computation, not a visualisation — fits
better in `squidpy.gr` or a CCC tool. Don't ship.

---

## C. Niche & neighborhood visualisation (18–25)

The CN/niche literature (Goltsev/Schürch CODEX, BANKSY, CellCharter,
UTAG, Spatial-LDA, NeST) all share an awkward viz step: "I have a
per-cell niche label, please draw it nicely". `render_shapes(color=niche)`
covers the baseline; the ideas below are visual primitives the niche
papers actually use.

**18. Niche-boundary isocontours.** Draw lines around contiguous niche
regions (concave hull / α-shape per label). Inspiration: BANKSY paper
figures, CODEX neighborhood papers, choropleth maps in cartography.
Critique: **high-value**. Niches are spatially contiguous by construction
yet currently shown as coloured dots that compete with the underlying
image. Boundaries on top of a faded fill is the right primitive. Cost:
needs alphashape or scipy.spatial; non-trivial when shapes are
disconnected or have holes. Worth doing if we accept the dependency.

**19. Per-cell neighbourhood-composition pie.** Replace each cell glyph
with a tiny pie chart of its k-NN type composition. Inspiration: CODEX
figures, `Voyager::plotPieGeom`. Critique: looks beautiful in
publications, performs catastrophically beyond ~2k cells. Needs LOD /
auto-aggregation to hexbins of pies. Real but expensive to do well.

**20. CN ribbon / iso-band map.** Continuous CN-similarity scalar →
filled iso-band style (think isobath maps). Inspiration: bathymetric maps,
Schürch CODEX. Critique: more aesthetic than informative; the niche
*label* is usually what people want. Skip unless we explicitly support
soft niche assignments.

**21. Rank-ordered neighbour stripe.** A vertical strip at the right of
the figure showing the most common neighbour cell-types of each cluster.
Inspiration: HoodScanPy, `nichepca`. Critique: belongs in a *separate*
panel; trying to inset this on the spatial axes is fragile. Cheap as a
plotting helper.

**22. Edge-coloured connectivity graph.** Extend `render_graph` with
`edge_color=<expression key>` (e.g. colour edges by L-R score, contact
length, or shared transcripts). Inspiration: napari edge colouring,
`stlearn`, NICHES. Critique: **easy win**, fits `render_graph` exactly.
Probably already half-implemented; this is the smallest delta with
highest user pull.

**23. Wasserstein-to-reference map.** Colour cells by EMD between their
local composition and a reference composition. Inspiration: UTAG,
"distance to healthy" maps in pathology. Critique: again a *statistic*,
not viz; we'd just be documenting the colour scale.

**24. Tissue-architecture cluster as a labels element.** Workflow helper
that promotes per-cell niche labels to a rasterised labels element
(Voronoi tessellation of cells, painted by niche). Inspiration:
CellCharter, Stardist label outputs. Critique: smart because labels
render fast and integrate cleanly into the existing layer stack — but
it's adding a *data transformation*, not a render fn. Better as a
`spatialdata.utils` helper that this package documents.

**25. Niche radar/spider panel.** Marginal radar plot per niche showing
its mean profile. Inspiration: HoodScanPy. Critique: same marginal-axis
caveat. Independent helper is fine.

---

## D. Subcellular / molecular geometry (26–32)

Sub-cellular spatial transcriptomics (Xenium, CosMx, MERFISH, Stereo-seq
+ Bento) introduce per-transcript and per-cell *geometry* that current
`render_points` flattens to a single dot.

**26. Subcellular pattern colouring.** Use Bento's per-cell pattern
classes (cell-edge, nuclear, cytoplasmic, none) as the colour of the
cell. Inspiration: Bento. Critique: again, computation upstream; the
viz is just a categorical render. Worth shipping as a documented
template, not API.

**27. Per-cell transcript-density label fill.** Compute local transcript
counts inside each segmentation and fill labels with the heatmap.
Inspiration: Xenium Explorer's count overlay. Critique: this *is* a new
render primitive — "label fill by aggregation of points" — and is hard
to express with the current chain. Reasonably novel.

**28. Nuclear vs cytoplasmic log-ratio.** Per-cell scalar of nuclear /
cytoplasmic counts → diverging colormap. Inspiration: Bento, CellProfiler
"intensity per compartment". Critique: needs both nuclear and cell masks
to exist as separate labels elements; restrictive but principled.

**29. Radial subcellular profile margin.** Per-cell density as a function
of normalised radius (0=nucleus, 1=membrane), aggregated by cell type.
Inspiration: Bento radial features, ImageJ "Radial Profile". Critique:
margin-plot dependency; useful in dev/QC of segmentation.

**30. Polarity vectors.** Per-cell arrow drawn from the centre of
transcripts to the centre of the segmentation, scaled by anisotropy.
Inspiration: Bento `nuclear_fraction` / polarity features, cell-polarity
papers. Critique: only meaningful with >50 molecules per cell — restrict
to that subset by default.

**31. Orphan-transcript highlight.** Render transcripts not assigned to
any segmentation in red on top of everything else. Inspiration: Sopa,
Baysor diagnostics, Xenium Explorer "unassigned" view. Critique:
**high-value QC tool** with almost no API surface (one boolean flag).
Should ship.

**32. Co-detected transcript-pair bipartite.** Link transcript species
that co-localise within ε. Inspiration: SpatialDE2, FISH co-detection
papers. Critique: looks pretty, rarely tells you anything you couldn't
get from a heatmap. Skip.

---

## E. Multi-modal / cross-modal integration (33–40)

SpatialData's killer feature is multi-modal data in one object; plotting
hasn't really exploited it.

**33. Auto-faceted modality grid.** `pl.show(facet="modality")` lays out
one subplot per modality with shared transforms. Inspiration: semla
`MapMultipleFeatures`, Vitessce side-by-side. Critique: subplot
choreography is already partly in `show()`; making the faceting axis
declarative is mostly UX polish, not a new primitive. Probably the
single most-requested QoL feature though.

**34. Per-pixel difference map.** `pl.diff(left, right)` renders the
signed difference between two registered rasters. Inspiration:
napari blend modes, ImageJ image calculator. Critique: needs careful
handling of dtype, NaN, and resolution. Definitely a new primitive,
not free.

**35. Registration QC: flicker or checkerboard.** Alternate two layers
in a static checkerboard or animated flicker. Inspiration: BigWarp,
ITK-SNAP, ANTs. Critique: GIF output complicates "show as image"
contract; checkerboard alone is fine and very useful.

**36. Cell2location-style stacked donut.** Replace points/shapes with
miniature donuts per spot showing cell-type abundances. Inspiration:
cell2location vignettes, SpatialPie, Voyager. Critique: see #19 — pies
at scale need binning. Provide via hexbin aggregation by default.

**37. Per-spot mixture pies.** Identical primitive to #36; deduplicate.
**Merge with #36.**

**38. Channel-sum auto-balance.** Compose all selected image channels
into one greyscale display with per-channel percentile balancing.
Inspiration: QuPath, Fiji multi-channel composite. Critique: there's
already multi-channel composition; the new thing is the *balancing
heuristic*. Worth a thin helper.

**39. Pre/post transform preview.** Side-by-side render of the same
element with two coordinate transforms applied. Inspiration: PASTE2,
STAlign QC figures, BigWarp. Critique: tightly coupled to SpatialData's
transform graph; this is one of the few features that's strictly
*spatialdata-plot's* job (no other tool has the transform abstraction).
High novelty value.

**40. Z-stack maximum-intensity projection.** `pl.render_images(z="mip")`
collapses Z to a single 2D display. Inspiration: ImageJ Z-project, napari.
Critique: ergonomic shortcut for a one-liner, ship it. Becomes more
interesting if we also offer `z="sum" | "std" | "argmax"`.

---

## F. Image quality & channel manipulation (41–50)

Each idea here invites the question "should this be in spatialdata-plot
or in upstream image processing?" I lean toward "documentation, not
features" for most of them.

**41. CLAHE per channel.** Inspiration: ImageJ, QuPath, OpenCV.
Critique: silently transforming displayed intensities is dangerous in a
publication-figure tool. **Skip** unless gated behind a very loud flag.

**42. Pseudo flat-field correction.** Inspiration: BaSiC, CellProfiler.
Critique: same concern. Belongs in upstream image processing.

**43. Per-channel gamma.** Inspiration: every imaging tool ever.
Critique: already covered by `norm`/`cmap` (PowerNorm). Documentation,
not feature.

**44. Auto "best contrast" percentile clip.** Inspiration: napari
contrast-limits auto, Fiji "auto B&C". Critique: ergonomic and harmless
since it doesn't change the underlying data — just the display window.
**Worth shipping.** A `contrast="auto"` mode that runs 1st–99th
percentile per channel.

**45. Spectral unmixing display.** Inspiration: Phenoptr, Pixelator,
Phenocycler. Critique: a real algorithm, not a render fn. Out of scope.

**46. Mask-image-by-segmentation.** "Show me the image, but only inside
cell masks." Inspiration: CellProfiler, QuPath. Critique: visually
striking, occasionally useful. Cheap: it's a masking step before display.
Ship as `image_mask=<labels_element>` kwarg.

**47. Scale-residual visualisation.** Show diff between resolutions in a
multiscale image. Inspiration: pyramid debugging in OME-Zarr tooling.
Critique: developer tool, not user feature. Skip.

**48. DAPI/membrane adaptive alpha.** Modulate per-channel alpha by
intensity so high-signal regions are opaque and low-signal transparent.
Inspiration: napari additive blending, Vitessce. Critique: nice for
overlays where one channel is "context" and the other "signal".
Probably trivially achievable with a colormap that has alpha varying
with value — see existing `set_zero_in_cmap_to_transparent`.

**49. Per-channel legend stickers.** A compact channel legend showing
"DAPI = blue, CD8 = green, …" overlaid on the figure. Inspiration:
multiplex IF figures everywhere. Critique: this *already exists*
(`_draw_channel_legend`); improvement target is placement + sizing.
Probably should be in this list under "polish", not "novel".

**50. Dark scientific theme.** Black canvas + tuned fluorescence
colours. Inspiration: napari default. Critique: one-liner with `style.use`;
not a feature.

---

## G. Cohort & multi-sample plotting (51–58)

Currently every user writes their own subplot loop. semla and
spatialLIBD demonstrate how much value sits here.

**51. `pl.cohort(...)` faceting helper.** Take a list of SpatialData
objects (or a single one with multiple samples) and render the same
declarative spec across all of them with shared color scaling.
Inspiration: semla `MapMultipleFeatures`, spatialLIBD `vis_grid_gene`,
ggplot facet_wrap. Critique: **the highest-leverage idea in this
document.** Today, multi-sample work in spatialdata-plot is awkward;
fixing it once eliminates dozens of users' boilerplate. Risk: the API
needs to handle shared norms, separate norms, per-sample masking, etc.
Not a one-afternoon job.

**52. Cohort QC dashboard.** One plot per sample + summary card (cell
counts, gene counts, region balance). Inspiration: scanpy's `qc_metrics`,
nf-core spatial pipelines. Critique: smells like a *report generator*
rather than a plot fn. Probably belongs in `spatialdata-io` or a separate
package.

**53. Per-sample normalised colour scaling.** Each panel uses its own
percentile range. Inspiration: heatmap conventions in MERFISH papers.
Critique: trivial mode of #51; do them together.

**54. Atlas-aligned cohort overlay.** Average expression across samples
after PASTE/STAlign alignment, rendered in atlas coordinates.
Inspiration: PASTE, STAlign, Allen CCF. Critique: needs the cohort to
have been aligned; we shouldn't ship the alignment itself. This is
genuinely novel as a visualisation (no tool I know of does it nicely),
but feels niche.

**55. Pseudobulk-by-region label fill.** Aggregate expression by region
ID and colour each region label by the mean. Inspiration: GTEx tissue
heatmaps, Visium tutorials. Critique: covered partially by
`render_labels(color=...)` when the table is region-level. Documentation
again.

**56. Replicate-disagreement variance map.** Per-pixel variance across
replicates. Inspiration: bioconductor `spatialLIBD::vis_grid_gene_diff`.
Critique: needs registered replicates — strong precondition. Worth a
helper *if* #54 ships.

**57. Donor metadata legend strip.** Sidebar of sample metadata as a
colour strip. Inspiration: complex-heatmap row annotations,
ComplexHeatmap. Critique: a heatmap convention forced onto spatial
figures; probably uncomfortable. Skip.

**58. PASTE warp-field arrow display.** Quiver plot of the warp field
between two slices. Inspiration: PASTE paper figures, ANTs warp viz.
Critique: novel and supported by no current tool. Same plumbing as #14,
#90. Reuse pays off.

---

## H. Interactivity bridges (59–64)

The package is a *static* plotter and should stay that way; "interactive"
is napari/Vitessce/TissUUmaps territory. The right move is making
hand-offs ergonomic, not adding in-tree interactivity.

**59. `.pl.to_vitessce()` config emitter.** Take the current plotting
tree and dump a Vitessce JSON. Inspiration: Vitessce config builder.
Critique: the declarative chain *almost* maps 1:1 to Vitessce
view-configs; this is novel and high-value. Risk: matplotlib parameters
(`cmap`, `norm`) translate imperfectly to Vitessce's colour controls —
expect lossy export. Worth doing precisely *because* the tree
exists.

**60. `.pl.to_napari()` handoff.** Open the current plotting tree in
napari-spatialdata with layers reflecting the chain. Inspiration:
napari-spatialdata. Critique: napari-spatialdata already works on the
SpatialData object directly; what's missing is *parameter inheritance*
(colormap, vmin/vmax). Useful, smaller than #59.

**61. Static+interactive twin.** Same call returns both an MPL figure
and an HTML widget. Inspiration: mpld3, bokeh. Critique: maintenance
nightmare — every feature would need to work in both backends. **Skip**;
do #59 and #60 instead.

**62. `.pl.to_tissuumaps()` config emitter.** Inspiration: TissUUmaps.
Critique: smaller user-base than Vitessce; do only if #59 succeeds and
the infrastructure is reusable.

**63. Plotly backend for `show()`.** Inspiration: plotly express.
Critique: see #61; double-engine maintenance burden. Skip.

**64. Quarto/Jupyter widget mode.** Inspiration: anywidget, ipympl.
Critique: ipympl handles this already; spatialdata-plot doesn't need
its own widget.

---

## I. 3D / volumetric / Z-stack (65–70)

The package is firmly 2D; the recent "raise on 3D" PR (#675) confirms
that. The viable strategy is "fail well, hand off to napari", not
"render in 3D ourselves".

**65. `.pl.to_napari(mode="3d")`.** Open Z-stacks in napari's 3D view.
Inspiration: napari, napari-spatialdata. Critique: a napari binding,
not a render fn. Subset of #60.

**66. MIP / sum / std Z-collapse.** Already covered in #40.

**67. Orthogonal slice triptych.** Render XY/XZ/YZ panels for a 3D
image. Inspiration: ITK-SNAP, napari, OHIF DICOM viewer. Critique:
useful for 3D image checks but conflicts with "we don't do 3D".
Reasonable middle ground.

**68. Z-stack animated GIF/MP4.** Inspiration: ImageJ animation,
napari screen-recording. Critique: see #35 — output type
mismatch with "static figure" contract. Bolt-on, not core.

**69. Isosurface mesh render.** Inspiration: pyvista, vedo, plotly.
Critique: 3D entirely; skip in favour of napari handoff.

**70. Notebook Z-slice slider.** Inspiration: ipywidgets, napari.
Critique: interactivity inside a static plotter contradicts the design.
Skip.

---

## J. Aesthetic & cartographic primitives (71–80)

This is where I think the most underrated wins live. Cartography has
solved many problems that spatial-omics keeps re-solving badly.

**71. Spaco-style spatial-aware palette generation.** Choose categorical
colours so spatially adjacent groups are visually distinct. Inspiration:
Spaco (Liu et al. 2024). Critique: **very high value**, fixes a real
problem (random palette → adjacent niches share colour). Compute cost
non-trivial; offer as `palette="spaco"` with caching on the cluster
labels.

**72. Color-blind-safety lint.** Refuse or warn on palettes that fail a
Daltonism simulation. Inspiration: viridis paper, ColorBrewer.
Critique: needs care not to be annoying. Useful as an opt-in
`palette_check=True`.

**73. Choropleth-style discrete breaks.** Jenks, quantile, equal-interval
bin breaks for continuous data. Inspiration: GeoPandas, mapclassify,
ArcGIS. Critique: a `breaks=` parameter on `cmap`/`norm`. Cheap,
mostly-useful win. Bio readers actually like discrete colour bands.

**74. Bivariate choropleth.** Two-variable colour grid (e.g.
expression × density). Inspiration: cartography, Joshua Stevens'
bivariate maps. Critique: striking but legendarily hard to read; ship
with caution and good docs. Very novel for the omics audience.

**75. Hex/square aggregation grid.** Independent of datashader — explicit
binning into hex/square with a documented bin size in microns.
Inspiration: deck.gl, ggplot `stat_summary_hex`, kepler.gl. Critique:
datashader already aggregates; this is about making the binning *user-
controllable in real units*. Worth a kwarg.

**76. Locator-overview inset.** Tiny thumbnail of the whole sample in
the corner, with a rectangle showing the current view. Inspiration: GIS
software, OpenSeadragon, any map app. Critique: **high-value**, low-cost
once axes-inset is plumbed. Currently users who crop have no idea where
they are.

**77. North arrow / orientation indicator.** Inspiration: every map ever.
Critique: meaningless for most lab samples but loved by neuroanatomy
(D-V, M-L, A-P arrows). A configurable indicator is cheap.

**78. Anatomical landmark calibration on scalebar.** Inspiration:
Allen Brain atlas, BrainGlobe. Critique: clever but couples to atlas
data; ship as a docs recipe with the existing scalebar.

**79. Patterned/hatched palette.** Add hatch textures as a second
information channel for accessibility. Inspiration: cartography of
B&W maps, matplotlib `hatch`. Critique: looks 1990s but genuinely helps
B&W printing and colour-blind viewers. Niche.

**80. Provenance watermark.** Auto-stamp figures with SpatialData
object hash, plotting-tree version, library version. Inspiration:
`watermark` package, MLflow. Critique: hugely useful for reproducibility,
trivially toggleable. Should be opt-in via config; nobody wants a hash
on a Nature figure.

---

## K. QC & diagnostic overlays (81–88)

QC is where matplotlib is currently weakest and where a coherent set of
primitives would unlock real workflow value.

**81. Tile/stitch-boundary overlay.** Show stitching seams from the
acquisition. Inspiration: BaSiC, Ashlar, Fiji Grid/Collection Stitching.
Critique: needs the boundaries to be in the SpatialData object —
they usually aren't. Useful **if** ingestion preserves them.

**82. Saturation/over-exposure overlay.** Red overlay on pixels at the
display maximum. Inspiration: every camera, ImageJ "Highlight Saturated".
Critique: textbook QC primitive that's missing from omics tools.
**High-value, low-cost. Ship.**

**83. Low-quality cell mask.** Grey-out cells failing a QC threshold
(min counts, max % MT, etc.). Inspiration: scanpy QC, MultiQC.
Critique: same as #13 — a documented `groups=` pattern. Helper, not
feature.

**84. Per-channel background overlay.** Render the estimated background
(rolling-ball or local-min). Inspiration: ImageJ subtract-background.
Critique: again, an image-processing computation. Out of scope.

**85. Distance-from-tissue-edge overlay.** Colour each cell by its
distance to the tissue boundary. Inspiration: Bento, MISTy, pathology
"distance to invasive front". Critique: **the cleanest novel feature
in this document.** Useful in tumour micro-environment, dev biology,
inflamed-vs-healthy comparisons. Requires a tissue mask; provide a
default that infers from cell density if no mask is present.

**86. Out-of-tissue warning.** Highlight points/cells outside the
inferred tissue extent. Inspiration: SpaceRanger, Loupe. Critique:
fast, helpful. Subset of #85's tissue-mask logic; ship together.

**87. Missing-data audit marker.** Distinct glyph (e.g. cross) for cells
with NaN in the chosen feature, instead of silent NA colour.
Inspiration: GeoDa, scanpy. Critique: useful but easily abused into
clutter. Opt-in flag.

**88. Transformation-chain annotation.** Print the CS transform chain in
the figure margin. Inspiration: SpatialData design philosophy; nobody
else has this. Critique: **uniquely a spatialdata-plot feature** —
nobody else can do it. Powerful for debugging multi-modal alignment.
Should ship.

---

## L. Trajectory & dynamic (89–92)

Spatial trajectory tools are immature; corresponding viz primitives are
correspondingly thin.

**89. Spatial pseudotime isocontours.** Draw isolines of pseudotime
across the tissue. Inspiration: stLearn, SpaceFlow. Critique: smoothing
choices dominate the picture. Worth doing once with sensible defaults.

**90. RNA-velocity vector field.** Quiver/streamlines overlay using
scVelo or VeloCyto outputs. Inspiration: scVelo, MultiVelo. Critique:
shares plumbing with #14 and #58. Implement the vector-field primitive
once. **Worth doing.**

**91. PAGA on spatial centroids.** Draw the PAGA graph at cluster
spatial centroids rather than UMAP centroids. Inspiration: scanpy
`pl.paga_compare`, but on tissue. Critique: striking and trivially
expressible if `render_graph` accepts arbitrary node positions. Probably
already feasible with the merged `render_graph`.

**92. Position-regressed gradient field.** Estimate ∂pseudotime/∂x,
∂pseudotime/∂y per neighborhood, draw as arrows. Inspiration: dev-bio
"gradient" papers, optical flow. Critique: same primitive as #14, #90,
#58. Trivial extension once vector fields exist.

---

## M. ROI, annotation & report (93–97)

**93. ROI summary cards.** For each ROI in a shapes element, render a
small statistics card (cell composition, top genes). Inspiration:
QuPath, HALO. Critique: looks more like a report generator than a plot
function; might overshoot scope. Useful in pathology workflows though.

**94. Anatomical atlas overlay.** Render Allen CCF/BrainGlobe atlas
contours on top of the tissue. Inspiration: BrainGlobe, ABBA, brainreg.
Critique: needs the atlas → sample registration to exist; we provide
the *display* only. Worth doing as an integration showcase with
BrainGlobe.

**95. QuPath/GeoJSON annotation import.** Read GeoJSON and render as a
shapes layer. Inspiration: QuPath, napari-annotator. Critique: more an
I/O job for `spatialdata-io`; we just consume the resulting shapes.

**96. Text callouts with leader lines.** Annotate regions with text +
arrow. Inspiration: every figure-making tool. Critique: matplotlib
already supports this; we'd ship a thin `pl.annotate(region, text)`
helper. Cheap polish, ship.

**97. Differential ROI side-by-side.** Render same gene in two ROIs +
bar of their difference. Inspiration: SpatialDM, custom pathology
figures. Critique: helper rather than primitive. Lives in a notebook.

---

## N. Power-user / infrastructure (98–100)

**98. `pl.transform(fn)` chain step.** Apply a callable to the feature
vector before colour mapping (e.g. log1p, sqrt, z-score per group).
Inspiration: ggplot `scale_*_continuous(trans=...)`, datashader pipeline.
Critique: `transfunc` already exists per-element. The novelty is making
it a chain-level step that applies to all subsequent renders. Mild
ergonomic win.

**99. Plot-spec (de)serialisation.** Dump the plotting tree as
JSON/YAML; reload to reproduce. Inspiration: ggplot grammar of graphics,
Vega-Lite, Altair. Critique: **uniquely possible because of the
declarative chain.** Implementing it forces the chain to be data, not
code — a healthy discipline. Risk: kwargs that are callables (`norm=`,
`transfunc=`) don't serialise; need a documented subset of
"serialisable specs". High value for reproducibility, reviews, CI
snapshot testing.

**100. Plot-spec diff.** Given two saved specs, show layer-by-layer
diff. Inspiration: nbdime, kitty diff. Critique: useful in PR review of
notebooks; without #99 it can't exist. Cheap once #99 ships.

---

## Critical cross-cutting observations

A few patterns emerged while writing this list. They're more important
than any individual idea.

**Pattern 1: The package is being asked to be three things at once.**
A renderer of SpatialData objects (its real job), a *statistics* layer
(LISA, neighborhood enrichment, etc.), and an *image-processing* layer
(CLAHE, flat-field, spillover). Most of category F and several entries
in A and K should be refused on principle and pushed to upstream
libraries. Saying "no" here protects the architecture.

**Pattern 2: Marginal axes are the single biggest unlock.** Ideas #4,
#5, #10, #16, #21, #25, #29 all want the same thing: a sanctioned way
to attach a small statistical panel (curve, distribution, profile) to a
spatial axes. Ship that primitive once and ~10 ideas become tractable.

**Pattern 3: Vector fields are the second biggest unlock.** Ideas #11,
#14, #30, #58, #90, #92 all want quiver/streamline rendering of
per-cell or per-pixel 2-vectors. One primitive, six use-cases.

**Pattern 4: Cohort/multi-sample faceting is the most user-visible win.**
Idea #51 alone replaces hundreds of lines of user boilerplate per
project. It's the single most likely "I keep telling people to install
spatialdata-plot just for this" feature.

**Pattern 5: SpatialData-unique features are under-exploited.** Ideas
#39 (transform preview), #88 (transform-chain annotation), and #99
(spec serialisation) are uniquely possible *because* of SpatialData's
abstractions. No competitor can ship these. They should rank above
generic plotting niceties.

**Pattern 6: Cartography has a 50-year head start.** Ideas #71 (Spaco),
#73 (Jenks breaks), #74 (bivariate choropleth), #76 (locator inset),
#79 (hatched palettes) come from GIS conventions and are well understood
there. We're rediscovering them.

**Pattern 7: Interactivity should leave the package.** Ideas #59–#64
either bridge to existing tools (good) or duplicate them (bad).
Don't build interactivity in-tree.

**Pattern 8: A lot of "ideas" are documentation.** Probably 25–30 of
the entries above are not features at all — they're recipes that work
with the current API. The package would benefit from a "cookbook" page
more than from those features being implemented.

---

## My shortlist if I could pick 10

Ordered by leverage × architectural fit × novelty:

1. **#51 Cohort faceting helper** — highest user pull.
2. **#99 Plot-spec (de)serialisation** — unlocks reproducibility, CI snapshots, PR review.
3. **Marginal-axes primitive** (unlocks #4, #5, #10, #16, #21, #25, #29).
4. **Vector-field primitive** (unlocks #11, #14, #30, #58, #90, #92).
5. **#18 Niche-boundary isocontours** — the iconic niche figure, well.
6. **#85 Distance-from-tissue-edge overlay** — novel, scientifically useful.
7. **#76 Locator-overview inset** — small change, large QoL.
8. **#71 Spaco-style spatial palette** — fixes a real and pervasive bug.
9. **#39 Pre/post-transform preview** — uniquely SpatialData.
10. **#82 Saturation overlay + #86 out-of-tissue warning** — QC primitives nobody ships.

Notable absences from the shortlist: subcellular pattern overlays (#26)
because they're upstream-defined; CCC arrows (#11) without bundling;
anything 3D; anything that silently transforms displayed data.
