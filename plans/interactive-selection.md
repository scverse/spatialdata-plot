# Interactive region selection in spatialdata-plot

Status: spec (v0). Materialized from session handoff on 2026-05-21.

## Goal

A minimal, in-notebook (Jupyter / VSCode-Remote-SSH) widget that lets the user
draw a region on a spatialdata-plot canvas and persist it back into the
SpatialData object as a ShapesModel element. Works over an SSH bridge to a
SLURM compute node. No napari, no desktop GUI.

## Confirmed design decisions

- Output: persisted ShapesModel written back to the on-disk zarr via
  `sdata.write_element`. Survives kernel restarts.
- Selector shapes in v0: rectangle, polygon (click vertices), lasso (freehand).
- Scale handling: auto-downsample on the fly. Pyramid-aware when available;
  `dask.coarsen` fallback when not.
- Layers in v0: images only. The image is rendered once via the existing
  `sdata.pl.render_images().pl.show()` pipeline into a matplotlib figure,
  exported to PNG, and laid under a client-side drawing canvas.
- Backend: **custom anywidget** with HTML5/SVG drawing tools (rectangle,
  polygon, freehand-lasso). All drawing happens in the browser; shape
  geometry is reported back to Python via traitlet sync. Image is sent
  once as a base64 data URL; mouse moves never round-trip the kernel.
  No bokeh/datashader.

### Why anywidget, not ipympl or plotly

The original spec called for `%matplotlib widget` (ipympl). The prototype
revealed two showstoppers over SSH:
1. **ipympl streams PNG frames per mouse-move** over websocket — every drag
   incurs SSH round-trip latency, making freehand drawing unusable.
2. **plotly's `FigureWidget`** has broken two-way shape sync in
   VSCode-Remote-SSH (regardless of plotly 5 vs 6 — different bugs each).

A small (~250-line) anywidget with traitlet-synced shape geometry was the
only architecture that worked reliably in VSCode-Remote and produced
responsive drawing. The image render still uses sdata-plot's matplotlib
pipeline; we just don't drive interaction through it.

## Resolved questions (locked 2026-05-21, task #1)

- **Q1 — Channel/contrast widgets**: **No live widgets in v0.** `channel=` and
  `clims=` remain optional kwargs that forward to `render_images`. No
  ipywidgets-driven controls. Widget toolbar deferred to v1.
- **Q2 — Auto-redraw on zoom**: **v1.** v0 renders once at the chosen scale;
  `xlim_changed`/`ylim_changed` does not re-pick pyramid level. Static extent
  ships sooner.
- **Q3 — Selector kind switching**: **One per call.** `selector=` is fixed at
  session construction; no mid-session switching. Switchable kinds deferred to
  v1.
- **Q4 — `name=` default**: **Required.** No default; omitting `name=` raises.
  Keeps persisted element names intentional and zarr listings legible.

## Public API sketch

```python
import spatialdata_plot  # registers .pl

session = sdata.pl.interactive(
    coordinate_system=None,   # optional pre-selection; None = let user pick in UI
    element=None,             # optional pre-selection; None = let user pick in UI
    persist=True,             # show "Write to disk" button (False = memory only)
)
session.show()                # renders the ipywidgets controls + draw canvas

# User picks CS + image, clicks Render, draws shapes, names + Saves each set.
# Each Save adds an entry to sdata.shapes (memory). Write to disk persists
# the most recent commit via sdata.write_element.

sdata["tumor_region"]         # ShapesModel
sub = sdata.query.polygon(sdata, sdata["tumor_region"])
```

Removed kwargs vs original spec:
- `selector=` — UI has a tool toggle (rect/polygon/lasso); no need to bind one
  selector at construction (Q3 resolution).
- `name=` — typed in the UI before each Save (Q4 resolution).
- `channel=`, `clims=` — deferred to v1 (Q1 resolution).
- `max_render_pixels=` — render is fixed at `figsize=(7,7), dpi=120` ≈ 840×840
  PNG; pyramid-aware downsampling deferred to v1.
- `overwrite=` — collision handling is automatic: same name → append UTC
  timestamp.

## Module layout

```
src/spatialdata_plot/pl/interactive/
  __init__.py        # exports interactive, InteractiveSession, DrawCanvas
  _session.py        # InteractiveSession class — ipywidgets controls
  _canvas.py         # DrawCanvas anywidget + traitlets
  _render.py         # render_to_png helper (sdata.pl → PNG + extent)
  _commit.py         # pixel-shape → CS-correct shapely Polygon → ShapesModel
  _persist.py        # write_element + collision/timestamp policy
  static/
    draw_canvas.js   # the ESM module; _esm = Path(...) reads at import

tests/test_interactive/
  test_commit.py     # pixel→CS conversion + ShapesModel correctness
  test_render.py     # render_to_png returns valid PNG + extent
  test_persist.py    # collision/timestamp policy
  test_canvas.py     # smoke: instantiate widget, check traitlet defaults
```

`sdata.pl.interactive(...)` is a method on `PlotAccessor` in
`src/spatialdata_plot/_accessor.py`. It constructs an `InteractiveSession`
and returns it; `session.show()` displays the controls + draw canvas.

Dropped from the original spec:
- `_downsample.py` — pyramid-aware downsampling deferred to v1; v0 renders
  at a fixed dpi (`figsize=(7,7), dpi=120`).
- `_selectors.py` — matplotlib selectors are replaced by the anywidget; the
  three drawing tools (rect/polygon/lasso) live in `static/draw_canvas.js`.

## Coordinate-system rules (highest-risk surface)

1. Session is bound to ONE coordinate system at construction.
2. Render is in that CS; axes coords on the canvas equal coords in the CS
   (1:1).
3. On commit, vertices are already in the rendered CS — no transform needed
   for the selection itself.
4. The committed ShapesModel is registered with `{cs_name: Identity()}`.
5. Cross-CS selection is the user's job downstream. Not v0.

Avoids the classic double-applied-transform bug.

## Rendering

`_render.render_to_png(sdata, element, coordinate_system) -> (png_bytes, image_w, image_h, xlim, ylim)`

- Uses `sdata.pl.render_images(element=...).pl.show(coordinate_systems=..., ax=...)`.
- Axes fills the figure (`ax.add_axes([0,0,1,1])`, `set_axis_off()`) so PNG pixel
  coordinates map exactly to data coordinates via `xlim`/`ylim`.
- Fixed at `figsize=(7,7)` × `dpi=120` ≈ 840×840 PNG for v0. Pyramid-aware
  downsampling deferred to v1.
- 3D / z-stacks: refused by `render_images` itself (commit 3ebefe1) — we
  propagate that error.

## Drawing tools (in `static/draw_canvas.js`)

| kind        | gesture                                        | commit trigger                                |
|-------------|------------------------------------------------|-----------------------------------------------|
| rectangle   | left-drag corner → corner                      | mouse release                                 |
| polygon     | click each vertex                              | snap-to-first-vertex (within 10 px) or Enter  |
| lasso       | left-drag freehand                             | mouse release                                 |

Plus client-side: wheel-zoom, shift-drag-pan, alt-click-shape-to-delete,
hover-highlight, Ctrl+Z undo, Delete clear, R/P/L tool shortcuts, F fit.

Lasso vertices are simplified server-side via `shapely.simplify(tolerance=0.5)`
in `_commit` before persisting.

## Persistence policy

- `sdata.path` set → `sdata.write_element(name)` on every commit.
- Not zarr-backed → warn once, keep in memory.
- `overwrite=False` default. Collision → rename to `"<name>_<UTC-ISO>"`.
- `session.commits` list tracks names committed this session.

## Risks (pre-mitigated)

1. CS mistakes → identity transform + unit tests.
2. Image too large → `max_render_pixels` hard cap with clear error.
3. ipympl flakiness in VSCode → documented fallback to browser-Jupyter via
   `ssh -L 8888:localhost:8888 node`.
4. Walltime kill → auto-persist every commit.
5. Lasso 10k vertices → `shapely.simplify`.
6. Concurrent zarr writers → documented, no locking in v0.
7. 3D / z-stacks → refuse with same error as static render (commit 3ebefe1).
8. Auto-zoom redraw not in v0 → static extent ships first.

## Test strategy

- Unit: `_commit` (synthetic pixel-coord shapes → CS-coord ShapesModel correctness).
- Unit: `_render` (returns valid PNG bytes + extent matching the axis limits).
- Unit: `_persist` (collision-rename + timestamp policy).
- Smoke: `_canvas` (instantiate `DrawCanvas`, check traitlet defaults).
- NO visual / live-canvas tests in v0 — the JS widget can't be driven from Python.
  Manual checklist in PR description covers the canvas behaviour.

## Dependencies

Exposed as `[project.optional-dependencies].interactive` so the feature is
opt-in (`pip install spatialdata-plot[interactive]`). Mirrors the pixi
`interactive` dep-group.

- `anywidget` (NEW) — the widget framework.
- `ipywidgets` (NEW or pin existing transitive) — for the controls VBox.
- `ipykernel` — needed by anywidget for comm channel.
- `shapely`, `geopandas` — already transitive via spatialdata.

`ipympl` and `plotly` are NOT runtime deps of the new architecture (we tried
both and rejected them). They remain in the prototype/pixi feature only for
historical comparison and may be dropped from the interactive feature later.

## v1 roadmap (after v0 ships)

1. Auto-downsample on zoom (pyramid-aware redraw on `xlim_changed`).
2. Channel + contrast widget controls in the figure toolbar.
3. Labels overlay (segmentation visible during selection).
4. Multiple selectors per session; switchable kinds.
5. Datashader path for points-heavy elements.

## Task queue

1. Resolve spec open questions Q1–Q4
2. Add ipympl dep + pixi interactive feature
3. Scaffold `pl/interactive` submodule
4. Wire `sdata.pl.interactive` entrypoint
5. Implement `_commit`: vertices → ShapesModel
6. Implement `_persist`: zarr write policy
7. Implement `_downsample`: scale picker + warn
8. Implement `_render`: image render to ax
9. Implement `_selectors`: Rectangle/Polygon/Lasso adapters
10. Wire `InteractiveSession` end-to-end
11. Manual end-to-end test on cluster
12. Document feature in module docstring + README

## Operating rules

- Repo CLAUDE.md rules apply: plan-first for multi-file work, no drive-by
  refactors, run pixi-defined tasks (lint/format/test) before commits, no
  pre-commit / no visual tests locally (CI only).
- Pixi only. No venv/pip. `dev-py313` environment.
- Don't stage with `-A`; stage only what's touched.
- Human drives the actual ipympl canvas; agent cannot see it. Agent can
  drive a parallel headless kernel on the same node for non-UI checks.
- If task #1 answers change the spec materially, update this file before
  starting #2.
