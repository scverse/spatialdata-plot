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
- Layers beneath the selector in v0: images only. Selector attaches to the
  `Axes` returned by the existing `sdata.pl.render_images().pl.show()` pipeline
  — we reuse the existing canvas, no duplicate render path.
- Backend: `%matplotlib widget` (ipympl) + `matplotlib.widgets.{Rectangle,
  Polygon,Lasso}Selector`. Pure server-side render, PNG frames over websocket.
  No bokeh/datashader.

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
    element="he_image",
    coordinate_system="global",
    channel=[0, 1, 2],            # optional
    clims=(0, 30000),             # optional
    selector="polygon",           # 'rectangle' | 'polygon' | 'lasso'
    name="tumor_region",
    overwrite=False,
    persist=True,
    max_render_pixels=2_000_000,
)
session.show()                    # returns the ipympl Figure
# user draws on canvas, double-click / release to commit
sdata["tumor_region"]             # ShapesModel
sub = sdata.query.polygon(sdata, sdata["tumor_region"])
```

## Module layout

```
src/spatialdata_plot/pl/interactive/
  __init__.py        # exports InteractiveSession
  _session.py        # InteractiveSession class, public entrypoint
  _render.py         # thin wrapper around existing render_images
  _downsample.py     # pyramid-aware scale picker; in-memory coarsen
  _selectors.py      # RectangleAdapter, PolygonAdapter, LassoAdapter
  _commit.py         # vertices → CS-correct shapely → ShapesModel
  _persist.py        # write_element + overwrite/timestamp policy

tests/test_interactive/
  test_commit.py
  test_downsample.py
  test_selectors_headless.py
```

`sdata.pl.interactive(...)` becomes a method on `PlotAccessor` in
`src/spatialdata_plot/_accessor.py`, returning an `InteractiveSession`.

## Coordinate-system rules (highest-risk surface)

1. Session is bound to ONE coordinate system at construction.
2. Render is in that CS; axes coords on the canvas equal coords in the CS
   (1:1).
3. On commit, vertices are already in the rendered CS — no transform needed
   for the selection itself.
4. The committed ShapesModel is registered with `{cs_name: Identity()}`.
5. Cross-CS selection is the user's job downstream. Not v0.

Avoids the classic double-applied-transform bug.

## Downsampling

`_downsample.pick_scale(image, bbox, max_pixels) -> (level_or_factor, array)`

- `MultiscaleSpatialImage`: walk scales coarse→fine, pick finest within budget.
- Single-scale: `dask.array.coarsen` with integer factor, warn once.
- Static extent in v0. Auto-redraw on `xlim_changed` is v1.
- Default `max_render_pixels ≈ 2M` (~1500×1500), tuned for ipympl PNG over SSH.

## Selector adapters

| kind        | matplotlib class      | commit trigger                |
|-------------|-----------------------|-------------------------------|
| rectangle   | `RectangleSelector`   | mouse release                 |
| polygon     | `PolygonSelector`     | close (double-click / enter)  |
| lasso       | `LassoSelector`       | mouse release                 |

Lasso vertices simplified via `shapely.simplify(tolerance=0.5px)` before
persist.

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

- Unit: `_commit` (synthetic vertices → ShapesModel correctness).
- Unit: `_downsample` (scale picker correctness on synthetic arrays).
- Headless: `_selectors` via programmatic `_press`/`_onmove`/`_release`.
- NO visual tests in v0. CI does not need a live canvas.
- Manual checklist in PR description for the canvas itself.

## Dependencies

`[project.dependencies]`:

- `ipympl` (NEW)
- `ipywidgets` (NEW or pin existing transitive)
- `shapely` (already transitive via geopandas)
- `geopandas` (already transitive via spatialdata)

Only `ipympl` is genuinely new.

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
