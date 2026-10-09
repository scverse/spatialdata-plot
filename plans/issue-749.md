# Issue #749: `pl.show()` ValueError — axes/panels count mismatch

## Root cause

When a SpatialElement carries transformations to **multiple coordinate systems**
(e.g. visium's `"<cs>"` and `"<cs>_downscaled_lowres"` pair), `filter_by_coordinate_system`
cannot strip the extra transformation (upstream spatialdata #176). `show()` then
auto-detects **more coordinate systems than the user intended**. When the user
passes a single `ax`, `_plan_panels` raises:

```
ValueError: Mismatch between number of matplotlib axes objects (1) and number of panels (2).
```

PR #580 added a `strict_cs` narrowing (keep only CS with element types for *all*
render commands), but it does **not** help here: both coordinate systems contain
the *same* element, so both survive the filter. The two panels are redundant —
they render the identical element set, differing only by a scale transform.

Confirmed reproducible on current `main` (see `/tmp/repro_issue_749.py`).

## Proposed changes (recommended: Approach 1 — scope to the erroring path)

Extend the existing `ax is not None and cs_was_auto` block in
`_resolve_coordinate_systems` (`src/spatialdata_plot/pl/basic.py`): after the
`strict_cs` step, if `len(coordinate_systems) > n_ax`, **deduplicate coordinate
systems by their renderable-element set** (via `_get_elements_to_be_rendered`),
keeping the first representative of each distinct set. If that brings the count
down to `<= n_ax`, use the deduplicated list and emit a `UserWarning` naming the
dropped (redundant) coordinate systems and pointing to `coordinate_systems=`.
If the sets are genuinely distinct (count still `> n_ax`), fall through to the
existing, correct `ValueError`.

| File | Change | Rationale |
|------|--------|-----------|
| `src/spatialdata_plot/pl/basic.py` (`_resolve_coordinate_systems`) | After `strict_cs`, dedup redundant CS by element set when `ax` given + auto-detected; warn | Fixes the mismatch only on the path that currently errors → strictly backward compatible |
| `tests/pl/test_show.py` | Regression test: multi-CS element + single `ax` no longer raises; distinct-CS case still raises | Lock behavior |

Why scope to the `ax`-provided path only: the no-`ax` case currently produces
one panel per coordinate system (redundant but not an error). Collapsing that
too would change existing multi-panel output / baselines — a backward-compat
break the reporter explicitly asked to avoid. The `ax` path currently *errors*,
so fixing it breaks nothing that worked before.

## Edge cases
- [ ] 2 CS, same element set, single `ax` → collapse to 1, warn, render (main fix).
- [ ] 2 CS, **different** element sets, single `ax` → still raise ValueError (correct; genuinely 2 panels).
- [ ] N CS redundant, `ax` is a list of N → counts already match, no change.
- [ ] N CS redundant, `ax` list shorter than distinct-set count → raise (correct).
- [ ] `coordinate_systems=` passed explicitly → `cs_was_auto` False, block skipped, unchanged.
- [ ] No `ax` → block skipped, unchanged (still multi-panel).
- [ ] Multi-panel `color=[...]` → single CS required already; unaffected.

## Downstream impact
- Public API unchanged (no new/changed kwargs).
- Only converts a previously-raised `ValueError` into a successful render + warning.
- No baseline image changes expected (new path only exercised by the regression test, which is non-visual).

## Test plan
- [ ] Regression test (non-visual): multi-CS element, `filter_by_coordinate_system`, `render_shapes().show(ax=ax)` → no raise; assert 1 axis used.
- [ ] Negative test: 2 CS with different elements + single `ax` → still raises `ValueError`.
- [ ] Assert a `UserWarning` is emitted mentioning the dropped CS.

## Risks
- Representative choice is "first in coordinate-system order", which may be the
  `_downscaled_lowres` variant. Visually identical for a standalone plot (axes
  autoscale), but a user overlaying multiple `show()` calls on one `ax` should
  still pass `coordinate_systems=`. The warning makes the choice explicit.
- Low risk overall: new code runs only where the code previously raised.

## Alternative (Approach 2 — not recommended)
Deduplicate redundant coordinate systems in auto-detect for **both** `ax` and
no-`ax` paths. More internally consistent, but changes existing no-`ax`
multi-panel output and possibly visual baselines → backward-incompatible.
