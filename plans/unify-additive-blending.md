# Plan: Unify Multi-Channel Image Compositing to Additive Blending

## Goal

Replace the inconsistent compositing formulas in `_render_images` with a single shared helper that implements standard additive blending with clamping (matching Napari/ImageJ/FIJI).

## Current State (investigation findings)

### Compositing paths in `render.py` `_render_images`

| Path | Lines | Condition | Formula | Alpha | Bug |
|------|-------|-----------|---------|-------|-----|
| **1** | 1232-1254 | 1 channel | Direct (no compositing) | Baked in cmap | None |
| **2A-RGB** | 1282-1283 | 3ch, default cmap | `np.stack(..., axis=-1)` | Via imshow | None |
| **2A-cmap** | 1284-1293 | 3ch, user cmap | `.sum(0) / n_channels` | Via imshow | **Averaging** |
| **2B** 2ch | 1314-1321 | 2ch, no palette | `.sum(0)`, no clip | Via imshow | **No clip** |
| **2B** 3ch | 1322-1329 | 3ch, no palette | `.sum(0)`, no clip | Via imshow | **No clip** |
| **2B** 4+ch | 1330-1363 | 4+ch, no palette | additive + `np.clip` | **Baked + imshow** | **Double alpha** |
| **2C** | 1374-1380 | palette | `.sum(0)`, no clip | Via imshow | **No clip + filter bug** |
| **2D** | 1390-1398 | multiple cmaps | `.sum(0) / n_channels` | Via imshow | **Averaging** |

### Key infrastructure details

- **`layers` dict** (L1269-1288): Keys are channel identifiers (str or int). Values are normalized arrays (norm applied before compositing). Constructed correctly.
- **`_get_linear_colormap`** (utils.py L1766-1767): `LinearSegmentedColormap.from_list(c, ["k", c], N=256)` — black-to-color LUTs. Returns RGBA (4 channels) when called. Correct for additive blending.
- **`_ax_show_and_transform`** (utils.py L2893-2936): When `cmap is None` and `alpha is not None`, passes `alpha` to `ax.imshow()`. When cmap is present, does NOT pass alpha. This means for composite RGB arrays, alpha is applied by imshow.
- **Palette validation**: `_type_check_params` already ensures palette contains only strings. The `if isinstance(c, str)` filter at L1378 is redundant and risks index misalignment.

### Alpha flow summary

All paths except 2B-4+ch correctly apply alpha once (via `_ax_show_and_transform` → `ax.imshow(alpha=...)`).

Path 2B-4+ch double-applies alpha:
1. L1356-1357: `rgba[..., 3] = render_params.alpha` then `comp_rgb += rgba[..., :3] * rgba[..., 3][..., None]`
2. L1369: passes `render_params.alpha` to `_ax_show_and_transform` → `ax.imshow(alpha=...)`

### Tests affected

Tests that will produce different (brighter) output when switching from averaging to additive:

| Baseline | Test | Path |
|----------|------|------|
| `Images_can_pass_str_cmap.png` | `test_plot_can_pass_str_cmap` | 2A-cmap |
| `Images_can_pass_cmap.png` | `test_plot_can_pass_cmap` | 2A-cmap |
| `Images_can_render_multiscale_image_with_custom_cmap.png` | `test_plot_can_render_multiscale_image_with_custom_cmap` | 2A-cmap |
| `Images_can_pass_str_cmap_list.png` | `test_plot_can_pass_str_cmap_list` | 2D |
| `Images_can_pass_cmap_list.png` | `test_plot_can_pass_cmap_list` | 2D |
| `Images_can_pass_cmap_to_each_channel.png` | `test_plot_can_pass_cmap_to_each_channel` | 2D |

Tests using 2B (2-3ch) or 2C may also shift slightly due to clip being added, but only if values currently exceed [0,1].

### Baseline regeneration

Per `docs/contributing.md`: baselines must be generated on Ubuntu in GitHub Actions (not locally). Push the change, let CI run, download `visual_test_results_*` artifact, review manually, copy to `tests/_images/`.

## Steps

### Step 1: Add `_additive_blend` helper

**File**: `src/spatialdata_plot/pl/render.py`

Place as a module-level function before `_render_images` (near other helpers).

```python
def _additive_blend(
    layers: dict,
    channels: list,
    channel_cmaps: list,
) -> np.ndarray:
    """Additive blend of colormapped channels, matching Napari's additive mode.

    Each channel is mapped through its colormap, the RGB components are summed,
    and the result is clamped to [0, 1].
    """
    H, W = next(iter(layers.values())).shape
    composite = np.zeros((H, W, 3), dtype=float)
    for ch_idx, ch in enumerate(channels):
        rgba = channel_cmaps[ch_idx](np.asarray(layers[ch]))
        composite += rgba[..., :3]
    return np.clip(composite, 0, 1)
```

No alpha parameter — alpha is handled uniformly by `_ax_show_and_transform` → `ax.imshow()`.

### Step 2: Update path 2A-cmap (L1284-1293)

Replace averaging with `_additive_blend`. Keep the warning about white cmaps.

### Step 3: Update path 2B 2ch (L1314-1321)

Replace `.sum(0)[:, :, :3]` with `_additive_blend(layers, channels, channel_cmaps)`.

### Step 4: Update path 2B 3ch (L1322-1329)

Same as step 3.

### Step 5: Simplify path 2B 4+ch (L1330-1363)

Replace inline loop with `_additive_blend(layers, channels, channel_cmaps)`. This removes the double-alpha bug (no more alpha bake-in; alpha is only applied via imshow).

### Step 6: Update path 2C palette (L1374-1380)

Replace `.sum(0)[:, :, :3]` with `_additive_blend`. Remove the `if isinstance(c, str)` filter.

### Step 7: Update path 2D multiple cmaps (L1390-1398)

Replace averaging with `_additive_blend`.

### Step 8: Regenerate baseline images

Push to PR branch, let CI run, download artifacts, review, commit new baselines.

## Not in scope

- PCA multichannel strategy / new parameters
- Changes to colormap selection logic
- Channel validation improvements (separate PR)
- The `norm.vmin` bug at L241 (functionally neutral when `vmin == vmax`)
- The `norm.vmax` bug from PR #451 at L258 (already fixed on main)

## Risks

1. **Visual change**: Averaging → additive makes composites brighter. Intentional and correct but requires baseline regeneration.
2. **Alpha semantics for 4+ch**: Removing double-alpha changes how `alpha < 1` looks. Since the current behavior is a bug, this is a fix.
3. **Edge case — saturated composites**: Additive sum of many channels can saturate to white. This matches Napari behavior and is expected.

## Test strategy

- Existing image comparison tests cover all paths.
- Add one direct unit test for `_additive_blend` with known inputs/outputs.
- Baselines regenerated via CI on Ubuntu.
