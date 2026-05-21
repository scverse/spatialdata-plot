"""Generate mock visualisations for the 10 proposed spatialdata-plot features.

These are *not* implementations — they use raw matplotlib to illustrate what
the proposed APIs would render.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy import ndimage
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

OUT = Path(__file__).parent
RNG = np.random.default_rng(7)

# --------------------------------------------------------------------------
# Synthetic spatial data shared across mocks
# --------------------------------------------------------------------------

H, W = 600, 800


def make_tissue() -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    cy, cx = H / 2, W / 2 + 30
    angle = np.arctan2(yy - cy, xx - cx)
    radius = 200 + 35 * np.sin(angle * 4) + 15 * np.cos(angle * 7)
    r = np.sqrt(((xx - cx) / 1.1) ** 2 + ((yy - cy) / 0.95) ** 2)
    return (r < radius).astype(bool)


def poisson_sample(mask: np.ndarray, n: int, min_d: float = 18) -> np.ndarray:
    ys, xs = np.where(mask)
    order = RNG.permutation(len(ys))
    chosen: list[np.ndarray] = []
    for i in order:
        p = np.array([xs[i], ys[i]], dtype=float)
        if not chosen:
            chosen.append(p)
            continue
        arr = np.array(chosen)
        if np.min(np.linalg.norm(arr - p, axis=1)) > min_d:
            chosen.append(p)
        if len(chosen) >= n:
            break
    return np.array(chosen)


def make_cell_types(cells: np.ndarray, n_types: int = 6) -> np.ndarray:
    # Niche-like blocks via KMeans on positions + jitter for realism
    feats = cells + RNG.normal(0, 25, size=cells.shape)
    km = KMeans(n_clusters=n_types, n_init=10, random_state=0).fit(feats)
    return km.labels_


def make_gene(cells: np.ndarray) -> np.ndarray:
    c1 = np.array([260, 200])
    c2 = np.array([560, 380])
    d1 = np.exp(-np.sum((cells - c1) ** 2, axis=1) / (2 * 110**2))
    d2 = np.exp(-np.sum((cells - c2) ** 2, axis=1) / (2 * 90**2))
    return np.clip(3 * d1 + 2 * d2 + RNG.normal(0, 0.18, size=len(cells)), 0, None)


def make_image(tissue: np.ndarray) -> np.ndarray:
    # DAPI-like nuclear field
    noise = RNG.normal(0.05, 0.02, size=tissue.shape)
    dist = ndimage.distance_transform_edt(tissue)
    base = np.clip(dist / dist.max(), 0, 1) * 0.6
    nuclei = np.zeros_like(base)
    n_nuclei = 1500
    ys, xs = np.where(tissue)
    idx = RNG.choice(len(ys), size=n_nuclei, replace=False)
    for y, x in zip(ys[idx], xs[idx]):
        nuclei[y, x] = 1
    nuclei = ndimage.gaussian_filter(nuclei, sigma=2.5)
    nuclei /= nuclei.max()
    img = (base * 0.3 + nuclei * 1.4 + noise) * tissue
    return np.clip(img, 0, 1)


print("generating synthetic data...")
TISSUE = make_tissue()
CELLS = poisson_sample(TISSUE, n=600, min_d=20)
TYPES = make_cell_types(CELLS, n_types=6)
GENE = make_gene(CELLS)
IMAGE = make_image(TISSUE)

# Cell-type palette used across mocks
PALETTE_DEFAULT = np.array(plt.get_cmap("tab10").colors)[:6]
CT_NAMES = [f"CT{i}" for i in range(6)]


def _setup_ax(ax, title: str | None = None) -> None:
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)


# --------------------------------------------------------------------------
# 1. pl.cohort(...) — multi-sample faceting with shared color scale
# --------------------------------------------------------------------------


def mock_01_cohort() -> None:
    samples = ["donor_A", "donor_B", "donor_C", "donor_D", "donor_E", "donor_F"]
    n = len(samples)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.3), constrained_layout=True)
    fig.suptitle(
        'sdata.pl.render_shapes(color="CD8A").pl.cohort(by="sample_id", ncols=3, share_color=True).pl.show()',
        fontsize=10, family="monospace",
    )

    vmin, vmax = 0, 4.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("magma")

    for k, (ax, name) in enumerate(zip(axes.flat, samples)):
        # Per-sample slight jitter to fake different tissues
        shift = RNG.normal(0, 30, size=(1, 2))
        scale = 0.85 + 0.15 * RNG.random()
        cells = (CELLS - [W / 2, H / 2]) * scale + [W / 2, H / 2] + shift
        gene = GENE * (0.7 + 0.5 * RNG.random()) + RNG.normal(0, 0.1, size=len(GENE))
        ax.scatter(cells[:, 0], cells[:, 1], c=gene, cmap=cmap, norm=norm,
                   s=10, edgecolors="none")
        _setup_ax(ax, name)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cbar.set_label("CD8A (shared scale)", fontsize=9)
    fig.savefig(OUT / "mock_01_cohort.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. pl.render_vectors(...) — vector field primitive
# --------------------------------------------------------------------------


def mock_02_vectors() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    fig.suptitle(
        'sdata.pl.render_shapes(color="cell_type", alpha=0.5).pl.render_vectors("commot_flux", style=...).pl.show()',
        fontsize=10, family="monospace",
    )

    # Common cell layer
    palette = ListedColormap(PALETTE_DEFAULT)
    for ax in axes:
        ax.scatter(CELLS[:, 0], CELLS[:, 1], c=TYPES, cmap=palette, s=8,
                   alpha=0.35, edgecolors="none")

    # Vector field — synthetic "signalling flux" radiating outward from a source
    src = np.array([260, 220])
    gy, gx = np.mgrid[40:H:40, 40:W:40]
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = TISSUE[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    pts = pts[inside]
    dx = pts[:, 0] - src[0]
    dy = pts[:, 1] - src[1]
    r = np.hypot(dx, dy) + 1e-3
    mag = np.exp(-r / 150)
    u, v = dx / r * mag, dy / r * mag

    # left: quiver
    axes[0].quiver(pts[:, 0], pts[:, 1], u, v, mag, cmap="viridis",
                   scale=8, width=0.004)
    _setup_ax(axes[0], 'style="quiver"  — directed CCC / velocity / warp arrows')

    # right: streamlines
    # Re-grid into a regular field for streamplot
    Y, X = np.mgrid[0:H:60j, 0:W:80j]
    DX = X - src[0]
    DY = Y - src[1]
    R = np.hypot(DX, DY) + 1e-3
    M = np.exp(-R / 150)
    U, V = DX / R * M, DY / R * M
    # blank outside tissue
    mask = TISSUE[Y.astype(int).clip(0, H - 1), X.astype(int).clip(0, W - 1)]
    U[~mask] = np.nan
    V[~mask] = np.nan
    axes[1].streamplot(X, Y, U, V, color=M, cmap="viridis", density=1.2,
                       linewidth=1.1, arrowsize=1.0)
    _setup_ax(axes[1], 'style="streamline"  — same primitive, integral curves')

    for ax in axes:
        ax.scatter(*src, marker="*", s=220, c="white", edgecolors="black", zorder=5)
        ax.text(src[0] + 8, src[1] - 12, "source", fontsize=8, color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))

    fig.savefig(OUT / "mock_02_vectors.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. pl.with_margins(...) — sanctioned marginal axes
# --------------------------------------------------------------------------


def mock_03_margins() -> None:
    fig = plt.figure(figsize=(9.5, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 4, 0.001],
                          width_ratios=[4, 1, 0.001])
    fig.suptitle(
        'sdata.pl.render_shapes(color="CD8A").pl.with_margins(top="histogram", right="distance_decay").pl.show()',
        fontsize=10, family="monospace",
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    sc = ax_main.scatter(CELLS[:, 0], CELLS[:, 1], c=GENE, cmap="magma",
                         s=14, edgecolors="none")
    _setup_ax(ax_main)
    ax_main.set_xlim(0, W)
    ax_main.set_ylim(H, 0)
    ax_main.set_aspect("equal")

    # Top marginal: expression histogram along X (binned by x)
    bins = np.linspace(0, W, 40)
    digit = np.digitize(CELLS[:, 0], bins) - 1
    mean_per_bin = np.array([GENE[digit == i].mean() if (digit == i).any() else 0
                             for i in range(len(bins) - 1)])
    ax_top.bar((bins[:-1] + bins[1:]) / 2, mean_per_bin,
               width=(bins[1] - bins[0]) * 0.9, color="#444")
    ax_top.set_ylabel("mean", fontsize=9)
    ax_top.tick_params(labelbottom=False, labelsize=8)
    ax_top.set_title("top = expression profile along x", fontsize=9, loc="left")

    # Right marginal: distance decay (gene vs distance to a source)
    src = np.array([260, 200])
    d = np.linalg.norm(CELLS - src, axis=1)
    order = np.argsort(d)
    smooth = np.convolve(GENE[order], np.ones(20) / 20, mode="same")
    ax_right.plot(smooth, d[order], color="#7a4cdb", lw=1.4)
    ax_right.set_xlabel("smoothed", fontsize=9)
    ax_right.set_title("right = distance-from-source decay",
                       fontsize=9, loc="left")
    ax_right.tick_params(labelleft=False, labelsize=8)
    ax_right.invert_yaxis()

    cbar = fig.colorbar(sc, ax=[ax_main, ax_right], shrink=0.7, pad=0.02)
    cbar.set_label("CD8A", fontsize=9)
    fig.savefig(OUT / "mock_03_margins.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. contour_by= — niche/cluster boundary contours
# --------------------------------------------------------------------------


def mock_04_contours() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    fig.suptitle(
        'sdata.pl.render_labels(color="cell_type", contour_by="niche").pl.show()',
        fontsize=10, family="monospace",
    )

    palette = ListedColormap(PALETTE_DEFAULT)

    # Left: current state (just colored dots, niches invisible)
    axes[0].imshow(IMAGE, cmap="gray", alpha=0.4, extent=(0, W, H, 0))
    axes[0].scatter(CELLS[:, 0], CELLS[:, 1], c=TYPES, cmap=palette,
                    s=12, edgecolors="none")
    _setup_ax(axes[0], "current — niches drown in colored dots")

    # Right: with boundary contours per niche label
    axes[1].imshow(IMAGE, cmap="gray", alpha=0.4, extent=(0, W, H, 0))
    axes[1].scatter(CELLS[:, 0], CELLS[:, 1], c=TYPES, cmap=palette,
                    s=10, alpha=0.55, edgecolors="none")
    for k in range(6):
        members = CELLS[TYPES == k]
        if len(members) < 6:
            continue
        try:
            hull = ConvexHull(members)
            poly = members[hull.vertices]
            # Smooth via mean of consecutive points
            smooth = np.vstack([
                (poly[i] + poly[(i + 1) % len(poly)]) / 2 for i in range(len(poly))
            ])
            patch = mpatches.Polygon(smooth, closed=True, fill=False,
                                     edgecolor="black", linewidth=1.7)
            axes[1].add_patch(patch)
            cx, cy = members.mean(axis=0)
            axes[1].text(cx, cy, CT_NAMES[k], ha="center", va="center",
                         fontsize=9, weight="bold",
                         bbox=dict(facecolor="white", edgecolor="black",
                                   boxstyle="round,pad=0.2", alpha=0.85))
        except Exception:
            continue
    _setup_ax(axes[1], 'contour_by="niche" — boundary is the figure, color the context')

    fig.savefig(OUT / "mock_04_contours.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. palette="spaco" — spatial-aware palette
# --------------------------------------------------------------------------


def mock_05_spaco() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    fig.suptitle(
        'sdata.pl.render_shapes(color="cell_type", palette="spaco")  — adjacent niches are distinguishable',
        fontsize=10, family="monospace",
    )

    # Bad palette: two adjacent niches get near-identical greens
    bad = np.array([
        [0.86, 0.20, 0.18],   # red
        [0.18, 0.45, 0.71],   # blue
        [0.27, 0.62, 0.32],   # green-A
        [0.31, 0.68, 0.36],   # green-B (intentionally close to A)
        [0.96, 0.55, 0.14],   # orange
        [0.55, 0.40, 0.78],   # purple
    ])

    # Spaco-style: greedy max-distance palette colors swapped so neighbours differ
    spaco = bad.copy()
    spaco[[2, 5]] = spaco[[5, 2]]  # swap so the two close greens aren't neighbours
    spaco[3] = [0.93, 0.78, 0.18]  # replace 2nd green with mustard

    for ax, pal, title in zip(axes, [bad, spaco],
                              ['palette="tab10" (default)',
                               'palette="spaco" (spatial-aware)']):
        cmap = ListedColormap(pal)
        ax.scatter(CELLS[:, 0], CELLS[:, 1], c=TYPES, cmap=cmap, s=20,
                   edgecolors="none")
        _setup_ax(ax, title)
        # Legend
        handles = [mpatches.Patch(facecolor=pal[i], label=CT_NAMES[i])
                   for i in range(6)]
        ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True,
                  ncol=2)

    axes[0].annotate("hard to tell\nthese two apart →",
                     xy=(420, 320), xytext=(80, 80),
                     fontsize=10, color="#b00",
                     arrowprops=dict(arrowstyle="->", color="#b00"))

    fig.savefig(OUT / "mock_05_spaco.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. highlight_unassigned=True — orphan transcripts
# --------------------------------------------------------------------------


def mock_06_orphan() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    fig.suptitle(
        'sdata.pl.render_points("transcripts", highlight_unassigned=True, na_color="red").pl.show()',
        fontsize=10, family="monospace",
    )

    # Make transcripts: most inside cells, ~12% orphan around edges
    n_tx = 4000
    cell_idx = RNG.integers(0, len(CELLS), size=n_tx)
    offsets = RNG.normal(0, 6, size=(n_tx, 2))
    tx = CELLS[cell_idx] + offsets
    # Mark ~12% as orphan by drifting them away from any cell
    n_orphan = int(0.12 * n_tx)
    orphan_idx = RNG.choice(n_tx, size=n_orphan, replace=False)
    tx[orphan_idx] += RNG.normal(0, 35, size=(n_orphan, 2))
    # Recompute "assigned" by distance to nearest cell
    from scipy.spatial import cKDTree
    tree = cKDTree(CELLS)
    d, _ = tree.query(tx)
    assigned = d < 12

    # Left: current — everything one color
    axes[0].imshow(IMAGE, cmap="gray", alpha=0.4, extent=(0, W, H, 0))
    axes[0].scatter(tx[:, 0], tx[:, 1], c="#4a9", s=1, alpha=0.4)
    _setup_ax(axes[0], f"current — {n_tx} transcripts, no QC signal")

    # Right: orphans highlighted
    axes[1].imshow(IMAGE, cmap="gray", alpha=0.4, extent=(0, W, H, 0))
    axes[1].scatter(tx[assigned, 0], tx[assigned, 1], c="#4a9", s=1, alpha=0.35,
                    label="assigned")
    axes[1].scatter(tx[~assigned, 0], tx[~assigned, 1], c="red", s=3,
                    alpha=0.95, label=f"orphan ({(~assigned).sum() / n_tx:.0%})")
    axes[1].legend(loc="lower right", fontsize=8, frameon=True)
    _setup_ax(axes[1], "highlight_unassigned=True — segmentation QC in one call")

    fig.savefig(OUT / "mock_06_orphan.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. locator_inset=True — overview thumbnail showing the current crop
# --------------------------------------------------------------------------


def mock_07_locator() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.3), constrained_layout=True)
    fig.suptitle(
        'sdata.pl.render_images().pl.render_shapes(color="CD8A").pl.show(crop=(...), locator_inset=True)',
        fontsize=10, family="monospace",
    )

    # Define a crop window
    crop = (380, 200, 700, 480)  # x0, y0, x1, y1
    x0, y0, x1, y1 = crop
    in_crop = ((CELLS[:, 0] > x0) & (CELLS[:, 0] < x1)
               & (CELLS[:, 1] > y0) & (CELLS[:, 1] < y1))
    ax.imshow(IMAGE[y0:y1, x0:x1], cmap="gray", alpha=0.6,
              extent=(x0, x1, y1, y0))
    sc = ax.scatter(CELLS[in_crop, 0], CELLS[in_crop, 1],
                    c=GENE[in_crop], cmap="magma", s=40, edgecolors="white",
                    linewidths=0.4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
    ax.set_title("zoomed crop", fontsize=10)

    # Inset locator
    ins = ax.inset_axes([0.02, 0.65, 0.27, 0.33])
    ins.imshow(IMAGE, cmap="gray", alpha=0.7, extent=(0, W, H, 0))
    ins.scatter(CELLS[:, 0], CELLS[:, 1], c="#999", s=1, alpha=0.6)
    ins.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                            fill=False, edgecolor="red", linewidth=1.8))
    ins.set_xlim(0, W)
    ins.set_ylim(H, 0)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_edgecolor("black")
    ins.set_title("locator", fontsize=8)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("CD8A", fontsize=9)
    fig.savefig(OUT / "mock_07_locator.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 8. render_images(image_mask=<labels>) — image restricted to segmentation
# --------------------------------------------------------------------------


def mock_08_image_mask() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    fig.suptitle(
        'sdata.pl.render_images("DAPI", image_mask="cells").pl.show()',
        fontsize=10, family="monospace",
    )

    # Build a cell mask = union of small circles around CELLS
    yy, xx = np.mgrid[:H, :W]
    cell_mask = np.zeros((H, W), dtype=bool)
    for cx, cy in CELLS:
        cell_mask |= ((xx - cx) ** 2 + (yy - cy) ** 2) < 8**2

    axes[0].imshow(IMAGE, cmap="magma", extent=(0, W, H, 0))
    _setup_ax(axes[0], "current — raw image incl. extracellular leak")

    masked = np.where(cell_mask, IMAGE, np.nan)
    axes[1].imshow(masked, cmap="magma", extent=(0, W, H, 0),
                   interpolation="nearest")
    _setup_ax(axes[1], 'image_mask="cells" — signal only inside segmentations')

    fig.savefig(OUT / "mock_08_image_mask.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 9. distance_from_edge=True — distance-to-tissue-boundary as a feature
# --------------------------------------------------------------------------


def mock_09_edge_distance() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    fig.suptitle(
        'sdata.pl.render_shapes(color="distance_from_edge", tissue_mask="tissue").pl.show()',
        fontsize=10, family="monospace",
    )

    # Compute edge distance via distance transform
    dist = ndimage.distance_transform_edt(TISSUE)
    cell_d = dist[CELLS[:, 1].astype(int).clip(0, H - 1),
                  CELLS[:, 0].astype(int).clip(0, W - 1)]

    # Find tissue boundary for outline
    edges = TISSUE ^ ndimage.binary_erosion(TISSUE)
    eys, exs = np.where(edges)

    # Left: tissue mask + edge
    axes[0].imshow(TISSUE.astype(float), cmap="gray", alpha=0.25,
                   extent=(0, W, H, 0))
    axes[0].scatter(exs, eys, s=0.3, c="black")
    axes[0].scatter(CELLS[:, 0], CELLS[:, 1], c="#666", s=8, edgecolors="none")
    _setup_ax(axes[0], "current — distance-from-edge is implicit / hard to see")

    # Right: cells colored by edge distance, with isobands
    axes[1].scatter(exs, eys, s=0.3, c="black")
    sc = axes[1].scatter(CELLS[:, 0], CELLS[:, 1], c=cell_d, cmap="viridis",
                         s=14, edgecolors="none")
    # Isolines of distance every 25 px
    Xg, Yg = np.meshgrid(np.arange(W), np.arange(H))
    levels = np.arange(25, dist.max(), 35)
    axes[1].contour(Xg, Yg, dist, levels=levels, colors="white",
                    linewidths=0.6, alpha=0.5)
    _setup_ax(axes[1], 'color="distance_from_edge" — first-class spatial covariate')
    cbar = fig.colorbar(sc, ax=axes[1], shrink=0.8, pad=0.02)
    cbar.set_label("distance to tissue edge (px)", fontsize=9)

    fig.savefig(OUT / "mock_09_edge_distance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 10. pl.to_yaml() / pl.from_yaml() — spec serialisation
# --------------------------------------------------------------------------


def mock_10_spec_yaml() -> None:
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    fig.suptitle(
        "sdata.pl.render_images(...).pl.render_shapes(...).pl.to_yaml()",
        fontsize=10, family="monospace",
    )

    # Left: produced figure
    ax_fig = fig.add_subplot(gs[0, 0])
    ax_fig.imshow(IMAGE, cmap="gray", alpha=0.5, extent=(0, W, H, 0))
    ax_fig.scatter(CELLS[:, 0], CELLS[:, 1], c=GENE, cmap="magma", s=12,
                   edgecolors="none")
    _setup_ax(ax_fig, "figure (what users see)")

    # Right: the YAML spec that produces the figure
    yaml_text = textwrap.dedent("""\
    # auto-generated by pl.to_yaml()
    spatialdata_plot:
      version: 1
      layers:
        - kind: render_images
          element: DAPI
          channel: 0
          cmap: gray
          alpha: 0.5
          norm: {type: linear, vmin: 0.0, vmax: 1.0}

        - kind: render_shapes
          element: cells
          color: CD8A
          table: gene_table
          cmap: magma
          colorbar: auto
          shape: circle
          fill_alpha: 1.0

      figure:
        figsize: [6, 4.5]
        dpi: 140
        scalebar:
          dx: 0.65   # microns / pixel
          units: um

    # round-trip:
    #   spec = sdata.pl....pl.to_yaml()
    #   fig  = sdata.pl.from_yaml(spec).pl.show()
    #
    # use cases:
    #   - snapshot tests in CI ("does the spec match golden?")
    #   - share a plot recipe in a PR or a paper supplement
    #   - diff two specs to review what changed""")

    ax_txt = fig.add_subplot(gs[0, 1])
    ax_txt.text(0.0, 1.0, yaml_text, family="monospace", fontsize=9,
                va="top", ha="left", color="#222")
    ax_txt.set_facecolor("#f7f7f0")
    ax_txt.set_xticks([])
    ax_txt.set_yticks([])
    for s in ax_txt.spines.values():
        s.set_edgecolor("#888")

    fig.savefig(OUT / "mock_10_spec_yaml.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("rendering mocks...")
    mock_01_cohort()
    print("  01 cohort        ✓")
    mock_02_vectors()
    print("  02 vectors       ✓")
    mock_03_margins()
    print("  03 margins       ✓")
    mock_04_contours()
    print("  04 contours      ✓")
    mock_05_spaco()
    print("  05 spaco         ✓")
    mock_06_orphan()
    print("  06 orphan        ✓")
    mock_07_locator()
    print("  07 locator       ✓")
    mock_08_image_mask()
    print("  08 image_mask    ✓")
    mock_09_edge_distance()
    print("  09 edge_distance ✓")
    mock_10_spec_yaml()
    print("  10 spec_yaml     ✓")
    print("done.")
