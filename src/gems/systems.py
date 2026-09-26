"""Fault-system decomposition and Leave-One-Fault-System-Out (LOFSO) folds.

Why this module exists
----------------------
The competition's scored population is **expert-mapped faults that are NOT in
the supplied USGS/INGENIOUS catalogue** (verified: problem description
"Competition structure", https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/),
and the supplied catalogue pixels are **masked out of scoring, pixel-exactly**
(verified: DrivenData staff, forum topic 11516 posts 2 and 4,
https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516):

  post 2 (chrisk-dd, 2026-09-16): "Pixels corresponding to known USGS/INGENIOUS
      faults are masked / excluded from evaluation, so they do not count towards
      penalty terms." (+ same masking in the Final Round re-evaluation)
  post 4 (chrisk-dd, 2026-09-21): "The mask is indeed pixel-exact - it is
      identical to the provided set of training fault labels." ... "A predicted
      pixel that is near a known fault trace but far from a new-fault ground
      truth pixel will be fully penalized, i.e., the buffer does not apply to
      known faults." ... "A new-fault ground truth pixel can indeed lie within
      300m of a known fault trace. Such pixels would constitute corrections or
      modifications to existing fault traces."

So a faithful local emulation of the leaderboard must satisfy all three at once:

  1. the held-out truth is *not* masked (it plays the role of "new faults"),
  2. everything else in the catalogue *is* masked pixel-exactly,
  3. the held-out truth may sit within 300 m of masked catalogue pixels
     (corrections / modifications), so folds must be cut at the level of
     *fault systems* (linked segments), not at the level of single pixels and
     not at the level of 100 km map blocks that slice traces in half.

Pixel-level or block-level holdouts both break this: block folds slice traces
and leave the two halves of one fault on opposite sides (train half leaks the
test half through a 3 px kernel), which is why earlier block-fold numbers in
this repository could not be compared with leaderboard scores.

`fault_systems()` groups catalogue pixels into systems by labelling the
`link_px`-dilated mask (default 3 px = 300 m, the metric kernel radius, so
segments that can already "see" each other through the kernel are one system).
`lofso_folds()` then assigns whole systems to folds using spatially clustered
grouping (k-means on system centroids) so each fold is a set of regions, and
removes a `collar_px` ring around each held-out system from the *training*
pool (never from scoring).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["fault_systems", "system_table", "lofso_folds", "collar"]

STRUCT_3X3 = np.ones((3, 3), dtype=bool)


def fault_systems(catalogue: np.ndarray, link_px: int = 3) -> np.ndarray:
    """Label catalogue pixels into fault systems.

    Args:
        catalogue: boolean HxW array of known-fault pixels.
        link_px: dilation radius (px) used to link nearby segments into one
            system before labelling. 3 px = 300 m = the metric kernel radius.

    Returns:
        int32 HxW array, 0 = not a catalogue pixel, 1..N = system id.
    """
    cat = np.asarray(catalogue).astype(bool)
    if link_px <= 0:
        lab, _ = ndimage.label(cat, structure=STRUCT_3X3)
        return lab.astype(np.int32)
    dil = ndimage.binary_dilation(cat, structure=STRUCT_3X3, iterations=link_px)
    lab_dil, _ = ndimage.label(dil, structure=STRUCT_3X3)
    return (lab_dil * cat).astype(np.int32)


def system_table(system_ids: np.ndarray) -> np.ndarray:
    """Return an (N, 4) array of [system_id, n_px, centroid_row, centroid_col]."""
    ids = np.asarray(system_ids)
    n = int(ids.max())
    if n == 0:
        return np.zeros((0, 4), dtype=np.float64)
    counts = np.bincount(ids.ravel(), minlength=n + 1)[1:]
    h, w = ids.shape
    rr, cc = np.mgrid[0:h, 0:w]
    flat = ids.ravel()
    sr = np.bincount(flat, weights=rr.ravel(), minlength=n + 1)[1:]
    sc = np.bincount(flat, weights=cc.ravel(), minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = np.where(counts > 0, sr / np.maximum(counts, 1), 0.0)
        cc_ = np.where(counts > 0, sc / np.maximum(counts, 1), 0.0)
    out = np.column_stack([np.arange(1, n + 1), counts, cr, cc_])
    return out[out[:, 1] > 0]


def collar(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Ring of `radius_px` around `mask` (excludes the mask itself)."""
    if radius_px <= 0:
        return np.zeros_like(np.asarray(mask), dtype=bool)
    dil = ndimage.binary_dilation(np.asarray(mask).astype(bool),
                                 structure=STRUCT_3X3, iterations=radius_px)
    return dil & ~np.asarray(mask).astype(bool)


def _kmeans_clusters(points: np.ndarray, k: int, seed: int = 0,
                     iters: int = 60) -> np.ndarray:
    """Tiny deterministic k-means (no sklearn dependency) -> cluster id per row."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    # k-means++ style seeding, deterministic given seed
    idx = [int(rng.integers(n))]
    d2 = np.sum((pts - pts[idx[0]]) ** 2, axis=1)
    for _ in range(k - 1):
        tot = d2.sum()
        if tot <= 0:
            idx.append(int(rng.integers(n)))
        else:
            idx.append(int(rng.choice(n, p=d2 / tot)))
        d2 = np.minimum(d2, np.sum((pts - pts[idx[-1]]) ** 2, axis=1))
    centres = pts[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.stack([np.sum((pts - c) ** 2, axis=1) for c in centres], axis=1)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                centres[j] = pts[m].mean(axis=0)
    return labels


def lofso_folds(catalogue: np.ndarray, footprint: np.ndarray,
                n_folds: int = 4, link_px: int = 3, collar_px: int = 3,
                seed: int = 11, spatial: bool = True) -> list[dict]:
    """Leave-One-Fault-System-Out folds.

    Each fold holds out whole fault systems (~1/n_folds of the catalogue pixel
    mass, balanced by size) as *pseudo-new-fault ground truth*. The rest of the
    catalogue is the pixel-exact scoring mask and the training label pool; a
    `collar_px` ring around the held-out systems is deleted from the training
    pool so no training pixel sits inside the metric kernel of a truth pixel.

    Returns a list of dicts:
      fold, gt (bool), known_mask (bool, masked from FP), train_labels (bool,
      catalogue pixels usable for training), train_pool (bool, all pixels usable
      for training), n_gt, n_known, systems (ids in this fold)
    """
    cat = np.asarray(catalogue).astype(bool)
    fp = np.asarray(footprint).astype(bool)
    ids = fault_systems(cat, link_px=link_px)
    tab = system_table(ids)
    n_sys = len(tab)
    if n_sys == 0:
        raise ValueError("no catalogue pixels -> no folds")

    # cluster systems into n_folds spatial groups (or random groups)
    if spatial:
        pts = tab[:, 2:4] / np.array([cat.shape[0], cat.shape[1]], float)
        cluster = _kmeans_clusters(pts, n_folds, seed=seed)
    else:
        rng = np.random.default_rng(seed)
        cluster = rng.integers(0, n_folds, size=n_sys)

    # balance pixel mass inside each cluster: order systems by size and deal
    # them round-robin into folds, keeping the cluster as the primary key so
    # folds stay spatially coherent but have comparable truth mass.
    fold_of = np.zeros(n_sys, dtype=int)
    for c in range(n_folds):
        members = np.flatnonzero(cluster == c)
        if len(members) == 0:
            continue
        order = members[np.argsort(-tab[members, 1], kind="stable")]
        # greedy: assign each system to the currently lightest fold
        load = np.zeros(n_folds, dtype=np.float64)
        for s in order:
            f = int(np.argmin(load))
            fold_of[s] = f
            load[f] += tab[s, 1]

    folds: list[dict] = []
    for f in range(n_folds):
        sys_here = np.flatnonzero(fold_of == f) + 1  # system ids are 1-based
        gt = np.isin(ids, sys_here)
        known = cat & ~gt
        ring = collar(gt, collar_px)
        train_labels = known
        train_pool = fp & ~gt & ~ring
        folds.append({
            "fold": f,
            "gt": gt,
            "known_mask": known,
            "train_labels": train_labels,
            "train_pool": train_pool,
            "ring": ring,
            "n_gt": int(gt.sum()),
            "n_known": int(known.sum()),
            "n_train_pool": int(train_pool.sum()),
            "systems": sys_here.tolist(),
            "link_px": link_px,
            "collar_px": collar_px,
        })
    return folds
