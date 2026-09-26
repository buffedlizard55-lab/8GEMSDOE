"""Pseudo-hidden-truth generators: local emulations of the competition's GT.

The scored population is **expert-mapped faults that are not in the supplied
catalogue** (problem description, "Competition structure"), and DrivenData staff
verified three properties of it (forum 11516 posts 2 and 4):

  1. catalogue pixels are masked out of the penalty terms, pixel-exactly
     ("the mask ... is identical to the provided set of training fault labels");
  2. a predicted pixel near a known trace but far from a new-fault GT pixel is
     *fully* penalised ("the buffer does not apply to known faults");
  3. GT pixels may lie within 300 m of a known trace — "such pixels would
     constitute corrections or modifications to existing fault traces".

Property 3 is the one that breaks naive holdouts: if folds are cut by linking
everything within 300 m into one "system" and holding whole systems out, then no
held-out pixel is ever within 3 px of a masked catalogue pixel, the
correction/modification regime is *structurally excluded*, and any
catalogue-hugging arm scores exactly 0.0 (measured 2026-09-26: HALO_r1..r5 gave
TP_w = 0.0 under such folds). The opposite cut (holding out along-strike
segments of every trace) excludes wholly-new faults.

`mixed_truth_folds()` therefore builds a **two-parameter family** of pseudo-GT
that spans both regimes:

  * `w` — fraction of the fold's truth pixels taken from whole held-out systems
    ("wholly new, previously unmapped faults");
  * the remainder taken from contiguous along-strike segments of the systems
    that stay in the catalogue ("corrections / modifications / extensions",
    which by construction sit within a pixel or two of masked catalogue).

  * `n_truth` — total truth pixels per fold, i.e. the local stand-in for the
    unknown |G| of the real test set.

`scripts/calibrate_protocol.py` fits (w, n_truth) so that the family's seven
publicly scored artifacts reproduce their real public scores; the fitted cell is
then the protocol used to compare candidate arms.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .systems import fault_systems, system_table

STRUCT = np.ones((3, 3), dtype=bool)

__all__ = ["along_strike_segments", "mixed_truth_folds"]


def _principal_axis(coords: np.ndarray) -> np.ndarray:
    """Unit vector of the first principal component of a pixel cloud."""
    c = coords - coords.mean(axis=0)
    if len(c) < 2:
        return np.array([1.0, 0.0])
    cov = (c.T @ c) / max(len(c) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, int(np.argmax(vals))]


def along_strike_segments(catalogue: np.ndarray, n_folds: int, fold: int,
                          ids: np.ndarray | None = None) -> np.ndarray:
    """Hold out one contiguous along-strike chunk of every fault trace.

    Each system's pixels are projected on their principal axis, sorted, split
    into `n_folds` contiguous bins, and bin `fold` is returned. The result is a
    set of catalogue pixels that are *adjacent to* retained catalogue pixels —
    the local stand-in for "corrections or modifications to existing fault
    traces" (forum 11516 post 4, bullet 3).
    """
    cat = np.asarray(catalogue).astype(bool)
    if ids is None:
        ids = fault_systems(cat, link_px=0)
    out = np.zeros(cat.shape, dtype=bool)
    h, w = cat.shape
    ys, xs = np.nonzero(ids)
    if len(ys) == 0:
        return out
    sys_ids = ids[ys, xs]
    order = np.argsort(sys_ids, kind="stable")
    ys, xs, sys_ids = ys[order], xs[order], sys_ids[order]
    bounds = np.flatnonzero(np.r_[True, sys_ids[1:] != sys_ids[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        sy, sx = ys[a:b], xs[a:b]
        n = b - a
        if n < n_folds:
            # tiny trace: deal pixels round-robin so every fold sees some
            take = np.arange(n) % n_folds == fold
        else:
            coords = np.column_stack([sy, sx]).astype(np.float64)
            axis = _principal_axis(coords)
            t = coords @ axis
            rank = np.argsort(t, kind="stable")
            bin_of = (np.arange(n) * n_folds) // n
            take = np.zeros(n, dtype=bool)
            take[rank[bin_of == fold]] = True
        out[sy[take], sx[take]] = True
    return out


def misalignment_ring(catalogue: np.ndarray, footprint: np.ndarray,
                      radius_px: int = 1) -> np.ndarray:
    """Pixels within `radius_px` of a catalogue trace but not on one.

    Local stand-in for the official statement that "portions of the existing
    fault data may be misaligned from the true location of the surface fault,
    which is the prediction target" (problem description, Performance metric
    section) and for staff bullet 3 of forum 11516 post 4 (GT may lie within
    300 m of a known trace as a correction or modification).
    """
    cat = np.asarray(catalogue).astype(bool)
    dil = ndimage.binary_dilation(cat, STRUCT, iterations=max(1, radius_px))
    return dil & ~cat & np.asarray(footprint).astype(bool)


def mixed_truth_folds(catalogue: np.ndarray, footprint: np.ndarray,
                      n_folds: int = 3, regime: tuple[float, float, float] = (0.34, 0.33, 0.33),
                      n_truth: int = 20_000, link_px: int = 0,
                      collar_px: int = 3, seed: int = 17,
                      w_new: float | None = None) -> list[dict]:
    """Build `n_folds` pseudo-hidden-truth folds spanning all three GT regimes.

    `regime` = (w_new, w_seg, w_off), normalised internally; `w_new` is accepted
    as a legacy 2-way shortcut for (w_new, 1 - w_new, 0).

    Per fold:
      * a seeded random subset of whole fault systems is held out until it
        supplies `w_new * n_truth` truth pixels  -> "wholly new faults";
      * contiguous along-strike segments of the *remaining* systems supply
        `w_seg * n_truth`                         -> "extensions / newly mapped
        geometry of an existing fault system" (forum 11536 staff answer);
      * a seeded sample of the 1 px misalignment ring around what is left
        supplies `w_off * n_truth`                -> "corrections / modifications
        of existing traces" (forum 11516 post 4 bullet 3);
      * truth is capped at `n_truth` px (last system / last segment trimmed);
      * `known_mask` = catalogue minus truth (pixel-exact scoring mask);
      * `train_pool` = footprint minus truth minus a `collar_px` ring around
        truth (so no training pixel sits inside the metric kernel of truth).
    """
    cat = np.asarray(catalogue).astype(bool)
    fp = np.asarray(footprint).astype(bool)
    ids = fault_systems(cat, link_px=link_px)
    tab = system_table(ids)
    if w_new is not None:
        regime = (float(w_new), float(1.0 - w_new), 0.0)
    tot = float(sum(regime)) or 1.0
    wnew, wseg,woff = (r / tot for r in regime)
    ring1 = misalignment_ring(cat, fp, 1)
    folds: list[dict] = []
    for f in range(n_folds):
        rng = np.random.default_rng(seed + 101 * f)
        gt = np.zeros(cat.shape, dtype=bool)
        budget = int(n_truth)
        n_sys = 0
        n_off = 0

        # ---- component 1: whole systems ("new, previously unmapped faults")
        if wnew > 0 and len(tab):
            target_new = int(round(wnew * n_truth))
            perm = rng.permutation(len(tab))
            sys_ids = tab[perm, 0].astype(int)
            sizes = tab[perm, 1].astype(int)
            taken = 0
            chosen: list[int] = []
            for sid, sz in zip(sys_ids, sizes):
                if taken >= target_new:
                    break
                chosen.append(int(sid))
                taken += int(sz)
            if chosen:
                new_mask = np.isin(ids, chosen)
                if taken > target_new:  # trim the last system to hit the budget
                    ys, xs = np.nonzero(new_mask)
                    keep = rng.permutation(len(ys))[:target_new]
                    trimmed = np.zeros_like(new_mask)
                    trimmed[ys[keep], xs[keep]] = True
                    new_mask = trimmed
                gt |= new_mask
                budget -= int(new_mask.sum())
                n_sys = int(new_mask.sum())

        # ---- component 2: along-strike segments ("extensions of known systems")
        if budget > 0 and wseg > 0:
            remaining = cat & ~gt
            if remaining.any():
                seg = along_strike_segments(remaining, n_folds, f,
                                            ids=fault_systems(remaining, link_px=0))
                n_seg = int(seg.sum())
                want_seg = int(round(wseg * n_truth))
                if n_seg > min(budget, want_seg):
                    budget_seg = min(budget, want_seg)
                    ys, xs = np.nonzero(seg)
                    keep = rng.permutation(len(ys))[:budget_seg]
                    seg2 = np.zeros_like(seg)
                    seg2[ys[keep], xs[keep]] = True
                    seg = seg2
                gt |= seg
                budget -= int(seg.sum())
        n_seg = int(gt.sum()) - n_sys

        # ---- component 3: misalignment ring ("corrections / modifications")
        n_off = 0
        if budget > 0 and woff > 0:
            cand = ring1 & ~gt
            idx = np.flatnonzero(cand.ravel())
            if idx.size:
                want = min(int(round(woff * n_truth)), budget, idx.size)
                take = rng.choice(idx, size=want, replace=False)
                off = np.zeros(cand.size, dtype=bool)
                off[take] = True
                off = off.reshape(cand.shape)
                gt |= off
                n_off = int(off.sum())
                budget -= n_off

        known = cat & ~gt
        ring = ndimage.binary_dilation(gt, STRUCT, iterations=collar_px) & ~gt & fp
        folds.append({
            "fold": f, "gt": gt, "known_mask": known,
            "train_labels": known, "train_pool": fp & ~gt & ~ring, "ring": ring,
            "n_gt": int(gt.sum()), "n_known": int(known.sum()),
            "n_truth_whole_system": n_sys,
            "n_truth_along_strike_segment": n_seg,
            "n_truth_misalignment_ring": n_off,
            "gt_touching_known": int((gt & ndimage.binary_dilation(known, STRUCT)).sum()),
            "regime_requested": [wnew, wseg, woff], "n_truth_requested": n_truth,
        })
    return folds
