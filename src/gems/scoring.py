"""Fold-level DTI scoring with precomputed truth terms + exact top-k emission.

`src/gems/metric.py` holds the reference implementation of the official metric
(and `tests/test_metric.py` pins it against the worked example in the problem
description). This module is the *fast* path used by the holdout scripts: it
hoists everything that depends only on the fold's truth and mask out of the
per-candidate loop, so a fold can be scored against dozens of candidate fields
in seconds instead of minutes.

Two details that were gotchas and are now pinned by tests:
  * the pixel-exact known-fault mask (forum 11516 post 4) zeroes the FP weight
    on catalogue pixels but does NOT remove them from the TP window maximum;
  * ranking by `predict_proba` saturates to exactly 1.0 in float32 over ~1% of
    the grid, which degenerates exact top-k selection to raster order. Rank by
    a continuous score (e.g. a boosting decision function) instead.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

R_PX = 3
ALPHA, BETA = 0.2, 0.8
STRUCT = np.ones((3, 3), dtype=bool)


class FoldScorer:
    """Official DTI with the fold's truth/mask precomputed (float32, ~1.5 s/eval)."""

    def __init__(self, gt: np.ndarray, known: np.ndarray, radius_px: int = R_PX):
        self.gt = gt.astype(bool)
        self.known = known.astype(bool)
        self.n_gt = int(self.gt.sum())
        dmin = ndimage.distance_transform_edt(~self.gt)
        kmax = np.maximum(1.0 - dmin / radius_px, 0.0)
        fpw = (1.0 - kmax).astype(np.float32)
        fpw[self.known] = 0.0          # pixel-exact mask: zero FP weight
        self.fpw = fpw
        self.offsets = []
        for oy in range(-radius_px + 1, radius_px):
            for ox in range(-radius_px + 1, radius_px):
                k = 1.0 - float(np.hypot(oy, ox)) / radius_px
                if k > 0:
                    self.offsets.append((oy, ox, np.float32(k)))
        del dmin, kmax

    def score(self, emitted: np.ndarray) -> dict:
        p = emitted.astype(np.float32)
        fp = float((p * self.fpw).sum())
        H, W = p.shape
        best = np.zeros((H, W), dtype=np.float32)
        for oy, ox, k in self.offsets:
            S = np.zeros((H, W), dtype=np.float32)
            ys_src = slice(max(0, oy), H + min(0, oy))
            ys_dst = slice(max(0, -oy), H + min(0, -oy))
            xs_src = slice(max(0, ox), W + min(0, ox))
            xs_dst = slice(max(0, -ox), W + min(0, -ox))
            S[ys_dst, xs_dst] = p[ys_src, xs_src]
            np.maximum(best, S * k, out=best)
            del S
        tp = float(best[self.gt].sum())
        fn = float(self.n_gt - tp)
        dti = tp / (tp + ALPHA * fp + BETA * fn + 1e-12)
        return {"DTI": float(dti), "TP_w": tp, "FP_w": fp, "FN_w": fn,
                "n_gt": self.n_gt,
                "emitted_px": int(p.sum()),
                "unmasked_emitted_px": int((p * (~self.known)).sum()),
                "hit_rate": tp / max(float((p * (~self.known)).sum()), 1.0)}


def exact_topk(score: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    s = np.where(eligible, np.nan_to_num(score, nan=-np.inf, posinf=np.inf), -np.inf)
    flat = s.ravel()
    idx = np.flatnonzero(eligible.ravel())
    k = int(min(max(k, 0), idx.size))
    if k == 0:
        return np.zeros(s.shape, dtype=bool)
    order = idx[np.lexsort((idx, -flat[idx]))][:k]
    out = np.zeros(s.size, dtype=bool)
    out[order] = True
    return out.reshape(s.shape)




