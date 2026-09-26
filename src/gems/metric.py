"""Distance-weighted Tversky index (DTI) — exact implementation of the official metric.

Official source: problem description, "Performance metric" section
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric

Official definitions (R = 300 m = 3 px at 100 m resolution, alpha = 0.2, beta = 0.8):

    k(d)  = max(1 - d / R, 0)                                  triangular kernel
    TP_w  = sum_{g in G} max_{x: d(x,g) <= R}  p(x) * k(d(x,g))
    FP_w  = sum_{x: p(x) > 0}        p(x) * [1 - max_{g in G} k(d(x,g))]
    FN_w  = sum_{g in G}           [1 - max_{x: d(x,g) <= R} p(x) * k(d(x,g))]
    DTI   = TP_w / (TP_w + alpha * FP_w + beta * FN_w + eps)

Organizer masking rule (verified in writing, forum 11516, chrisk-dd 2026-09-16):
"Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from
evaluation, so they do not count towards penalty terms." — and the same masking
applies to the Final Round re-evaluation.

What the masking statement does NOT specify (flagged as unverified detail):
whether only the exact known-fault pixels are excluded, or also a neighbourhood.
This module implements exact-pixel exclusion of the FP term by default
(predictions on known-fault pixels contribute 0 to FP_w), with an optional
`mask_dilation_px` parameter so the alternative reading (dilated exclusion zone)
can be measured explicitly. Holdout reports always print BOTH the unmasked and
the exact-masked DTI so the assumption is never hidden.

NaN convention: predictions may be NaN outside the scored footprint (per the
submission format). For scoring, NaN is treated as 0.0. NOTE: the claim "the
official scorer reads NaN as 0.0" is inherited from sibling-site reasoning, NOT
from an official sentence — flagged in docs/sources.html. Our submission
builder never relies on it: inside-footprint pixels are always finite.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

ALPHA = 0.2
BETA = 0.8
RADIUS_PX = 3  # 300 m at 100 m resolution
EPS = 1e-12


def triangular_kernel(radius_px: int = RADIUS_PX) -> np.ndarray:
    """Return the (2R+1)x(2R+1) triangular kernel k(d) = max(1 - d/R, 0)."""
    ax = np.arange(-radius_px, radius_px + 1, dtype=np.float64)
    yy, xx = np.meshgrid(ax, ax)
    d = np.sqrt(xx**2 + yy**2)
    return np.maximum(1.0 - d / radius_px, 0.0)


def _as_finite_prob(pred: np.ndarray) -> np.ndarray:
    """NaN -> 0.0 (outside-footprint convention); clip to [0, 1]; float64."""
    p = np.asarray(pred, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, 1.0)


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool)
    size = 2 * radius_px + 1
    return ndimage.binary_dilation(mask.astype(bool),
                                   structure=np.ones((size, size), dtype=bool))


def distance_weighted_tversky(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    known_mask: np.ndarray | None = None,
    mask_dilation_px: int = 0,
    alpha: float = ALPHA,
    beta: float = BETA,
    radius_px: int = RADIUS_PX,
    eps: float = EPS,
) -> dict:
    """Compute the official distance-weighted Tversky index and its terms.

    Args:
        pred: HxW predicted probabilities in [0, 1] (NaN allowed outside footprint).
        truth: HxW binary ground truth (nonzero = fault). NaN treated as 0.
        known_mask: optional HxW boolean — True on known USGS/INGENIOUS fault
            pixels. When given, predictions there are excluded from the FP term
            (organizer masking rule), optionally dilated by `mask_dilation_px`.
        mask_dilation_px: extra exclusion neighbourhood around known faults
            (0 = exact pixels only, the only organizer-verified reading).

    Returns:
        dict with TP_w, FP_w, FN_w, DTI, n_truth, n_pred_pos, masked_fp_pixels.
    """
    p = _as_finite_prob(pred)
    g = np.asarray(truth)
    g = (np.isfinite(g) & (g != 0))

    H, W = p.shape
    if g.shape != (H, W):
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {g.shape}")

    kernel = triangular_kernel(radius_px)
    R = radius_px

    # --- FP term: 1 - max_g k(d(x,g)) for every x.
    # max_g k(d) = 1 - dmin/R clipped at 0  <=>  from distance-to-nearest-truth.
    if g.any():
        # distance (px) from every pixel to nearest truth pixel
        dmin = ndimage.distance_transform_edt(~g)
        kmax_fp = np.maximum(1.0 - dmin / R, 0.0)
    else:
        kmax_fp = np.zeros_like(p)

    fp_weight = 1.0 - kmax_fp
    if known_mask is not None:
        km = np.asarray(known_mask).astype(bool)
        if km.shape != (H, W):
            raise ValueError("known_mask shape mismatch")
        excl = _dilate_mask(km, mask_dilation_px)
        fp_weight = np.where(excl, 0.0, fp_weight)
        masked_fp_pixels = int(excl.sum())
    else:
        masked_fp_pixels = 0

    pos = p > 0
    fp_w = float(np.sum(p[pos] * fp_weight[pos]))

    # --- TP / FN terms: per-truth-pixel max of p(x)*k over the (2R+1) window.
    ys, xs = np.nonzero(g)
    n_truth = len(ys)
    if n_truth == 0:
        return {
            "TP_w": 0.0, "FP_w": fp_w, "FN_w": 0.0,
            "DTI": 0.0 if fp_w > 0 else 0.0,
            "n_truth": 0, "n_pred_pos": int(pos.sum()),
            "masked_fp_pixels": masked_fp_pixels,
        }

    # Pad p so every window is in-bounds.
    pp = np.pad(p, R, mode="constant", constant_values=0.0)
    best = np.empty(n_truth, dtype=np.float64)
    for i in range(n_truth):
        y, x = int(ys[i]), int(xs[i])
        win = pp[y:y + 2 * R + 1, x:x + 2 * R + 1]
        best[i] = float(np.max(win * kernel))
    tp_w = float(best.sum())
    fn_w = float(n_truth - tp_w)

    denom = tp_w + alpha * fp_w + beta * fn_w + eps
    dti = tp_w / denom if denom > 0 else 0.0
    return {
        "TP_w": tp_w, "FP_w": fp_w, "FN_w": fn_w,
        "DTI": float(dti),
        "n_truth": n_truth, "n_pred_pos": int(pos.sum()),
        "masked_fp_pixels": masked_fp_pixels,
    }


def degenerate_baselines(truth: np.ndarray, footprint: np.ndarray,
                         **kwargs) -> dict:
    """DTI of the degenerate strategies on a given truth+footprint.

    Strategies: all_zero, all_one_inside_footprint, catalogue_copy (truth itself).
    (all_one_everywhere is only meaningful on a full grid; footprint==full here
    when the caller passes an all-finite footprint.)
    """
    g = (np.isfinite(np.asarray(truth)) & (np.asarray(truth) != 0)).astype(np.float64)
    fp = np.asarray(footprint).astype(bool)
    out = {}
    out["all_zero"] = distance_weighted_tversky(np.zeros_like(g), g, **kwargs)["DTI"]
    one = np.where(fp, 1.0, 0.0)
    out["all_one_inside_footprint"] = distance_weighted_tversky(one, g, **kwargs)["DTI"]
    out["catalogue_copy"] = distance_weighted_tversky(g, g, **kwargs)["DTI"]
    # Closed form for uniform prediction c=1 inside footprint:
    # DTI = TP/(TP + a*FP + b*FN) with TP=n, FN=0 -> n/(n + a*(N-n)) approx
    # (exact only ignoring distance weighting; reported for the audit trail).
    n = float(g.sum())
    N = float(fp.sum())
    out["closed_form_uniform_c1"] = (n / (n + ALPHA * (N - n))) if N > n else 1.0
    out["n_truth"] = int(n)
    out["n_footprint"] = int(N)
    return out
