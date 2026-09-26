"""Distance-weighted Tversky index (DTI) — exact implementation of the official metric.

Official source: problem description, "Performance metric" section
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric

Official definitions (R = 300 m = 3 px at 100 m resolution, alpha = 0.2, beta = 0.8):

    k(d)  = max(1 - d / R, 0)                                  triangular kernel
    TP_w  = sum_{g in G} max_{x: d(x,g) <= R}  p(x) * k(d(x,g))
    FP_w  = sum_{x: p(x) > 0}        p(x) * [1 - max_{g in G} k(d(x,g))]
    FN_w  = sum_{g in G}           [1 - max_{x: d(x,g) <= R} p(x) * k(d(x,g))]
    DTI   = TP_w / (TP_w + alpha * FP_w + beta * FN_w + eps)

Organizer masking rules — BOTH now verified in writing from DrivenData staff
(chrisk-dd, flair "drivendata-staff") in forum topic 11516
https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516

  post 2 (2026-09-16): "Pixels corresponding to known USGS/INGENIOUS faults are
      masked / excluded from evaluation, so they do not count towards penalty
      terms." and "Re-evaluation will also mask/exclude the existing
      USGS/INGENIOUS faults."
  post 4 (2026-09-21) — resolves the geometry question that this module
      previously carried as UNVERIFIED:
      "1. The mask is indeed pixel-exact - it is identical to the provided set
       of training fault labels.
       2. Only new-fault ground truth is considered for scoring purposes. A
       predicted pixel that is near a known fault trace but far from a new-fault
       ground truth pixel will be fully penalized, i.e., the buffer does not
       apply to known faults.
       3. A new-fault ground truth pixel can indeed lie within 300m of a known
       fault trace. Such pixels would constitute corrections or modifications
       to existing fault traces."

Consequences implemented here (and the reason `mask_dilation_px` defaults to 0):
  * exclusion is exact-pixel: predictions on catalogue pixels contribute 0 to
    FP_w, but they still contribute to TP_w for ground-truth pixels within R
    (they are inside the max over x in the TP term);
  * there is NO discounted ring around known faults — a predicted pixel 1 px
    off a known trace pays the full FP weight unless it is within R of a
    *new-fault* ground-truth pixel;
  * ground truth may sit inside R of known traces (corrections), so holdout
    protocols must not delete that zone from the truth. See `src/gems/systems.py`
    for the LOFSO folds that emulate all three rules at once.
`mask_dilation_px` is retained only so the falsified buffered-mask reading can
still be reproduced for audit purposes.

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


def distance_weighted_tversky_fast(
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
    """Vectorised twin of :func:`distance_weighted_tversky` (same numbers).

    The reference implementation loops over every truth pixel to take the
    window maximum; that is exact but O(|G| * (2R+1)^2) in Python. This version
    computes the same per-truth-pixel window maximum with ``|offsets|`` whole
    array shifts (29 of them for R = 3, since k(d) = 0 for d >= R), which is
    ~500x faster on the 3292x3730 competition grid and makes fold-wise
    evaluation of many candidate fields practical.

    ``tests/test_metric.py`` asserts the two agree to < 1e-9 on random fixtures
    and on the real grid, so the fast path can never silently drift from the
    official formula.
    """
    p = _as_finite_prob(pred)
    g = np.asarray(truth)
    g = (np.isfinite(g) & (g != 0))
    H, W = p.shape
    if g.shape != (H, W):
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {g.shape}")

    R = radius_px
    # --- FP term (identical maths to the reference implementation)
    if g.any():
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

    n_truth = int(g.sum())
    if n_truth == 0:
        return {"TP_w": 0.0, "FP_w": fp_w, "FN_w": 0.0, "DTI": 0.0,
                "n_truth": 0, "n_pred_pos": int(pos.sum()),
                "masked_fp_pixels": masked_fp_pixels}

    # --- TP term: per-truth-pixel max of p(x) * k(d(x, g)) over d < R
    best = np.zeros((H, W), dtype=np.float64)
    for oy in range(-R + 1, R):
        for ox in range(-R + 1, R):
            d = float(np.hypot(oy, ox))
            k = 1.0 - d / R
            if k <= 0.0:
                continue
            # shifted view: S[y, x] = p[y + oy, x + ox] (0 outside the grid)
            S = np.zeros((H, W), dtype=np.float64)
            ys_src = slice(max(0, oy), H + min(0, oy))
            ys_dst = slice(max(0, -oy), H + min(0, -oy))
            xs_src = slice(max(0, ox), W + min(0, ox))
            xs_dst = slice(max(0, -ox), W + min(0, -ox))
            S[ys_dst, xs_dst] = p[ys_src, xs_src]
            np.maximum(best, S * k, out=best)
    tp_w = float(best[g].sum())
    fn_w = float(n_truth - tp_w)
    denom = tp_w + alpha * fp_w + beta * fn_w + eps
    dti = tp_w / denom if denom > 0 else 0.0
    return {"TP_w": tp_w, "FP_w": fp_w, "FN_w": fn_w, "DTI": float(dti),
            "n_truth": n_truth, "n_pred_pos": int(pos.sum()),
            "masked_fp_pixels": masked_fp_pixels}


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
