"""Distance-Weighted Tversky Index (DTI) - Official GEMS Competition Metric.

Implemented verbatim from:
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric

Parameters:
  alpha = 0.2 (penalty for False Positives)
  beta = 0.8  (penalty for False Negatives)
  R = 300 meters = 3 pixels (at 100m resolution)
  Triangular kernel: k(d) = max(1 - d/R, 0)

Formulas:
  TP_w = sum_{g in G} max_{x : d(x, g) <= R} p(x) * k(d(x, g))
  FP_w = sum_{x : p(x) > 0} p(x) * [1 - max_{g in G} k(d(x, g))]
  FN_w = sum_{g in G} [1 - max_{x : d(x, g) <= R} p(x) * k(d(x, g))]
  DTI = TP_w / (TP_w + alpha * FP_w + beta * FN_w + eps)

Identity:
  TP_w + FN_w == |G| (total positive ground truth pixels)
  When alpha = 1 - beta (0.2 + 0.8 = 1.0):
    denom = 0.2 * (TP_w + FP_w) + 0.8 * |G|
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

ALPHA = 0.2
BETA = 0.8
R_PIXELS = 3.0
EPS = 1e-10


def _kernel_offsets(r_pixels: float = R_PIXELS):
    """Integer offsets (dy, dx) within kernel radius R and corresponding weight k(d)."""
    r = int(np.floor(r_pixels))
    offs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = float(np.hypot(dx, dy))
            if d <= r_pixels:
                offs.append((dy, dx, d, max(1.0 - d / r_pixels, 0.0)))
    return offs


_OFFSETS = _kernel_offsets()


def _shift(arr: np.ndarray, dy: int, dx: int, fill: float = 0.0) -> np.ndarray:
    """Shift 2D array by dy, dx with border fill."""
    out = np.full_like(arr, fill, dtype=np.float64)
    h, w = arr.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = arr[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def dtvi_components(pred: np.ndarray, truth: np.ndarray, r_pixels: float = R_PIXELS) -> dict:
    """Compute distance-weighted components TP_w, FP_w, FN_w."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {truth.shape}")

    p = np.where(np.isfinite(pred), pred, 0.0)
    if p.min() < -1e-6 or p.max() > 1.0 + 1e-6:
        raise ValueError("Predicted probabilities must be within [0, 1]")
    p = np.clip(p, 0.0, 1.0)

    g_mask = truth > 0
    n_truth = int(g_mask.sum())

    # TP_w and FN_w
    contrib = np.zeros(p.shape, dtype=np.float64)
    if n_truth > 0:
        for dy, dx, _d, w in _OFFSETS:
            if w == 0.0:
                continue
            shifted = _shift(p, dy, dx)
            contrib = np.maximum(contrib, shifted * w)
        tp_w = float(contrib[g_mask].sum())
    else:
        tp_w = 0.0
    fn_w = float(n_truth - tp_w)

    # FP_w
    if n_truth > 0:
        dist = distance_transform_edt(~g_mask)
        kern = np.maximum(1.0 - dist / r_pixels, 0.0)
    else:
        kern = np.zeros(p.shape, dtype=np.float64)
    fp_mask = p > 0
    fp_w = float((p[fp_mask] * (1.0 - kern[fp_mask])).sum())

    return {
        "TP_w": tp_w,
        "FP_w": fp_w,
        "FN_w": fn_w,
        "n_truth": n_truth,
        "predicted_mass": float(p.sum()),
    }


def dtvi(
    pred: np.ndarray,
    truth: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
    r_pixels: float = R_PIXELS,
    eps: float = EPS,
) -> float:
    """Distance-Weighted Tversky Index."""
    c = dtvi_components(pred, truth, r_pixels=r_pixels)
    denom = c["TP_w"] + alpha * c["FP_w"] + beta * c["FN_w"] + eps
    return float(c["TP_w"] / denom)
