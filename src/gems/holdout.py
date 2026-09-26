"""Spatially-blocked, buffered holdout harness.

Why: fault pixels are spatially autocorrelated (a trace runs through adjacent
pixels). Random pixel splits leak near-duplicates between train and test and
overstate DTI. The honest protocol — used by the sibling 6GEMSDOE line
(4x4 blocks, 300 m buffer) and standard in geospatial ML — is:

  * tile the footprint into BxB spatial blocks,
  * hold out whole blocks as test folds,
  * delete a buffer ring (default 3 px = 300 m, the metric kernel radius)
    around each test block from the TRAINING pool, so no training pixel sits
    inside the metric kernel of a test truth pixel.

Labels for proxy scoring:
  * catalogue labels (downloadable INGENIOUS/USGS raster) — the WRONG
    population (scored-against-new-faults asymmetry), reported as a monitor.
  * independent-compilation proxy (SGMC / NBMG faults rasterized, or the
    catalogue pixels excluded from training as "unmapped stand-ins") — the
    RIGHT population stand-in. Selection decisions use ONLY the proxy DTI.

This module has no data dependency: it operates on arrays + masks, so the
full protocol runs on synthetic fixtures in CI and on real bands once
`data/` is populated.
"""

from __future__ import annotations

import numpy as np

from .metric import distance_weighted_tversky


def block_folds(shape: tuple[int, int], footprint: np.ndarray,
                n_blocks: int = 4, buffer_px: int = 3,
                seed: int = 8) -> list[dict]:
    """Yield {train_mask, test_mask, buffer_mask} for each fold.

    Blocks are assigned to folds by shuffled round-robin so every fold holds
    out a spatially distributed set of blocks (default: n_blocks^2 blocks over
    K=n_blocks folds… here K folds each holding 1/n_blocks of blocks).
    Simpler and leak-free: fold k holds out blocks where (bi + bj) % K == k.
    """
    H, W = shape
    fp = np.asarray(footprint).astype(bool)
    K = n_blocks
    ys = np.linspace(0, H, n_blocks + 1).astype(int)
    xs = np.linspace(0, W, n_blocks + 1).astype(int)
    block_id = np.full(shape, -1, dtype=int)
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            block_id[ys[bi]:ys[bi + 1], xs[bj]:xs[bj + 1]] = (bi + bj) % K
    rng = np.random.default_rng(seed)
    perm = rng.permutation(K)
    block_id = np.where(block_id >= 0, perm[block_id], -1)
    folds = []
    for k in range(K):
        test = (block_id == k) & fp
        # buffer = dilation of test by buffer_px, minus test itself
        from scipy import ndimage
        dil = ndimage.binary_dilation(test,
                                      structure=np.ones((2 * buffer_px + 1,
                                                         2 * buffer_px + 1)))
        buffer = dil & ~test & fp
        train = fp & ~test & ~buffer
        folds.append({"fold": k, "train_mask": train, "test_mask": test,
                      "buffer_mask": buffer,
                      "n_train": int(train.sum()), "n_test": int(test.sum()),
                      "n_buffer": int(buffer.sum())})
    return folds


def score_on_fold(pred: np.ndarray, truth: np.ndarray,
                  test_mask: np.ndarray,
                  known_mask: np.ndarray | None = None) -> dict:
    """DTI restricted to a test fold.

    Restriction is honest for TP/FN (truth outside the fold is ignored) and
    conservative for FP (predictions outside the fold are ignored too — the
    reported number is the fold-local DTI, comparable across folds).
    Both unmasked and known-masked DTI are reported.
    """
    tm = np.asarray(test_mask).astype(bool)
    g = (np.isfinite(truth) & (truth != 0) & tm).astype(np.float64)
    p = np.where(tm, np.nan_to_num(np.asarray(pred, dtype=np.float64)), 0.0)
    out = {"unmasked": distance_weighted_tversky(p, g)}
    if known_mask is not None:
        km = np.asarray(known_mask).astype(bool) & tm
        out["known_masked_exact"] = distance_weighted_tversky(p, g, known_mask=km)
        out["known_masked_dilated3"] = distance_weighted_tversky(
            p, g, known_mask=km, mask_dilation_px=3)
    out["n_truth_in_fold"] = int(g.sum())
    return out


def topk_field(score: np.ndarray, footprint: np.ndarray,
               frac: float) -> np.ndarray:
    """Binary field: 1.0 on the top `frac` of footprint pixels by score.

    Rationale (verified algebra, see docs): with binary {0,1} emission the
    metric is monotone increasing in the emitted value, so fractional
    confidence gives score away; the only decision is WHERE to spend pixels.
    """
    fp = np.asarray(footprint).astype(bool)
    s = np.where(fp, np.nan_to_num(np.asarray(score, dtype=np.float64),
                                   nan=-np.inf), -np.inf)
    k = max(1, int(round(frac * fp.sum())))
    thr = np.partition(s[fp], -k)[-k] if k <= fp.sum() else np.inf
    out = np.where(fp & (s >= thr), 1.0, 0.0)
    return out
