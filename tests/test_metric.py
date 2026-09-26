"""Metric tests — including the official worked example as ground truth.

Official worked example (problem description, "Scoring example"):
  TP_w = 3.00, FP_w = 1.89, FN_w = 2.00,
  DTI(0.2, 0.8) = 3.00 / (3.00 + 0.2*1.89 + 0.8*2.00) = 0.60.
We assert the closed-form combination (the per-pixel weights come from the
organizers' schematic, so the exact TP/FP/FN reproduction is asserted on a
hand-computed micro-grid below instead).
"""
import numpy as np

from gems.metric import (ALPHA, BETA, RADIUS_PX, degenerate_baselines,
                         distance_weighted_tversky, triangular_kernel)


def test_official_worked_example_algebra():
    tp, fp, fn = 3.00, 1.89, 2.00
    dti = tp / (tp + ALPHA * fp + BETA * fn)
    assert abs(dti - 0.60) < 5e-3  # 0.6002…; official text rounds to 0.60
    assert ALPHA == 0.2 and BETA == 0.8 and RADIUS_PX == 3


def test_triangular_kernel_values():
    k = triangular_kernel(3)
    assert k.shape == (7, 7)
    assert abs(k[3, 3] - 1.0) < 1e-12
    assert abs(k[3, 4] - (1 - 1 / 3)) < 1e-12
    assert abs(k[3, 6] - 0.0) < 1e-12
    assert abs(k[0, 0] - 0.0) < 1e-12  # corner d=4.24 > 3


def test_single_pixel_exact_hit():
    # One truth pixel, prediction 1.0 exactly on it: TP=1, FN=0, FP=0 -> DTI=1.
    p = np.zeros((11, 11))
    g = np.zeros((11, 11))
    p[5, 5] = 1.0
    g[5, 5] = 1.0
    r = distance_weighted_tversky(p, g)
    assert abs(r["TP_w"] - 1.0) < 1e-9
    assert abs(r["FN_w"] - 0.0) < 1e-9
    assert abs(r["FP_w"] - 0.0) < 1e-9
    assert abs(r["DTI"] - 1.0) < 1e-9


def test_single_pixel_offset_by_one():
    # Truth at (5,5), prediction 1.0 at (5,6): d=1 -> k=2/3.
    # TP=2/3, FN=1/3, FP = 1*(1-2/3)=1/3.
    p = np.zeros((11, 11))
    g = np.zeros((11, 11))
    p[5, 6] = 1.0
    g[5, 5] = 1.0
    r = distance_weighted_tversky(p, g)
    assert abs(r["TP_w"] - 2 / 3) < 1e-9
    assert abs(r["FN_w"] - 1 / 3) < 1e-9
    assert abs(r["FP_w"] - 1 / 3) < 1e-9
    expected = (2 / 3) / (2 / 3 + 0.2 / 3 + 0.8 / 3)
    assert abs(r["DTI"] - expected) < 1e-9


def test_prediction_outside_kernel_is_full_fp():
    # Prediction 5px away: k=0 -> TP=0, FN=1, FP=1.
    p = np.zeros((15, 15))
    g = np.zeros((15, 15))
    p[7, 12] = 1.0
    g[7, 7] = 1.0
    r = distance_weighted_tversky(p, g)
    assert abs(r["TP_w"]) < 1e-9
    assert abs(r["FN_w"] - 1.0) < 1e-9
    assert abs(r["FP_w"] - 1.0) < 1e-9


def test_nan_treated_as_zero_and_masking():
    p = np.full((9, 9), np.nan)
    g = np.zeros((9, 9))
    g[4, 4] = 1.0
    r = distance_weighted_tversky(p, g)
    assert r["TP_w"] == 0.0 and r["FP_w"] == 0.0
    # known-fault masking: prediction on a known pixel costs no FP.
    p2 = np.zeros((9, 9))
    p2[0, 0] = 1.0  # far from truth
    known = np.zeros((9, 9), dtype=bool)
    known[0, 0] = True
    r_nomask = distance_weighted_tversky(p2, g)
    r_mask = distance_weighted_tversky(p2, g, known_mask=known)
    assert abs(r_nomask["FP_w"] - 1.0) < 1e-9
    assert abs(r_mask["FP_w"] - 0.0) < 1e-9


def test_degenerate_baselines_catalogue_copy_is_one():
    rng = np.random.default_rng(0)
    g = (rng.random((20, 20)) < 0.05).astype(float)
    fp = np.ones((20, 20), dtype=bool)
    b = degenerate_baselines(g, fp)
    assert abs(b["catalogue_copy"] - 1.0) < 1e-9
    assert b["all_zero"] == 0.0
    assert 0.0 < b["all_one_inside_footprint"] < 1.0
