"""Tests for the H7-H10 transforms (synthetic grids, no data needed)."""
import numpy as np

from gems import features as F


def test_h7_tdr_zero_contour_tracks_magnetic_step():
    # vertical magnetic step: hg peaks at edge, vg changes sign
    hg = np.zeros((41, 41))
    hg[:, 19:23] = 5.0  # horizontal-gradient high at the edge
    vg = np.zeros((41, 41))
    vg[:, :21] = 2.0
    vg[:, 21:] = -2.0  # vertical gradient sign change at x=21
    out = F.magnetic_tdr_as(hg, vg)
    assert np.isfinite(np.nan_to_num(out["score"])).all()
    # score concentrates at the edge columns, not everywhere
    col = np.nanmean(np.nan_to_num(out["score"]), axis=0)
    assert col.argmax() in (19, 20, 21, 22, 23)
    assert out["score"][np.isfinite(out["score"])].mean() < 0.5
    # analytic signal peaks where gradients are strong
    assert np.nanmean(out["as_norm"][:, 19:23]) > np.nanmean(
        out["as_norm"][:, 0:5])


def test_h7_all_nan_safe():
    a = np.full((9, 9), np.nan)
    out = F.magnetic_tdr_as(a, a)
    assert np.isnan(out["score"]).all()


def test_h8_endpoints_and_strike_vectors():
    f = np.zeros((21, 21), dtype=bool)
    f[10, 5:10] = True  # horizontal segment x=5..9 at y=10
    tips, vy, vx = F.fault_endpoints(f)
    assert tips.sum() == 2
    # left tip (10,5) points -x; right tip (10,9) points +x
    assert vx[10, 5] == -1.0 and vy[10, 5] == 0.0
    assert vx[10, 9] == 1.0 and vy[10, 9] == 0.0


def test_h8_links_collinear_tips_across_gap():
    f = np.zeros((21, 31), dtype=bool)
    f[10, 2:8] = True    # left segment, right tip at (10,7)
    f[10, 12:18] = True  # right segment, left tip at (10,12), gap 4px
    fp = np.ones_like(f, dtype=bool)
    out = F.tip_linkage_corridors(f, min_gap_px=3, max_gap_px=20,
                                  footprint=fp)
    # corridor pixels between the tips must be positive
    assert out[10, 8] > 0 and out[10, 11] > 0
    # far from the gap: zero
    assert out[0, 0] == 0.0
    assert np.isfinite(out).all()


def test_h8_rejects_same_component_and_far_pairs():
    f = np.zeros((21, 61), dtype=bool)
    f[10, 2:8] = True
    f[10, 40:46] = True  # gap of 32px > max_gap 20
    fp = np.ones_like(f, dtype=bool)
    out = F.tip_linkage_corridors(f, min_gap_px=3, max_gap_px=20,
                                  footprint=fp)
    assert np.nan_to_num(out).sum() == 0.0


def test_h8_rejects_non_collinear_tips():
    f = np.zeros((31, 31), dtype=bool)
    f[5, 2:8] = True    # horizontal segment, tips point +-x
    f[20:26, 25] = True  # vertical segment far below-right, tips point +-y
    fp = np.ones_like(f, dtype=bool)
    out = F.tip_linkage_corridors(f, min_gap_px=3, max_gap_px=20,
                                  footprint=fp, collinearity_min=0.9)
    # tips ~20px apart diagonally and not pointing at each other
    assert np.nan_to_num(out).sum() == 0.0


def test_h9_coherence_high_on_linear_valley():
    # diagonal linear valley in elevation over incoherent noise
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:61, 0:61]
    dem = -8.0 * np.exp(-((xx - yy) ** 2) / (2 * 2.0 ** 2))  # valley along x=y
    dem = dem + 0.3 * rng.standard_normal(dem.shape)  # incoherent texture
    out = F.valley_axis_coherence(dem)
    coh = np.nan_to_num(out["coherence"])
    on = coh[5:15, 5:15].mean()     # on the valley axis (off-centre)
    off = coh[0:8, 50:58].mean()    # noise-only corner
    assert on > off + 0.1
    assert np.isfinite(np.nan_to_num(out["score"])).all()


def test_h10_coherence_selects_linear_ridge_over_blob():
    yy, xx = np.mgrid[0:51, 0:51].astype(float)
    ridge = 5.0 * np.exp(-((xx - 25) ** 2) / (2 * 1.5 ** 2))  # vertical ridge
    blob = 5.0 * np.exp(-(((xx - 10) ** 2 + (yy - 10) ** 2)) / (2 * 1.5 ** 2))
    field = ridge + blob
    dq = np.full_like(field, 100.0)
    out = F.seismic_ridge_coherence(field, dq)
    coh = np.nan_to_num(out["coherence"])
    # mid-ridge coherence exceeds blob-centre coherence (isotropic -> low)
    assert coh[25, 25] > coh[10, 10] + 0.15
    assert out["score"][25, 25] > out["score"][10, 10]
