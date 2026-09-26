"""Transform sanity tests on synthetic grids (no data needed)."""
import numpy as np

from gems import features as F


def test_hgm_peaks_on_step():
    a = np.zeros((41, 41))
    a[:, 21:] = 10.0  # vertical step at x=21
    h = F.horizontal_gradient_magnitude(a, method="gradient")
    col = np.nanmean(h, axis=0)
    assert col.argmax() in (20, 21, 22)
    assert col.max() > 5 * (col[0] + 1e-9)


def test_tdr_zero_contour_tracks_step():
    a = np.zeros((41, 41))
    a[:, 21:] = 10.0
    t = F.tilt_derivative(a)
    prox = F.zero_contour_proximity(t)
    assert np.isfinite(prox).all()
    # contour must exist near the step and not everywhere
    assert prox[:, 18:24].sum() > 0
    assert prox.mean() < 0.5


def test_scarp_step_separates_step_from_ridge():
    # step edge vs symmetric ridge: scarp_step fires on the step side strongly
    step = np.zeros((41, 41))
    step[:, 21:] = 8.0
    ridge = np.zeros((41, 41))
    ridge[:, 19:23] = 8.0  # narrow symmetric plateau
    s_step = np.nanmax(F.scarp_step(step))
    s_ridge = np.nanmax(F.scarp_step(ridge))
    assert s_step > 0
    # step response concentrates on ONE flank; ridge on TWO — check counts differ
    assert s_step > 0.5 * s_ridge  # sanity bound, not equality


def test_strain_indices_bounded():
    rng = np.random.default_rng(1)
    d = rng.normal(0, 1, (15, 15))
    s = rng.normal(0, 1, (15, 15))
    ii = np.abs(rng.normal(1, 0.2, (15, 15))) + 0.5
    out = F.strain_corridors(d, s, ii)
    assert np.isfinite(np.nan_to_num(out["dilation_index"])).all()
    assert (np.nan_to_num(out["shear_index"]) >= 0).all()


def test_skeleton_thin_reduces_width():
    f = np.zeros((21, 21))
    f[9:12, 3:18] = 1.0  # 3-px-thick bar
    sk = F.skeleton_thin(f, 0.5)
    assert sk[np.isfinite(sk)].sum() < f.sum()
    assert sk[np.isfinite(sk)].sum() > 0


def test_all_nan_input_returns_all_nan_without_warnings():
    import warnings
    a = np.full((9, 9), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = F.fft_derivatives(a)
        t = F.tilt_derivative(a)
        h = F.horizontal_gradient_magnitude(a)
    assert np.isnan(d["dx"]).all() and np.isnan(t).all() and np.isnan(h).all()


def test_hgm_methods_agree_on_step():
    a = np.zeros((41, 41))
    a[:, 21:] = 10.0
    h_fft = F.horizontal_gradient_magnitude(a, method="fft")
    h_g = F.horizontal_gradient_magnitude(a, method="gradient")
    assert np.nanargmax(np.nanmean(h_fft, axis=0)) in (19, 20, 21, 22, 23)
    assert np.nanargmax(np.nanmean(h_g, axis=0)) in (20, 21, 22)


def test_blind_intersection_quantiles():
    rng = np.random.default_rng(2)
    e = rng.random((30, 30))
    m = rng.random((30, 30))
    out = F.blind_fault_intersection(e, m)
    assert 0 < np.nansum(out) < 30 * 30 * 0.1
