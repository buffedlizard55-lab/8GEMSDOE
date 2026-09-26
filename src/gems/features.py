"""Geophysical / topographic transforms for the 8GEMSDOE hypothesis portfolio.

All transforms are pure numpy/scipy (no GDAL, no torch) so they run on CPU in
this sandbox and in CI. Each function documents:
  * which competition band(s) it consumes (see problem description
    "Provided features": conductivity, depth-to-conductive-base, detrended
    elevation + slope, dilatation / shear / 2nd-invariant strain rates,
    isostatic gravity + slope, RTP magnetics, TMI + vertical/horizontal slope,
    magnetic source depth, earthquake density),
  * the physical signature targeted,
  * which portfolio hypothesis it serves (H1..H5, see docs/hypotheses.html).

Conventions: inputs are 2D float arrays with NaN outside the footprint;
`dx_m = 100.0` (competition resolution). NaN-aware: NaN is filled by nearest
neighbour for derivative computation and the footprint mask is re-applied to
outputs so NaN never leaks into scored pixels silently.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

DX_M = 100.0


# ---------------------------------------------------------------- utilities

def _fill_nan_nearest(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (filled array, valid mask). Nearest-neighbour fill for NaN."""
    a = np.asarray(a, dtype=np.float64)
    valid = np.isfinite(a)
    if valid.all():
        return a.copy(), valid
    if not valid.any():
        return np.zeros_like(a), valid
    idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
    filled = a[tuple(idx)]
    return filled, valid


def _reapply(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64)
    return np.where(valid, out, np.nan)


# ------------------------------------------------- FFT potential-field ops

def _wavenumbers(shape: tuple[int, int], dx: float = DX_M):
    ny, nx = shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    return np.meshgrid(kx, ky)


def _fft_spec(filled: np.ndarray, dx: float, upward_m: float = 0.0):
    """Spectral derivatives with mirror padding.

    The FFT assumes a periodic grid; raw competition grids are not periodic,
    which creates spurious edge gradients (Gibbs/wrap artefacts). Mirror
    (reflect) padding to >=1.5x size makes the extended grid approximately
    even-periodic; derivatives are cropped back to the original extent.
    Returns (dx, dy, dz) cropped arrays.
    """
    H, W = filled.shape
    ph, pw = H // 2 + 1, W // 2 + 1
    ext = np.pad(filled - np.mean(filled), ((ph, ph), (pw, pw)), mode="reflect")
    F = np.fft.fft2(ext)
    KX, KY = _wavenumbers(ext.shape, dx)
    K = np.sqrt(KX**2 + KY**2)
    if upward_m > 0:
        F = F * np.exp(-K * upward_m)
    dx_f = np.real(np.fft.ifft2(1j * KX * F))[ph:ph + H, pw:pw + W]
    dy_f = np.real(np.fft.ifft2(1j * KY * F))[ph:ph + H, pw:pw + W]
    dz_f = np.real(np.fft.ifft2(K * F))[ph:ph + H, pw:pw + W]
    return dx_f, dy_f, dz_f


def fft_derivatives(grid: np.ndarray, dx: float = DX_M) -> dict:
    """Spectral x/y/vertical derivatives of a potential-field grid.

    Standard potential-field practice: in the wavenumber domain,
    d/dx -> i*kx, d/dy -> i*ky, d/dz (vertical derivative) -> |k|.
    Used for the H1 gravity-edge family and the magnetic-edge ablations.
    """
    filled, valid = _fill_nan_nearest(grid)
    shape = np.asarray(grid).shape
    if not valid.any():
        nan = np.full(shape, np.nan)
        return {"dx": nan.copy(), "dy": nan.copy(), "dz": nan.copy()}
    dx_f, dy_f, dz_f = _fft_spec(filled, dx)
    return {
        "dx": _reapply(dx_f, valid),
        "dy": _reapply(dy_f, valid),
        "dz": _reapply(dz_f, valid),
    }


def horizontal_gradient_magnitude(grid: np.ndarray, dx: float = DX_M,
                                  method: str = "fft") -> np.ndarray:
    """HGM = sqrt(dx^2 + dy^2). Edges of causative bodies (H1, H4).

    method="fft" (spectral derivatives, mirror-padded) or "gradient"
    (central differences). Both re-apply the footprint mask.
    """
    g = np.asarray(grid, dtype=np.float64)
    if method == "fft":
        d = fft_derivatives(g, dx)
        hgm = np.sqrt(np.nan_to_num(d["dx"])**2 + np.nan_to_num(d["dy"])**2)
        return _reapply(hgm, np.isfinite(g))
    elif method == "gradient":
        filled, valid = _fill_nan_nearest(g)
        gy, gx = np.gradient(filled, dx)
        hgm = np.sqrt(gx**2 + gy**2)
        return _reapply(hgm, valid)
    raise ValueError(f"unknown method: {method!r} (use 'fft' or 'gradient')")


def analytic_signal_amplitude(grid: np.ndarray, dx: float = DX_M) -> np.ndarray:
    """AS = sqrt(dx^2 + dy^2 + dz^2). Maxima sit over magnetic/gravity sources.

    Note: AS of TMI and AS of RTP are mathematically related but not identical
    on real gridded data; we compute AS on the RTP band and on isostatic
    gravity separately and keep both.
    """
    g = np.asarray(grid, dtype=np.float64)
    d = fft_derivatives(g, dx)
    asg = np.sqrt(np.nan_to_num(d["dx"])**2 + np.nan_to_num(d["dy"])**2
                  + np.nan_to_num(d["dz"])**2)
    return _reapply(asg, np.isfinite(g))


def tilt_derivative(grid: np.ndarray, dx: float = DX_M,
                    upward_m: float = 0.0) -> np.ndarray:
    """TDR = atan2(dz, HGM), in radians (-pi/2..pi/2).

    Enhances weak trends independent of anomaly amplitude; the ZERO CONTOUR
    tracks edges of causative units. H1 scores pixels by proximity to the TDR
    zero contour of isostatic gravity (+ upward-continued variant for the deep
    / buried-fault arm). Optional upward continuation (in metres) stabilises
    the deep-source arm.
    """
    filled, valid = _fill_nan_nearest(grid)
    if not valid.any():
        return np.full(np.asarray(grid).shape, np.nan)
    dx_f, dy_f, dz_f = _fft_spec(filled, dx, upward_m)
    hgm = np.sqrt(dx_f**2 + dy_f**2)
    tdr = np.arctan2(dz_f, hgm + 1e-30)
    return _reapply(tdr, valid)


def zero_contour_proximity(tdr: np.ndarray, tolerance_rad: float = 0.15) -> np.ndarray:
    """1.0 on pixels within `tolerance_rad` of the TDR zero contour.

    Implemented as: |TDR| < tol  OR  sign change within the 3x3 neighbourhood
    (catches the contour between pixels). Output is a {0,1} float field, NaN
    outside the footprint.
    """
    t = np.asarray(tdr, dtype=np.float64)
    valid = np.isfinite(t)
    f = np.nan_to_num(t)
    near = np.abs(f) < tolerance_rad
    # sign-change detector: min and max over 3x3 straddle zero
    lo = ndimage.minimum_filter(f, size=3)
    hi = ndimage.maximum_filter(f, size=3)
    cross = (lo < 0) & (hi > 0)
    out = (near | cross).astype(np.float64)
    return np.where(valid, out, np.nan)


# ------------------------------------------------------- topographic family

def slope_magnitude(dem: np.ndarray, dx: float = DX_M) -> np.ndarray:
    """Gradient magnitude of elevation (m/m). Sanity/comparison band."""
    filled, valid = _fill_nan_nearest(dem)
    gy, gx = np.gradient(filled, dx)
    return _reapply(np.sqrt(gx**2 + gy**2), valid)


def curvature_zevenbergen_thorne(dem: np.ndarray, dx: float = DX_M) -> dict:
    """Profile + plan curvature by the Zevenbergen–Thorne 3x3 quadratic.

    Returns dict(profile, plan, total). Positive profile = concave-up.
    Serves H3 (scarp shoulders) and the curvature ablations.
    """
    filled, valid = _fill_nan_nearest(dem)
    z1, z2, z3 = filled[:-2, :-2], filled[:-2, 1:-1], filled[:-2, 2:]
    z4, z5, z6 = filled[1:-1, :-2], filled[1:-1, 1:-1], filled[1:-1, 2:]
    z7, z8, z9 = filled[2:, :-2], filled[2:, 1:-1], filled[2:, 2:]
    L = dx
    B = (z3 + z6 + z9 - z1 - z4 - z7) / (6 * L)       # dz/dx
    C = (z1 + z2 + z3 - z7 - z8 - z9) / (6 * L)       # dz/dy
    D = (z1 + z3 + z4 + z6 + z7 + z9 - 2 * (z2 + z5 + z8)) / (3 * L * L)
    E = (z1 + z2 + z3 + z7 + z8 + z9 - 2 * (z4 + z5 + z6)) / (3 * L * L)
    p2q2 = B**2 + C**2
    denom = np.maximum(p2q2, 1e-12)
    profile = -(D * B**2 + E * C**2) / (denom * (1 + p2q2) ** 1.5 + 1e-30)
    plan = -(D * C**2 + E * B**2) / (denom**1.5 + 1e-30)
    pad = dict(pad_width=1, mode="edge")
    out = {
        "profile": _reapply(np.pad(profile, **pad), valid),
        "plan": _reapply(np.pad(plan, **pad), valid),
    }
    out["total"] = np.where(valid, np.nan_to_num(out["profile"]) + np.nan_to_num(out["plan"]), np.nan)
    return out


def break_in_slope(dem: np.ndarray, dx: float = DX_M) -> np.ndarray:
    """Magnitude of the slope-of-slope (max 3x3 slope range / dx).

    Scarps and range fronts appear as break-in-slope lineaments even where
    absolute elevation contrast is small. H3 input.
    """
    s = slope_magnitude(dem, dx)
    f = np.nan_to_num(s)
    rng = ndimage.maximum_filter(f, size=3) - ndimage.minimum_filter(f, size=3)
    return np.where(np.isfinite(s), rng / dx, np.nan)


def scarp_step(dem: np.ndarray, dx: float = DX_M, half_len_px: int = 2) -> np.ndarray:
    """Asymmetric scarp-step response (H3 core signature).

    Along the local gradient direction, compare mean elevation `half_len_px`
    pixels upslope vs downslope:
        step = |mean_up - mean_down| / (2 * half_len_px * dx)
    A fault scarp is a STEP (antisymmetric), whereas a ridge/valley is
    symmetric — this separates scarps from generic curvature/ridgeness, which
    is exactly what the sibling Sato/structure-tensor lineaments do NOT do.
    """
    filled, valid = _fill_nan_nearest(dem)
    gy, gx = np.gradient(filled, dx)
    mag = np.sqrt(gx**2 + gy**2) + 1e-12
    ux, uy = gx / mag, gy / mag  # unit vector along gradient
    H, W = filled.shape
    yy, xx = np.mgrid[0:H, 0:W]
    acc_u = np.zeros_like(filled)
    acc_d = np.zeros_like(filled)
    for k in range(1, half_len_px + 1):
        xu = np.clip((xx + ux * k).round().astype(int), 0, W - 1)
        yu = np.clip((yy + uy * k).round().astype(int), 0, H - 1)
        xd = np.clip((xx - ux * k).round().astype(int), 0, W - 1)
        yd = np.clip((yy - uy * k).round().astype(int), 0, H - 1)
        acc_u += filled[yu, xu]
        acc_d += filled[yd, xd]
    step = np.abs(acc_u - acc_d) / (2 * half_len_px * dx)
    # Suppress where there is no coherent gradient (flats): weight by slope.
    step = step * np.tanh(mag * 50.0)
    return _reapply(step, valid)


# ------------------------------------------------------------- strain family

def strain_corridors(dilatation: np.ndarray, shear: np.ndarray,
                     second_invariant: np.ndarray) -> dict:
    """H2 signatures from the three INGENIOUS strain-rate bands.

    * dilation_index = dilat / (II + eps): extension-dominated corridors
      (positive = dilating stepovers/relays where fluids ascend).
    * shear_index    = |shear| / (II + eps): strike-slip partitioning.
    * strain_rate    = II itself (total deformation intensity).
    Catalogue traces mark discrete mapped faults; these fields mark the
    DISTRIBUTED actively-deforming corridors between them — the population a
    discrete-trace catalogue systematically misses.
    """
    d = np.asarray(dilatation, dtype=np.float64)
    s = np.asarray(shear, dtype=np.float64)
    ii = np.asarray(second_invariant, dtype=np.float64)
    valid = np.isfinite(d) & np.isfinite(s) & np.isfinite(ii)
    denom = np.abs(np.nan_to_num(ii)) + 1e-30
    dilation_index = np.nan_to_num(d) / denom
    shear_index = np.abs(np.nan_to_num(s)) / denom
    out = {
        "dilation_index": np.where(valid, dilation_index, np.nan),
        "shear_index": np.where(valid, shear_index, np.nan),
        "strain_rate": np.where(valid, np.nan_to_num(ii), np.nan),
    }
    return out


# ------------------------------------------------------ conductivity family

def conductivity_edges(conductivity: np.ndarray,
                       depth_to_base: np.ndarray | None = None,
                       dx: float = DX_M) -> dict:
    """H4 signatures: clay-alteration / upflow conductive lineaments.

    Fault gouge + hydrothermal clay cap = conductive lineaments with no
    necessary magnetic expression. Fields:
      * cond_hgm: HGM of surface conductivity (lateral alteration boundaries)
      * base_relief: HGM of depth-to-base (basement/cover steps)
      * alignment: product of the two normalised HGMs (both change together)
    """
    ch = horizontal_gradient_magnitude(conductivity, dx, method="gradient")
    out = {"cond_hgm": ch}
    if depth_to_base is not None:
        bh = horizontal_gradient_magnitude(depth_to_base, dx, method="gradient")
        out["base_relief"] = bh
        cn = ch / (np.nanmax(ch) + 1e-30)
        bn = bh / (np.nanmax(bh) + 1e-30)
        out["alignment"] = np.where(np.isfinite(ch) & np.isfinite(bh),
                                    np.nan_to_num(cn) * np.nan_to_num(bn), np.nan)
    return out


# --------------------------------------------------- seismicity + depth (H5)

def blind_fault_intersection(eq_density: np.ndarray,
                             mag_source_depth: np.ndarray,
                             eq_quantile: float = 0.90,
                             depth_quantile: float = 0.25) -> np.ndarray:
    """H5: microseismic lineaments where magnetic basement shallows.

    Blind active faults: earthquake density concentrates along the buried
    structure while the top-of-crustal magnetic source depth shallows
    (uplifted basement block). Thresholds are quantiles over the valid
    footprint so the operator size adapts to the data, not to magic numbers.
    """
    e = np.asarray(eq_density, dtype=np.float64)
    m = np.asarray(mag_source_depth, dtype=np.float64)
    valid = np.isfinite(e) & np.isfinite(m)
    if not valid.any():
        return np.full_like(e, np.nan)
    e_thr = np.quantile(e[valid], eq_quantile)
    m_thr = np.quantile(m[valid], depth_quantile)
    hit = (e >= e_thr) & (m <= m_thr)
    return np.where(valid, hit.astype(np.float64), np.nan)


# ------------------------------------------------------------- line cleaning

def skeleton_thin(field: np.ndarray, threshold: float) -> np.ndarray:
    """Binary thin of `field >= threshold` to a 1-px skeleton (Zhang-Suen).

    The metric credits each truth pixel from its single best prediction inside
    300 m, so dense bands pay repeated FP cost for one coverage; all H1..H5
    emission policies thin to skeletons before budgeting (unlike the dense
    ridge control, which exists to measure exactly this effect).
    Pure-numpy Zhang–Suen implementation (no skimage dependency).
    """
    img = (np.isfinite(field) & (field >= threshold)).astype(np.uint8)
    valid = np.isfinite(field)
    prev = np.zeros_like(img)
    P = np.pad(img, 1)
    while True:
        for step in (0, 1):
            N = P[:-2, 1:-1]
            S = P[2:, 1:-1]
            E = P[1:-1, 2:]
            W_ = P[1:-1, :-2]
            NE = P[:-2, 2:]
            NW = P[:-2, :-2]
            SE = P[2:, 2:]
            SW = P[2:, :-2]
            C = P[1:-1, 1:-1]
            nn = N + S + E + W_ + NE + NW + SE + SW
            seq = np.stack([N, NE, E, SE, S, SW, W_, NW, N], axis=0)
            trans = np.sum((seq[:-1] == 0) & (seq[1:] == 1), axis=0)
            if step == 0:
                cond = (C == 1) & (nn >= 2) & (nn <= 6) & (trans == 1) \
                    & ((N * S * E) == 0) & ((E * S * W_) == 0)
            else:
                cond = (C == 1) & (nn >= 2) & (nn <= 6) & (trans == 1) \
                    & ((N * S * W_) == 0) & ((N * E * W_) == 0)
            P[1:-1, 1:-1][cond] = 0
        cur = P[1:-1, 1:-1]
        if np.array_equal(cur, prev):
            break
        prev = cur.copy()
    out = P[1:-1, 1:-1].astype(np.float64)
    return np.where(valid, out, np.nan)
