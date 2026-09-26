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


# ------------------------------------------------- H7 magnetic family (new)

def magnetic_tdr_as(tmi_hg: np.ndarray, tmi_vg: np.ndarray,
                    tolerance_rad: float = 0.15) -> dict:
    """H7: magnetic basement-lineament score from the survey's own gradients.

    Consumes official bands tmi_hg (3) and tmi_vg (9) — the contractor's
    line-data gradients of total magnetic intensity. Elementwise (no FFT):
      * tdr = atan2(vg, |hg|) — tilt derivative of the magnetic field; its
        ZERO CONTOUR tracks lateral edges of magnetic units independent of
        anomaly amplitude (same signature family as H1, on magnetics).
      * as_norm = sqrt(hg^2 + vg^2), normalised — analytic-signal amplitude
        from provided gradients; maxima sit over magnetic sources / edges.
      * score = zero_contour_proximity(tdr) * (0.5 + 0.5 * as_norm).
    Buried basement faults juxtapose units of contrasting magnetization with
    no surface expression; the Quaternary catalogue requires geologic
    evidence of surface deformation (USGS faults page), so this population is
    systematically under-mapped. Novel vs H1 (gravity-only TDR), vs Apex
    (Sobel on tmi_hg used only as a GBT input, never as an emitted skeleton
    field), vs siblings (mag HGM/tilt as blender inputs, never the
    TDR-zero-contour x AS primary field).
    """
    hg = np.asarray(tmi_hg, dtype=np.float64)
    vg = np.asarray(tmi_vg, dtype=np.float64)
    valid = np.isfinite(hg) & np.isfinite(vg)
    if not valid.any():
        nan = np.full(hg.shape, np.nan)
        return {"tdr": nan.copy(), "as_norm": nan.copy(),
                "score": nan.copy()}
    h = np.abs(np.nan_to_num(hg))
    dz = np.nan_to_num(vg)
    tdr = np.arctan2(dz, h + 1e-30)
    asg = np.sqrt(h ** 2 + dz ** 2)
    as_norm = asg / (np.nanmax(np.where(valid, asg, np.nan)) + 1e-30)
    prox = zero_contour_proximity(np.where(valid, tdr, np.nan),
                                  tolerance_rad=tolerance_rad)
    score = np.nan_to_num(prox) * (0.5 + 0.5 * np.nan_to_num(as_norm))
    return {
        "tdr": np.where(valid, tdr, np.nan),
        "as_norm": np.where(valid, as_norm, np.nan),
        "score": np.where(valid, score, np.nan),
    }


# ------------------------------------------------- H8 linkage family (new)

def fault_endpoints(faults: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
    """Tip pixels of a binary fault raster + their outward strike unit vectors.

    A tip is a fault pixel with exactly one 8-neighbour fault pixel. The
    outward strike vector points from the single neighbour through the tip
    (tip - neighbour), normalised. Returns (tips_mask, vy, vx) with NaN
    vectors off tips.
    """
    f = np.asarray(faults).astype(bool)
    k = np.ones((3, 3), dtype=np.int32)
    k[1, 1] = 0
    deg = ndimage.convolve(f.astype(np.int32), k, mode="constant", cval=0)
    tips = f & (deg == 1)
    H, W = f.shape
    vy = np.full((H, W), np.nan)
    vx = np.full((H, W), np.nan)
    ys, xs = np.nonzero(tips)
    # neighbour offsets in fixed order (deterministic)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),
            (1, 1)]
    for y, x in zip(ys.tolist(), xs.tolist()):
        for dy, dx in offs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and f[ny, nx]:
                ddy, ddx = float(y - ny), float(x - nx)
                n = (ddy ** 2 + ddx ** 2) ** 0.5
                if n > 0:
                    vy[y, x], vx[y, x] = ddy / n, ddx / n
                break
    return tips, vy, vx


def tip_linkage_corridors(faults: np.ndarray,
                          dilatation: np.ndarray | None = None,
                          min_gap_px: int = 3,
                          max_gap_px: int = 20,
                          footprint: np.ndarray | None = None,
                          collinearity_min: float = 0.5) -> np.ndarray:
    """H8: tip-to-tip relay-linkage corridors between catalogue fault tips.

    For every pair of tips (see fault_endpoints) belonging to DIFFERENT
    connected components with separation in [min_gap_px, max_gap_px], rasterize
    the straight segment between them and weight it by:
      * gap decay: 1 - (gap - min_gap) / (max_gap - min_gap + eps),
      * collinearity: mean of cos(angle between each tip's outward strike
        and the segment direction); pairs below `collinearity_min` are
        rejected (tips must point at each other, i.e. a plausible relay),
      * extension bonus: mean normalised positive dilatation sampled along
        the segment (1.0 if no dilatation band given).
    Output is a float field (max over corridors), NaN outside `footprint`
    (or outside the finite faults grid when footprint is None).

    This is the linkage object the Apex pipeline lacks: its rays project
    outward from SINGLE tips and STOP at catalogue pixels (src/geology.py
    `break`), so no inter-tip corridor is ever emitted; its "relay" halo is
    an intersection (deg>=3) buffer, not a tip-pair linkage. Relay/linkage
    faults between mapped segments are exactly "newly mapped geometry of an
    existing fault system" (forum 11516 verified definition).
    """
    f = np.asarray(faults).astype(bool)
    H, W = f.shape
    if footprint is None:
        fp = np.isfinite(np.asarray(faults, dtype=np.float64)) | f
        fp = np.ones((H, W), dtype=bool) if not fp.any() else fp
    else:
        fp = np.asarray(footprint).astype(bool)
    out = np.zeros((H, W), dtype=np.float64)
    tips, vy, vx = fault_endpoints(f)
    comp, _ = ndimage.label(f, structure=np.ones((3, 3), dtype=int))
    ys, xs = np.nonzero(tips)
    n = len(ys)
    if n < 2:
        return np.where(fp, out, np.nan)
    comp_id = comp[ys, xs]
    tvy = vy[ys, xs]
    tvx = vx[ys, xs]
    if dilatation is not None:
        d = np.asarray(dilatation, dtype=np.float64)
        dpos = np.clip(np.nan_to_num(d), 0.0, None)
        dmax = dpos.max() + 1e-30
    else:
        dpos, dmax = None, 1.0
    # spatial hash so pairing is O(n * local) instead of O(n^2)
    cell = max_gap_px
    grid: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        key = (int(ys[i]) // cell, int(xs[i]) // cell)
        grid.setdefault(key, []).append(i)
    span = max_gap_px - min_gap_px + 1e-9
    for i in range(n):
        cy, cx = int(ys[i]) // cell, int(xs[i]) // cell
        for gy in (cy - 1, cy, cy + 1):
            for gx in (cx - 1, cx, cx + 1):
                for j in grid.get((gy, gx), ()):
                    if j <= i:
                        continue
                    if comp_id[i] == comp_id[j]:
                        continue
                    dy = float(ys[j] - ys[i])
                    dx = float(xs[j] - xs[i])
                    gap = (dy ** 2 + dx ** 2) ** 0.5
                    if not (min_gap_px <= gap <= max_gap_px):
                        continue
                    uy, ux = dy / gap, dx / gap
                    # each tip's outward strike vs segment direction
                    c1 = tvy[i] * uy + tvx[i] * ux
                    c2 = tvy[j] * (-uy) + tvx[j] * (-ux)
                    if not (np.isfinite(c1) and np.isfinite(c2)):
                        continue
                    col = 0.5 * (c1 + c2)
                    if col < collinearity_min:
                        continue
                    # rasterize segment (exclusive of endpoints: corridor only)
                    steps = int(max(abs(dy), abs(dx))) + 1
                    w_decay = max(0.0, 1.0 - (gap - min_gap_px) / span)
                    w = w_decay * (0.5 + 0.5 * min(1.0, max(0.0, col)))
                    pts = []
                    for s in range(1, steps):
                        t = s / steps
                        ry = int(round(ys[i] + dy * t))
                        rx = int(round(xs[i] + dx * t))
                        if 0 <= ry < H and 0 <= rx < W and fp[ry, rx]:
                            pts.append((ry, rx))
                    if not pts:
                        continue
                    if dpos is not None:
                        ext = float(np.mean([dpos[r, c] for r, c in pts])
                                    ) / dmax
                        w *= (0.5 + 0.5 * min(1.0, max(0.0, ext)))
                    for ry, rx in pts:
                        if w > out[ry, rx]:
                            out[ry, rx] = w
    return np.where(fp, out, np.nan)


# --------------------------------------- structure-tensor helper (H9 / H10)

def structure_coherence(field: np.ndarray, sigma: float = 1.5,
                        energy_floor_frac: float = 1e-6) -> dict:
    """Orientation coherence of a scalar field via its structure tensor.

    Gradients (gy, gx) -> tensor [[Jxx, Jxy],[Jxy, Jyy]] with Gaussian
    smoothing sigma -> eigenvalues l1 >= l2 ->
    coherence = (l1 - l2) / (l1 + l2 + eps) in [0, 1].
    High coherence = locally parallel structure (a lineament); low = flat or
    isotropic texture. Pixels whose tensor energy (l1+l2) is below
    `energy_floor_frac` of the field maximum are forced to 0: with no
    gradient energy there is no measurable orientation (otherwise the ratio
    is noise-dominated and spuriously ~1 on flats). NaN-aware via nearest
    fill; mask re-applied.
    """
    filled, valid = _fill_nan_nearest(field)
    gy, gx = np.gradient(filled)
    jxx = ndimage.gaussian_filter(gx * gx, sigma)
    jyy = ndimage.gaussian_filter(gy * gy, sigma)
    jxy = ndimage.gaussian_filter(gx * gy, sigma)
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    energy = np.nan_to_num(l1 + l2)
    coh = (l1 - l2) / (energy + 1e-30)
    emax = energy.max()
    if emax > 0:
        coh = np.where(energy < energy_floor_frac * emax, 0.0, coh)
    coh = np.clip(np.nan_to_num(coh), 0.0, 1.0)
    return {"coherence": _reapply(coh, valid),
            "energy": _reapply(energy, valid)}


def valley_axis_coherence(dem: np.ndarray, dx: float = DX_M,
                          sigma: float = 1.5) -> dict:
    """H9: valley-axis alignment lineaments (map-view orientation coherence).

    Active faults deflect drainages and etch linear valleys / shutter ridges
    that persist after the scarp step itself erodes below 100 m
    detectability. Signature: structure-tensor coherence of the detrended
    elevation gradient field (long parallel contour bands = coherent
    orientation) MULTIPLIED by normalised break-in-slope (topographic
    discontinuity present). Sign-free: no curvature-sign convention is
    asserted. Same bands as H3 (det_elev 12, det_elev_slope 19) but an
    orthogonal operator: H3 samples an antisymmetric step ALONG the gradient
    (cross-scarp profile); H9 measures orientation coherence IN MAP VIEW
    (along-valley alignment). Sibling Sato/structure-tensor lineament
    detectors (GEMSDOE4) run on edge bands generally, never coherence-gated
    break-in-slope on detrended elevation.
    """
    bis = break_in_slope(dem, dx)
    st = structure_coherence(dem, sigma=sigma)
    coh = np.nan_to_num(st["coherence"])
    bn = np.nan_to_num(bis) / (np.nanmax(np.nan_to_num(bis)) + 1e-30)
    score = coh * bn
    valid = np.isfinite(np.asarray(dem, dtype=np.float64))
    return {"coherence": _reapply(coh, valid),
            "break_in_slope": bis,
            "score": _reapply(score, valid)}


def seismic_ridge_coherence(eq_density: np.ndarray,
                            dist_to_quake: np.ndarray | None = None,
                            sigma: float = 1.5) -> dict:
    """H10: orientation-coherent microseismic ridges (blind seismogenic faults).

    Blind faults lack surface expression, so the Quaternary catalogue —
    which requires geologic evidence of surface deformation (USGS faults
    page) — systematically misses them; microseismic lineaments are the only
    surface-observable witness. Signature: Gaussian-smoothed earthquake
    density ridge (normalised) x structure-tensor coherence of the density
    gradient field (linear ridges, not blobs) x proximity weight
    1/(1 + deq/median(deq)) when dist_to_quake (band 10) is given.
    Novel vs H5 (binary quantile INTERSECTION of ieq with shallow basement:
    blob gating, no orientation, no band 10): H10 is continuous,
    orientation-selective, and basement-free. Band deq_n100a15 is used
    nowhere else in this repo.
    """
    e = np.asarray(eq_density, dtype=np.float64)
    valid = np.isfinite(e)
    filled, _ = _fill_nan_nearest(e)
    sm = ndimage.gaussian_filter(np.nan_to_num(filled), sigma)
    ridge = sm / (np.nanmax(sm) + 1e-30)
    st = structure_coherence(np.where(valid, filled, np.nan), sigma=sigma)
    coh = np.nan_to_num(st["coherence"])
    score = np.clip(ridge, 0, None) * coh
    if dist_to_quake is not None:
        dq = np.asarray(dist_to_quake, dtype=np.float64)
        dqv = dq[np.isfinite(dq)]
        med = float(np.median(dqv)) if dqv.size else 1.0
        prox = 1.0 / (1.0 + np.nan_to_num(dq, nan=med * 4) / (med + 1e-30))
        score = score * prox
        valid = valid & np.isfinite(dq)
    return {"ridge": _reapply(np.clip(ridge, 0, None), valid),
            "coherence": _reapply(coh, valid),
            "score": _reapply(score, valid)}
