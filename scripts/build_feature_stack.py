#!/usr/bin/env python3
"""Build the shared derived-feature plane stack used by the LOFSO experiments.

Output: ``data/derived/planes.f32`` (float32 memmap, shape (P, H, W)) plus
``data/derived/planes.json`` (ordered plane names + provenance). Both are
git-ignored: they are recomputed from ``data/training_features.tif``.

Plane families (each named for the hypothesis it serves):
  raw b01..b19      the 19 supplied GeoDAWN/INGENIOUS bands, verbatim
  grad_*            supplied-band gradient magnitudes (edge detectors)
  H11_*             gravity-magnetic edge co-location and phase agreement
  H12_*             gravity/magnetic curvature (Laplacian, Hessian ridge)
  H13_*             DEM scale-space lineament persistence + tensor anisotropy
  H14_*             seismic/geophysical concordance + strain corridors

Memory safety: the 19-band stack is 932 MB as float32 and this sandbox has
~3 GB, so raw bands are written straight into the on-disk memmap and every
derived plane is computed from memmap slices, one plane at a time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA = REPO / "data"
OUT_DIR = DATA / "derived"
DX = 100.0  # metres per pixel (verified: scripts/prepare_data.py, EPSG:32611 @100 m)

NODATA_F32_MIN = -1e38


def _clean(a: np.ndarray) -> np.ndarray:
    """Sentinel nodata -> NaN (the file uses -3.4028234663852886e+38)."""
    a = np.asarray(a, dtype=np.float32)
    a[a < NODATA_F32_MIN] = np.nan
    return a


def _grad_mag(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.nan_to_num(a, nan=0.0), DX)
    m = np.hypot(gy, gx).astype(np.float32)
    m[~np.isfinite(a)] = np.nan
    return m


def _grad_vec(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.nan_to_num(a, nan=0.0)
    gy, gx = np.gradient(b, DX)
    return gy.astype(np.float32), gx.astype(np.float32)


def _lap(a: np.ndarray, sigma: float) -> np.ndarray:
    b = np.nan_to_num(a, nan=0.0)
    s = ndimage.gaussian_filter(b, sigma)
    l = ndimage.gaussian_laplace(b, sigma)
    out = l.astype(np.float32)
    # zero out the smoothing bleed where the input was nodata
    valid = ndimage.gaussian_filter(np.isfinite(a).astype(np.float32), sigma) > 0.99
    out[~valid] = np.nan
    del s
    return out


def _hessian_ridge(a: np.ndarray, sigma: float) -> np.ndarray:
    """Larger eigenvalue of the Hessian of a Gaussian-smoothed field.

    Negative lambda2 = ridge/valley (elongated) structure; positive = blob.
    Fault-lineaments show up as sustained negative lambda2 corridors.
    """
    b = np.nan_to_num(a, nan=0.0)
    g = ndimage.gaussian_filter(b, sigma)
    gyy = ndimage.sobel(ndimage.sobel(g, axis=0), axis=0) / 4.0
    gxx = ndimage.sobel(ndimage.sobel(g, axis=1), axis=1) / 4.0
    gxy = ndimage.sobel(ndimage.sobel(g, axis=0), axis=1) / 4.0
    tr = gyy + gxx
    det = gyy * gxx - gxy * gxy
    disc = np.maximum(tr * tr / 4.0 - det, 0.0)
    lam2 = tr / 2.0 + np.sqrt(disc)   # algebraically larger eigenvalue
    lam1 = tr / 2.0 - np.sqrt(disc)
    out = np.minimum(lam1, lam2).astype(np.float32)  # most negative = strongest ridge
    valid = ndimage.gaussian_filter(np.isfinite(a).astype(np.float32), sigma) > 0.99
    out[~valid] = np.nan
    return out


def _structure_coherence(a: np.ndarray, sigma_i: float = 1.0,
                         sigma_w: float = 3.0) -> np.ndarray:
    """Anisotropy of the local structure tensor of a field, in [0, 1].

    1 = perfectly lineated (gradient direction constant over the window),
    0 = isotropic. Long, straight fault scarps/lineaments score high; this is
    the standard coherence/lineament operator (e.g.structure-tensor based
    lineament enhancement), applied here to the detrended-elevation gradient.
    """
    gy, gx = _grad_vec(a)
    jyy = ndimage.gaussian_filter(np.nan_to_num(gy * gy), sigma_w)
    jxx = ndimage.gaussian_filter(np.nan_to_num(gx * gx), sigma_w)
    jxy = ndimage.gaussian_filter(np.nan_to_num(gy * gx), sigma_w)
    tr = jyy + jxx
    det = jyy * jxx - jxy * jxy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    l1 = (tr + np.sqrt(disc)) / 2.0
    l2 = (tr - np.sqrt(disc)) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(l1 > 0, (l1 - l2) / l1, 0.0)
    return coh.astype(np.float32)


def _z(a: np.ndarray, n_sample: int = 400_000, seed: int = 3) -> np.ndarray:
    """Robust z-score (median / IQR) estimated on a subsample; NaN preserved."""
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a)
    rng = np.random.default_rng(seed)
    if v.size > n_sample:
        v = rng.choice(v, size=n_sample, replace=False)
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    iqr = float(q3 - q1)
    scale = iqr if iqr > 0 else float(v.std() + 1e-9)
    out = (a - med) / (1.349 * scale)
    return np.clip(np.nan_to_num(out, nan=0.0), -6, 6).astype(np.float32)


def _tensor(a: np.ndarray, sigma_w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Structure tensor of a field, smoothed over `sigma_w` px -> (jyy, jxx, jxy)."""
    gy, gx = _grad_vec(a)
    jyy = ndimage.gaussian_filter(np.nan_to_num(gy * gy), sigma_w)
    jxx = ndimage.gaussian_filter(np.nan_to_num(gx * gx), sigma_w)
    jxy = ndimage.gaussian_filter(np.nan_to_num(gy * gx), sigma_w)
    return jyy.astype(np.float32), jxx.astype(np.float32), jxy.astype(np.float32)


def _coherence_from_tensor(jyy, jxx, jxy) -> np.ndarray:
    tr = jyy + jxx
    det = jyy * jxx - jxy * jxy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    l1 = (tr + np.sqrt(disc)) / 2.0
    l2 = (tr - np.sqrt(disc)) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(l1 > 0, (l1 - l2) / l1, 0.0)
    return coh.astype(np.float32)


def _strike_field(a: np.ndarray, sigma_w: float = 3.0) -> np.ndarray:
    """Azimuth (radians) of the locally dominant LINEAMENT direction.

    The structure tensor's major axis is perpendicular to the lineament, so the
    lineament azimuth is theta + pi/2 where theta = 0.5*atan2(2 Jxy, Jyy - Jxx).
    """
    jyy, jxx, jxy = _tensor(a, sigma_w)
    theta = 0.5 * np.arctan2(2.0 * jxy, jyy - jxx)
    return (theta + np.pi / 2.0).astype(np.float32)


def _along_strike_persistence(field: np.ndarray, strike: np.ndarray,
                              steps: tuple[int, ...] = (3, 6)) -> np.ndarray:
    """Mean of `field` sampled along the local strike, relative to the pixel value.

    A real fault trace keeps its signature for kilometres; noise and short-lived
    topography do not. Ratio in [0, ~1+]: high = the edge continues along its
    own strike, which is the map-view definition of a lineament.
    """
    f = np.nan_to_num(np.asarray(field, dtype=np.float32), nan=0.0)
    H, W = f.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dy = np.sin(strike).astype(np.float32)
    dx = np.cos(strike).astype(np.float32)
    acc = np.zeros_like(f)
    n = 0
    for t in steps:
        for sgn in (1.0, -1.0):
            sy = (yy + sgn * t * dy).astype(np.float32)
            sx = (xx + sgn * t * dx).astype(np.float32)
            acc += ndimage.map_coordinates(f, [sy, sx], order=1, mode="constant",
                                           prefilter=False).astype(np.float32)
            n += 1
            del sy, sx
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(f > 1e-6, (acc / n) / f, 0.0)
    return np.clip(np.nan_to_num(ratio), 0.0, 4.0).astype(np.float32)


def main() -> int:
    feat = DATA / "training_features.tif"
    if not feat.exists():
        print(f"FAIL: {feat} missing - run scripts/download_competition_data.sh")
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(feat) as src:
        H, W, nb = src.height, src.width, src.count
        tags = [src.tags(i) for i in range(1, nb + 1)]
    print(f"features {W}x{H} bands={nb}")

    t0 = time.time()
    MAX_PLANES = 64
    planes_path = OUT_DIR / "planes.f32"
    planes_path.write_bytes(b"")
    with open(planes_path, "r+b") as fh:
        fh.truncate(4 * MAX_PLANES * H * W)
    planes: list[str] = []

    def add(name: str, arr: np.ndarray) -> None:
        """Append one plane to the on-disk stack (never holds two planes in RAM)."""
        if len(planes) >= MAX_PLANES:
            raise RuntimeError("plane budget exhausted - raise MAX_PLANES")
        idx = len(planes)
        planes.append(name)
        st = np.memmap(planes_path, dtype=np.float32, mode="r+",
                       shape=(MAX_PLANES, H, W))
        st[idx] = np.asarray(arr, dtype=np.float32)
        st.flush()
        del st
        print(f"  [{idx+1:>2}] {name:32s} {time.time()-t0:6.1f}s")

    # ---- raw bands (verbatim, nodata -> NaN)
    with rasterio.open(feat) as src:
        for i in range(1, nb + 1):
            name = tags[i - 1].get("description", f"band{i}").split(" - ")[0]
            short = f"b{i:02d}_" + "".join(c if c.isalnum() else "_" for c in name)[:26]
            add(short, _clean(src.read(i)))

    def plane(idx0: int) -> np.ndarray:  # 1-based band index -> raw plane
        st = np.memmap(planes_path, dtype=np.float32, mode="r",
                       shape=(MAX_PLANES, H, W))
        out = np.array(st[idx0 - 1])
        del st
        return out

    det_elev = plane(12)          # Detrended elevation
    det_slope = plane(19)         # Detrended elevation slope
    grav = plane(13)              # Isostatic gravity anomaly
    grav_hg = plane(18)           # Isostatic gravity anomaly horizontal gradient
    tmi = plane(14)               # Total magnetic intensity
    tmi_hg = plane(3)             # TMI horizontal gradient
    tmi_vg = plane(9)             # TMI vertical gradient
    cond = plane(17)              # Conductivity surface
    depth_base = plane(15)        # Depth to basement surface
    ieq = plane(16)               # Earthquake intensity/density
    deq = plane(10)               # Distance to earthquake
    dil = plane(8)                # Geodetic dilatation rate
    shear = plane(7)              # Geodetic shear rate
    sec_inv = plane(4)            # Geodetic second invariant

    # ---- supplied-band gradient magnitudes
    add("grad_det_elev", _grad_mag(det_elev))
    add("grad_grav", _grad_mag(grav))
    add("grad_tmi", _grad_mag(tmi))
    add("grad_cond", _grad_mag(cond))
    add("grad_depth_base", _grad_mag(depth_base))
    add("mean5_det_elev", ndimage.uniform_filter(np.nan_to_num(det_elev), 5).astype(np.float32))

    # ---- H11: gravity-magnetic edge co-location and phase agreement
    zg, zm = _z(grav_hg), _z(tmi_hg)
    add("H11_coloc_min", np.minimum(zg, zm))
    add("H11_coloc_prod", zg * zm)
    gyy, gxx = _grad_vec(grav)
    myy, mxx = _grad_vec(tmi)
    dot = gyy * myy + gxx * mxx
    ng = np.hypot(gyy, gxx) + 1e-9
    nm = np.hypot(myy, mxx) + 1e-9
    add("H11_phase_cos", (dot / (ng * nm)).astype(np.float32))
    add("H11_analytic_sig", np.hypot(_grad_mag(tmi), tmi_vg).astype(np.float32))

    # ---- H12: curvature / ridge-valley differential (gravity + magnetics)
    add("H12_lap_grav_s1", _lap(grav, 1.0))
    add("H12_lap_grav_s3", _lap(grav, 3.0))
    add("H12_ridge_grav_s2", _hessian_ridge(grav, 2.0))
    add("H12_ridge_tmi_s2", _hessian_ridge(tmi, 2.0))

    # ---- H13: DEM lineament persistence and tensor anisotropy
    coh = _structure_coherence(det_elev, 1.0, 3.0)
    add("H13_coherence_det_elev", coh)
    g1 = _z(_grad_mag(det_elev))
    g2 = _z(_grad_mag(ndimage.gaussian_filter(np.nan_to_num(det_elev), 2.0)))
    g4 = _z(_grad_mag(ndimage.gaussian_filter(np.nan_to_num(det_elev), 4.0)))
    add("H13_scale_persistence", np.minimum(np.minimum(g1, g2), g4))
    add("H13_slope_x_coherence", (_z(det_slope) * coh).astype(np.float32))

    # ---- H14: seismic / strain concordance
    add("H14_ieq_x_coherence", (_z(ieq) * coh).astype(np.float32))
    add("H14_log_deq", np.log1p(np.nan_to_num(deq, nan=0.0)).astype(np.float32))
    add("H14_strain_corridor", (_z(np.hypot(dil, shear)) * _z(sec_inv)).astype(np.float32))


    # ---- C family: map-view context (emulates what a CNN sees, cheaply)
    gm_det = _grad_mag(det_elev)
    coh5 = _coherence_from_tensor(*_tensor(det_elev, 5.0))
    coh9 = _coherence_from_tensor(*_tensor(det_elev, 9.0))
    add("C_coherence_w5", coh5)
    add("C_coherence_w9", coh9)
    strike = _strike_field(det_elev, 3.0)
    add("C_alongstrike_dem", _along_strike_persistence(gm_det, strike))
    add("C_alongstrike_gravhg", _along_strike_persistence(grav_hg, strike))
    ms = [_z(_grad_mag(ndimage.gaussian_filter(np.nan_to_num(det_elev), sg)))
          for sg in (1.0, 2.0, 4.0, 8.0)]
    add("C_multiscale_grad_max", np.maximum.reduce(ms))
    # fast windowed std: sqrt(E[x^2] - E[x]^2) via two box filters
    _de = np.nan_to_num(det_elev)
    _m1 = ndimage.uniform_filter(_de, 9, mode="nearest")
    _m2 = ndimage.uniform_filter(_de * _de, 9, mode="nearest")
    add("C_relief_std9", np.sqrt(np.maximum(_m2 - _m1 * _m1, 0.0)).astype(np.float32))
    add("C_edge_density5", ndimage.uniform_filter(
        np.nan_to_num(_z(grav_hg) + _z(tmi_hg)), 5).astype(np.float32))

    with open(planes_path, "r+b") as fh:
        fh.truncate(4 * len(planes) * H * W)
    meta = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "scripts/build_feature_stack.py",
        "source": "data/training_features.tif",
        "shape": [len(planes), H, W],
        "n_planes": len(planes),
        "dtype": "float32",
        "dx_m": DX,
        "planes": planes,
        "band_tags": [t.get("description") for t in tags],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "planes.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {planes_path} ({len(planes)} planes, "
          f"{planes_path.stat().st_size/1e9:.2f} GB) in {meta['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
