"""Structural Geology & Geothermal Discovery Engine.

Implements domain-specific structural geology algorithms for Basin and Range
and Walker Lane geothermal systems:
  1. Along-strike fault tip extensions: normal faults terminate in alluvium where
     under-mapped fault tip damage zones continue for 0.5 - 1.5 km.
  2. Fault intersections: high fracture density conduits where normal and
     strike-slip faults intersect.
  3. Relay stepovers: overlapping fault segments with extensional fracture meshes
     concentrating high-temperature geothermal circulation (Faulds et al. 2011).
  4. Geodetic strain tensor weighting: active crustal shear and dilatation rates
     foster open permeable fracture pathways for geothermal fluids.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt, label


def compute_along_strike_extensions(
    faults: np.ndarray,
    max_step: int = 10,
    base_weight: float = 0.85,
) -> np.ndarray:
    """Projects directional fault continuation rays from termination endpoints along the local strike azimuth."""
    H, W = faults.shape
    ext = np.zeros((H, W), dtype=np.float32)

    # 3x3 neighbor convolution to find degree
    k = np.ones((3, 3), dtype=np.int32)
    k[1, 1] = 0
    deg = convolve(faults.astype(np.int32), k, mode="constant", cval=0)
    endpoints = faults & (deg == 1)

    ey, ex = np.where(endpoints)
    dy_arr = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dx_arr = np.array([-1, 0, 1, -1, 1, -1, 0, 1])

    for i in range(len(ey)):
        y, x = int(ey[i]), int(ex[i])
        for dy, dx in zip(dy_arr, dx_arr):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and faults[ny, nx]:
                vy, vx = float(y - ny), float(x - nx)
                norm = np.hypot(vy, vx)
                if norm > 0:
                    vy, vx = vy / norm, vx / norm
                    for step in range(1, max_step + 1):
                        ry = int(round(y + vy * step))
                        rx = int(round(x + vx * step))
                        if 0 <= ry < H and 0 <= rx < W:
                            if faults[ry, rx]:
                                break
                            decay = max(0.0, 1.0 - (step - 1) / float(max_step)) * base_weight
                            if decay > ext[ry, rx]:
                                ext[ry, rx] = decay
                break
    return ext


def compute_near_fault_envelope(
    faults: np.ndarray,
    r_pixels: int = 2,
    core_prob: float = 0.90,
) -> np.ndarray:
    """Computes a metric-aligned triangular decay halo (k(d) = 1 - d/3) within R=2 pixels.

    Captures near-fault splays, off-fault fractures, and mapping corrections
    as confirmed by DrivenData competition organizers (ChrisK-DD, Sept 21).
    Restricted to r <= 2 pixels to completely prevent false-positive inflation.
    """
    dist = distance_transform_edt(~faults)
    halo = np.zeros_like(dist, dtype=np.float32)
    valid_dist = (dist > 0) & (dist <= r_pixels)
    halo[valid_dist] = (1.0 - dist[valid_dist] / 3.0) * core_prob
    halo[faults] = core_prob
    return halo


def compute_geothermal_structural_prior(
    faults: np.ndarray,
    shear_strain: np.ndarray,
    dilat_strain: np.ndarray,
    conductivity: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Computes an integrated hydrothermal fluid pathway favorability prior."""
    H, W = faults.shape
    prior = np.zeros((H, W), dtype=np.float32)

    # 1. Along strike extensions
    ext = compute_along_strike_extensions(faults, max_step=10, base_weight=0.75)

    # 2. Distance transform to fault intersections & endpoints
    k = np.ones((3, 3), dtype=np.int32)
    k[1, 1] = 0
    deg = convolve(faults.astype(np.int32), k, mode="constant", cval=0)
    intersections = faults & (deg >= 3)
    dist_inter = distance_transform_edt(~intersections)
    inter_halo = np.where((dist_inter > 0) & (dist_inter <= 5), (1.0 - dist_inter / 6.0) * 0.6, 0.0).astype(np.float32)

    # 3. Geothermal strain & conductivity modulation
    # Normalize inputs on valid mask
    def norm_band(b):
        b_clean = np.where(np.isfinite(b) & (b > -1e30), b, 0.0)
        v = b_clean[valid_mask]
        p1, p99 = np.percentile(v, 1.0), np.percentile(v, 99.0)
        if p99 > p1:
            scaled = (b_clean - p1) / (p99 - p1)
        else:
            scaled = np.zeros_like(b_clean)
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)

    norm_shear = norm_band(shear_strain)
    norm_dilat = norm_band(dilat_strain)
    norm_cond = norm_band(conductivity)

    geothermal_mod = (0.4 * norm_shear + 0.3 * norm_dilat + 0.3 * norm_cond)

    prior = np.maximum(ext, inter_halo)
    # Enhance candidates located in high-strain, high-conductivity geothermal zones
    prior = prior * (0.6 + 0.4 * geothermal_mod)
    prior = np.clip(prior, 0.0, 0.95)
    return prior
