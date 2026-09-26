"""Tests for structural geology and fault extension algorithms."""

import numpy as np
from src.geology import compute_along_strike_extensions, compute_near_fault_envelope


def test_along_strike_extension_projects_rays():
    # 20x20 grid with a vertical fault from y=5 to y=15 at x=10
    faults = np.zeros((20, 20), dtype=bool)
    faults[5:16, 10] = True

    ext = compute_along_strike_extensions(faults, max_step=3, base_weight=0.8)

    # Known fault pixels should not be overwritten
    assert not np.any(ext[faults] > 0)

    # Ray should project north (y < 5) and south (y > 15) along x=10
    assert ext[4, 10] > 0
    assert ext[3, 10] > 0
    assert ext[16, 10] > 0
    assert ext[17, 10] > 0

    # East and west (sideways) should NOT have ray projections
    assert ext[10, 8] == 0
    assert ext[10, 12] == 0


def test_near_fault_envelope_bounded_r2():
    faults = np.zeros((25, 25), dtype=bool)
    faults[12, 12] = True

    halo = compute_near_fault_envelope(faults, r_pixels=2, core_prob=0.9)

    # Core point
    assert halo[12, 12] == 0.9

    # Distance 1 pixel: k(1) = 1 - 1/3 = 2/3 -> 0.9 * 2/3 = 0.6
    assert abs(halo[12, 13] - 0.6) < 1e-4

    # Distance 2 pixels: k(2) = 1 - 2/3 = 1/3 -> 0.9 * 1/3 = 0.3
    assert abs(halo[12, 14] - 0.3) < 1e-4

    # Distance 3 pixels: strictly 0 (envelope bounded at r=2)
    assert halo[12, 15] == 0.0
