"""Tests for submission template conformance (needs data/ template).

Skipped when data/ is absent (hermetic CI without the 400 MB download) —
same pattern as tests/test_bandlayout.py.
"""
from pathlib import Path

import numpy as np
import pytest
from src.submission_io import conform_to_template
from src.dataset import EXPECTED_HEIGHT, EXPECTED_WIDTH, load_footprint_mask

TEMPLATE = (Path(__file__).resolve().parents[1] / "data"
            / "sample_submission.tif")
TEMPLATE_ALT = (Path(__file__).resolve().parents[1] / "data"
                / "example_submission.tif")


@pytest.mark.skipif(not (TEMPLATE.exists() or TEMPLATE_ALT.exists()),
                    reason="data/ template not placed in this env")
def test_conform_to_template_sanitizes_nans():
    footprint = load_footprint_mask()
    raw = np.zeros((EXPECTED_HEIGHT, EXPECTED_WIDTH), dtype=np.float32)

    # Inject invalid values inside footprint: NaNs, Infs, < 0, > 1
    foot_indices = np.where(footprint)
    raw[foot_indices[0][:100], foot_indices[1][:100]] = np.nan
    raw[foot_indices[0][100:200], foot_indices[1][100:200]] = np.inf
    raw[foot_indices[0][200:300], foot_indices[1][200:300]] = -5.0
    raw[foot_indices[0][300:400], foot_indices[1][300:400]] = 2.5

    # Inject finite values outside footprint
    out_indices = np.where(~footprint)
    raw[out_indices[0][:100], out_indices[1][:100]] = 0.8

    # Apply conformation
    conformed = conform_to_template(raw, fill_value=0.0)

    # 1. Shape is preserved
    assert conformed.shape == (EXPECTED_HEIGHT, EXPECTED_WIDTH)

    # 2. Inside footprint: strictly finite, strictly in [0, 1]
    inside = conformed[footprint]
    assert not np.isnan(inside).any(), "Found NaNs inside valid footprint!"
    assert not np.isinf(inside).any(), "Found Infs inside valid footprint!"
    assert inside.min() >= 0.0, f"Found negative values: {inside.min()}"
    assert inside.max() <= 1.0, f"Found values > 1.0: {inside.max()}"

    # 3. Outside footprint: strictly NaN
    outside = conformed[~footprint]
    assert np.isnan(outside).all(), "Found finite values outside valid footprint!"
