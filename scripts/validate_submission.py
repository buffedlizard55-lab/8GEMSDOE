#!/usr/bin/env python3
"""Format and Conformance Validator for DrivenData GEMS Submissions.

Runs 13 independent checks to guarantee that a candidate GeoTIFF passes
DrivenData's automated submission ingestion without errors.

Specifically guards against the platform error:
  "Predicted values must be in range [0, 1]"
which occurs when NaN or non-finite values sit inside the scored footprint.

Usage:
  python scripts/validate_submission.py downloads/submission.tif
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.dataset import (
    EXPECTED_EPSG,
    EXPECTED_HEIGHT,
    EXPECTED_RES,
    EXPECTED_TRANSFORM,
    EXPECTED_WIDTH,
    load_footprint_mask,
)

EXPECTED_VALID_PX = 5167373
EXPECTED_NAN_PX = 7111787


def validate(tif_path: str | Path) -> bool:
    tif_path = Path(tif_path)
    print(f"=== Validating Submission File: {tif_path.name} ===")
    if not tif_path.exists():
        print(f"FAIL [Gate 1]: File not found: {tif_path}")
        return False
    size_bytes = tif_path.stat().st_size
    print(f"PASS [Gate 1]: File exists ({size_bytes:,} bytes)")

    footprint = load_footprint_mask()

    try:
        with rasterio.open(tif_path) as src:
            driver = src.driver
            count = src.count
            dtype = src.dtypes[0]
            width, height = src.width, src.height
            crs = src.crs.to_epsg() if src.crs else None
            res = (float(src.res[0]), float(src.res[1]))
            transform = tuple(src.transform)[:6]
            nodata = src.nodata
            arr = src.read(1)
    except Exception as e:
        print(f"FAIL [Gate 2]: Unable to open GeoTIFF with rasterio: {e}")
        return False

    print(f"PASS [Gate 2]: Valid GeoTIFF driver ({driver})")

    # Gate 3: Band count
    if count != 1:
        print(f"FAIL [Gate 3]: Expected 1 band, got {count}")
        return False
    print("PASS [Gate 3]: Single band raster")

    # Gate 4: dtype float32
    if dtype != "float32":
        print(f"FAIL [Gate 4]: Expected dtype float32, got {dtype}")
        return False
    print("PASS [Gate 4]: Datatype is float32")

    # Gate 5: Dimensions
    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        print(f"FAIL [Gate 5]: Expected dimensions ({EXPECTED_WIDTH}, {EXPECTED_HEIGHT}), got ({width}, {height})")
        return False
    print(f"PASS [Gate 5]: Grid shape is {width} x {height}")

    # Gate 6: CRS EPSG:32611
    if crs != EXPECTED_EPSG:
        print(f"FAIL [Gate 6]: Expected CRS EPSG:{EXPECTED_EPSG}, got EPSG:{crs}")
        return False
    print(f"PASS [Gate 6]: Projected CRS is EPSG:{EXPECTED_EPSG}")

    # Gate 7: Resolution & Transform
    if res != (EXPECTED_RES, EXPECTED_RES):
        print(f"FAIL [Gate 7]: Expected resolution {EXPECTED_RES}m, got {res}")
        return False
    if transform != EXPECTED_TRANSFORM:
        print(f"FAIL [Gate 7]: Transform mismatch: expected {EXPECTED_TRANSFORM}, got {transform}")
        return False
    print("PASS [Gate 7]: Affine geotransform matches template exactly")

    # Gate 8: Nodata tag
    if nodata is None or not np.isnan(nodata):
        print(f"FAIL [Gate 8]: Nodata must be NaN, got {nodata}")
        return False
    print("PASS [Gate 8]: Nodata tag is set to NaN")

    # Gate 9: Footprint mask alignment
    nan_mask = np.isnan(arr)
    valid_mask = ~nan_mask
    valid_px = int(valid_mask.sum())
    nan_px = int(nan_mask.sum())

    if valid_px != EXPECTED_VALID_PX:
        print(f"FAIL [Gate 9]: Valid pixel count mismatch: {valid_px} (expected {EXPECTED_VALID_PX})")
        return False
    print(f"PASS [Gate 9]: Valid footprint count is exactly {valid_px:,}")

    if nan_px != EXPECTED_NAN_PX:
        print(f"FAIL [Gate 10]: NaN pixel count mismatch: {nan_px} (expected {EXPECTED_NAN_PX})")
        return False
    print(f"PASS [Gate 10]: Nodata NaN count is exactly {nan_px:,}")

    # Gate 11: No NaNs inside template valid region ("Predicted values must be in range [0, 1]")
    inside_vals = arr[footprint]
    n_internal_nans = int(np.isnan(inside_vals).sum())
    if n_internal_nans > 0:
        print(f"FAIL [Gate 11]: {n_internal_nans} NaN values found INSIDE valid footprint.")
        print("  -> DrivenData will reject this file with 'Predicted values must be in range [0, 1]'!")
        return False
    print("PASS [Gate 11]: Zero NaN or Inf values inside valid footprint")

    # Gate 12: Value range [0.0, 1.0]
    min_val, max_val = float(inside_vals.min()), float(inside_vals.max())
    if min_val < 0.0 or max_val > 1.0:
        print(f"FAIL [Gate 12]: Predictions out of range [0, 1]: min={min_val}, max={max_val}")
        return False
    print(f"PASS [Gate 12]: All predictions in valid range [0, 1] (min={min_val:.6f}, max={max_val:.6f})")

    # Gate 13: Zero finite pixels outside footprint
    outside_vals = arr[~footprint]
    n_external_finite = int(np.isfinite(outside_vals).sum())
    if n_external_finite > 0:
        print(f"FAIL [Gate 13]: {n_external_finite} finite values found OUTSIDE valid footprint.")
        return False
    print("PASS [Gate 13]: Zero finite values outside footprint")

    sha256 = hashlib.sha256(tif_path.read_bytes()).hexdigest()
    print(f"\nALL 13 GATES PASSED! SHA-256: {sha256}")
    print("This file is 100% compliant and ready for immediate DrivenData upload.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_submission.py <path_to_submission.tif>")
        sys.exit(1)
    success = validate(sys.argv[1])
    sys.exit(0 if success else 1)
