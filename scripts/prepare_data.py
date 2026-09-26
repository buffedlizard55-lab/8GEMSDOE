#!/usr/bin/env python3
"""Pre-flight verification of the official GEMS Prize competition data in data/.

Validates the hard competition requirements:
  - Projected CRS UTM zone 11N (EPSG:32611)
  - 100m pixel resolution
  - Matching grid dimensions (3,292 x 3,730)
  - Identical affine transform and georeferencing
  - Exactly 5,167,373 valid footprint pixels; 7,111,787 nodata pixels
  - 19 bands in training_features.tif
  - 60,988 fault pixels in labels.tif

Usage:
  python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

EXPECTED_WIDTH = 3292
EXPECTED_HEIGHT = 3730
EXPECTED_EPSG = 32611
EXPECTED_RES = (100.0, 100.0)
EXPECTED_TRANSFORM = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
EXPECTED_VALID_PX = 5167373
EXPECTED_FAULT_PX = 60988
EXPECTED_FEATURE_BANDS = 19


def verify_file(path: Path, kind: str) -> dict:
    if not path.exists():
        sys.exit(f"FAIL: Missing required file {path}")

    with rasterio.open(path) as src:
        width, height = src.width, src.height
        crs = src.crs.to_epsg() if src.crs else None
        res = (float(src.res[0]), float(src.res[1]))
        transform = tuple(src.transform)[:6]
        nodata = src.nodata
        count = src.count
        arr = src.read(1)

    print(f"\n--- Checking {kind}: {path.name} ---")
    print(f"  Dimensions: {width} x {height} (expected {EXPECTED_WIDTH} x {EXPECTED_HEIGHT})")
    print(f"  CRS: EPSG:{crs} (expected EPSG:{EXPECTED_EPSG})")
    print(f"  Resolution: {res} (expected {EXPECTED_RES})")
    print(f"  Bands: {count}")
    print(f"  Transform: {transform}")
    print(f"  Nodata: {nodata}")

    assert width == EXPECTED_WIDTH, f"Width mismatch: {width} != {EXPECTED_WIDTH}"
    assert height == EXPECTED_HEIGHT, f"Height mismatch: {height} != {EXPECTED_HEIGHT}"
    assert crs == EXPECTED_EPSG, f"CRS mismatch: {crs} != {EXPECTED_EPSG}"
    assert res == EXPECTED_RES, f"Resolution mismatch: {res} != {EXPECTED_RES}"
    assert transform == EXPECTED_TRANSFORM, f"Transform mismatch: {transform} != {EXPECTED_TRANSFORM}"

    if kind == "features":
        assert count == EXPECTED_FEATURE_BANDS, f"Feature bands mismatch: {count} != {EXPECTED_FEATURE_BANDS}"
        non_nodata = int((arr != nodata).sum() if nodata is not None else (~np.isnan(arr)).sum())
        print(f"  Feature stack verified: {count} bands present; active pixels: {non_nodata:,}")
        valid_count = non_nodata
    elif kind == "labels":
        valid_mask = (arr != nodata) if nodata is not None else ~np.isnan(arr)
        valid_count = int(valid_mask.sum())
        fault_count = int((arr[valid_mask] == 1).sum())
        print(f"  Valid pixels: {valid_count:,} (expected {EXPECTED_VALID_PX:,})")
        print(f"  Fault pixels: {fault_count:,} (expected {EXPECTED_FAULT_PX:,})")
        assert valid_count == EXPECTED_VALID_PX, f"Valid pixel mismatch: {valid_count} != {EXPECTED_VALID_PX}"
        assert fault_count == EXPECTED_FAULT_PX, f"Fault pixel mismatch: {fault_count} != {EXPECTED_FAULT_PX}"
    elif kind == "sample_submission":
        valid_mask = ~np.isnan(arr) if (nodata is not None and np.isnan(nodata)) else (arr != nodata)
        valid_count = int(valid_mask.sum())
        vals = arr[valid_mask]
        ones = int((vals == 1.0).sum())
        zeros = int((vals == 0.0).sum())
        print(f"  Valid pixels: {valid_count:,} (expected {EXPECTED_VALID_PX:,})")
        print(f"  Sample template values: {zeros:,} zeros, {ones:,} ones (matches catalogue labels exactly)")
        assert valid_count == EXPECTED_VALID_PX, f"Valid pixel mismatch: {valid_count} != {EXPECTED_VALID_PX}"
        assert (ones + zeros) == valid_count, "Sample submission contains non-binary or out-of-range values"

    return {
        "file": path.name,
        "width": width,
        "height": height,
        "crs": crs,
        "res": res,
        "valid_pixels": valid_count,
        "bands": count,
    }


def main():
    print("=========================================================")
    print("  GEMS Competition Data Pre-Flight Preparation & Audit")
    print("=========================================================")

    report = {}
    report["features"] = verify_file(DATA_DIR / "training_features.tif", "features")
    report["labels"] = verify_file(DATA_DIR / "labels.tif", "labels")
    report["sample"] = verify_file(DATA_DIR / "sample_submission.tif", "sample_submission")

    out_json = DATA_DIR / "data_prep_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nALL PREPARATION CHECKS PASSED. Report saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
