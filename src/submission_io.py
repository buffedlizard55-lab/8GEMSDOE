"""Submission raster I/O, format conformation, and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio

from src.dataset import (
    EXPECTED_EPSG,
    EXPECTED_HEIGHT,
    EXPECTED_RES,
    EXPECTED_WIDTH,
    get_template_path,
    load_footprint_mask,
)


def conform_to_template(raw_pred: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Conforms a prediction array to the exact competition template specification:
      - Shape: (3730, 3292)
      - dtype: float32
      - Inside valid footprint: strictly finite in [0.0, 1.0] (no NaNs, no Infs)
      - Outside valid footprint: strictly np.nan (nodata)

    This completely eliminates the DrivenData error:
      "Predicted values must be in range [0, 1]"
    which is triggered whenever a NaN or Inf occurs inside the scored area.
    """
    if raw_pred.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise ValueError(f"Shape mismatch: {raw_pred.shape} != {(EXPECTED_HEIGHT, EXPECTED_WIDTH)}")

    footprint = load_footprint_mask()
    out = np.full((EXPECTED_HEIGHT, EXPECTED_WIDTH), np.nan, dtype=np.float32)

    # Extract inside-footprint values
    inside_vals = raw_pred[footprint].astype(np.float32)

    # Replace any non-finite with fill_value
    invalid_mask = ~np.isfinite(inside_vals)
    if np.any(invalid_mask):
        inside_vals[invalid_mask] = fill_value

    # Clip strictly to [0.0, 1.0]
    inside_vals = np.clip(inside_vals, 0.0, 1.0)

    out[footprint] = inside_vals
    return out


def write_submission_geotiff(
    pred: np.ndarray,
    output_path: Path,
    compress: str = "deflate",
) -> dict:
    """Writes a conformant submission GeoTIFF and validates every requirement."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conformed = conform_to_template(pred)

    template_path = get_template_path()
    with rasterio.open(template_path) as tmpl:
        meta = tmpl.meta.copy()

    meta.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress=compress,
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(conformed, 1)

    # Independent post-write verification gate
    with rasterio.open(output_path) as src:
        arr = src.read(1)
        w, h = src.width, src.height
        crs = src.crs.to_epsg() if src.crs else None
        res = (float(src.res[0]), float(src.res[1]))
        nodata = src.nodata

    assert w == EXPECTED_WIDTH, f"Width mismatch: {w}"
    assert h == EXPECTED_HEIGHT, f"Height mismatch: {h}"
    assert crs == EXPECTED_EPSG, f"CRS mismatch: {crs}"
    assert res == (EXPECTED_RES, EXPECTED_RES), f"Resolution mismatch: {res}"
    assert np.isnan(nodata), "Nodata is not NaN"

    valid_mask = ~np.isnan(arr)
    valid_count = int(valid_mask.sum())
    nan_count = int((~valid_mask).sum())
    assert valid_count == 5167373, f"Valid pixel count mismatch: {valid_count}"
    assert nan_count == 7111787, f"NaN pixel count mismatch: {nan_count}"

    vals = arr[valid_mask]
    assert np.all(np.isfinite(vals)), "Found non-finite values inside valid footprint"
    min_v, max_v = float(vals.min()), float(vals.max())
    assert min_v >= 0.0 and max_v <= 1.0, f"Values out of range [0, 1]: min={min_v}, max={max_v}"

    # Calculate SHA-256
    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()

    info = {
        "path": str(output_path),
        "filename": output_path.name,
        "sha256": sha256,
        "bytes": output_path.stat().st_size,
        "width": w,
        "height": h,
        "crs": f"EPSG:{crs}",
        "valid_pixels": valid_count,
        "nan_pixels": nan_count,
        "min_val": min_v,
        "max_val": max_v,
        "verified_conformant": True,
    }
    return info
