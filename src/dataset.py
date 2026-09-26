"""Dataset utilities and raster I/O for GEMS."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("GEMS_DATA_DIR", REPO_ROOT / "data"))

EXPECTED_WIDTH = 3292
EXPECTED_HEIGHT = 3730
EXPECTED_EPSG = 32611
EXPECTED_RES = 100.0
EXPECTED_TRANSFORM = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)


def get_template_path() -> Path:
    candidates = [
        DATA_DIR / "sample_submission.tif",
        DATA_DIR / "example_submission.tif",
        REPO_ROOT / "data" / "sample_submission.tif",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find sample_submission.tif template.")


def get_labels_path() -> Path:
    candidates = [
        DATA_DIR / "labels.tif",
        DATA_DIR / "existing_faults.tif",
        REPO_ROOT / "data" / "labels.tif",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find labels.tif.")


def get_features_path() -> Path:
    candidates = [
        DATA_DIR / "training_features.tif",
        DATA_DIR / "gems-geodawn-numerical-features.tif",
        REPO_ROOT / "data" / "training_features.tif",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find training_features.tif.")


def load_footprint_mask() -> np.ndarray:
    """Returns boolean mask (H, W) where True = valid scored GeoDAWN footprint (5,167,373 px)."""
    tmpl = get_template_path()
    with rasterio.open(tmpl) as src:
        arr = src.read(1)
        nodata = src.nodata
    if nodata is not None and np.isnan(nodata):
        return ~np.isnan(arr)
    elif nodata is not None:
        return arr != nodata
    return ~np.isnan(arr)


def load_known_faults() -> np.ndarray:
    """Returns binary mask (H, W) where True = known catalogue fault pixel (60,988 px)."""
    lp = get_labels_path()
    with rasterio.open(lp) as src:
        arr = src.read(1)
        nod = src.nodata
    if nod is not None:
        valid = arr != nod
    else:
        valid = ~np.isnan(arr)
    return (arr == 1) & valid
