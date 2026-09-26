"""Submission GeoTIFF builder + 13-gate validator (no GDAL required).

Submission format (official, problem description "Submission format"):
  * same CRS as training data: EPSG:32611 (UTM 11N)
  * same resolution: 100 m
  * same bounds; data outside the bounds is null/NaN
  * single layer, float32, values in [0, 1]

Grid constants (shape / transform / footprint) are NEVER hard-coded as truth:
they are read from the official template file (`sample_submission.tif`) at
runtime. Fallback constants below are labelled UNVERIFIED (quoted from sibling
measurements) and trigger a loud warning whenever they are used.

The 13 gates mirror the sibling 6GEMSDOE validator (whose gate list is the
best-tested public statement of the platform's checks) plus the duplicate
guard:
  1 file-exists   2 single-band   3 dtype-float32   4 crs-epsg32611
  5 resolution-100m   6 shape   7 geotransform   8 nodata-nan
  9 values-in-0-1   10 no-inf   11 template-verified
  12 NAN-INSIDE-FOOTPRINT (the exact condition behind the platform's
     "Predicted values must be in range [0, 1]" rejection)
  13 footprint-matches-official
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

# --- UNVERIFIED fallback grid (sibling-site measurements, NOT official) ------
# Used ONLY when no template file is available (synthetic self-tests).
# Every use is logged and the output is stamped UNVERIFIED in its description.
FALLBACK_SHAPE_HW = (3730, 3292)            # UNVERIFIED
FALLBACK_TRANSFORM = (100.0, 0.0, 243350.0,  # UNVERIFIED (a, b, x0, d, e, y0)
                      0.0, -100.0, 4508550.0)
FALLBACK_CRS_EPSG = 32611                   # VERIFIED (official format text)

# GeoTIFF tag ids
TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922
TAG_GEO_KEY_DIRECTORY = 34735
TAG_GEO_ASCII_PARAMS = 34737
TAG_GDAL_NODATA = 42113


def _utc_now_iso() -> str:
    return _dt.datetime.now(_tz := _dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _geokeys_epsg32611() -> tuple[list, str]:
    # GeoKeyDirectory: header + 4 keys (GTModelType, GTRasterType, ProjectedCSType, ProjLinearUnits?)
    # Keep minimal but valid: model=Projected(1), raster=PixelIsArea(1), PCS=32611.
    header = [1, 1, 0, 3]
    keys = [
        1024, 0, 1, 1,      # GTModelTypeGeoKey = Projected
        1025, 0, 1, 1,      # GTRasterTypeGeoKey = PixelIsArea
        3072, 0, 1, 32611,  # ProjectedCSTypeGeoKey = EPSG:32611
    ]
    return header + keys, "WGS 84 / UTM zone 11N|WGS 84|"


def write_geotiff(path: str | os.PathLike, array: np.ndarray,
                  transform: tuple, epsg: int = 32611,
                  nodata: float = float("nan"),
                  description: str = "") -> None:
    """Write a single-band float32 GeoTIFF with tifffile (uncompressed)."""
    if tifffile is None:
        raise RuntimeError("tifffile is required (pip install tifffile)")
    a, b, x0, d, e, y0 = (float(v) for v in transform)
    if b != 0.0 or d != 0.0:
        raise ValueError("only north-up grids supported")
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("array must be 2D (single band)")
    scale = (abs(a), abs(e), 0.0)
    tiepoint = (0.0, 0.0, 0.0, x0, y0, 0.0)
    keys, ascii_params = _geokeys_epsg32611()
    if epsg != 32611:  # pragma: no cover — builder always uses 32611
        raise ValueError("only EPSG:32611 supported")
    extratags = [
        (TAG_MODEL_PIXEL_SCALE, "d", 3, scale, False),
        (TAG_MODEL_TIEPOINT, "d", 6, tiepoint, False),
        (TAG_GEO_KEY_DIRECTORY, "H", len(keys), tuple(keys), False),
        (TAG_GEO_ASCII_PARAMS, "s", 0, ascii_params, False),
        (TAG_GDAL_NODATA, "s", 0, "nan", False),
    ]
    tifffile.imwrite(path, arr, description=description or "8GEMSDOE submission",
                     extratags=extratags, metadata=None)


def read_geotiff(path: str | os.PathLike) -> dict:
    """Read array + geo tags. Returns dict(array, shape, dtype, transform, epsg, nodata)."""
    if tifffile is None:
        raise RuntimeError("tifffile is required (pip install tifffile)")
    with tifffile.TiffFile(path) as tf:
        if len(tf.pages) != 1:
            raise ValueError(f"expected 1 page, found {len(tf.pages)}")
        page = tf.pages[0]
        arr = page.asarray()
        tags = {t.code: t.value for t in page.tags.values()}
    H, W = arr.shape[-2], arr.shape[-1]
    # transform from scale/tiepoint
    scale = tags.get(TAG_MODEL_PIXEL_SCALE, (100.0, 100.0, 0.0))
    tie = tags.get(TAG_MODEL_TIEPOINT, (0, 0, 0, 0, 0, 0))
    transform = (float(scale[0]), 0.0, float(tie[3]),
                 0.0, -float(scale[1]), float(tie[4]))
    # epsg from GeoKeyDirectory
    epsg = None
    gkd = tags.get(TAG_GEO_KEY_DIRECTORY)
    if gkd is not None:
        vals = list(gkd)
        n = vals[3]
        for i in range(n):
            key, loc, count, val = vals[4 + 4 * i:8 + 4 * i]
            if key == 3072:  # ProjectedCSTypeGeoKey
                epsg = int(val)
    nodata = tags.get(TAG_GDAL_NODATA, None)
    if isinstance(nodata, bytes):
        nodata = nodata.decode("ascii", "ignore")
    return {"array": np.asarray(arr), "shape": (H, W),
            "dtype": str(np.asarray(arr).dtype), "transform": transform,
            "epsg": epsg, "nodata": nodata, "nbands": 1}


def footprint_of_template(template_array: np.ndarray) -> np.ndarray:
    """Scored footprint = finite pixels of the official template.

    The sample submission predicts total fault absence: 0.0 inside, NaN
    outside — so finite() is exactly the scored region.
    """
    return np.isfinite(np.asarray(template_array, dtype=np.float64))


def conform_to_template(pred: np.ndarray, template_array: np.ndarray,
                        clip: bool = True) -> tuple[np.ndarray, dict]:
    """Force finite-inside / NaN-outside / [0,1].

    Returns (conformed float32-ready array, report dict with counts).
    This is the structural fix for the platform's
    "Predicted values must be in range [0, 1]" rejection.
    """
    fp = footprint_of_template(template_array)
    p = np.asarray(pred, dtype=np.float64)
    if p.shape != fp.shape:
        raise ValueError(f"pred shape {p.shape} != template shape {fp.shape}")
    report = {}
    if clip:
        lo = int(np.sum(np.isfinite(p) & (p < 0.0)))
        hi = int(np.sum(np.isfinite(p) & (p > 1.0)))
        p = np.clip(p, 0.0, 1.0)
        report["clipped_below_0"] = lo
        report["clipped_above_1"] = hi
    nan_inside_before = int(np.sum(~np.isfinite(p) & fp))
    p = np.where(fp, np.where(np.isfinite(p), p, 0.0), np.nan)
    report["nan_inside_filled_with_0"] = nan_inside_before
    report["finite_inside"] = int(np.sum(np.isfinite(p) & fp))
    report["nan_outside"] = int(np.sum(~np.isfinite(p) & ~fp))
    return p.astype(np.float32), report


GATES = ["file-exists", "single-band", "dtype-float32", "crs-epsg32611",
         "resolution-100m", "shape", "geotransform", "nodata-nan",
         "values-in-0-1", "no-inf", "template-verified",
         "NAN-INSIDE-FOOTPRINT", "footprint-matches-official"]


def validate_submission(path: str | os.PathLike,
                        template: dict | None,
                        template_sha256: str | None = None) -> dict:
    """Run the 13 gates. Returns {pass: bool, gates: [{id, ok, detail}], stats}."""
    gates = []

    def gate(gid, ok, detail=""):
        gates.append({"id": gid, "ok": bool(ok), "detail": str(detail)})

    p = Path(path)
    if not p.exists():
        gate("file-exists", False, "missing file")
        return {"pass": False, "gates": gates, "stats": {}}
    gate("file-exists", True, f"{p.stat().st_size} bytes")

    try:
        got = read_geotiff(p)
    except Exception as e:  # noqa: BLE001
        gate("single-band", False, f"unreadable: {e}")
        return {"pass": False, "gates": gates, "stats": {}}
    gate("single-band", True, "band count = 1")
    gate("dtype-float32", got["dtype"] == "float32", f"dtype = {got['dtype']}")
    gate("crs-epsg32611", got["epsg"] == 32611, f"epsg = {got['epsg']}")
    a, b, x0, d, e, y0 = got["transform"]
    gate("resolution-100m", abs(abs(a) - 100.0) < 1e-6 and abs(abs(e) - 100.0) < 1e-6,
         f"resolution = ({abs(a)}, {abs(e)})")

    arr = np.asarray(got["array"], dtype=np.float64)
    H, W = arr.shape
    stats = {"shape": [H, W], "finite": int(np.isfinite(arr).sum()),
             "nan": int(np.isnan(arr).sum()),
             "inf": int(np.isinf(arr).sum()),
             "min": float(np.nanmin(arr)) if np.isfinite(arr).any() else None,
             "max": float(np.nanmax(arr)) if np.isfinite(arr).any() else None,
             "positive": int(np.sum(arr == 1.0))}

    if template is None:
        for gid in ("shape", "geotransform", "nodata-nan", "values-in-0-1",
                    "no-inf", "template-verified", "NAN-INSIDE-FOOTPRINT",
                    "footprint-matches-official"):
            gate(gid, False, "no template provided — grid UNVERIFIED")
        return {"pass": False, "gates": gates, "stats": stats}

    tarr = np.asarray(template["array"], dtype=np.float64)
    gate("shape", (H, W) == tarr.shape, f"shape = {(H, W)}, expected {tarr.shape}")
    teps = template.get("transform")
    same_tf = teps is not None and all(abs(x - y) < 1e-6 for x, y in zip(got["transform"], teps))
    gate("geotransform", same_tf, f"transform = {got['transform']}")
    nd = got.get("nodata")
    gate("nodata-nan", isinstance(nd, str) and nd.strip().lower() == "nan",
         f"declared nodata = {nd}")
    finite = arr[np.isfinite(arr)]
    in_range = finite.size == 0 or bool(np.all((finite >= 0.0) & (finite <= 1.0)))
    gate("values-in-0-1", in_range,
         f"finite range = [{finite.min()}, {finite.max()}]" if finite.size
         else "no finite pixels")
    gate("no-inf", stats["inf"] == 0, f"inf pixels = {stats['inf']}")
    if template_sha256:
        h = hashlib.sha256(Path(template["_path"]).read_bytes()).hexdigest() \
            if template.get("_path") else None
        gate("template-verified", h == template_sha256,
             f"template sha256 {'ok' if h == template_sha256 else 'MISMATCH'}")
    else:
        gate("template-verified", True, "template loaded (hash not pinned)")
    fp = footprint_of_template(tarr)
    nan_inside = int(np.sum(~np.isfinite(arr) & fp))
    gate("NAN-INSIDE-FOOTPRINT", nan_inside == 0,
         f"{nan_inside} non-finite pixels inside the scored footprint"
         " (this is the exact condition that makes the submission form answer"
         " 'Predicted values must be in range [0, 1]')")
    finite_outside = int(np.sum(np.isfinite(arr) & ~fp))
    gate("footprint-matches-official", finite_outside == 0,
         f"{finite_outside} finite pixels outside the official footprint")
    ok = all(g["ok"] for g in gates)
    return {"pass": ok, "gates": gates, "stats": stats}


def unique_submission_name(strategy: str, payload_sha8: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in strategy)
    return f"gems8-{safe}-{_utc_now_iso()}-{payload_sha8}.tif"


def suggest_note(strategy: str, payload_sha8: str, extra: str = "") -> str:
    base = f"8GEMSDOE {strategy} | {payload_sha8}"
    return f"{base} | {extra}" if extra else base


def record_submission(log_path: str | os.PathLike, entry: dict,
                      allow_duplicate: bool = False) -> dict:
    """Append to the submissions log; refuse duplicate payload hashes."""
    lp = Path(log_path)
    log = {"submissions": []}
    if lp.exists():
        log = json.loads(lp.read_text())
    for prev in log.get("submissions", []):
        if prev.get("payload_sha256") == entry.get("payload_sha256"):
            if not allow_duplicate:
                raise ValueError(
                    "duplicate payload hash already in log "
                    f"({prev.get('filename')}); refusing to emit the same bytes "
                    "twice (GEMSDOE vs 5GEMSDOE lesson). Pass allow_duplicate=True "
                    "to override with a logged warning.")
            entry["duplicate_warning"] = f"same bytes as {prev.get('filename')}"
    log.setdefault("submissions", []).append(entry)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps(log, indent=2))
    return entry
