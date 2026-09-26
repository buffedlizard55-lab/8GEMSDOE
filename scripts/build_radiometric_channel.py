#!/usr/bin/env python3
"""GeoDAWN airborne radiometrics (K, eTh, eU and ratios) resampled onto the competition grid.

HYPOTHESIS H2 (docs/STRATEGY.md §4), finally executable
------------------------------------------------------
GeoDAWN's own release (Glen & Earney 2024, doi:10.5066/P93LGLVQ, ScienceBase item
657e1d85d34e23d3533209f7) ships GeoTIFF grids of the radiometric survey in two zips:

    22103_area1_tiffs.zip   45,685,879 B   MD5 56d8450884ec737a10ad6031056c22e2
    22103_area2_tiffs.zip  241,738,690 B   MD5 2b927f17b8261380b72c2be9e1d46490

(sizes and MD5s read from the item's JSON on 2026-09-25).  Area 1 (Clayton Valley) was flown at
200 m line spacing, Area 2 (the rest, the geothermal focus) at 400 m.  Potassium, equivalent
thorium and equivalent uranium map near-surface lithology and hydrothermal alteration; a fault
that juxtaposes units, or a fault zone that channelled fluids, is a radiometric EDGE.  The 19
official bands contain magnetics, gravity, strain, seismicity, conductivity and 100 m topography,
and - as this script also tests - at most ONE radiometric band (band 6, `tc`).

A SECOND MEASUREMENT IN THE SAME RUN: WHAT IS OFFICIAL BAND 6?
-------------------------------------------------------------
Band 6 of training_features.tif is named `tc` but tagged `data_category=magnetic_data`,
"Tilt angle or total curvature - magnetic field derivative".  In GeoDAWN's own file naming
`22103_tc_a*.tif` is the radiometric TOTAL COUNT, the problem page's figure shows "total
radiometric counts", and on the bytes band 6 spans 3.1 .. 62.2 with median 18.5 - strictly
positive, which a tilt angle (symmetric about 0, bounded by +-90 deg) cannot be.  This script
correlates band 6 with the resampled GeoDAWN total-count grid and records the answer, so the
irregularity is resolved by measurement rather than argument.

OUTPUT
    data/external/radiometric_100m.tif        float32, N bands, EXACT competition grid, NaN outside
    data/evidence/radiometric/radiometric_channel.report.json   zip listing, per-band coverage,
                                              band-6 identity test, provenance

USAGE (runner; the sandbox cannot reach ScienceBase)
    python scripts/build_radiometric_channel.py --zip-dir /tmp/geodawn
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]

ZIPS = {
    "22103_area1_tiffs.zip": {
        "url": "https://www.sciencebase.gov/catalog/file/get/657e1d85d34e23d3533209f7?f=__disk__51%2Fa9%2F11%2F51a9112b3035464a94e3b1d1e6e6efd1e2ff80ae",
        "bytes": 45685879, "md5": "56d8450884ec737a10ad6031056c22e2"},
    "22103_area2_tiffs.zip": {
        "url": "https://www.sciencebase.gov/catalog/file/get/657e1d85d34e23d3533209f7?f=__disk__35%2F69%2F6a%2F35696a36c88459bb8d2bd1a628c3306bc9749c2f",
        "bytes": 241738690, "md5": "2b927f17b8261380b72c2be9e1d46490"},
}
SOURCE_ITEM = "https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7"
SOURCE_DOI = "https://doi.org/10.5066/P93LGLVQ"

# GeoDAWN grid codes -> output band names.  Codes come from the Area-1 file list this repository
# already recorded (data/reconstructed/provenance.json: 22103_{k,th,u,tc,thk,uk,uth}_a1.tif).
RAD_CODES = [("k", "rad_k"), ("th", "rad_th"), ("u", "rad_u"), ("tc", "rad_tc"),
             ("thk", "rad_thk"), ("uk", "rad_uk"), ("uth", "rad_uth")]
BAD_ABS = 1e20  # Geosoft / GDAL dummies (-1e32, -3.4e38 ...) are not data


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_grid(root: Path, code: str, area: int) -> Path | None:
    """First *.tif whose stem ends in _<code>_a<area> (case-insensitive), e.g. 22103_k_a2.tif."""
    pat = re.compile(rf"(^|_){re.escape(code)}_a{area}$", re.IGNORECASE)
    hits = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".tif", ".tiff") and pat.search(p.stem))
    return hits[0] if hits else None


def to_grid(src_path: Path, tprof: dict) -> np.ndarray:
    """Reproject one source grid onto the template grid.  'average' when the source is finer than
    100 m (Area 1), bilinear otherwise; non-finite / dummy values become NaN first."""
    with rasterio.open(src_path) as s:
        a = s.read(1).astype(np.float32)
        bad = ~np.isfinite(a) | (np.abs(a) > BAD_ABS)
        if s.nodata is not None and np.isfinite(s.nodata):
            bad |= a == np.float32(s.nodata)
        a[bad] = np.nan
        finer = abs(s.transform.a) < 0.75 * abs(tprof["transform"].a)
        dst = np.full((tprof["height"], tprof["width"]), np.nan, dtype=np.float32)
        reproject(a, dst, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                  dst_transform=tprof["transform"], dst_crs=tprof["crs"], dst_nodata=np.nan,
                  resampling=Resampling.average if finer else Resampling.bilinear)
    return dst


def band6_identity(features: Path, tc_grid: np.ndarray, foot: np.ndarray, seed: int = 0) -> dict:
    """Pearson + Spearman between official band 6 and the resampled GeoDAWN total-count grid."""
    from scipy.stats import pearsonr, spearmanr
    with rasterio.open(features) as f:
        b6 = f.read(6).astype(np.float64)
        tags = f.tags(6)
    m = foot & np.isfinite(b6) & (b6 > -1e30) & np.isfinite(tc_grid)
    idx = np.flatnonzero(m.ravel())
    if idx.size < 1000:
        return {"n": int(idx.size), "verdict": "insufficient overlap"}
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(200_000, idx.size), replace=False)
    x = b6.ravel()[pick]
    y = tc_grid.astype(np.float64).ravel()[pick]
    pr = float(pearsonr(x, y)[0])
    sr = float(spearmanr(x, y)[0])
    return {
        "official_band6_tags": tags,
        "n_pixels_sampled": int(pick.size), "n_pixels_overlap": int(idx.size),
        "pearson": round(pr, 4), "spearman": round(sr, 4),
        "band6_range": [float(np.min(x)), float(np.median(x)), float(np.max(x))],
        "geodawn_tc_range": [float(np.min(y)), float(np.median(y)), float(np.max(y))],
        "verdict": ("band 6 IS the GeoDAWN radiometric total count (tag description is wrong)"
                    if sr >= 0.9 else
                    "band 6 is NOT the GeoDAWN total count" if sr < 0.5 else
                    "band 6 is related to, but not identical with, the GeoDAWN total count"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip-dir", required=True, help="directory holding the two GeoDAWN tiff zips")
    ap.add_argument("--template", default=str(ROOT / "data/sample_submission.tif"))
    ap.add_argument("--features", default=str(ROOT / "data/training_features.tif"))
    ap.add_argument("--out", default=str(ROOT / "data/external/radiometric_100m.tif"))
    ap.add_argument("--report", default=str(ROOT / "data/evidence/radiometric/radiometric_channel.report.json"))
    a = ap.parse_args()

    zdir = Path(a.zip_dir)
    report: dict = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/build_radiometric_channel.py",
        "source": {"item": SOURCE_ITEM, "doi": SOURCE_DOI,
                   "citation": "Glen, J.M.G., and Earney, T.E., 2024, GeoDAWN: Airborne magnetic and radiometric "
                               "surveys of the northwestern Great Basin, Nevada and California: U.S. Geological "
                               "Survey data release, https://doi.org/10.5066/P93LGLVQ.",
                   "license": "USGS data release (public domain, U.S. Government work)"},
        "zips": {},
    }
    ext = zdir / "extracted"
    ext.mkdir(parents=True, exist_ok=True)
    for zname, meta in ZIPS.items():
        zp = zdir / zname
        if not zp.exists():
            raise SystemExit(f"FAIL: {zp} missing (download {meta['url']})")
        got_md5, got_bytes = md5_file(zp), zp.stat().st_size
        ok = got_md5 == meta["md5"] and got_bytes == meta["bytes"]
        with zipfile.ZipFile(zp) as z:
            listing = [(i.filename, i.file_size) for i in z.infolist()]
            z.extractall(ext / zp.stem)
        report["zips"][zname] = {"url": meta["url"], "bytes": got_bytes, "md5": got_md5,
                                 "md5_expected": meta["md5"], "verified": ok, "listing": listing}
        if not ok:
            raise SystemExit(f"FAIL: {zname} md5/size mismatch ({got_md5}, {got_bytes})")

    with rasterio.open(a.template) as t:
        tprof = t.profile.copy()
        foot = np.isfinite(t.read(1))
    bands, names, per_band = [], [], []
    for code, out_name in RAD_CODES:
        g2 = find_grid(ext / "22103_area2_tiffs", code, 2)
        g1 = find_grid(ext / "22103_area1_tiffs", code, 1)
        if g2 is None and g1 is None:
            per_band.append({"code": code, "status": "absent from both zips"})
            continue
        base = to_grid(g2, tprof) if g2 is not None else np.full(foot.shape, np.nan, np.float32)
        n2 = int(np.isfinite(base[foot]).sum())
        filled = 0
        if g1 is not None:
            a1 = to_grid(g1, tprof)
            fill = ~np.isfinite(base) & np.isfinite(a1)
            base[fill] = a1[fill]
            filled = int((fill & foot).sum())
        base[~foot] = np.nan
        cov = float(np.isfinite(base[foot]).mean())
        bands.append(base)
        names.append(out_name)
        per_band.append({"code": code, "band_name": out_name,
                         "area2_file": str(g2.relative_to(ext)) if g2 else None,
                         "area1_file": str(g1.relative_to(ext)) if g1 else None,
                         "area2_px_in_footprint": n2, "area1_fill_px": filled,
                         "coverage_of_footprint": round(cov, 4),
                         "p1_p50_p99": [float(v) for v in np.nanpercentile(base[foot], [1, 50, 99])]
                         if cov > 0 else None})
        print(f"  {out_name:8s} coverage {cov:.3f}  (area2 {n2} px, area1 fill {filled} px)")
    report["bands"] = per_band
    if not bands:
        report["status"] = "FAIL: no radiometric grid matched the expected names; see zip listings"
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=1) + "\n")
        print(report["status"])
        return 2

    prof = tprof.copy()
    prof.update(driver="GTiff", dtype="float32", count=len(bands), nodata=np.nan, compress="deflate",
                predictor=3, tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(a.out, "w", **prof) as dst:
        for i, (arr, nm) in enumerate(zip(bands, names), start=1):
            dst.write(arr, i)
            dst.update_tags(i, band_name=nm, source=SOURCE_DOI)
            dst.set_band_description(i, nm)
    report["out"] = a.out
    report["band_names"] = names
    if "rad_tc" in names:
        report["band6_identity_test"] = band6_identity(Path(a.features), bands[names.index("rad_tc")], foot)
        print("  band-6 identity:", report["band6_identity_test"].get("verdict"),
              report["band6_identity_test"].get("spearman"))
    report["status"] = "OK"
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1) + "\n")
    print(f"OK   wrote {a.out} ({len(bands)} bands) and {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
