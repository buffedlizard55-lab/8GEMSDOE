#!/usr/bin/env python3
"""Topographic scarp channel: 10 m 3DEP DEM derivatives → sub-pixel statistics on the 100 m grid.

THE HYPOTHESIS (docs/STRATEGY.md §4, H1)
----------------------------------------
Quaternary faults in the Basin and Range are mapped by experts from scarps and lineaments seen
in high-resolution topography; the organizers hand out 1 m lidar links for exactly that reason.
The supplied 19-band stack carries only a 100 m detrended elevation and its slope.  A 2–10 m
scarp inside a 100 m cell barely moves the 100 m slope, but it dominates the DISTRIBUTION of
10 m slopes inside that cell.  So instead of averaging the DEM down to 100 m and differentiating
(what ``src/external_data.py`` does, and what every 100 m product does), this script
differentiates at 10 m and aggregates *statistics* per 100 m cell:

  b1  slope_p90        90th-percentile 10 m slope (deg) in the cell          – scarp presence
  b2  slope_max        max 10 m slope (deg)                                    – scarp presence
  b3  steep_frac       fraction of 10 m sub-cells with slope > 8 deg           – scarp extent
  b4  slope_std        std of 10 m slope                                        – texture / scarp vs uniform hillside
  b5  relief_local     max − min 10 m elevation in the cell (m)                – local relief
  b6  curv_prof_absmax max |profile curvature| (1/m ×1e3)                      – scarp crest/base breaks
  b7  aspect_coherence resultant length of 10 m aspect unit vectors (0–1),    – one-sided (scarp) vs
                       weighted by slope                                          multi-sided (hill) cells
  b8  hs_lineament     max over 4 azimuths of |mean(hillshade) − hillshade|    – illumination-direction
                       contrast, i.e. how strongly a linear break shows           dependent break
  b9  dem_mean         mean 10 m elevation (m) (for detrending downstream)

Everything is float32 with NaN where the DEM has no data; the output raster is on the EXACT
competition grid (shape, CRS, transform copied from the sample submission), so
``data.aux_feature_paths`` can stack it with zero resampling at train and inference time.

SOURCES (public domain, no auth; verified 2026-09-25)
-----------------------------------------------------
* USGS 3DEP seamless 1/3 arc-second (~10 m) DEM, staged GeoTIFFs + a VRT that GDAL reads over HTTP:
  https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt
  collection record: https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5
  data catalogue:    https://data.usgs.gov/datacatalog/data/USGS:3a81321b-c153-416f-98b7-cc8e5f0e17c3
* The 1 m tiles the organizers list (data/dem_links.json) cover only parts of the region; the 1/3"
  product is seamless across all of it, which is why it is the first channel to build.  1 m can be
  layered on later for the covered projects with the same aggregation code (``--dem`` accepts any
  GDAL-readable source, so a local 1 m mosaic works unchanged with ``--fine-res 1``).

RUNTIME
-------
The sandbox that wrote this cannot reach S3 (egress allow-list), so the whole-region run is a
GitHub Actions job (.github/workflows/build-topo-features.yml).  It processes the grid in chunks
of ``--chunk`` 100 m cells (default 256 → 2560×2560 10 m px ≈ 26 MB float32 per chunk) and skips
chunks that are entirely outside the template's valid footprint.  The local test uses a synthetic
DEM (tests/test_topo_features.py) and a tiny real-grid smoke on the committed fixture.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_VRT = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt"
BANDS = ["slope_p90", "slope_max", "steep_frac", "slope_std", "relief_local",
         "curv_prof_absmax", "aspect_coherence", "hs_lineament", "dem_mean"]
BAND_DESC = {
    "slope_p90": "90th percentile of fine-resolution slope (deg) within the 100 m cell",
    "slope_max": "max fine-resolution slope (deg) within the cell",
    "steep_frac": "fraction of fine sub-cells with slope > steep_deg",
    "slope_std": "std of fine-resolution slope (deg) within the cell",
    "relief_local": "max - min fine elevation within the cell (m)",
    "curv_prof_absmax": "max |profile curvature| within the cell (1/m x 1e3)",
    "aspect_coherence": "slope-weighted resultant length of fine aspect vectors (0..1)",
    "hs_lineament": "max over 4 azimuths of |cell-mean hillshade - hillshade| (0..1)",
    "dem_mean": "mean fine elevation (m)",
}
STEEP_DEG = 8.0
AZIMUTHS = (315.0, 45.0, 0.0, 90.0)


# ------------------------------------------------------------------ fine-grid operators
def _gradients(dem: np.ndarray, res: float):
    # central differences with edge padding; NaN-safe via nan-fill then re-mask
    gy, gx = np.gradient(dem, res, res)
    return gx, gy


def slope_deg(gx, gy):
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def aspect_rad(gx, gy):
    return np.arctan2(-gx, gy)  # downslope direction, geographic-ish convention


def profile_curvature(dem: np.ndarray, res: float):
    """Zevenbergen–Thorne style profile curvature (curvature along the slope direction)."""
    gy, gx = np.gradient(dem, res, res)
    gyy, gyx = np.gradient(gy, res, res)
    gxy, gxx = np.gradient(gx, res, res)
    p, q = gx, gy
    denom = (p * p + q * q)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = -(p * p * gxx + 2 * p * q * gxy + q * q * gyy) / (denom * np.power(1 + denom, 1.5))
    k[denom < 1e-12] = 0.0
    return k * 1e3


def hillshade(gx, gy, azimuth_deg: float, altitude_deg: float = 45.0):
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0.0, 1.0)


def block_reduce(a: np.ndarray, f: int, fn):
    """Reduce (H*f, W*f) → (H, W) with fn over the last two axes of a (H, W, f, f) view."""
    H, W = a.shape[0] // f, a.shape[1] // f
    v = a[:H * f, :W * f].reshape(H, f, W, f).transpose(0, 2, 1, 3).reshape(H, W, f * f)
    return fn(v, axis=-1)


def features_from_fine_dem(dem: np.ndarray, fine_res: float, factor: int, steep_deg: float = STEEP_DEG) -> np.ndarray:
    """(H*f, W*f) fine DEM (NaN = no data) → (len(BANDS), H, W) float32."""
    valid = np.isfinite(dem)
    if not valid.any():
        H, W = dem.shape[0] // factor, dem.shape[1] // factor
        return np.full((len(BANDS), H, W), np.nan, dtype=np.float32)
    fill = float(np.nanmedian(dem))
    d = np.where(valid, dem, fill).astype(np.float64)
    gx, gy = _gradients(d, fine_res)
    slp = slope_deg(gx, gy)
    asp = aspect_rad(gx, gy)
    curv = np.abs(profile_curvature(d, fine_res))
    # mask fine invalid px back to NaN before aggregation
    slp[~valid] = np.nan
    curv[~valid] = np.nan
    dm = np.where(valid, dem, np.nan)

    w = np.where(valid, np.tan(np.radians(np.nan_to_num(slp))), 0.0)
    cx = block_reduce(w * np.cos(asp), factor, np.nansum)
    cy = block_reduce(w * np.sin(asp), factor, np.nansum)
    wsum = block_reduce(w, factor, np.nansum)
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.where(wsum > 0, np.hypot(cx, cy) / wsum, np.nan)

    hs_contrast = None
    for az in AZIMUTHS:
        hs = hillshade(gx, gy, az)
        hs[~valid] = np.nan
        mean_hs = block_reduce(hs, factor, np.nanmean)
        dev = np.abs(hs - np.repeat(np.repeat(mean_hs, factor, 0), factor, 1)[:hs.shape[0], :hs.shape[1]])
        c = block_reduce(dev, factor, np.nanmax)
        hs_contrast = c if hs_contrast is None else np.fmax(hs_contrast, c)

    n_valid = block_reduce(valid.astype(np.float32), factor, np.sum)
    with np.errstate(all="ignore"):
        out = np.stack([
            block_reduce(slp, factor, lambda v, axis: np.nanpercentile(v, 90, axis=axis)),
            block_reduce(slp, factor, np.nanmax),
            block_reduce((slp > steep_deg).astype(np.float32), factor, np.nansum) / np.where(n_valid > 0, n_valid, np.nan),
            block_reduce(slp, factor, np.nanstd),
            block_reduce(dm, factor, np.nanmax) - block_reduce(dm, factor, np.nanmin),
            block_reduce(curv, factor, np.nanmax),
            coherence,
            hs_contrast,
            block_reduce(dm, factor, np.nanmean),
        ]).astype(np.float32)
    # a 100 m cell with < half its fine sub-cells valid is not trustworthy → NaN
    out[:, n_valid < 0.5 * factor * factor] = np.nan
    return out


# ------------------------------------------------------------------ driver
def read_fine_window(src, dst_crs, dst_transform: Affine, shape, resampling=Resampling.bilinear):
    dst = np.full(shape, np.nan, dtype=np.float32)
    reproject(source=rasterio.band(src, 1), destination=dst,
              src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
              dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
              resampling=resampling, num_threads=2)
    dst[~np.isfinite(dst)] = np.nan
    if src.nodata is not None and np.isfinite(src.nodata):
        dst[dst == src.nodata] = np.nan
    dst[dst < -1000] = np.nan            # 3DEP uses -999999 in places; nothing on Earth's land is < -1000 m
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dem", default=DEFAULT_VRT, help="any GDAL-readable DEM (URL/VRT/vsicurl or local file)")
    ap.add_argument("--template", default=str(ROOT / "data/sample_submission.tif"),
                    help="raster that defines the output grid (shape/CRS/transform) and the valid footprint")
    ap.add_argument("--out", default=str(ROOT / "data/external/topo_features_100m.tif"))
    ap.add_argument("--fine-res", type=float, default=10.0, help="fine grid cell size in metres (10 for 1/3 arc-second)")
    ap.add_argument("--chunk", type=int, default=256, help="chunk edge in coarse (100 m) cells")
    ap.add_argument("--steep-deg", type=float, default=STEEP_DEG)
    ap.add_argument("--max-chunks", type=int, default=None, help="stop after N non-empty chunks (smoke runs)")
    ap.add_argument("--window", default=None, metavar="ROW0,COL0,H,W",
                    help="only process this coarse window (smoke runs); output still spans the full grid")
    ap.add_argument("--report", default=None, help="JSON run report path (default: next to --out)")
    a = ap.parse_args(argv)

    with rasterio.open(a.template) as t:
        H, W = t.height, t.width
        crs, tr = t.crs, t.transform
        footprint = np.isfinite(t.read(1))
    coarse = abs(tr.a)
    factor = int(round(coarse / a.fine_res))
    if abs(factor * a.fine_res - coarse) > 1e-6:
        raise SystemExit(f"fine-res {a.fine_res} must divide the coarse cell {coarse}")

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", dtype="float32", count=len(BANDS), height=H, width=W, crs=crs, transform=tr,
                   nodata=np.nan, compress="lzw", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    r0 = c0 = 0
    r1, c1 = H, W
    if a.window:
        r0, c0, h, w = (int(x) for x in a.window.split(","))
        r1, c1 = min(H, r0 + h), min(W, c0 + w)

    t_start = time.time()
    done = skipped = 0
    chunks_log = []
    with rasterio.open(a.dem) as src, rasterio.open(out_path, "w", **profile) as dst:
        for i in range(1, len(BANDS) + 1):
            dst.set_band_description(i, BANDS[i - 1])
            dst.update_tags(i, band_name=BANDS[i - 1], description=BAND_DESC[BANDS[i - 1]],
                            data_category="topographic_fine", source=a.dem, fine_res_m=a.fine_res)
        dst.update_tags(generated_by="scripts/build_topo_features.py", fine_res_m=a.fine_res,
                        steep_deg=a.steep_deg, dem_source=a.dem)
        for row in range(r0, r1, a.chunk):
            for col in range(c0, c1, a.chunk):
                h = min(a.chunk, r1 - row)
                w = min(a.chunk, c1 - col)
                if not footprint[row:row + h, col:col + w].any():
                    skipped += 1
                    continue
                fine_tr = Affine(a.fine_res, 0, tr.c + col * coarse, 0, -a.fine_res, tr.f - row * coarse)
                fine = read_fine_window(src, crs, fine_tr, (h * factor, w * factor))
                feats = features_from_fine_dem(fine, a.fine_res, factor, a.steep_deg)
                feats[:, ~footprint[row:row + h, col:col + w]] = np.nan
                dst.write(feats, window=((row, row + h), (col, col + w)))
                done += 1
                chunks_log.append({"row": row, "col": col, "h": h, "w": w,
                                   "fine_valid_frac": float(np.isfinite(fine).mean()),
                                   "slope_p90_median": float(np.nanmedian(feats[0])) if np.isfinite(feats[0]).any() else None})
                if a.max_chunks and done >= a.max_chunks:
                    break
            if a.max_chunks and done >= a.max_chunks:
                break

    # read-back verification
    with rasterio.open(out_path) as ds:
        assert (ds.height, ds.width) == (H, W) and ds.count == len(BANDS)
        b1 = ds.read(1)
        finite_px = int(np.isfinite(b1).sum())
        stats = {}
        for i, name in enumerate(BANDS, 1):
            b = ds.read(i)
            f = b[np.isfinite(b)]
            stats[name] = ({"finite_px": int(f.size), "min": float(f.min()), "p50": float(np.median(f)),
                            "p99": float(np.percentile(f, 99)), "max": float(f.max())} if f.size else {"finite_px": 0})
    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/build_topo_features.py",
        "dem_source": a.dem, "template": str(a.template), "out": str(out_path),
        "fine_res_m": a.fine_res, "factor": factor, "chunk_coarse_cells": a.chunk, "steep_deg": a.steep_deg,
        "grid": {"height": H, "width": W, "crs": str(crs), "transform": list(tr)[:6]},
        "chunks_done": done, "chunks_skipped_outside_footprint": skipped,
        "footprint_px": int(footprint.sum()), "finite_px_band1": finite_px,
        "coverage_of_footprint": finite_px / float(footprint.sum()) if footprint.any() else None,
        "seconds": round(time.time() - t_start, 1), "band_stats": stats, "bands": BANDS,
        "chunks": chunks_log[:64],
    }
    rp = Path(a.report) if a.report else out_path.with_suffix(".report.json")
    rp.write_text(json.dumps(report, indent=1))
    print(f"wrote {out_path}  bands={len(BANDS)} chunks={done} skipped={skipped} "
          f"coverage={report['coverage_of_footprint']}  in {report['seconds']} s; report {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
