#!/usr/bin/env python3
"""Aux-channel git bridge: bring runner-built rasters back into the egress-restricted sandbox.

WHY THIS EXISTS (session 4, 2026-09-25)
---------------------------------------
The development sandbox reaches only api.github.com and pypi.org (measured this session: every
USGS / ScienceBase / S3 / OpenTopography host returns curl code 000).  GitHub-hosted runners reach
all of them, but Actions *artifacts* are served from an Azure blob host the sandbox cannot resolve
(measured session 3), so a 194 MB topographic channel built on a runner in 874 s sat unreachable
for two sessions.  The one channel that works in both directions is git itself - exactly how the
official rasters already travel (`data/bridge/`, `scripts/assemble_data_bridge.py`).

This script is that channel for auxiliary channels.  `pack` (runs on the runner):

  1. reads a float32 multi-band raster on the EXACT competition grid;
  2. quantises every band to uint8 with a per-band robust linear map
       q = 1 + round(254 * clip((v - lo) / (hi - lo), 0, 1)),   lo/hi = p0.5 / p99.5 over finite px
     and 0 = nodata (non-finite, or outside the template footprint);
  3. writes one DEFLATE-compressed, tiled, predictor-2 GeoTIFF with the grid, band names and the
     (lo, hi) of every band in its tags;
  4. splits it into <= 45 MiB parts (GitHub rejects blobs >= 100 MB and warns above 50 MB) and
     writes a manifest with the sha256 of every part and of the whole file.

`unpack` (runs here): verifies every part's size + sha256, reassembles, verifies the whole-file
sha256, checks the grid against the template, and writes a float32 raster (NaN = nodata, values
de-quantised to bin centres) that `scripts/train_context_detector.py --aux` and
`src/dataset.py` accept unchanged.

WHY uint8 LOSES (ALMOST) NOTHING HERE
-------------------------------------
The consumer is a histogram gradient-boosted tree: sklearn's HistGradientBoosting bins every
feature into at most 255 quantile bins before it learns anything, and trees are invariant to any
monotone transform.  A 254-level monotone quantisation therefore changes the model's input only
where two quantile bins would fall inside one uint8 level (tails beyond p0.5/p99.5).  The CNN
path percentile-normalises to [0, 1] anyway.  The quantisation error bound is (hi-lo)/508 per band
and is recorded in the manifest so nothing about it is implicit.

USAGE
    python scripts/aux_bridge.py pack   --src data/external/topo_features_100m.tif --name topo
    python scripts/aux_bridge.py unpack --name topo          # -> data/external/topo_features_100m.tif
    python scripts/aux_bridge.py verify --name topo          # parts + whole-file sha256 only
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = ROOT / "data" / "aux_bridge"
PART_BYTES = 45 * 1024 * 1024
ROW_BLOCK = 256
Q_LEVELS = 254  # 1..255 carry data, 0 = nodata


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _footprint(template: Path) -> np.ndarray:
    with rasterio.open(template) as t:
        a = t.read(1)
    return np.isfinite(a)


def _grid(ds) -> dict:
    return {"height": ds.height, "width": ds.width, "crs": str(ds.crs),
            "transform": [round(float(v), 6) for v in tuple(ds.transform)[:6]]}


def _band_name(ds, b: int) -> str:
    return ds.tags(b).get("band_name") or (ds.descriptions[b - 1] if ds.descriptions[b - 1] else f"band{b}")


def band_ranges(src_path: Path, foot: np.ndarray) -> list[tuple[float, float]]:
    """Robust (p0.5, p99.5) of every band over finite pixels inside the footprint."""
    out = []
    with rasterio.open(src_path) as ds:
        for b in range(1, ds.count + 1):
            acc = []
            for r0 in range(0, ds.height, ROW_BLOCK):
                r1 = min(ds.height, r0 + ROW_BLOCK)
                blk = ds.read(b, window=((r0, r1), (0, ds.width))).astype(np.float64)
                m = foot[r0:r1] & np.isfinite(blk) & (blk > -1e30)
                if m.any():
                    v = blk[m]
                    # a systematic 1-in-7 subsample keeps memory flat on 12 M px bands
                    acc.append(v[::7] if v.size > 70_000 else v)
            if not acc:
                out.append((0.0, 1.0))
                continue
            v = np.concatenate(acc)
            lo, hi = np.percentile(v, [0.5, 99.5])
            if not hi > lo:
                hi = lo + 1.0
            out.append((float(lo), float(hi)))
    return out


def quantise(blk: np.ndarray, lo: float, hi: float, valid: np.ndarray) -> np.ndarray:
    blk = blk.astype(np.float64)
    ok = valid & np.isfinite(blk) & (blk > -1e30)
    x = np.clip((np.where(ok, blk, lo) - lo) / (hi - lo), 0.0, 1.0)
    q = (1 + np.rint(Q_LEVELS * x)).astype(np.uint8)
    q[~ok] = 0
    return q


def dequantise(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = lo + (q.astype(np.float32) - 1.0) / Q_LEVELS * (hi - lo)
    out = out.astype(np.float32)
    out[q == 0] = np.nan
    return out


def pack(src: Path, name: str, template: Path, bridge_root: Path = BRIDGE_ROOT,
         part_bytes: int = PART_BYTES, provenance: dict | None = None) -> dict:
    foot = _footprint(template)
    with rasterio.open(template) as t:
        tgrid = _grid(t)
    with rasterio.open(src) as ds:
        sgrid = _grid(ds)
        if sgrid != tgrid:
            raise SystemExit(f"FAIL: {src} grid {sgrid} != template grid {tgrid}")
        names = [_band_name(ds, b) for b in range(1, ds.count + 1)]
        profile = ds.profile.copy()
        n_bands = ds.count
    ranges = band_ranges(src, foot)

    out_dir = bridge_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{name}_u8.tif.part-*"):
        old.unlink()
    whole = out_dir / f"{name}_u8.tif"
    profile.update(driver="GTiff", dtype="uint8", nodata=0, count=n_bands, compress="deflate",
                   predictor=2, zlevel=9, tiled=True, blockxsize=256, blockysize=256,
                   interleave="band", BIGTIFF="IF_SAFER")
    with rasterio.open(src) as ds, rasterio.open(whole, "w", **profile) as dst:
        for b in range(1, n_bands + 1):
            lo, hi = ranges[b - 1]
            for r0 in range(0, ds.height, ROW_BLOCK):
                r1 = min(ds.height, r0 + ROW_BLOCK)
                win = ((r0, r1), (0, ds.width))
                q = quantise(ds.read(b, window=win), lo, hi, foot[r0:r1])
                dst.write(q, b, window=rasterio.windows.Window(0, r0, ds.width, r1 - r0))
            dst.update_tags(b, band_name=names[b - 1], q_lo=repr(lo), q_hi=repr(hi),
                            q_rule="v = lo + (q-1)/254*(hi-lo); q=0 nodata")
            dst.set_band_description(b, names[b - 1])

    whole_sha = sha256_file(whole)
    whole_bytes = whole.stat().st_size
    parts = []
    with open(whole, "rb") as f:
        i = 0
        while True:
            chunk = f.read(part_bytes)
            if not chunk:
                break
            p = out_dir / f"{name}_u8.tif.part-{i:03d}"
            p.write_bytes(chunk)
            parts.append({"name": p.name, "bytes": len(chunk),
                          "sha256": hashlib.sha256(chunk).hexdigest()})
            i += 1
    whole.unlink()
    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/aux_bridge.py pack",
        "source_raster": str(src),
        "source_sha256": sha256_file(src),
        "grid": tgrid,
        "encoding": {"dtype": "uint8", "nodata": 0, "levels": Q_LEVELS,
                     "rule": "q = 1 + round(254*clip((v-lo)/(hi-lo),0,1)); v' = lo + (q-1)/254*(hi-lo)",
                     "range_rule": "lo, hi = p0.5, p99.5 of finite in-footprint pixels"},
        "bands": [{"index": i + 1, "band_name": names[i], "lo": ranges[i][0], "hi": ranges[i][1],
                   "max_abs_quantisation_error_inside_range": (ranges[i][1] - ranges[i][0]) / (2 * Q_LEVELS)}
                  for i in range(n_bands)],
        "whole": {"name": whole.name, "bytes": whole_bytes, "sha256": whole_sha},
        "parts": parts,
        "provenance": provenance or {},
        "unpack_command": f"python scripts/aux_bridge.py unpack --name {name}",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _load_manifest(name: str, bridge_root: Path) -> tuple[Path, dict]:
    d = bridge_root / name
    mf = d / "manifest.json"
    if not mf.exists():
        raise SystemExit(f"FAIL: no manifest at {mf} (has the runner committed the bridge yet?)")
    return d, json.loads(mf.read_text())


def verify(name: str, bridge_root: Path = BRIDGE_ROOT) -> dict:
    d, m = _load_manifest(name, bridge_root)
    h = hashlib.sha256()
    total = 0
    for p in m["parts"]:
        path = d / p["name"]
        if not path.exists():
            raise SystemExit(f"FAIL: missing part {path}")
        b = path.read_bytes()
        if len(b) != p["bytes"] or hashlib.sha256(b).hexdigest() != p["sha256"]:
            raise SystemExit(f"FAIL: part {path} size/sha256 mismatch")
        h.update(b)
        total += len(b)
    if total != m["whole"]["bytes"] or h.hexdigest() != m["whole"]["sha256"]:
        raise SystemExit("FAIL: reassembled bytes do not match the whole-file sha256")
    print(f"OK   {name}: {len(m['parts'])} parts, {total} B, whole sha256 {m['whole']['sha256'][:12]}")
    return m


def unpack(name: str, out: Path, template: Path, bridge_root: Path = BRIDGE_ROOT) -> Path:
    m = verify(name, bridge_root)
    d = bridge_root / name
    whole = d / m["whole"]["name"]
    with open(whole, "wb") as f:
        for p in m["parts"]:
            f.write((d / p["name"]).read_bytes())
    with rasterio.open(template) as t:
        tgrid = _grid(t)
        tprof = t.profile.copy()
    try:
        with rasterio.open(whole) as q:
            if _grid(q) != tgrid:
                raise SystemExit(f"FAIL: bridge grid {_grid(q)} != template {tgrid}")
            n = q.count
            prof = tprof
            prof.update(driver="GTiff", dtype="float32", count=n, nodata=np.nan, compress="deflate",
                        predictor=3, tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
            out.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out, "w", **prof) as dst:
                for b in range(1, n + 1):
                    info = m["bands"][b - 1]
                    for r0 in range(0, q.height, ROW_BLOCK):
                        r1 = min(q.height, r0 + ROW_BLOCK)
                        blk = q.read(b, window=((r0, r1), (0, q.width)))
                        dst.write(dequantise(blk, info["lo"], info["hi"]), b,
                                  window=rasterio.windows.Window(0, r0, q.width, r1 - r0))
                    dst.update_tags(b, band_name=info["band_name"], source="aux_bridge:" + name)
                    dst.set_band_description(b, info["band_name"])
    finally:
        whole.unlink(missing_ok=True)
    print(f"OK   wrote {out} ({n} bands float32, de-quantised from the {name} bridge)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--src", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--template", default=str(ROOT / "data/sample_submission.tif"))
    p.add_argument("--provenance", default=None, help="JSON file merged into the manifest")
    u = sub.add_parser("unpack")
    u.add_argument("--name", required=True)
    u.add_argument("--out", default=None)
    u.add_argument("--template", default=str(ROOT / "data/sample_submission.tif"))
    v = sub.add_parser("verify")
    v.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.cmd == "pack":
        prov = json.loads(Path(a.provenance).read_text()) if a.provenance else None
        m = pack(Path(a.src), a.name, Path(a.template), provenance=prov)
        print(f"OK   packed {a.name}: {len(m['parts'])} parts, {m['whole']['bytes']} B")
    elif a.cmd == "unpack":
        default_out = {"topo": "data/external/topo_features_100m.tif",
                       "radiometric": "data/external/radiometric_100m.tif"}.get(a.name,
                                                                                f"data/external/{a.name}_100m.tif")
        unpack(a.name, Path(a.out or ROOT / default_out), Path(a.template))
    else:
        verify(a.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
