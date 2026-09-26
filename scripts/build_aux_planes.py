"""Append the two EXTERNAL feature channels to the derived plane stack.

Input   data/derived/planes.f32 + planes.json   (46 planes, in-file features)
        data/external/topo_features_100m.tif    (9 bands, 3DEP 1/3" DEM derived)
        data/external/radiometric_100m.tif      (7 bands, USGS GeoDAWN release)
Output  data/derived/planes_ext.f32 + planes_ext.json   (62 planes)

Plane naming: in-file planes keep their names; the external ones are prefixed
`T_` (topo / scarp channel) and `R_` (radiometric channel) so that
`scripts/lofso_train_eval.py --sets BASE+T+R` can select them.

WHY THIS EXISTS.  The competition's 19 bands are 100 m geophysics. Fault
scarps are a *topographic* expression: a 10 m step over 100s of metres is
invisible after the data are gridded to 100 m and detrended, but it survives in
slope/curvature/aspect statistics computed on the ~10 m 3DEP DEM and then
aggregated back to the competition grid. The radiometric channels (K, Th, U and
their ratios) map surficial lithology and alteration, which is independent
information about where a fault cuts the surface. Both are free official USGS
public-domain products; provenance URLs and bridge sha256 are recorded in the
output JSON so nothing about the lineage is implicit.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
EXTERNAL = REPO / "data" / "external"
ROW_BLOCK = 256

AUX = [
    {
        "prefix": "T",
        "path": EXTERNAL / "topo_features_100m.tif",
        "bridge": "data/aux_bridge/topo (sha256 a6398d9950965dec6aae6ccecdaa6ced48645d133eab222cbdd11def9bdabfa4)",
        "source": "USGS 3DEP 1/3 arc-second (~10 m) seamless DEM, "
                  "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt",
    },
    {
        "prefix": "R",
        "path": EXTERNAL / "radiometric_100m.tif",
        "bridge": "data/aux_bridge/radiometric (sha256 6cb051f70f94...)",
        "source": "USGS GeoDAWN airborne magnetic and radiometric surveys, "
                  "https://doi.org/10.5066/P93LGLVQ (ScienceBase item 657e1d85d34e23d3533209f7)",
    },
]


def sha8(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    meta = json.loads((DERIVED / "planes.json").read_text())
    P0, H, W = meta["shape"]
    names = list(meta["planes"])
    src = np.memmap(DERIVED / "planes.f32", dtype=np.float32, mode="r", shape=(P0, H, W))

    with rasterio.open(REPO / "data/labels.tif") as lr:
        lab = lr.read(1)
    # IRREGULARITY FIXED 2026-09-26: labels.tif is finite EVERYWHERE and carries
    # the sentinel -1 outside the scored footprint, so `isfinite` alone selects all
    # 12,279,160 px and a coverage census against it reports 42.08% (= 5,167,373 /
    # 12,279,160) for a channel that in fact covers 100% of the footprint. The
    # footprint is `isfinite & >= 0`, exactly as scripts/prepare_data.py defines it.
    foot = np.isfinite(lab) & (lab >= 0)
    assert int(foot.sum()) == 5_167_373, f"footprint {int(foot.sum())} != 5,167,373"
    del lab

    aux_names: list[str] = []
    aux_specs: list[dict] = []
    for spec in AUX:
        with rasterio.open(spec["path"]) as ds:
            if (ds.height, ds.width) != (H, W):
                raise SystemExit(f"FAIL: {spec['path']} grid {ds.height}x{ds.width} != {H}x{W}")
            if str(ds.crs) != "EPSG:32611":
                raise SystemExit(f"FAIL: {spec['path']} crs {ds.crs} != EPSG:32611")
            tr = [round(float(v), 6) for v in tuple(ds.transform)[:6]]
            if tr != [100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0]:
                raise SystemExit(f"FAIL: {spec['path']} transform {tr}")
            for b in range(1, ds.count + 1):
                nm = ds.descriptions[b - 1] or ds.tags(b).get("band_name") or f"band{b}"
                aux_names.append(f"{spec['prefix']}_{nm}")
                aux_specs.append({"prefix": spec["prefix"], "band": b, "name": nm,
                                  "path": str(spec["path"].relative_to(REPO)),
                                  "source": spec["source"], "bridge": spec["bridge"],
                                  "file_sha256": sha8(spec["path"])})
    P = P0 + len(aux_names)
    print(f"in-file planes={P0} aux planes={len(aux_names)} total={P}")

    out_path = DERIVED / "planes_ext.f32"
    if out_path.exists():
        out_path.unlink()
    dst = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(P, H, W))

    # copy the existing stack in row blocks (keeps RSS flat on a 3 GB box)
    for p in range(P0):
        for y0 in range(0, H, 4096):
            y1 = min(H, y0 + 4096)
            dst[p, y0:y1, :] = src[p, y0:y1, :]
        if p % 10 == 0:
            print(f"  copied plane {p+1}/{P0}", flush=True)
    dst.flush()

    stats = {}
    for j, sp in enumerate(aux_specs):
        pi = P0 + j
        with rasterio.open(sp["path"]) as ds:
            acc = []
            nan_in = 0
            for y0 in range(0, H, ROW_BLOCK):
                y1 = min(H, y0 + ROW_BLOCK)
                blk = ds.read(sp["band"], window=((y0, y1), (0, W))).astype(np.float32)
                dst[pi, y0:y1, :] = blk
                m = foot[y0:y1] & np.isfinite(blk)
                nan_in += int((foot[y0:y1] & ~np.isfinite(blk)).sum())
                if m.any():
                    acc.append(blk[m][::7])
            v = np.concatenate(acc) if acc else np.zeros(1, np.float32)
        stats[aux_names[j]] = {
            "nan_inside_footprint": nan_in,
            "coverage_pct": round(100.0 * (1 - nan_in / int(foot.sum())), 3),
            "p0.5": float(np.percentile(v, 0.5)), "p50": float(np.percentile(v, 50)),
            "p99.5": float(np.percentile(v, 99.5)),
        }
        print(f"  wrote {aux_names[j]:28s} coverage={stats[aux_names[j]]['coverage_pct']}%",
              flush=True)
    dst.flush()
    del dst

    out_meta = dict(meta)
    out_meta.update({
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/build_aux_planes.py",
        "shape": [P, H, W],
        "n_planes": P,
        "planes": names + aux_names,
        "n_in_file_planes": P0,
        "aux_planes": aux_names,
        "aux_sources": [{k: v for k, v in s.items() if k != "band"} for s in aux_specs],
        "aux_stats": stats,
        "source_f32_sha256": sha8(DERIVED / "planes.f32"),
        "out_f32_bytes": out_path.stat().st_size,
    })
    (DERIVED / "planes_ext.json").write_text(json.dumps(out_meta, indent=1) + "\n")
    print(f"OK wrote {out_path} ({out_path.stat().st_size/1e9:.2f} GB) + planes_ext.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
