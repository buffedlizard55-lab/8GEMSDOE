"""Measured group audit: hash + catalogue-monitor DTI for every sibling artifact.

Reads the actual committed bytes (via local clones of the sibling repos —
github egress works in this sandbox) and measures, for each artifact:
  * sha256, bytes, value histogram, budget (positive px / probability mass),
  * full-grid catalogue DTI (public labels) — the WRONG population for the
    leaderboard, reported strictly as a monitor + duplicate detector,
  * overlap with the known catalogue (n positive pixels on catalogue faults).

This INDEPENDENTLY reverifies (2026-09-26, direct byte comparison) the
GEMSDOE == 5GEMSDOE duplicate: identical sha256 on both submission.tif
(7f00890a...) and submission_field.bin (b966d47c...).

Usage:
  python scripts/audit_siblings.py /tmp/sib [report.json]
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import rasterio  # noqa: E402

from gems.metric import distance_weighted_tversky  # noqa: E402

ARTIFACTS = [
    ("GEMSDOE-ens12", "GEMSDOE/data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"),
    ("5GEMSDOE-ens12", "5GEMSDOE/data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"),
    ("GEMSDOE2-union", "GEMSDOE2/docs/gemsdoe2-dual-family-union-f68e590f.tif"),
    ("GEMSDOE2-precision", "GEMSDOE2/docs/gemsdoe2-precision-arm-8bce5dfe.tif"),
    ("GEMSDOE2-extension", "GEMSDOE2/docs/gemsdoe2-extension-arm-ad5ba911.tif"),
    ("GEMSDOE2-recall", "GEMSDOE2/docs/gemsdoe2-recall-arm-7f00890a.tif"),
    ("GEMSDOE3-pindrop-nodes", "GEMSDOE3/docs/downloads/pindrop-v4-nodes-20260925T152420Z-f347b70daa.tif"),
    ("GEMSDOE3-pindrop-discovery", "GEMSDOE3/docs/downloads/pindrop-v4-discovery-20260925T152423Z-37f9d5b855.tif"),
    ("GEMSDOE3-pindrop-ridge", "GEMSDOE3/docs/downloads/pindrop-v4-ridge-20260925T152422Z-4e03fc9705.tif"),
    ("6GEMSDOE-hgb88", "6GEMSDOE/downloads/gems6_hgb88-topk03_33cec71ff0.tif"),
    ("GEMSDOE4-combined", "GEMSDOE4/data/evidence/combined/submission.tif"),
]

LOCAL = [("8GEMSDOE-apex", REPO_ROOT / "downloads" / "submission.tif")]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    sib = Path(argv[1] if len(argv) > 1 else "/tmp/sib")
    out = Path(argv[2] if len(argv) > 2 else "reports/sibling_audit.json")

    labels_path = REPO_ROOT / "data" / "labels.tif"
    if not labels_path.exists():
        print("FATAL: data/labels.tif missing")
        return 2
    with rasterio.open(labels_path) as src:
        la = src.read(1)
        nod = src.nodata
    valid = la != nod if nod is not None else ~np.isnan(la.astype(float))
    truth = ((la == 1) & valid).astype(np.float64)
    known = truth > 0
    print(f"catalogue: {int(known.sum())} fault px, "
          f"{int(valid.sum())} valid px")

    rows = []
    payloads: dict[str, str] = {}

    def measure(name: str, p: Path):
        if not p.exists():
            print(f"  SKIP {name}: missing {p}")
            return
        with rasterio.open(p) as src:
            a = src.read(1).astype(np.float64)
            meta = (src.width, src.height, str(src.crs),
                    src.dtypes[0])
        finite = np.isfinite(a)
        pos = (a == 1.0) & finite
        mass = float(np.nan_to_num(a)[valid].sum())
        n_pos = int(pos.sum())
        n_on_known = int((pos & known).sum())
        # payload hash = raw scored-pixel bytes (container-independent)
        scored = np.where(valid, np.nan_to_num(a), np.nan).astype(np.float32)
        payloads[name] = hashlib.sha256(scored.tobytes()).hexdigest()
        r = distance_weighted_tversky(np.nan_to_num(a), truth)
        rm = distance_weighted_tversky(np.nan_to_num(a), truth,
                                       known_mask=known)
        rows.append({
            "artifact": name,
            "path": str(p),
            "file_sha256": sha256_file(p),
            "payload_sha256": payloads[name],
            "bytes": p.stat().st_size,
            "grid": {"width": meta[0], "height": meta[1], "crs": meta[2],
                     "dtype": meta[3]},
            "finite_px": int(finite.sum()),
            "positive_px_eq1": n_pos,
            "mass_in_footprint": mass,
            "positive_on_catalogue": n_on_known,
            "catalogue_monitor_dti": r["DTI"],
            "catalogue_monitor_TP": r["TP_w"],
            "catalogue_monitor_FP": r["FP_w"],
            "catalogue_monitor_FN": r["FN_w"],
            "catalogue_monitor_dti_masked": rm["DTI"],
        })
        print(f"  {name:28} sha={rows[-1]['file_sha256'][:12]}... "
              f"pos={n_pos:>7} mass={mass:>12,.0f} "
              f"on_known={n_on_known:>6} catDTI={r['DTI']:.4f}")

    print("== sibling artifacts ==")
    for name, rel in ARTIFACTS:
        measure(name, sib / rel)
    print("== local ==")
    for name, p in LOCAL:
        measure(name, p)

    # duplicate detection on payload hashes
    print("\n== duplicate scan (payload sha256) ==")
    seen: dict[str, str] = {}
    dups = []
    for name, h in payloads.items():
        if h in seen:
            dups.append([seen[h], name, h[:16]])
            print(f"  DUPLICATE: {seen[h]} == {name} (payload {h[:16]}...)")
        else:
            seen[h] = name
    if not dups:
        print("  no payload duplicates found")

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "direct byte reads of committed sibling artifacts; "
                  "full-grid catalogue DTI is a MONITOR (wrong population), "
                  "not a leaderboard claim",
        "artifacts": rows,
        "payload_duplicates": dups,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
