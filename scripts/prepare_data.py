"""Verify data/ placement and write data/manifest.json (hashes + grid facts).

Reads ONLY the files listed by download_competition_data.sh. Never fabricates
missing files. Exits nonzero with a clear message when required files are
absent, so CI and the holdout script fail loudly instead of hallucinating.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems.submission import read_geotiff  # noqa: E402

REQUIRED = ["training_features.tif", "sample_submission.tif"]
OPTIONAL = ["training_labels.tif", "training_labels.geojson", "training_labels.gpkg",
            "training_labels.zip", "1m_DEM_links.csv", "SHA256SUMS.txt"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    data = Path("data")
    manifest = {"required": {}, "optional": {}, "grid": None, "ok": False}
    missing = [f for f in REQUIRED if not (data / f).exists()]
    for f in REQUIRED + OPTIONAL:
        p = data / f
        if p.exists():
            manifest["required" if f in REQUIRED else "optional"][f] = {
                "bytes": p.stat().st_size, "sha256": sha256(p)}
    if missing:
        print(f"MISSING required files in data/: {missing}")
        print("Run: bash scripts/download_competition_data.sh")
        (data / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return 1
    info = read_geotiff(data / "sample_submission.tif")
    import numpy as np
    fp = np.isfinite(np.asarray(info["array"], dtype=np.float64))
    manifest["grid"] = {"shape_hw": list(info["shape"]), "dtype": info["dtype"],
                        "transform": list(info["transform"]), "epsg": info["epsg"],
                        "nodata": info["nodata"],
                        "footprint_px": int(fp.sum()),
                        "total_px": int(fp.size)}
    manifest["ok"] = True
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
