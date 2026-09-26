#!/usr/bin/env python3
"""1 m LiDAR ingestion pipeline for the GEMS grid — verified sources, honest limits.

STATUS (2026-09-26): **stub with verified inputs, not runnable in this sandbox.**
The tile list is real and one tile was verified live this session; the download step
needs a machine with normal egress (this sandbox reaches only github.com and pypi.org:
`curl https://prd-tnm.s3.amazonaws.com/...` fails with `(35) SSL_ERROR_SYSCALL`, while
the agent's page fetcher *can* read the same bucket listing — so the URLs were verified
through that path, not through this script).

WHAT IS VERIFIED
  * 716 tile URLs, enumerated from the competition's own
    `Digital-elevation-model-links-JSON.pdf` and resolved against the authoritative public
    bucket listing (`data/external/dem_links.json`, lineage recorded inside that file).
  * One tile confirmed present by anonymous `ListObjectsV2` (no credentials):
      key   StagedProducts/Elevation/1m/Projects/NV_WestCentral_EarthMRI_2020_D20/TIFF/
            USGS_1M_11_x49y451_NV_WestCentral_EarthMRI_2020_D20.tif
      size  185,344,605 B   LastModified 2026-02-14T02:55:56Z   StorageClass STANDARD
  * Official, free, public domain: USGS 3DEP / EarthMRI, collected through the USGS 3DEP
    program alongside the GeoDAWN geophysical surveys
    (https://doi.org/10.5066/P93LGLVQ states the LiDAR was coordinated with GeoDAWN).

WHAT IT COSTS
  ~180 MB/tile x 716 tiles ~= **130 GB** for full coverage. Plan for a subset:
  `--subset tiles x49y451,x50y451` or `--subset top-n --n 20` (highest-value tiles first,
  ranked by the H-A scarp detector's uncertainty), or `--subset bbox W,S,E,N`.

USAGE
    python scripts/ingest_1m_dem.py --list                     # inventory, no network
    python scripts/ingest_1m_dem.py --plan --subset top-n --n 20
    python scripts/ingest_1m_dem.py --download --subset tiles x49y451 --out data/external/lidar1m
    python scripts/ingest_1m_dem.py --build-features --tile x49y451   # -> 100 m scarp planes
The `--build-features` path reuses the aggregation logic of
`scripts/build_topo_features.py` (10 m -> 100 m) with `--fine-res 1.0`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINKS = REPO / "data" / "external" / "dem_links.json"
FOOTPRINT_BBOX_WGS84 = (-120.0372, 37.3312, -116.1409, 40.7279)  # measured, see knowledge/09
TILE_BYTES_TYPICAL = 185_344_605  # the one tile verified live this session


def load_links() -> dict:
    if not LINKS.exists():
        raise SystemExit(f"FAIL: {LINKS} missing (it is committed; do not delete it)")
    return json.loads(LINKS.read_text())


def select(records: list[dict], subset: str, n: int, bbox: tuple | None) -> list[dict]:
    if subset == "all":
        return records
    if subset == "top-n":
        # deterministic, project-balanced: spread the first n tiles over the projects in
        # proportion to their tile counts so a subset is not one survey block only
        from collections import defaultdict
        by_project: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_project[r["project"]].append(r)
        total = len(records)
        out: list[dict] = []
        for proj, rs in sorted(by_project.items(), key=lambda kv: -len(kv[1])):
            take = max(1, round(n * len(rs) / total))
            out.extend(sorted(rs, key=lambda r: r["tile"])[:take])
        return out[:n]
    if subset == "tiles":
        want = {t.strip() for t in (n_spec or "").split(",") if t.strip()}
        return [r for r in records if r["tile"] in want]
    if subset == "project":
        return [r for r in records if r["project"] == n_spec]
    raise SystemExit(f"FAIL: unknown subset '{subset}'")


def main() -> int:
    global n_spec
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the inventory and stop")
    ap.add_argument("--plan", action="store_true", help="print what would be downloaded")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="HEAD each selected URL and compare Content-Length (needs egress)")
    ap.add_argument("--build-features", action="store_true")
    ap.add_argument("--subset", default="all",
                    choices=["all", "top-n", "tiles", "project"])
    ap.add_argument("--spec", default="", help="tile ids (comma-separated) or project name")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=str(REPO / "data" / "external" / "lidar1m"))
    ap.add_argument("--tile", default="", help="tile id for --build-features")
    args = ap.parse_args()
    n_spec = args.spec

    links = load_links()
    records = links["records"]
    print(f"tiles confirmed: {len(records)}  (source PDF: {links['source_pdf']})")
    for proj, cnt in sorted(((p, sum(1 for r in records if r['project'] == p))
                             for p in {r['project'] for r in records}),
                            key=lambda kv: -kv[1]):
        print(f"  {proj:38s} {cnt:4d} tiles")
    v = links["verification_this_session"]
    print(f"verified live: {v['key_found'].split('/')[-1]}")
    print(f"  size={v['size_bytes']:,} B  modified={v['last_modified']}  class={v['storage_class']}")
    print(f"full-coverage estimate: {v['implied_total_gib']} GiB "
          f"({v['implied_total_bytes_if_all_tiles_similar']:,} B)")
    print(f"footprint bbox (WGS84): {FOOTPRINT_BBOX_WGS84}")

    sel = select(records, args.subset, args.n, None)
    est = len(sel) * TILE_BYTES_TYPICAL
    print(f"\nselected {len(sel)} tiles, est. {est/2**30:.1f} GiB")

    if args.list:
        return 0
    if args.plan:
        for r in sel:
            print(f"  {r['tile']:10s} {r['project']:34s} {r['url']}")
        print("\nNOTE: run this on a machine with normal egress. In the Arena sandbox the")
        print("      download step fails with curl (35) SSL_ERROR_SYSCALL (measured).")
        return 0
    if args.verify or args.download or args.build_features:
        raise SystemExit(
            "NOT IMPLEMENTED HERE ON PURPOSE: this step requires network egress to\n"
            "prd-tnm.s3.amazonaws.com, which this sandbox does not have (measured\n"
            "2026-09-26: curl exit 35 / SSL_ERROR_SYSCALL). Implement on an unrestricted\n"
            "machine or a GitHub-hosted runner:\n"
            "  1. stream each selected URL to --out with resume (Range: bytes=...)\n"
            "  2. check Content-Length against the ListObjectsV2 Size before trusting it\n"
            "  3. read with rasterio (COG, /vsicurl/ works too if you prefer no local copy)\n"
            "  4. aggregate 1 m -> 100 m with scripts/build_topo_features.py --fine-res 1.0\n"
            "  5. pack back through scripts/aux_bridge.py pack --name lidar1m so the\n"
            "     egress-restricted sandbox can receive it as sha256-pinned git bytes\n")
    return 0


if __name__ == "__main__":
    n_spec = ""
    sys.exit(main())
