"""CLI: 13-gate submission validator (hard gate for the builder).

Usage:
  python scripts/validate_submission.py submissions/*.tif --template data/sample_submission.tif
  python scripts/validate_submission.py --self-test   # synthetic grids, no data needed
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems.submission import (conform_to_template, footprint_of_template,
                             read_geotiff, validate_submission, write_geotiff)


def self_test() -> int:
    """Build synthetic template+submission grids and exercise every gate."""
    rng = np.random.default_rng(8)
    H, W = 60, 50
    tf = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
    template = np.zeros((H, W), dtype=np.float32)
    template[5:10, :] = np.nan  # outside footprint: top stripe
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td) / "template.tif"
        write_geotiff(tp, template, tf)
        tinfo = read_geotiff(tp)
        tinfo["_path"] = str(tp)

        # GOOD file: conformed random field
        raw = rng.random((H, W))
        good, rep = conform_to_template(raw, template)
        gp = Path(td) / "good.tif"
        write_geotiff(gp, good, tf)
        res = validate_submission(gp, tinfo)
        assert res["pass"], f"GOOD file must pass: {res['gates']}"

        # BAD file 1: NaN inside footprint -> must trip gate 12 (the [0,1] rejection)
        bad1 = good.copy()
        bad1[20, 20] = np.nan
        b1 = Path(td) / "bad1.tif"
        write_geotiff(b1, bad1, tf)
        r1 = validate_submission(b1, tinfo)
        g12 = [g for g in r1["gates"] if g["id"] == "NAN-INSIDE-FOOTPRINT"][0]
        assert not r1["pass"] and not g12["ok"], "gate 12 must catch NaN-inside"

        # BAD file 2: value > 1 inside -> must trip gate 9
        bad2 = good.copy()
        bad2[21, 21] = 1.5
        b2 = Path(td) / "bad2.tif"
        write_geotiff(b2, bad2, tf)
        r2 = validate_submission(b2, tinfo)
        g9 = [g for g in r2["gates"] if g["id"] == "values-in-0-1"][0]
        assert not r2["pass"] and not g9["ok"], "gate 9 must catch value > 1"

        # BAD file 3: finite outside footprint -> must trip gate 13
        bad3 = good.copy()
        bad3[6, 6] = 0.5
        b3 = Path(td) / "bad3.tif"
        write_geotiff(b3, bad3, tf)
        r3 = validate_submission(b3, tinfo)
        g13 = [g for g in r3["gates"] if g["id"] == "footprint-matches-official"][0]
        assert not r3["pass"] and not g13["ok"], "gate 13 must catch finite-outside"

        print("self-test: GOOD passes 13/13; BAD1/BAD2/BAD3 correctly rejected.")
        print(json.dumps({"good_gates": [(g['id'], g['ok']) for g in res["gates"]],
                          "conform_report": rep}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="13-gate submission validator")
    ap.add_argument("files", nargs="*", help="submission .tif files")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.files:
        ap.error("no files given (or use --self-test)")
    tinfo = None
    if Path(args.template).exists():
        tinfo = read_geotiff(args.template)
        tinfo["_path"] = args.template
    else:
        print(f"WARNING: template {args.template} missing — grid UNVERIFIED, "
              "all grid gates will FAIL by design.")
    rc = 0
    for f in args.files:
        res = validate_submission(f, tinfo)
        print(f"== {f}: {'PASS 13/13' if res['pass'] else 'FAIL'}")
        for g in res["gates"]:
            print(f"   [{'ok' if g['ok'] else 'FAIL'}] {g['id']}: {g['detail']}")
        print(f"   stats: {json.dumps(res['stats'])}")
        rc = rc or (0 if res["pass"] else 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
