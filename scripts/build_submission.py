"""CLI: build a uniquely-named, hard-gated submission GeoTIFF.

Strategies:
  topk_hard@FRAC   1.0 on top FRAC of footprint by score, else 0.0 (needs --scores)
  threshold@T      1.0 where score >= T (needs --scores)
  zeros            all-zero inside footprint (format smoke test; DTI = 0)

--scores is a .npy file with the score field (same shape as template).
Without real data, use --demo to build from a synthetic fixture (stamped DEMO,
never uploadable as a real entry — shape will not match the template).

Every build: conform (finite-inside/NaN-outside/clip) -> 13-gate validate ->
unique name -> duplicate-hash guard -> submissions log + paste-ready Note.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems.submission import (conform_to_template, read_geotiff,
                             record_submission, suggest_note,
                             unique_submission_name, validate_submission,
                             write_geotiff)


def parse_strategy(s: str):
    if s == "zeros":
        return ("zeros", None)
    if "@" in s:
        name, val = s.split("@", 1)
        if name in ("topk_hard", "threshold"):
            return (name, float(val))
    raise ValueError(f"unknown strategy: {s}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a gated submission GeoTIFF")
    ap.add_argument("--strategy", default="topk_hard@0.03")
    ap.add_argument("--scores", default=None, help=".npy score field")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--out", default="submissions")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic fixture demo (NOT a real submission)")
    ap.add_argument("--allow-duplicate", action="store_true")
    ap.add_argument("--note-extra", default="")
    args = ap.parse_args(argv)

    name, val = parse_strategy(args.strategy)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.demo and not Path(args.template).exists():
        ap.error("official template missing; refusing to build a submission (use --demo for fixture)")
    if not args.demo and name != "zeros" and not args.scores:
        ap.error("real submission requires --scores; refusing synthetic random predictions")
    if args.demo or not Path(args.template).exists():
        if not args.demo:
            print(f"template {args.template} missing — falling back to --demo "
                  "fixture. NOT a real submission.")
        H, W = 60, 50
        tf = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
        template = np.zeros((H, W), dtype=np.float32)
        template[0:5, :] = np.nan
        tinfo = {"array": template, "shape": (H, W), "transform": tf,
                 "epsg": 32611, "nodata": "nan", "_path": None}
        demo = True
    else:
        tinfo = read_geotiff(args.template)
        tinfo["_path"] = args.template
        template = np.asarray(tinfo["array"])
        tf = tinfo["transform"]
        demo = False

    fp = np.isfinite(np.asarray(template, dtype=np.float64))
    if name == "zeros":
        field = np.zeros_like(template, dtype=np.float64)
    else:
        if args.scores:
            score = np.load(args.scores).astype(np.float64)
            if score.shape != template.shape:
                ap.error(f"scores shape {score.shape} != template {template.shape}")
        else:
            rng = np.random.default_rng(8)
            score = rng.random(template.shape)
            print("no --scores given: using deterministic synthetic scores (demo).")
        if name == "topk_hard":
            s = np.where(fp, np.nan_to_num(score, nan=-np.inf), -np.inf)
            k = max(1, int(round(val * fp.sum())))
            thr = np.partition(s[fp], -k)[-k]
            field = np.where(fp & (s >= thr), 1.0, 0.0)
        else:
            field = np.where(fp & (np.nan_to_num(score) >= val), 1.0, 0.0)

    conformed, rep = conform_to_template(field, template)
    payload = conformed.tobytes()
    sha = hashlib.sha256(payload).hexdigest()
    fname = unique_submission_name(
        f"{args.strategy}-{'demo' if demo else 'real'}", sha[:8])
    fpath = outdir / fname
    if fpath.exists():
        ap.error(f"output already exists: {fpath}")
    write_geotiff(fpath, conformed, tf,
                  description=f"8GEMSDOE {args.strategy} {'DEMO' if demo else ''} {sha[:16]}")
    res = validate_submission(fpath, tinfo)
    if not res["pass"]:
        fpath.unlink(missing_ok=True)
        print("BUILD REJECTED by validator:")
        for g in res["gates"]:
            if not g["ok"]:
                print(f"  FAIL {g['id']}: {g['detail']}")
        return 1
    note = suggest_note(args.strategy, sha[:8],
                        ("DEMO — do not upload" if demo else args.note_extra))
    entry = {"filename": fname, "strategy": args.strategy, "demo": demo,
             "payload_sha256": hashlib.sha256(
                 np.asarray(conformed).tobytes()).hexdigest(),
             "file_sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
             "bytes": fpath.stat().st_size, "note": note,
             "conform_report": rep, "stats": res["stats"]}
    try:
        record_submission(Path("reports/submissions_log.json"), entry,
                          allow_duplicate=args.allow_duplicate)
    except Exception:
        fpath.unlink(missing_ok=True)
        raise
    print(f"BUILT: {fpath} ({entry['bytes']} bytes, payload {sha[:16]}…)")
    print(f"NOTE (paste into DrivenData dialog): {note}")
    print("gates: 13/13 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
