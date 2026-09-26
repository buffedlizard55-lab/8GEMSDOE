#!/usr/bin/env python3
"""LOFSO anchor calibration — turn every publicly scored family artifact into a
local, competition-faithful measurement.

Problem this solves
-------------------
The family's local proxies have repeatedly failed to predict the public board
(5GEMSDOE `leaderboard_anchor.json` deduction D3: the local proxy over-rewarded
sparsity by ~23x). The reason is protocol, not arithmetic: block folds slice
fault traces and score against a truth the detector was trained on.

This script scores each artifact under **Leave-One-Fault-System-Out (LOFSO)**
folds (`src/gems/systems.py`), which emulate all three staff-verified scoring
rules at once:

  * held-out whole fault *systems* are the truth and are NOT masked
    (they play the role of expert-mapped "new faults");
  * the remaining catalogue is masked pixel-exactly (forum 11516 post 4);
  * truth may lie within 300 m of masked catalogue (corrections), because
    systems are linked at 300 m before the fold cut.

Artifact -> public score pairs are taken from the family's own verified anchor
ledger (5GEMSDOE `data/evidence/leaderboard_anchor/leaderboard_anchor.json`,
read 2026-09-25/26) plus the two extra rows whose identity is CERTAIN because
the site offers exactly one file (6GEMSDOE, GEMSDOE4). Identities that are
inferred rather than read from a DrivenData submissions page are flagged.

CONTAMINATION CAVEAT (stated in every report): these artifacts were trained on
the *whole* catalogue, so under LOFSO they can partly recall held-out traces
from memory. LOFSO on pre-existing artifacts is therefore a **ranking
diagnostic**, not an unbiased score estimate. Unbiased estimates require
per-fold retraining — `scripts/lofso_train_eval.py`.

Usage:  python scripts/lofso_anchor.py [--folds 6] [--out reports/lofso_anchor.json]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems.metric import distance_weighted_tversky_fast as dti_fast  # noqa: E402
from gems.systems import lofso_folds  # noqa: E402

from gems.anchors import ANCHORS as ANCHOR_CANDIDATES  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(1, len(x) + 1, dtype=float)
        # average ties
        vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--link-px", type=int, default=3)
    ap.add_argument("--collar-px", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "reports/lofso_anchor.json"))
    args = ap.parse_args()

    labels_path = REPO / "data/labels.tif"
    if not labels_path.exists():
        print(f"FAIL: {labels_path} missing - run scripts/download_competition_data.sh")
        return 2
    with rasterio.open(labels_path) as src:
        lab = src.read(1).astype(np.float32)
    footprint = np.isfinite(lab) & (lab >= 0)
    catalogue = lab == 1
    print(f"catalogue px={catalogue.sum():,}  footprint px={footprint.sum():,}")

    folds = lofso_folds(catalogue, footprint, n_folds=args.folds,
                        link_px=args.link_px, collar_px=args.collar_px)
    print("folds:", [(f["fold"], f["n_gt"], f["n_known"], len(f["systems"])) for f in folds])

    results = []
    for path_s, score, account, identity in ANCHOR_CANDIDATES:
        path = Path(path_s)
        if not path.exists():
            print(f"SKIP (absent): {path}")
            continue
        with rasterio.open(path) as src:
            pred = src.read(1).astype(np.float64)
        pred = np.where(np.isfinite(pred), pred, 0.0)
        pred = np.clip(pred, 0.0, 1.0)
        finite_px = int(np.isfinite(pred).sum())
        pos_px = int((pred > 0).sum())
        on_known = int(((pred > 0) & catalogue).sum())

        per_fold = []
        for f in folds:
            r = dti_fast(pred, f["gt"].astype(np.float64), known_mask=f["known_mask"])
            per_fold.append({"fold": f["fold"], "DTI": r["DTI"], "TP_w": r["TP_w"],
                             "FP_w": r["FP_w"], "FN_w": r["FN_w"], "n_gt": r["n_truth"]})
        mean_dti = float(np.mean([p["DTI"] for p in per_fold]))
        gt_weighted = float(np.sum([p["DTI"] * p["n_gt"] for p in per_fold]) /
                            max(1, np.sum([p["n_gt"] for p in per_fold])))
        sum_tp = float(np.sum([p["TP_w"] for p in per_fold]))
        sum_fp = float(np.sum([p["FP_w"] for p in per_fold]))
        sum_fn = float(np.sum([p["FN_w"] for p in per_fold]))
        K = len(folds)
        # pooled approximation: truth sets are disjoint so TP/FN add exactly;
        # each fold's FP term covers the whole grid, so the FP sum is divided by
        # K. This OVER-states FP (max over a subset <= max over the union), so
        # pooled_approx is a lower bound on the full-catalogue DTI.
        pooled = sum_tp / (sum_tp + 0.2 * sum_fp / K + 0.8 * sum_fn + 1e-12)
        full_cat = dti_fast(pred, catalogue.astype(np.float64))["DTI"]

        rec = {
            "file": path.name,
            "sha256_file": _sha256(path),
            "path": str(path),
            "public_score": score,
            "account": account,
            "identity_basis": identity,
            "positive_px": pos_px,
            "finite_px": finite_px,
            "on_known_fault_px": on_known,
            "lofso_mean_dti": mean_dti,
            "lofso_gt_weighted_dti": gt_weighted,
            "lofso_pooled_approx_dti": float(pooled),
            "full_catalogue_monitor_dti": float(full_cat),
            "per_fold": per_fold,
        }
        results.append(rec)
        s = f"{score:.4f}" if score is not None else "  --  "
        print(f"{path.name[:52]:54s} pub={s} lofso_mean={mean_dti:.4f} "
              f"pooled={pooled:.4f} cat={full_cat:.4f} px={pos_px:,} onKnown={on_known:,}")

    scored = [r for r in results if r["public_score"] is not None]
    corr = {}
    if len(scored) >= 3:
        pub = np.array([r["public_score"] for r in scored])
        for key in ("lofso_mean_dti", "lofso_gt_weighted_dti", "lofso_pooled_approx_dti",
                    "full_catalogue_monitor_dti"):
            v = np.array([r[key] for r in scored])
            corr[key] = {"pearson": float(np.corrcoef(pub, v)[0, 1]),
                         "spearman": spearman(pub, v)}
        # also: does emitted mass alone explain the board?
        mass = np.array([r["positive_px"] for r in scored], dtype=float)
        corr["positive_px"] = {"pearson": float(np.corrcoef(pub, mass)[0, 1]),
                               "spearman": spearman(pub, mass)}
    print("\ncorrelation with public score (n=%d scored artifacts):" % len(scored))
    for k, v in corr.items():
        print(f"  {k:32s} pearson={v['pearson']:+.3f} spearman={v['spearman']:+.3f}")

    out = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/lofso_anchor.py",
        "protocol": {
            "folds": args.folds, "link_px": args.link_px, "collar_px": args.collar_px,
            "truth": "held-out whole fault systems (unmasked)",
            "mask": "remaining catalogue, pixel-exact (forum 11516 post 4)",
            "metric": "official DTI alpha=0.2 beta=0.8 R=3px (src/gems/metric.py fast path)",
            "fold_truth_sizes": [f["n_gt"] for f in folds],
        },
        "contamination_caveat": (
            "All scored artifacts were trained on the ENTIRE supplied catalogue, so "
            "LOFSO lets them recall held-out traces from memory. Numbers are a "
            "ranking diagnostic, not unbiased estimates of the public score."),
        "correlations": corr,
        "artifacts": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
