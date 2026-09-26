#!/usr/bin/env python3
"""Calibrate a local pseudo-truth protocol against the family's REAL public scores.

The project's recurring failure mode is that local holdouts do not predict the
board (5GEMSDOE anchor ledger, deduction D3: a local proxy over-rewarded
sparsity by ~23x; this session: 6GEMSDOE `hgb88` has the highest local
catalogue-recall DTI of all ten artifacts, 0.1527, and the LOWEST public score,
0.0286). Rather than trust a protocol by construction, this script *fits* it.

Unknown facts about the hidden test set, all consistent with the official rules:
  * |G| — how many pixels the expert-labelled new faults occupy;
  * the mixture of three verified-possible regimes (src/gems/pseudo_truth.py):
      new  = wholly unmapped faults (held-out whole systems)
      seg  = newly mapped geometry of an existing system (along-strike chunks;
             forum 11536: "'new fault' ... can include newly mapped geometry of
             an existing fault system")
      off  = corrections/modifications of existing traces, i.e. the supplied
             trace is misaligned from the true surface fault (forum 11516 post 4
             bullet 3 + the problem description's misalignment sentence)

Method: for each cell of a (regime, |G|) grid, build the pseudo-truth folds,
score every publicly-scored family artifact, and compare the predicted mean DTI
with its real public score. The cell with the lowest SSE / highest Spearman is
the protocol the project should use for candidate selection.

CAVEAT carried into every output: the artifacts were trained on the whole
supplied catalogue, so their local numbers are contaminated upward; the fitted
cell is the best *available* mapping, not an unbiased generative model of G.

Usage:  python scripts/calibrate_protocol.py [--folds 2] [--quick]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems.anchors import ANCHORS  # noqa: E402
from gems.pseudo_truth import mixed_truth_folds  # noqa: E402
from gems.scoring import FoldScorer  # noqa: E402

FULL_GRID = [
    ((1.0, 0.0, 0.0), 20_000), ((1.0, 0.0, 0.0), 60_000),
    ((0.0, 1.0, 0.0), 20_000), ((0.0, 1.0, 0.0), 60_000),
    ((0.0, 0.0, 1.0), 20_000), ((0.0, 0.0, 1.0), 60_000),
    ((0.34, 0.33, 0.33), 20_000), ((0.34, 0.33, 0.33), 60_000),
    ((0.60, 0.20, 0.20), 20_000), ((0.60, 0.20, 0.20), 60_000),
    ((0.20, 0.40, 0.40), 20_000), ((0.20, 0.40, 0.40), 60_000),
]
QUICK_GRID = [
    ((1.0, 0.0, 0.0), 20_000), ((0.0, 1.0, 0.0), 20_000),
    ((0.0, 0.0, 1.0), 20_000), ((0.34, 0.33, 0.33), 20_000),
    ((0.34, 0.33, 0.33), 60_000), ((0.60, 0.20, 0.20), 60_000),
]


def spearman(a, b) -> float:
    def rank(x):
        x = np.asarray(x, float)
        vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[order] = np.arange(1, len(x) + 1, dtype=float)
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(REPO / "reports/protocol_calibration.json"))
    args = ap.parse_args()
    t_start = time.time()

    with rasterio.open(REPO / "data/labels.tif") as src:
        lab = src.read(1).astype(np.float32)
    footprint = np.isfinite(lab) & (lab >= 0)
    catalogue = lab == 1

    anchors = []
    for path_s, score, account, identity in ANCHORS:
        p = Path(path_s)
        if not p.exists() or score is None:
            continue
        with rasterio.open(p) as src:
            pred = src.read(1).astype(np.float32)
        pred = np.clip(np.nan_to_num(pred, nan=0.0), 0.0, 1.0)
        anchors.append({"name": p.name, "score": float(score), "account": account,
                        "identity": identity, "pred": pred,
                        "positive_px": int((pred > 0).sum())})
    if len(anchors) < 4:
        print(f"FAIL: only {len(anchors)} scored anchors found locally; clone the "
              "sibling repos first (see scripts/audit_siblings.py)")
        return 2
    print(f"anchors with public scores: {len(anchors)}")

    grid = QUICK_GRID if args.quick else FULL_GRID
    cells = []
    pub = np.array([a["score"] for a in anchors])
    for regime, n_truth in grid:
        t0 = time.time()
        folds = mixed_truth_folds(catalogue, footprint, n_folds=args.folds,
                                  regime=regime, n_truth=n_truth)
        pred_scores = np.zeros(len(anchors))
        per_fold = []
        detail = []
        for f in folds:
            sc = FoldScorer(f["gt"], f["known_mask"])
            for i, a in enumerate(anchors):
                r = sc.score(a["pred"] > 0)
                pred_scores[i] += r["DTI"] / len(folds)
                detail.append({"fold": f["fold"], "artifact": a["name"],
                               "DTI": r["DTI"], "TP_w": r["TP_w"], "FP_w": r["FP_w"],
                               "n_gt": r["n_gt"]})
            del sc
            per_fold.append({"fold": f["fold"], "n_gt": f["n_gt"],
                             "n_truth_whole_system": f["n_truth_whole_system"],
                             "n_truth_along_strike_segment": f["n_truth_along_strike_segment"],
                             "n_truth_misalignment_ring": f["n_truth_misalignment_ring"],
                             "gt_touching_known": f["gt_touching_known"]})
        sse = float(np.sum((pred_scores - pub) ** 2))
        mae = float(np.mean(np.abs(pred_scores - pub)))
        rho = spearman(pred_scores, pub)
        pear = float(np.corrcoef(pred_scores, pub)[0, 1])
        # best affine rescale (local DTI -> public score), reported as a diagnostic
        A = np.vstack([pred_scores, np.ones_like(pred_scores)]).T
        coef, res, *_ = np.linalg.lstsq(A, pub, rcond=None)
        sse_affine = float(np.sum((A @ coef - pub) ** 2))
        cells.append({
            "regime_new_seg_off": list(regime), "n_truth_per_fold": n_truth,
            "sse": sse, "mae": mae, "spearman": rho, "pearson": pear,
            "affine_slope": float(coef[0]), "affine_intercept": float(coef[1]),
            "sse_after_affine": sse_affine,
            "predicted": {a["name"]: float(v) for a, v in zip(anchors, pred_scores)},
            "actual": {a["name"]: a["score"] for a in anchors},
            "folds": per_fold, "detail": detail,
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"regime={tuple(round(x,2) for x in regime)} |G|={n_truth:>6,} "
              f"SSE={sse:.5f} MAE={mae:.4f} rho={rho:+.3f} "
              f"affine(SSE={sse_affine:.5f}, slope={coef[0]:.3f}) {time.time()-t0:.0f}s")
        del folds

    best_sse = min(cells, key=lambda c: c["sse"])
    best_rho = max(cells, key=lambda c: (-1 if np.isnan(c["spearman"]) else c["spearman"]))
    best_aff = min(cells, key=lambda c: c["sse_after_affine"])
    print("\n=== best cells ===")
    for tag, c in (("lowest SSE", best_sse), ("highest Spearman", best_rho),
                   ("lowest SSE after affine rescale", best_aff)):
        print(f"  {tag}: regime={tuple(round(x,2) for x in c['regime_new_seg_off'])} "
              f"|G|={c['n_truth_per_fold']:,} SSE={c['sse']:.5f} rho={c['spearman']:+.3f} "
              f"affine_slope={c['affine_slope']:.3f}")
        for a in anchors:
            print(f"      {a['name'][:44]:46s} pred={c['predicted'][a['name']]:.4f} "
                  f"actual={a['score']:.4f}")

    out = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/calibrate_protocol.py",
        "n_folds": args.folds,
        "anchors": [{k: v for k, v in a.items() if k != "pred"} for a in anchors],
        "cells": cells,
        "best": {"lowest_sse": {k: v for k, v in best_sse.items() if k != "detail"},
                 "highest_spearman": {k: v for k, v in best_rho.items() if k != "detail"},
                 "lowest_sse_affine": {k: v for k, v in best_aff.items() if k != "detail"}},
        "contamination_caveat": (
            "Anchor artifacts were trained on the entire supplied catalogue, so "
            "their local DTI is inflated relative to an honest per-fold retrain. "
            "The fitted cell is the best available local-to-board mapping, not an "
            "unbiased generative model of the hidden label set."),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out} ({out['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
