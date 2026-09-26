#!/usr/bin/env python3
"""Honest Leave-One-Fault-System-Out (LOFSO) bake-off: model arms vs geometric arms.

Why this script exists
----------------------
`scripts/lofso_anchor.py` measured the family's *pre-existing* artifacts under
LOFSO and exposed the contamination trap: 6GEMSDOE `hgb88`, which puts 23,605 px
on the supplied catalogue, posts the highest local DTI of all ten artifacts
(0.1527) and the LOWEST public score (0.0286). Any detector trained on the
whole catalogue can recall held-out catalogue traces from memory, so scoring it
against held-out catalogue traces measures memorisation, not discovery.

This script retrains inside every fold, and also evaluates *label-free*
geometric arms (which cannot memorise anything) in the same folds:

  truth   = fold's held-out whole fault systems, NOT masked
  mask    = remaining catalogue, pixel-exact (DrivenData staff, forum 11516 post 4)
  train   = remaining catalogue positives + background negatives from the fold's
            training pool (held-out systems and a 3 px collar removed)
  score   = official DTI (alpha 0.2, beta 0.8, R = 3 px), fast fold-level path

Ranking uses the boosting *decision function*, not `predict_proba`: the sigmoid
saturates to exactly 1.0 in float32 for ~1% of the grid, and those ties made
top-k selection degenerate to raster order (found 2026-09-26: BASE and H13
returned byte-identical emission masks and identical DTI).

Usage
-----
    python scripts/lofso_train_eval.py --folds 4 --budgets 0.01,0.03
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
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems.systems import collar  # noqa: E402
from gems.pseudo_truth import mixed_truth_folds  # noqa: E402

from gems.scoring import FoldScorer, exact_topk, STRUCT  # noqa: E402

DERIVED = REPO / "data" / "derived"


def load_planes(stem: str = "planes"):
    """`planes` = the 46 in-file planes; `planes_ext` = those 46 plus the 16
    EXTERNAL channel planes (T_* = 3DEP 10 m DEM scarp/topo, R_* = GeoDAWN
    radiometric) built by scripts/build_aux_planes.py."""
    meta = json.loads((DERIVED / f"{stem}.json").read_text())
    P, H, W = meta["shape"]
    st = np.memmap(DERIVED / f"{stem}.f32", dtype=np.float32, mode="r", shape=(P, H, W))
    return st, meta["planes"]


def family_indices(names: list[str], family: str) -> list[int]:
    if family == "BASE":
        return [i for i, n in enumerate(names) if n.startswith(("b0", "b1", "grad_", "mean5_"))]
    if family == "ALL":
        return list(range(len(names)))
    return [i for i, n in enumerate(names) if n.startswith(family + "_")]


def set_indices(names: list[str], spec: str) -> list[int]:
    """Composite feature sets: 'BASE+T+R' = in-file bands + topo + radiometric."""
    out: set[int] = set()
    for tok in spec.split("+"):
        tok = tok.strip().upper()
        if tok:
            out |= set(family_indices(names, tok))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--regime", default="1,0,0",
                    help="pseudo-truth mixture w_new,w_seg,w_off (see "
                         "src/gems/pseudo_truth.py). Default 1,0,0 = whole "
                         "held-out fault systems ('find a trace you never saw')")
    ap.add_argument("--n-truth", type=int, default=10_000,
                    help="pseudo-truth pixels per fold")
    ap.add_argument("--sets", default="BASE,H11,H12,H13,H14,C,ALL",
                    help="comma-separated; each entry may be composite, e.g. "
                         "BASE+T (in-file + topo scarp), BASE+T+R, BASE+C+T+R")
    ap.add_argument("--budgets", default="0.01,0.03")
    ap.add_argument("--neg", type=int, default=80_000)
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--block-rows", type=int, default=256)
    ap.add_argument("--use-dknown", action="store_true",
                    help="append distance-to-known-catalogue as a feature. "
                         "OFF by default: measured 2026-09-26, it makes the "
                         "model degenerate onto the 1 px catalogue ring "
                         "(identical emission for every feature set, DTI 0.0088 "
                         "vs random 0.0596 under the whole-system regime)")
    ap.add_argument("--planes-stem", default="planes",
                    help="'planes' (46 in-file) or 'planes_ext' (46 in-file + 16 "
                         "external: T_* 3DEP 10 m DEM scarp/topo, R_* GeoDAWN radiometric)")
    ap.add_argument("--restrict-aux", action="store_true",
                    help="ALSO score every arm with the emission pool restricted to "
                         "pixels where the external channels have data (T_* finite). "
                         "The aux channels cover only 42.08%% of the footprint "
                         "(measured 2026-09-26), so an unrestricted comparison "
                         "dilutes their contribution by 58%% of unusable area. "
                         "RAND_cov is the matched control.")
    ap.add_argument("--geo-arms", action="store_true", default=True)
    ap.add_argument("--out", default=str(REPO / "reports/lofso_train_eval.json"))
    args = ap.parse_args()

    t_start = time.time()
    st, names = load_planes(args.planes_stem)
    with rasterio.open(REPO / "data/labels.tif") as src:
        lab = src.read(1).astype(np.float32)
    footprint = np.isfinite(lab) & (lab >= 0)
    catalogue = lab == 1
    H, W = footprint.shape
    n_unmasked_total = int(footprint.sum())
    regime = tuple(float(x) for x in args.regime.split(","))
    folds = mixed_truth_folds(catalogue, footprint, n_folds=args.folds,
                              regime=regime, n_truth=args.n_truth)
    sets = [s.strip().upper() for s in args.sets.split(",") if s.strip()]
    budgets = [float(b) for b in args.budgets.split(",") if b.strip()]
    base_idx = family_indices(names, "BASE")
    print(f"planes={len(names)} base_planes={len(base_idx)} folds={args.folds} "
          f"sets={sets} budgets={budgets}")

    aux_cover = None
    if args.restrict_aux:
        ti = next((i for i, n in enumerate(names) if n.startswith("T_")), None)
        if ti is None:
            raise SystemExit("FAIL: --restrict-aux needs the external stack "
                             "(--planes-stem planes_ext, no T_ planes found)")
        aux_cover = np.isfinite(np.array(st[ti])) & footprint
        cov = int(aux_cover.sum())
        print(f"aux coverage: {cov:,} px = {100.0*cov/int(footprint.sum()):.2f}% of footprint")

    arms: dict[str, list[dict]] = {}

    def record(arm, fold, sc, extra=None):
        rec = {"fold": fold, "arm": arm}
        rec.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in sc.items()})
        if extra:
            rec.update(extra)
        arms.setdefault(arm, []).append(rec)
        print(f"  {arm:22s} DTI={sc['DTI']:.4f} TP={sc['TP_w']:8.1f} "
              f"FP={sc['FP_w']:9.1f} emit={sc['emitted_px']:>8,} hit={sc['hit_rate']:.4f}")

    for f in folds:
        gt, known = f["gt"], f["known_mask"]
        unmasked = footprint & ~known
        n_unmasked = int(unmasked.sum())
        scorer = FoldScorer(gt, known)
        print(f"\n[fold {f['fold']}] gt={f['n_gt']:,} (sys={f['n_truth_whole_system']:,} "
              f"seg={f['n_truth_along_strike_segment']:,} off={f['n_truth_misalignment_ring']:,} "
              f"adj_known={f['gt_touching_known']:,}) known={f['n_known']:,} "
              f"unmasked={n_unmasked:,} ({time.time()-t_start:.0f}s)")

        # ---------------- label-free geometric arms (cannot memorise) --------
        if args.geo_arms:
            rng = np.random.default_rng(1000 + f["fold"])
            r = rng.random((H, W), dtype=np.float32)
            for b in budgets:
                em = exact_topk(r, unmasked, int(round(b * n_unmasked))) | known
                record(f"RAND@{b:g}", f["fold"], scorer.score(em))
            if aux_cover is not None:
                pool = unmasked & aux_cover
                npool = int(pool.sum())
                for b in budgets:
                    em = exact_topk(r, pool, int(round(b * npool))) | known
                    record(f"RAND_cov@{b:g}", f["fold"], scorer.score(em),
                           {"pool_px": npool})
            for rad in (1, 2, 3, 5):
                ring = ndimage.binary_dilation(known, STRUCT, iterations=rad) & footprint
                record(f"HALO_r{rad}", f["fold"], scorer.score(ring))

        # ---------------- model arms (retrained inside the fold) -------------
        if args.use_dknown:
            d_known = ndimage.distance_transform_edt(~known).astype(np.float32)
            d_known[~footprint] = np.nan
        else:
            d_known = None
        neg_pool = f["train_pool"] & ~known & ~collar(known, 1)
        pos_idx = np.flatnonzero((f["train_labels"] & footprint).ravel())
        rng = np.random.default_rng(args.seed + f["fold"])
        n_neg = min(args.neg, int(neg_pool.sum()))
        neg_idx = rng.choice(np.flatnonzero(neg_pool.ravel()), size=n_neg, replace=False)
        rows = np.concatenate([pos_idx, neg_idx])
        y = np.zeros(len(rows), dtype=np.uint8)
        y[: len(pos_idx)] = 1
        rr, cc = np.divmod(rows, W)
        w = np.ones(len(rows))
        w[: len(pos_idx)] = n_neg / max(len(pos_idx), 1)

        # ONE sequential pass over every plane builds the sample matrix for all
        # arms at once (23 MB); gathering per arm instead scattered random reads
        # over the 2.26 GB memmap and cost ~10 min per arm-fold (measured).
        t0 = time.time()
        Xall = np.empty((len(rows), len(names) + (1 if d_known is not None else 0)),
                        dtype=np.float32)
        for pi in range(len(names)):
            Xall[:, pi] = np.array(st[pi])[rr, cc]
        if d_known is not None:
            Xall[:, -1] = d_known[rr, cc]
        print(f"  sample matrix {Xall.shape} built in {time.time()-t0:.0f}s")

        for s in sets:
            idx = set_indices(names, s)
            cols = idx + ([len(names)] if d_known is not None else [])
            ncol = len(cols)
            X = Xall[:, cols]
            t0 = time.time()
            clf = HistGradientBoostingClassifier(
                max_iter=args.max_iter, learning_rate=args.lr, max_leaf_nodes=31,
                min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
                random_state=args.seed)
            clf.fit(X, y, sample_weight=w)
            t_fit = time.time() - t0
            del X
            score = np.zeros((H, W), dtype=np.float32)
            t0 = time.time()
            for y0 in range(0, H, args.block_rows):
                y1 = min(H, y0 + args.block_rows)
                nb = y1 - y0
                Xb = np.empty((nb * W, ncol), dtype=np.float32)
                for j, pi in enumerate(idx):
                    Xb[:, j] = np.asarray(st[pi, y0:y1, :]).ravel()
                if d_known is not None:
                    Xb[:, -1] = d_known[y0:y1, :].ravel()
                score[y0:y1, :] = clf.decision_function(Xb).reshape(nb, W)
                del Xb
            t_pred = time.time() - t0
            score[~footprint] = -np.inf
            # honest generalisation diagnostic: AUC of the score field against
            # the fold's HELD-OUT truth (never seen in training), computed on a
            # seeded sample of unmasked pixels. 0.5 = no better than random.
            rs = np.random.default_rng(7 + f["fold"])
            cand = np.flatnonzero(unmasked.ravel())
            take = cand[rs.choice(cand.size, size=min(300_000, cand.size), replace=False)]
            y_auc = gt.ravel()[take].astype(int)
            auc = float("nan")
            if 0 < y_auc.sum() < len(y_auc):
                r_ = score.ravel()[take]
                pos, neg = r_[y_auc == 1], r_[y_auc == 0]
                # rank-based AUC (no sklearn metric import needed at this size)
                allv = np.concatenate([pos, neg])
                order = np.argsort(allv, kind="mergesort")
                ranks = np.empty(len(allv), dtype=np.float64)
                ranks[order] = np.arange(1, len(allv) + 1, dtype=np.float64)
                # average ties
                uv, ui, uc = np.unique(allv, return_index=True, return_counts=True)
                csum = np.cumsum(uc)
                avg = (csum - (uc - 1) / 2.0)
                ranks = avg[np.searchsorted(uv, allv)]
                rpos = ranks[: len(pos)].sum()
                auc = float((rpos - len(pos) * (len(pos) + 1) / 2.0) /
                            (len(pos) * len(neg)))
            for b in budgets:
                em = exact_topk(score, unmasked, int(round(b * n_unmasked))) | known
                record(f"{s}@{b:g}", f["fold"], scorer.score(em),
                       {"n_features": ncol, "fit_s": round(t_fit, 1),
                        "predict_s": round(t_pred, 1), "auc_vs_heldout_truth": auc})
                if aux_cover is not None:
                    pool = unmasked & aux_cover
                    npool = int(pool.sum())
                    em2 = exact_topk(score, pool, int(round(b * npool))) | known
                    record(f"{s}_cov@{b:g}", f["fold"], scorer.score(em2),
                           {"n_features": ncol, "pool_px": npool,
                            "auc_vs_heldout_truth": auc})
        del scorer, Xall

    summary = {}
    for arm, recs in arms.items():
        summary[arm] = {
            "mean_dti": float(np.mean([r["DTI"] for r in recs])),
            "per_fold_dti": [r["DTI"] for r in recs],
            "mean_hit_rate": float(np.mean([r["hit_rate"] for r in recs])),
            "mean_emitted_px": float(np.mean([r["emitted_px"] for r in recs])),
            "mean_unmasked_emitted_px": float(np.mean([r["unmasked_emitted_px"] for r in recs])),
            "mean_TP_w": float(np.mean([r["TP_w"] for r in recs])),
            "mean_auc_vs_heldout_truth": (
                float(np.nanmean([r["auc_vs_heldout_truth"] for r in recs]))
                if any("auc_vs_heldout_truth" in r for r in recs) else None),
            "kind": "geometric" if arm.startswith(("RAND", "HALO")) else "model",
        }
    ranked = sorted(summary.items(), key=lambda kv: -kv[1]["mean_dti"])
    print("\n=== LOFSO mean DTI (honest: models retrained per fold) ===")
    for arm, v in ranked:
        print(f"  {arm:22s} DTI={v['mean_dti']:.4f} hit={v['mean_hit_rate']:.4f} "
              f"emit={v['mean_emitted_px']:>9,.0f} (unmasked {v['mean_unmasked_emitted_px']:>9,.0f})")

    out = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/lofso_train_eval.py",
        "protocol": {
            "folds": args.folds, "budgets": budgets, "regime_new_seg_off": list(regime),
            "n_truth_per_fold": args.n_truth, "link_px": 0, "collar_px": 3,
            "neg_samples": args.neg, "max_iter": args.max_iter,
            "learning_rate": args.lr, "seed": args.seed,
            "detector": "sklearn HistGradientBoostingClassifier (decision_function ranking)",
            "distance_to_known_feature": bool(args.use_dknown),
            "emission": "exact top-k over unmasked footprint, plus 1.0 on masked "
                        "catalogue pixels (zero FP cost per forum 11516 post 4)",
            "metric": "official DTI alpha=0.2 beta=0.8 R=3px (fold-level fast path)",
            "fold_truth_sizes": [f["n_gt"] for f in folds],
            "fold_truth_composition": [[f["n_truth_whole_system"],
                                        f["n_truth_along_strike_segment"],
                                        f["n_truth_misalignment_ring"]] for f in folds],
            "note": "geometric arms use no labels at all, so they are "
                    "contamination-free by construction",
        },
        "feature_planes": names,
        "summary": summary,
        "ranked": [a for a, _ in ranked],
        "per_fold": arms,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}  ({out['elapsed_s']}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
