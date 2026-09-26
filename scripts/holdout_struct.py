"""Out-of-sample near-catalogue skill test (structural, unsupervised).

The Apex GBT's catalogue-fold DTI (0.89) is in-sample (trained on all
labels) — uninformative. This script isolates the UNSUPERVISED structural
prior (near-fault envelope r=2 + along-strike tip extensions, src/geology.py:
a fixed transform, no training, out-of-sample by construction) and measures
whether near-catalogue geometry predicts HELD-OUT catalogue faults — the
direct analogue of "newly mapped geometry of an existing fault system"
(forum 11536, verified). Arms (3% unless noted):

  S-structural-topk03  per-fold top 3% of max(envelope, extensions) built
                       from TRAIN-known only, incl. train-known px
  S-structural-offknown per-fold top 3% of the train-built ranking EXCLUDING
                       ALL known px (pure extension/splay prediction — the
                       honest new-geometry skill number)
  H6b-hedge            per-fold 2% train-structural (off-known) + 0.5% H1 +
                       0.5% H3 (global far-field), disjoint
  Z-random-03          uniform random baseline

Per-fold construction is REQUIRED for honesty: a global envelope built from
all labels covers test faults with their own halo (self-envelope), which
inflated an earlier draft of this script to 0.58 (caught in review
2026-09-26; that number is discarded, not reported). Train-only construction
measures true cross-boundary generalization (continuations/splays reaching
across fold boundaries), the analogue of new-geometry prediction.

Evaluated on catalogue folds (primary: new-geometry skill) and proxy folds
(context). Composite = mean of the two fold-means.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rasterio  # noqa: E402

from gems.holdout import block_folds, score_on_fold, topk_field  # noqa: E402
from src.geology import (compute_along_strike_extensions,  # noqa: E402
                         compute_near_fault_envelope)

from holdout_real import (BANDS_NEEDED, load_band_name_map, provided_tdr_score,  # noqa: E402
                          read_band_as_nan, read_labels, tile_apply)
from gems import features as F  # noqa: E402
from holdout_hedge import topk_disjoint  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Structural skill test")
    ap.add_argument("--report", default="reports/holdout_struct.json")
    args = ap.parse_args(argv)
    t0 = time.time()

    data = REPO_ROOT / "data"
    truth_cat, footprint = read_labels(data / "labels.tif")
    known = truth_cat > 0
    with rasterio.open(data / "proxy" / "proxy_catalogue_sgmc.tif") as src:
        proxy_only = ((src.read(1) == 2) & footprint).astype(np.float64)

    mapping, nodata = load_band_name_map(data / "training_features.tif")
    print("== H1/H3 far-field (global, label-free) ==", flush=True)
    h1 = tile_apply(
        provided_tdr_score, truth_cat.shape, tile_rows=4, overlap=4,
        vg=read_band_as_nan(data / "training_features.tif",
                            mapping["iso_grav_anom_vg"], nodata),
        hg=read_band_as_nan(data / "training_features.tif",
                            mapping["iso_grav_anom_hg"], nodata))
    dem = read_band_as_nan(data / "training_features.tif",
                           mapping["det_elev"], nodata)
    h3 = tile_apply(F.scarp_step, truth_cat.shape, tile_rows=4, overlap=8,
                    dem=dem)
    del dem
    rand_field = topk_field(
        np.where(footprint, np.random.default_rng(8).random(
            truth_cat.shape), np.nan), footprint, 0.03)

    folds = block_folds(truth_cat.shape, footprint, n_blocks=4, buffer_px=3,
                        seed=8)
    pops = {"catalogue": (truth_cat, None), "proxy": (proxy_only, known)}
    summary = {a: {p: {"mean": 0.0, "per_fold": []}
                   for p in pops} for a in (
                       "S-structural-topk03", "S-structural-offknown",
                       "H6b-hedge-struct", "Z-random-03")}
    for f in folds:
        tm = f["test_mask"]
        train_known = known & f["train_mask"]
        print(f"== fold {f['fold']}: train-built structural "
              f"(train-known px={int(train_known.sum())}) ==", flush=True)
        env = compute_near_fault_envelope(train_known, r_pixels=2,
                                          core_prob=0.92)
        ext = compute_along_strike_extensions(train_known, max_step=12,
                                              base_weight=0.85)
        struct = np.maximum(np.asarray(env, dtype=np.float64),
                            np.asarray(ext, dtype=np.float64))
        struct = np.where(footprint, struct, np.nan)
        del env, ext
        h6b = np.zeros(truth_cat.shape, dtype=bool)
        h6b |= topk_disjoint(struct, footprint, 0.020, h6b | known)
        h6b |= topk_disjoint(h1, footprint, 0.005, h6b | known)
        h6b |= topk_disjoint(h3, footprint, 0.005, h6b | known)
        fields = {
            "S-structural-topk03": topk_field(struct, footprint, 0.03),
            "S-structural-offknown": topk_field(
                np.where(~known, struct, np.nan), footprint, 0.03),
            "H6b-hedge-struct": np.where(footprint & h6b, 1.0, 0.0),
            "Z-random-03": rand_field,
        }
        del struct, h6b
        ys, xs = np.nonzero(tm)
        pad = 6
        y0, y1 = max(0, ys.min() - pad), min(tm.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(tm.shape[1], xs.max() + pad + 1)
        for pop, (g_full, km_full) in pops.items():
            for arm, field in fields.items():
                r = score_on_fold(field[y0:y1, x0:x1], g_full[y0:y1, x0:x1],
                                  tm[y0:y1, x0:x1],
                                  known_mask=(km_full[y0:y1, x0:x1]
                                              if km_full is not None else None))
                key = ("known_masked_exact" if km_full is not None
                       else "unmasked")
                summary[arm][pop]["per_fold"].append(float(r[key]["DTI"]))
    for arm in summary:
        for pop in summary[arm]:
            ds = summary[arm][pop]["per_fold"]
            summary[arm][pop]["mean"] = float(np.mean(ds))
    fields = summary  # for the print loop below
    del h1, h3
    print("\n== structural skill ==")
    for arm in fields:
        c = summary[arm]["catalogue"]["mean"]
        p = summary[arm]["proxy"]["mean"]
        print(f"  {arm:>22} cat={c:.4f} {summary[arm]['catalogue']['per_fold']} "
              f"proxy={p:.4f} composite={(c + p) / 2:.4f}")

    off = summary["S-structural-offknown"]["catalogue"]["mean"]
    rnd = summary["Z-random-03"]["catalogue"]["mean"]
    if off > rnd + 0.02:
        verdict = (f"structural off-known {off:.4f} beats random {rnd:.4f}: "
                   "near-catalogue geometry GENERALIZES to held-out faults — "
                   "supports the apex/extension line")
    else:
        verdict = (f"structural off-known {off:.4f} vs random {rnd:.4f}: no "
                   "measured out-of-sample near-catalogue skill")
    print(f"\nverdict: {verdict}")
    report = {"mode": "structural out-of-sample skill (catalogue+proxy folds)",
              "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
              "summary": summary, "verdict": verdict,
              "elapsed_s": round(time.time() - t0, 1)}
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
