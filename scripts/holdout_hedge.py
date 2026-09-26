"""H6 dual-population hedge test: near-catalogue + far-field in one 3% field.

Motivation (measured 2026-09-26): catalogue-fold DTI structurally rewards
near-catalogue mass (apex 0.89 in-sample) while SGMC-proxy-only DTI
structurally rewards catalogue-avoiding mass (pindrop-ridge 0.30 full-grid)
— because proxy-only truth DEFINES AWAY near-catalogue signal (code 2
excludes everything within 300 m of a label). Each proxy is blind to the
other population. The hidden new faults plausibly contain BOTH near-catalogue
geometry (extensions/corrections/splays — explicitly blessed in forum 11536:
"newly mapped geometry of an existing fault system" counts) and far-field
buried structures (competition "hidden systems" narrative). Under this
population uncertainty, a hedged field may beat concentrated bets.

H6 construction (3% total, disjoint):
  * 2.0% top apex-soft ranking (near-catalogue envelope + extensions +
    GBT anomalies — the verified-unique b83ea0e7 field's ranking)
  * 0.5% top H1 gravity-TDR ranking (far-field buried edges)
  * 0.5% top H3 scarp-step ranking (far-field subtle scarps)
Far-field parts exclude catalogue pixels AND already-chosen pixels.

Selection objective (balanced uncertainty): composite = mean(catalogue-fold
mean-DTI, proxy-fold mean-DTI). H6 spends a slot only if its composite beats
every single arm's composite by a clear margin AND the operator accepts the
50/50 population weighting as a judgment call (it is NOT measured — the true
near/far mix is unknown; forum 11527 unanswered).

Caveats: apex ranking is in-sample on catalogue (optimistic half of the
composite); SGMC resemblance to hidden faults is questioned by in-group
evidence (see holdout_proxy docstring). Composite is a decision aid under
uncertainty, not a leaderboard prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rasterio  # noqa: E402

from gems import features as F  # noqa: E402
from gems.holdout import block_folds, score_on_fold, topk_field  # noqa: E402

from holdout_real import (BANDS_NEEDED, load_band_name_map, provided_tdr_score,  # noqa: E402
                          read_band_as_nan, read_labels, tile_apply)


def topk_disjoint(score: np.ndarray, footprint: np.ndarray, frac: float,
                  exclude: np.ndarray) -> np.ndarray:
    """Top-frac mask over footprint pixels not already excluded (exact-k)."""
    fp = np.asarray(footprint).astype(bool) & ~np.asarray(exclude).astype(bool)
    s = np.where(fp, np.nan_to_num(np.asarray(score, dtype=np.float64),
                                   nan=-np.inf), -np.inf)
    n_avail = int(fp.sum())
    n_fp = int(np.asarray(footprint).astype(bool).sum())
    k = max(1, min(n_avail, int(round(frac * n_fp))))
    flat = s.ravel()
    fp_idx = np.flatnonzero(fp.ravel())
    order = fp_idx[np.lexsort((fp_idx, -flat[fp_idx]))][:k]
    out = np.zeros(s.shape, dtype=bool)
    out.ravel()[order] = True
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="H6 hedge composite test")
    ap.add_argument("--report", default="reports/holdout_hedge.json")
    args = ap.parse_args(argv)
    t0 = time.time()

    data = REPO_ROOT / "data"
    feat_path = data / "training_features.tif"
    labels_path = data / "labels.tif"
    proxy_path = data / "proxy" / "proxy_catalogue_sgmc.tif"
    for p in (feat_path, labels_path, proxy_path):
        if not p.exists():
            print(f"FATAL: missing {p}")
            return 2

    mapping, nodata = load_band_name_map(feat_path)
    for name, exp_idx in BANDS_NEEDED.items():
        assert mapping.get(name) == exp_idx, (name, mapping.get(name))
    truth_cat, footprint = read_labels(labels_path)
    known = truth_cat > 0
    with rasterio.open(proxy_path) as src:
        proxy = src.read(1)
    proxy_only = ((proxy == 2) & footprint).astype(np.float64)

    with rasterio.open(REPO_ROOT / "downloads" / "submission.tif") as src:
        apex = src.read(1).astype(np.float64)
    apex = np.where(footprint, np.nan_to_num(apex), np.nan)
    print("== H1 field ==", flush=True)
    vg = read_band_as_nan(feat_path, mapping["iso_grav_anom_vg"], nodata)
    ghg = read_band_as_nan(feat_path, mapping["iso_grav_anom_hg"], nodata)
    h1 = tile_apply(provided_tdr_score, truth_cat.shape, tile_rows=4,
                    overlap=4, vg=vg, hg=ghg)
    del vg, ghg
    print("== H3 field ==", flush=True)
    dem = read_band_as_nan(feat_path, mapping["det_elev"], nodata)
    h3 = tile_apply(F.scarp_step, truth_cat.shape, tile_rows=4, overlap=8,
                    dem=dem)
    del dem

    h6 = np.zeros(truth_cat.shape, dtype=bool)
    h6 |= topk_disjoint(apex, footprint, 0.020, h6)
    h6 |= topk_disjoint(h1, footprint, 0.005, h6 | known)
    h6 |= topk_disjoint(h3, footprint, 0.005, h6 | known)
    h6_field = np.where(footprint & h6, 1.0, 0.0)
    print(f"H6 budget: {int(h6.sum())} px "
          f"({int(h6.sum()) / int(footprint.sum()) * 100:.2f}% of footprint)")
    del h1, h3

    fields = {
        "H6-hedge-2/0.5/0.5": h6_field,
        "A-apex-topk03": topk_field(apex, footprint, 0.03),
        "H1-topk03": topk_field(
            tile_apply(provided_tdr_score, truth_cat.shape, tile_rows=4,
                       overlap=4,
                       vg=read_band_as_nan(feat_path, mapping["iso_grav_anom_vg"], nodata),
                       hg=read_band_as_nan(feat_path, mapping["iso_grav_anom_hg"], nodata)),
            footprint, 0.03),
        "Z-random-03": topk_field(
            np.where(footprint, np.random.default_rng(8).random(truth_cat.shape),
                     np.nan), footprint, 0.03),
    }
    del apex

    folds = block_folds(truth_cat.shape, footprint, n_blocks=4, buffer_px=3,
                        seed=8)
    pops = {"catalogue": (truth_cat, None), "proxy": (proxy_only, known)}
    summary = {}
    for pop, (g_full, km_full) in pops.items():
        for arm, field in fields.items():
            ds = []
            for f in folds:
                tm = f["test_mask"]
                ys, xs = np.nonzero(tm)
                pad = 6
                y0, y1 = max(0, ys.min() - pad), min(tm.shape[0], ys.max() + pad + 1)
                x0, x1 = max(0, xs.min() - pad), min(tm.shape[1], xs.max() + pad + 1)
                r = score_on_fold(field[y0:y1, x0:x1], g_full[y0:y1, x0:x1],
                                  tm[y0:y1, x0:x1],
                                  known_mask=(km_full[y0:y1, x0:x1]
                                              if km_full is not None else None))
                key = "known_masked_exact" if km_full is not None else "unmasked"
                ds.append(r[key]["DTI"])
            summary.setdefault(arm, {})[pop] = {
                "mean": float(np.mean(ds)),
                "per_fold": [float(v) for v in ds]}
    print("\n== composite (mean of catalogue-mean and proxy-mean) ==")
    comp = {}
    for arm in fields:
        c = float(np.mean([summary[arm]["catalogue"]["mean"],
                           summary[arm]["proxy"]["mean"]]))
        comp[arm] = c
        print(f"  {arm:>22} composite={c:.4f} "
              f"(cat={summary[arm]['catalogue']['mean']:.4f} "
              f"proxy={summary[arm]['proxy']['mean']:.4f})")
    ranked = sorted(comp.items(), key=lambda kv: kv[1], reverse=True)
    best, best_c = ranked[0]
    h6_c = comp["H6-hedge-2/0.5/0.5"]
    if best == "H6-hedge-2/0.5/0.5" and h6_c > ranked[1][1] + 0.02:
        verdict = (f"H6 validated as uncertainty-hedge (composite {h6_c:.4f} "
                   f"beats {ranked[1][0]} {ranked[1][1]:.4f}) — slot "
                   "candidate under 50/50 population judgment")
    else:
        verdict = (f"H6 composite {h6_c:.4f}; best is {best} ({best_c:.4f}) — "
                   "no H6 slot on composite evidence")
    print(f"\nverdict: {verdict}")

    report = {"mode": "H6 hedge composite (catalogue + SGMC-proxy folds)",
              "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
              "h6_budget_px": int(h6.sum()),
              "summary": summary, "composite": comp,
              "ranked": [a for a, _ in ranked],
              "verdict": verdict,
              "elapsed_s": round(time.time() - t0, 1)}
    rp = Path(args.report)
    rp.write_text(json.dumps(report, indent=2))
    print(f"report: {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
