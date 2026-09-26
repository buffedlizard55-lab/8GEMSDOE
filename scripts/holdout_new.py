"""Blocked holdout validation for the H7-H10 portfolio (new hypotheses).

H7  magnetic basement lineaments (tmi_hg 3 + tmi_vg 9, provided-gradient
    TDR x analytic signal) — global, label-free.
H8  tip-to-tip relay-linkage corridors (catalogue tips + dilatation 8) —
    per-fold train-built, the TOP candidate (near-catalogue population).
H9  valley-axis alignment lineaments (det_elev 12: coherence x
    break-in-slope) — global, label-free.
H10 orientation-coherent microseismic ridges (ieq 16 + deq 10) — global,
    label-free.

Fold design (two geometries, each honest for its arms):
  * Component folds (H8 + extension reference + random): connected
    catalogue components are assigned whole to 4 folds by centroid block,
    with a 3 px buffer around test components. A linkage corridor from
    train tips that bridges a held-out component scores — the direct
    analogue of predicting an unmapped relay fault between mapped segments.
    Block folds would split components across train/test (self-envelope
    leakage for structural arms); component folds never split one.
  * Block folds 4x4 + 3 px buffer (H7/H9/H10 + random, catalogue monitor
    and SGMC-proxy selection): identical protocol to holdout_real.py /
    holdout_proxy.py so numbers are comparable.

Sparse-arm honesty: H8/ext corridors are scored at NATIVE mass against a
MATCHED-BUDGET random baseline (same pixel count), plus a padded-to-3%
variant vs random@3%. Padding a sparse field to 3% with arbitrary fill
would measure the fill, not the corridors.

Binding rule: no slot until an arm beats the holdout best (random 0.170 /
0.171) on a hidden-truth-like population. This script prints the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rasterio  # noqa: E402

from gems import features as F  # noqa: E402
from gems.holdout import block_folds, score_on_fold, topk_field  # noqa: E402
from gems.metric import distance_weighted_tversky  # noqa: E402
from holdout_real import (EXPECTED_SHA, load_band_name_map,  # noqa: E402
                          read_band_as_nan, read_labels, sha256_file)
from src.geology import compute_along_strike_extensions  # noqa: E402

BANDS_NEEDED = {
    "tmi_hg": 3,
    "tmi_vg": 9,
    "det_elev": 12,
    "ieq_n100a15": 16,
    "deq_n100a15": 10,
    "geod_dilaterate": 8,
    "iso_grav_anom_hg": 18,
}

BUDGET = 0.03


def component_folds(known: np.ndarray, footprint: np.ndarray,
                    n_blocks: int = 4, buffer_px: int = 3,
                    seed: int = 8) -> tuple[list[dict], np.ndarray]:
    """Whole-component folds: never split a connected fault component.

    Components are assigned to folds by centroid block ((bi+bj) % K,
    shuffled), exactly the block geometry of block_folds but with
    membership by component. Returns (folds, comp_id_grid).
    """
    H, W = known.shape
    fp = np.asarray(footprint).astype(bool)
    comp, ncomp = ndimage.label(np.asarray(known).astype(bool),
                                structure=np.ones((3, 3), dtype=int))
    ys_e = np.linspace(0, H, n_blocks + 1).astype(int)
    xs_e = np.linspace(0, W, n_blocks + 1).astype(int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_blocks)
    comp_fold = np.full(ncomp + 1, -1, dtype=int)
    # centroid of each component -> block -> fold
    for c in range(1, ncomp + 1):
        yy, xx = np.nonzero(comp == c)
        cy, cx = float(yy.mean()), float(xx.mean())
        bi = min(int(np.searchsorted(ys_e, cy, side="right") - 1),
                 n_blocks - 1)
        bj = min(int(np.searchsorted(xs_e, cx, side="right") - 1),
                 n_blocks - 1)
        comp_fold[c] = int(perm[(bi + bj) % n_blocks])
    folds = []
    for k in range(n_blocks):
        test_comp = np.isin(comp, np.flatnonzero(comp_fold == k))
        test = test_comp & fp
        dil = ndimage.binary_dilation(
            test, structure=np.ones((2 * buffer_px + 1, 2 * buffer_px + 1)))
        buffer = dil & ~test & fp
        train = fp & ~test & ~buffer
        folds.append({"fold": k, "train_mask": train, "test_mask": test,
                      "buffer_mask": buffer,
                      "n_train": int(train.sum()),
                      "n_test": int(test.sum()),
                      "n_buffer": int(buffer.sum()),
                      "n_test_components": int((comp_fold == k).sum())})
    return folds, comp


def random_exact_k(footprint: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Binary field with EXACTLY k footprint pixels (seeded, deterministic)."""
    fp = np.asarray(footprint).astype(bool)
    n_fp = int(fp.sum())
    k = max(1, min(n_fp, int(k)))
    rng = np.random.default_rng(seed)
    score = np.where(fp, rng.random(fp.shape), -np.inf)
    flat = score.ravel()
    fp_idx = np.flatnonzero(fp.ravel())
    order = fp_idx[np.lexsort((fp_idx, -flat[fp_idx]))][:k]
    out = np.zeros(fp.shape, dtype=np.float64)
    out.ravel()[order] = 1.0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="H7-H10 holdout validation")
    ap.add_argument("--report", default="reports/holdout_new.json")
    args = ap.parse_args(argv)
    t0 = time.time()

    data = REPO_ROOT / "data"
    feat_path = data / "training_features.tif"
    labels_path = data / "labels.tif"
    proxy_path = data / "proxy" / "proxy_catalogue_sgmc.tif"
    for p in (feat_path, labels_path, proxy_path):
        if not p.exists():
            print(f"FATAL: required data file missing: {p}")
            return 2

    print("== data hash verification ==")
    for name, exp in EXPECTED_SHA.items():
        p = data / name
        if not p.exists():
            print(f"  SKIP {name}: not present")
            continue
        got = sha256_file(p)
        print(f"  [{'OK' if got == exp else 'MISMATCH'}] {name}: "
              f"{got[:16]}...")
        if got != exp:
            return 2

    print("== band layout assertion ==")
    mapping, nodata = load_band_name_map(feat_path)
    for name, exp_idx in BANDS_NEEDED.items():
        got_idx = mapping.get(name)
        if got_idx != exp_idx:
            print(f"FATAL: band {name!r}: expected {exp_idx}, got "
                  f"{got_idx}")
            return 2
        print(f"  [{got_idx:>2}] {name} OK")

    print("== loading labels + proxy ==")
    truth_cat, footprint = read_labels(labels_path)
    known = truth_cat > 0
    with rasterio.open(proxy_path) as src:
        proxy_only = ((src.read(1) == 2) & footprint).astype(np.float64)
    print(f"  footprint={int(footprint.sum())} "
          f"catalogue={int(known.sum())} "
          f"proxy-code2={int((proxy_only > 0).sum())}")

    # ---- global label-free arms ----
    print("== H7: magnetic TDR x AS (bands 3+9) ==", flush=True)
    tmi_hg = read_band_as_nan(feat_path, mapping["tmi_hg"], nodata)
    tmi_vg = read_band_as_nan(feat_path, mapping["tmi_vg"], nodata)
    h7 = F.magnetic_tdr_as(tmi_hg, tmi_vg)["score"].astype(np.float32)
    del tmi_hg, tmi_vg

    print("== H9: valley-axis coherence (band 12) ==", flush=True)
    dem = read_band_as_nan(feat_path, mapping["det_elev"], nodata)
    h9 = F.valley_axis_coherence(dem)["score"].astype(np.float32)
    del dem

    print("== H10: seismic ridge coherence (bands 16+10) ==", flush=True)
    ieq = read_band_as_nan(feat_path, mapping["ieq_n100a15"], nodata)
    deq = read_band_as_nan(feat_path, mapping["deq_n100a15"], nodata)
    h10 = F.seismic_ridge_coherence(ieq, deq)["score"].astype(np.float32)
    del ieq, deq

    print("== H8 cost band: dilatation (band 8) ==", flush=True)
    dilat = read_band_as_nan(feat_path, mapping["geod_dilaterate"], nodata)

    rng = np.random.default_rng(8)
    z_rand = np.where(footprint, rng.random(truth_cat.shape),
                      np.nan).astype(np.float32)
    fields_3 = {"H7-mag-TDR-AS": topk_field(h7.astype(np.float64),
                                            footprint, BUDGET),
                "H9-valley-coherence": topk_field(h9.astype(np.float64),
                                                 footprint, BUDGET),
                "H10-seismic-coherence": topk_field(h10.astype(np.float64),
                                                   footprint, BUDGET),
                "Z-random-03": topk_field(z_rand.astype(np.float64),
                                          footprint, BUDGET)}
    del h7, h9, h10, z_rand

    # ---- component folds: H8 vs extensions vs matched random ----
    print("== component folds (whole-component, buffer 3px) ==")
    comp_folds, _ = component_folds(known, footprint, n_blocks=4,
                                    buffer_px=3, seed=8)
    for f in comp_folds:
        print(f"  fold {f['fold']}: train={f['n_train']} "
              f"test={f['n_test']} buffer={f['n_buffer']} "
              f"test_components={f['n_test_components']}")

    comp_rows = []
    for f in comp_folds:
        tm = f["test_mask"]
        train_known = known & f["train_mask"]
        print(f"== fold {f['fold']}: train-built H8 "
              f"(train-known px={int(train_known.sum())}) ==", flush=True)
        h8 = F.tip_linkage_corridors(train_known, dilatation=dilat,
                                     min_gap_px=3, max_gap_px=20,
                                     footprint=footprint,
                                     collinearity_min=0.5)
        h8_native = np.where(np.nan_to_num(h8) > 0, 1.0, 0.0)
        h8_mass = int(h8_native.sum())
        print(f"   H8 corridors: {h8_mass} px "
              f"({h8_mass / footprint.sum() * 100:.3f}%)", flush=True)
        ext = compute_along_strike_extensions(train_known, max_step=12,
                                              base_weight=0.85)
        ext_native = np.where(footprint & (np.asarray(ext) > 0), 1.0, 0.0)
        ext_mass = int(ext_native.sum())
        print(f"   EXT rays: {ext_mass} px "
              f"({ext_mass / footprint.sum() * 100:.3f}%)", flush=True)
        k3 = int(round(BUDGET * footprint.sum()))
        h8_pad = h8_native.copy()
        need = k3 - int(h8_pad.sum())
        if need > 0:
            fill = random_exact_k(footprint & (h8_pad == 0), need,
                                  seed=100 + f["fold"])
            h8_pad = np.clip(h8_pad + fill, 0, 1)
        arms = {
            "H8-linkage-native": (h8_native, h8_mass),
            "H8-linkage-padded03": (h8_pad, int(h8_pad.sum())),
            "Z-random-matchH8": (random_exact_k(
                footprint, h8_mass, seed=200 + f["fold"]), h8_mass),
            "EXT-rays-native": (ext_native, ext_mass),
            "Z-random-matchEXT": (random_exact_k(
                footprint, ext_mass, seed=300 + f["fold"]), ext_mass),
            "Z-random-03": (random_exact_k(footprint, k3,
                                           seed=400 + f["fold"]), k3),
        }
        del h8, ext
        # Fair component-fold scoring: truth = test components ONLY, but
        # predictions are admitted on test UNION the 3 px buffer so the
        # official 300 m kernel can credit near-hits (a corridor 100 m from
        # a held-out component earns k=0.67). Restricting pred to the sparse
        # test pixels would zero every near-hit by construction. Both the
        # kernel-credited (eval) and exact-hit-only DTI are recorded.
        ev = tm | f["buffer_mask"]
        g = (truth_cat > 0) & tm
        for arm, (field, mass) in arms.items():
            r_eval = distance_weighted_tversky(
                np.where(ev, field, 0.0), g.astype(np.float64))
            r_exact = score_on_fold(np.where(tm, field, 0.0),
                                    np.where(tm, truth_cat, 0.0), tm)
            comp_rows.append({"fold": f["fold"], "arm": arm,
                              "mass": mass,
                              "dti": float(r_eval["DTI"]),
                              "dti_exact": float(
                                  r_exact["unmasked"]["DTI"]),
                              "TP_w": float(r_eval["TP_w"]),
                              "FP_w": float(r_eval["FP_w"]),
                              "FN_w": float(r_eval["FN_w"]),
                              "n_truth": int(g.sum()),
                              "n_pred_in_eval": int(field[ev].sum())})
        means = {a: np.mean([x["dti"] for x in comp_rows
                             if x["arm"] == a and x["fold"] <= f["fold"]])
                 for a in arms}
        print("   fold-DTI: " + " ".join(f"{a}={means[a]:.4f}"
                                         for a in arms), flush=True)
    del dilat

    # ---- block folds: H7/H9/H10 on catalogue monitor + proxy selection ----
    print("== block folds (4x4, buffer 3px) for H7/H9/H10 ==")
    bfolds = block_folds(truth_cat.shape, footprint, n_blocks=4,
                         buffer_px=3, seed=8)
    block_rows = []
    for f in bfolds:
        tm = f["test_mask"]
        ys, xs = np.nonzero(tm)
        pad = 6
        y0, y1 = max(0, ys.min() - pad), min(tm.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(tm.shape[1], xs.max() + pad + 1)
        sl = (slice(y0, y1), slice(x0, x1))
        for arm, field in fields_3.items():
            rc = score_on_fold(field[sl], truth_cat[sl], tm[sl])
            rp = score_on_fold(field[sl], proxy_only[sl], tm[sl],
                               known_mask=known[sl])
            block_rows.append(
                {"fold": f["fold"], "arm": arm,
                 "dti_catalogue": float(rc["unmasked"]["DTI"]),
                 "dti_proxy_masked": float(
                     rp["known_masked_exact"]["DTI"])})
    arm_names = list(fields_3.keys())

    def mean_d(rows, arm, key):
        ds = [r[key] for r in rows if r["arm"] == arm]
        return float(np.mean(ds)), [float(v) for v in ds]

    print("\n== component-fold results (H8 population) ==")
    comp_arms = sorted({r["arm"] for r in comp_rows},
                       key=lambda a: -np.mean([x["dti"] for x in comp_rows
                                               if x["arm"] == a]))
    comp_summary = {}
    for a in comp_arms:
        ds = [x["dti"] for x in comp_rows if x["arm"] == a]
        comp_summary[a] = {"mean_dti": float(np.mean(ds)),
                           "per_fold": [float(v) for v in ds]}
        print(f"  {a:>28} mean={np.mean(ds):.4f} "
              f"folds={[round(v, 4) for v in ds]}")

    print("\n== block-fold results (catalogue monitor / proxy selection) ==")
    block_summary = {}
    for a in arm_names:
        mc, pc = mean_d(block_rows, a, "dti_catalogue")
        mp, pp = mean_d(block_rows, a, "dti_proxy_masked")
        block_summary[a] = {"catalogue_mean": mc, "catalogue_folds": pc,
                            "proxy_mean": mp, "proxy_folds": pp}
        print(f"  {a:>22} cat={mc:.4f} proxy={mp:.4f}")

    # ---- verdicts ----
    h8m = comp_summary["H8-linkage-native"]["mean_dti"]
    h8_match = comp_summary["Z-random-matchH8"]["mean_dti"]
    extm = comp_summary["EXT-rays-native"]["mean_dti"]
    ext_match = comp_summary["Z-random-matchEXT"]["mean_dti"]
    h8pad = comp_summary["H8-linkage-padded03"]["mean_dti"]
    rnd3 = comp_summary["Z-random-03"]["mean_dti"]
    if h8m > h8_match + 0.02 and h8m > extm + 0.01:
        v_comp = (f"H8 VALIDATED on component folds: native {h8m:.4f} vs "
                  f"matched random {h8_match:.4f} and rays {extm:.4f} — "
                  f"linkage corridors generalize to held-out components")
    else:
        v_comp = (f"H8 NOT validated: native {h8m:.4f} vs matched random "
                  f"{h8_match:.4f}, rays {extm:.4f} "
                  f"(padded03 {h8pad:.4f} vs random03 {rnd3:.4f}) — "
                  f"NO SLOT on this evidence")
    print(f"\nverdict (H8): {v_comp}")

    prox_best, prox_best_a = 0.0, None
    for a in ("H7-mag-TDR-AS", "H9-valley-coherence", "H10-seismic-coherence"):
        m = block_summary[a]["proxy_mean"]
        if m > prox_best:
            prox_best, prox_best_a = m, a
    prox_rnd = block_summary["Z-random-03"]["proxy_mean"]
    if prox_best > prox_rnd:
        v_far = (f"{prox_best_a} beats proxy random ({prox_best:.4f} vs "
                 f"{prox_rnd:.4f}) — far-field candidate for review")
    else:
        v_far = (f"no far-field arm beats proxy random "
                 f"(best {prox_best_a} {prox_best:.4f} vs {prox_rnd:.4f}) "
                 f"— NO SLOT")
    print(f"verdict (far-field): {v_far}")

    report = {
        "mode": "REAL DATA H7-H10 (component folds for H8, block folds "
                "for H7/H9/H10)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_sha256": {k: sha256_file(data / k) for k in EXPECTED_SHA
                        if (data / k).exists()},
        "band_layout_asserted_from_tags": BANDS_NEEDED,
        "protocol": "component folds: whole-component 4-fold + 3px buffer; "
                    "block folds: 4x4 + 3px buffer; top-3% emission for "
                    "H7/H9/H10; native+matched-budget for H8/EXT",
        "caveats": [
            "Component-fold truth = held-out whole catalogue components, "
            "NOT new faults. H8 DTI measures gap-bridging generalization; "
            "it is not a leaderboard claim.",
            "H8/ext are train-built per fold (out-of-sample by "
            "construction). H7/H9/H10 + random are unsupervised "
            "(out-of-sample by construction).",
            "Proxy DTI structurally favors catalogue-avoiding fields "
            "(see knowledge/05); proxy numbers select far-field arms only.",
        ],
        "component_summary": comp_summary,
        "component_rows": comp_rows,
        "block_summary": block_summary,
        "block_rows": block_rows,
        "verdict_component_H8": v_comp,
        "verdict_farfield": v_far,
        "elapsed_s": round(time.time() - t0, 1),
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {rp} (elapsed {report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
