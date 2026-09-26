"""Real-data spatially-blocked holdout validation for the H1..H5 portfolio.

Protocol (binding, see docs/hypotheses.html):
  * 4x4 spatial blocks, 300 m (3 px) train-exclusion buffer (src/gems/holdout.py).
  * Fixed 3% pixel budget for every ranked arm (topk_field) — ranking comparison.
  * Per-fold DTI via the exact official metric (src/gems/metric.py), reported
    unmasked + known-masked-exact + known-masked-dilated-3px.
  * Decision rule: a hypothesis spends a weekly submission slot only if it beats
    the current holdout best on proxy DTI.

Band identities are read from the file's own `band_name` tags at runtime and
asserted — indices are NEVER trusted from memory. Verified layout (2026-09-26,
training_features.tif sha256 4371c82e...):
  13 iso_grav_anom, 5 iso_grav_anom_slope, 18 iso_grav_anom_hg, 11 iso_grav_anom_vg,
  12 det_elev, 19 det_elev_slope, 8 geod_dilaterate, 7 geod_shearrate,
  4 geod_2ndinv, 17 cond_surf, 15 depth_to_base_surf, 16 ieq_n100a15,
  10 deq_n100a15, 3 tmi_hg, 6 tc, 9 tmi_vg, 14 tmi, 2 rtp, 1 mag_anom.

IMPORTANT — what this measures and what it does NOT:
  * Truth = catalogue faults (labels.tif) restricted to held-out geography.
    H1..H5 are unsupervised transforms (they never see the catalogue), so their
    scores are genuine out-of-sample geographic-generalization numbers.
  * The Apex GBT arms are IN-SAMPLE: that model trained on all catalogue faults
    with no holdout, so its catalogue-fold DTI is optimistic (memorization
    possible). If an unsupervised arm beats Apex here despite that handicap,
    that is strong evidence; an Apex win here is weak evidence.
  * Catalogue DTI is the WRONG population for the leaderboard (scored on
    new faults). These numbers rank methods, they are NOT leaderboard claims.
  * When truth == known catalogue, the three masking columns are identical by
    construction (FP weight on truth pixels is already 0). They become
    informative only with independent proxy labels (SGMC) — kept for protocol
    consistency.

H5 REDEFINITION (verified band inventory 2026-09-26): the problem description
lists a "top-of-crustal magnetic source depth estimate" band, but NO such band
exists in training_features.tif (19 tags inventoried, none is a source-depth
band). H5 is therefore redefined to intersect earthquake density (ieq_n100a15,
band 16) with shallow basement from depth_to_base_surf (band 15). Likewise the
description's "depth to conductive base surface" has no exact tag match — band
15 ("Depth to basement surface - thickness of sedimentary cover") is the only
depth band and is used for H4/H5 with this wording mismatch flagged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import rasterio  # noqa: E402

from gems import features as F  # noqa: E402
from gems.holdout import block_folds, score_on_fold, topk_field  # noqa: E402
from gems.metric import degenerate_baselines  # noqa: E402

BUDGET_FRAC = 0.03

EXPECTED_SHA = {
    "training_features.tif": "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",
    "labels.tif": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
    "sample_submission.tif": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
}

# band_name tag -> 1-based index, asserted at runtime
BANDS_NEEDED = {
    "iso_grav_anom": 13,
    "iso_grav_anom_hg": 18,
    "iso_grav_anom_vg": 11,
    "det_elev": 12,
    "geod_dilaterate": 8,
    "geod_shearrate": 7,
    "geod_2ndinv": 4,
    "cond_surf": 17,
    "depth_to_base_surf": 15,
    "ieq_n100a15": 16,
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_band_name_map(feat_path: Path) -> dict[str, int]:
    with rasterio.open(feat_path) as src:
        mapping = {}
        for i in range(1, src.count + 1):
            nm = src.tags(i).get("band_name")
            if nm:
                mapping[nm] = i
        nodata = src.nodata
    return mapping, nodata


def read_band_as_nan(feat_path: Path, idx: int, nodata: float) -> np.ndarray:
    with rasterio.open(feat_path) as src:
        a = src.read(idx).astype(np.float64)
    return np.where(a == nodata, np.nan, a)


def read_labels(labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(labels_path) as src:
        a = src.read(1)
        nod = src.nodata
    if nod is not None:
        valid = a != nod
    else:
        valid = ~np.isnan(a.astype(np.float64))
    truth = ((a == 1) & valid).astype(np.float64)
    return truth, valid


def crop_to_fold(arrays: dict[str, np.ndarray], test_mask: np.ndarray,
                 pad: int = 6) -> tuple[dict[str, np.ndarray], np.ndarray]:
    ys, xs = np.nonzero(test_mask)
    y0, y1 = max(0, ys.min() - pad), min(test_mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(test_mask.shape[1], xs.max() + pad + 1)
    sl = (slice(y0, y1), slice(x0, x1))
    return {k: v[sl] for k, v in arrays.items()}, test_mask[sl]


def tile_apply(func, shape: tuple[int, int], tile_rows: int = 4,
               overlap: int = 8, out_dtype=np.float32, **arrays):
    """Apply a neighbourhood op in row strips (memory-safe on 12.3M grids).

    `func` takes the arrays as kwargs (one extended strip each) and returns an
    array (or dict of arrays) with the same 2D shape. `overlap` must exceed the
    operator radius (gradient=1, 3x3 filters=1, scarp half_len=2).
    """
    H, W = shape
    edges = np.linspace(0, H, tile_rows + 1).astype(int)
    out = None
    out_dict = None
    for t in range(tile_rows):
        y0, y1 = int(edges[t]), int(edges[t + 1])
        ey0, ey1 = max(0, y0 - overlap), min(H, y1 + overlap)
        kw = {k: v[ey0:ey1] for k, v in arrays.items()}
        res = func(**kw)
        ry0 = y0 - ey0
        ry1 = ry0 + (y1 - y0)
        if isinstance(res, dict):
            if out_dict is None:
                out_dict = {k: np.full(shape, np.nan, dtype=out_dtype)
                            for k in res}
            for k, arr in res.items():
                out_dict[k][y0:y1] = np.asarray(arr)[ry0:ry1]
        else:
            if out is None:
                out = np.full(shape, np.nan, dtype=out_dtype)
            out[y0:y1] = np.asarray(res)[ry0:ry1]
    return out_dict if out_dict is not None else out


def provided_tdr_score(vg: np.ndarray, hg: np.ndarray) -> np.ndarray:
    """TDR from the survey's own published gradients (no FFT).

    TDR = atan2(dz, HGM) with dz = iso_grav_anom_vg (band 11) and
    HGM = |iso_grav_anom_hg| (band 18). Same physical signature as the
    spectral TDR in the fixture pipeline (H1 spec), but computed from the
    contractor's line-data gradients (higher fidelity than re-deriving from
    the 100 m grid) and memory-safe (elementwise, no padded FFT).
    """
    from scipy import ndimage
    dz = np.asarray(vg, dtype=np.float64)
    h = np.abs(np.asarray(hg, dtype=np.float64))
    valid = np.isfinite(dz) & np.isfinite(h)
    tdr = np.arctan2(np.nan_to_num(dz), np.nan_to_num(h) + 1e-30)
    # zero-contour proximity (inlined 3x3 sign-change detector)
    near = np.abs(tdr) < 0.15
    lo = ndimage.minimum_filter(tdr, size=3)
    hi = ndimage.maximum_filter(tdr, size=3)
    cross = (lo < 0) & (hi > 0)
    prox = (near | cross).astype(np.float64)
    hn = h / (np.nanmax(h) + 1e-30)
    out = np.nan_to_num(prox) * (0.5 + 0.5 * np.nan_to_num(hn))
    return np.where(valid, out, np.nan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Real-data blocked holdout for H1..H5")
    ap.add_argument("--report", default="reports/holdout_real.json")
    ap.add_argument("--budget", type=float, default=BUDGET_FRAC)
    ap.add_argument("--folds", type=int, default=4, help="n_blocks (K folds)")
    args = ap.parse_args(argv)
    t0 = time.time()

    data = REPO_ROOT / "data"
    feat_path = data / "training_features.tif"
    labels_path = data / "labels.tif"
    for p in (feat_path, labels_path):
        if not p.exists():
            print(f"FATAL: required data file missing: {p}")
            print("Run scripts/download_competition_data.sh (or the bridge "
                  "reassembly) then scripts/prepare_data.py first.")
            return 2

    # --- 1. verify data hashes (fail loud on mismatch) ---
    print("== data hash verification ==")
    for name, exp in EXPECTED_SHA.items():
        p = data / name
        if not p.exists():
            print(f"  SKIP {name}: not present")
            continue
        got = sha256_file(p)
        status = "OK " if got == exp else "MISMATCH"
        print(f"  [{status}] {name}: {got[:16]}...")
        if got != exp:
            print(f"FATAL: hash mismatch for {name}")
            return 2

    # --- 2. assert band layout from tags ---
    print("== band layout assertion ==")
    mapping, nodata = load_band_name_map(feat_path)
    print(f"  nodata = {nodata}")
    for name, exp_idx in BANDS_NEEDED.items():
        got_idx = mapping.get(name)
        if got_idx != exp_idx:
            print(f"FATAL: band {name!r}: expected index {exp_idx}, "
                  f"tag map says {got_idx}")
            return 2
        print(f"  [{got_idx:>2}] {name} OK")
    if "mag_source_depth" in mapping or "tmi_source_depth" in mapping:
        print("NOTE: a magnetic source-depth band exists after all — H5 "
              "redefinition should be revisited.")
    else:
        print("  confirmed: NO magnetic-source-depth band in tags "
              "(H5 redefinition stands)")

    # --- 3. load truth + footprint ---
    print("== loading labels ==")
    truth, valid = read_labels(labels_path)
    footprint = valid
    print(f"  footprint px: {int(footprint.sum())}  "
          f"fault px: {int((truth > 0).sum())}")

    # --- 4. build score fields (one band at a time, float64->float32) ---
    scores: dict[str, np.ndarray] = {}

    print("== H1: gravity TDR zero-contour x HGM (bands 11+18, provided) ==",
          flush=True)
    vg = read_band_as_nan(feat_path, mapping["iso_grav_anom_vg"], nodata)
    ghg = read_band_as_nan(feat_path, mapping["iso_grav_anom_hg"], nodata)
    h1 = tile_apply(provided_tdr_score, truth.shape, tile_rows=4,
                    overlap=4, vg=vg, hg=ghg)
    scores["H1-gravity-TDR"] = h1
    del vg, h1

    print("== H1b: provided gravity HG (band 18, direct) ==", flush=True)
    scores["H1b-provided-grav-HG"] = np.where(
        np.isfinite(ghg), np.abs(np.nan_to_num(ghg)), np.nan).astype(np.float32)
    del ghg

    print("== H1c: gradient HGM recomputed on band 13 (ablation) ==",
          flush=True)
    grav = read_band_as_nan(feat_path, mapping["iso_grav_anom"], nodata)
    h1c = tile_apply(
        lambda grid: F.horizontal_gradient_magnitude(grid, method="gradient"),
        truth.shape, tile_rows=4, overlap=4, grid=grav)
    scores["H1c-recomputed-HGM"] = h1c
    del grav, h1c

    print("== H3: scarp step on detrended elevation (band 12) ==", flush=True)
    dem = read_band_as_nan(feat_path, mapping["det_elev"], nodata)
    scarp = tile_apply(F.scarp_step, truth.shape, tile_rows=4, overlap=8,
                       dem=dem)
    scarp = scarp / (np.nanmax(np.asarray(scarp, dtype=np.float64)) + 1e-30)
    scores["H3-scarp-step"] = np.asarray(scarp, dtype=np.float32)
    del dem, scarp

    print("== H2: strain dilation corridors (bands 8/7/4) ==")
    dilat = read_band_as_nan(feat_path, mapping["geod_dilaterate"], nodata)
    shear = read_band_as_nan(feat_path, mapping["geod_shearrate"], nodata)
    sec = read_band_as_nan(feat_path, mapping["geod_2ndinv"], nodata)
    sc = F.strain_corridors(dilat, shear, sec)
    dil = np.nan_to_num(sc["dilation_index"])
    dil_pos = np.clip(dil, 0, None)  # extension (opening) only
    valid_s = np.isfinite(dilat) & np.isfinite(shear) & np.isfinite(sec)
    scores["H2-dilation"] = np.where(valid_s, dil_pos, np.nan).astype(np.float32)
    del dilat, shear, sec, sc, dil, dil_pos, valid_s

    print("== H4: conductivity x basement-relief alignment (bands 17+15) ==",
          flush=True)
    cond = read_band_as_nan(feat_path, mapping["cond_surf"], nodata)
    depth = read_band_as_nan(feat_path, mapping["depth_to_base_surf"], nodata)
    ce = tile_apply(
        lambda conductivity, depth_to_base: F.conductivity_edges(
            conductivity, depth_to_base)["alignment"],
        truth.shape, tile_rows=4, overlap=4, conductivity=cond,
        depth_to_base=depth)
    scores["H4-cond-align"] = ce
    del cond, ce

    print("== H5 (redefined): top-decile ieq where basement shallow (16+15) ==")
    ieq = read_band_as_nan(feat_path, mapping["ieq_n100a15"], nodata)
    h5 = F.blind_fault_intersection(ieq, depth)
    # blind_fault_intersection is binary; rank hits by ieq for top-k
    h5score = np.where(np.nan_to_num(h5) > 0, np.nan_to_num(ieq), 0.0)
    scores["H5-blind-seismic"] = np.where(
        np.isfinite(ieq) & np.isfinite(depth), h5score, np.nan).astype(np.float32)
    del ieq, depth, h5, h5score

    print("== baselines: uniform-random + apex arms ==")
    rng = np.random.default_rng(8)
    rnd = rng.random(truth.shape)
    scores["Z-random"] = np.where(footprint, rnd, np.nan).astype(np.float32)
    del rnd

    apex_path = REPO_ROOT / "downloads" / "submission.tif"
    apex_native = None
    if apex_path.exists():
        with rasterio.open(apex_path) as src:
            apex_native = src.read(1).astype(np.float64)
        apex_native = np.where(footprint, np.nan_to_num(apex_native), np.nan)
        scores["A-apex-topk03"] = apex_native.astype(np.float32)
        apex_mass = float(np.nan_to_num(apex_native)[footprint].sum())
        apex_frac = apex_mass / float(footprint.sum())
        print(f"  apex native mass={apex_mass:,.0f} "
              f"(equiv frac={apex_frac:.4f})")
    else:
        apex_frac = None
        print("  WARNING: downloads/submission.tif missing — apex arms skipped")

    # --- 5. folds ---
    print(f"== building {args.folds}x{args.folds} blocked folds (buffer 3px) ==")
    folds = block_folds(truth.shape, footprint, n_blocks=args.folds,
                        buffer_px=3, seed=8)
    for f in folds:
        print(f"  fold {f['fold']}: train={f['n_train']} "
              f"test={f['n_test']} buffer={f['n_buffer']}")

    # --- 6. score every arm on every fold (bbox-cropped for speed) ---
    rows = []
    arm_names = list(scores.keys())
    for f in folds:
        tm = f["test_mask"]
        crop_arrays, tm_c = crop_to_fold(
            {"truth": truth, **{f"p::{k}": v for k, v in scores.items()}}, tm)
        g_c = crop_arrays.pop("truth")
        for arm in arm_names:
            p_full = scores[arm]
            if arm == "A-apex-topk03":
                field_full = topk_field(p_full, footprint, args.budget)
            else:
                field_full = topk_field(
                    np.asarray(p_full, dtype=np.float64), footprint,
                    args.budget)
            _, tm_c2 = crop_to_fold({"x": field_full}, tm)
            # recompute crop consistently: crop field with same bbox
            ys, xs = np.nonzero(tm)
            pad = 6
            y0 = max(0, ys.min() - pad)
            y1 = min(tm.shape[0], ys.max() + pad + 1)
            x0 = max(0, xs.min() - pad)
            x1 = min(tm.shape[1], xs.max() + pad + 1)
            field_c = field_full[y0:y1, x0:x1]
            r = score_on_fold(field_c, g_c, tm_c, known_mask=(g_c > 0))
            rows.append({
                "fold": f["fold"], "arm": arm,
                "budget_frac": args.budget,
                "dti_unmasked": r["unmasked"]["DTI"],
                "dti_known_masked_exact": r["known_masked_exact"]["DTI"],
                "dti_known_masked_dil3": r["known_masked_dilated3"]["DTI"],
                "TP_w": r["unmasked"]["TP_w"],
                "FP_w": r["unmasked"]["FP_w"],
                "FN_w": r["unmasked"]["FN_w"],
                "n_truth_in_fold": r["n_truth_in_fold"],
                "n_pred_pos_in_fold": r["unmasked"]["n_pred_pos"],
            })
        # apex native-soft arm (same fold crop, no hardening)
        if apex_native is not None:
            ys, xs = np.nonzero(tm)
            pad = 6
            y0 = max(0, ys.min() - pad)
            y1 = min(tm.shape[0], ys.max() + pad + 1)
            x0 = max(0, xs.min() - pad)
            x1 = min(tm.shape[1], xs.max() + pad + 1)
            r = score_on_fold(apex_native[y0:y1, x0:x1], g_c, tm_c,
                              known_mask=(g_c > 0))
            rows.append({
                "fold": f["fold"], "arm": "A-apex-soft-native",
                "budget_frac": None,
                "dti_unmasked": r["unmasked"]["DTI"],
                "dti_known_masked_exact": r["known_masked_exact"]["DTI"],
                "dti_known_masked_dil3": r["known_masked_dilated3"]["DTI"],
                "TP_w": r["unmasked"]["TP_w"],
                "FP_w": r["unmasked"]["FP_w"],
                "FN_w": r["unmasked"]["FN_w"],
                "n_truth_in_fold": r["n_truth_in_fold"],
                "n_pred_pos_in_fold": r["unmasked"]["n_pred_pos"],
            })
        # mass-matched hardened apex (isolates binary-vs-soft at fixed mass)
        if apex_native is not None and apex_frac is not None:
            ys, xs = np.nonzero(tm)
            pad = 6
            y0 = max(0, ys.min() - pad)
            y1 = min(tm.shape[0], ys.max() + pad + 1)
            x0 = max(0, xs.min() - pad)
            x1 = min(tm.shape[1], xs.max() + pad + 1)
            field_mm = topk_field(
                np.asarray(apex_native, dtype=np.float64), footprint,
                apex_frac)[y0:y1, x0:x1]
            r = score_on_fold(field_mm, g_c, tm_c, known_mask=(g_c > 0))
            rows.append({
                "fold": f["fold"], "arm": "A-apex-topk-massmatch",
                "budget_frac": apex_frac,
                "dti_unmasked": r["unmasked"]["DTI"],
                "dti_known_masked_exact": r["known_masked_exact"]["DTI"],
                "dti_known_masked_dil3": r["known_masked_dilated3"]["DTI"],
                "TP_w": r["unmasked"]["TP_w"],
                "FP_w": r["unmasked"]["FP_w"],
                "FN_w": r["unmasked"]["FN_w"],
                "n_truth_in_fold": r["n_truth_in_fold"],
                "n_pred_pos_in_fold": r["unmasked"]["n_pred_pos"],
            })

    # --- 7. aggregate + degenerate baselines (full grid) ---
    arms = sorted({r["arm"] for r in rows})
    summary = {}
    for arm in arms:
        ds = [r["dti_unmasked"] for r in rows if r["arm"] == arm]
        summary[arm] = {
            "mean_dti": float(np.mean(ds)),
            "min_dti": float(np.min(ds)),
            "max_dti": float(np.max(ds)),
            "per_fold": [float(v) for v in ds],
        }
    base = degenerate_baselines(truth, footprint)
    ranked = sorted(summary.items(), key=lambda kv: kv[1]["mean_dti"],
                    reverse=True)

    print("\n== holdout results (mean DTI over folds, budget 3%) ==")
    for arm, s in ranked:
        print(f"  {arm:>22} mean={s['mean_dti']:.4f} "
              f"folds={[round(v, 4) for v in s['per_fold']]}")
    print(f"\ndegenerate baselines (full grid): {json.dumps(base, indent=2)}")

    winner, wsum = ranked[0]
    h1_mean = summary.get("H1-gravity-TDR", {}).get("mean_dti")
    verdict = {
        "winner": winner,
        "winner_mean_dti": wsum["mean_dti"],
        "h1_mean_dti": h1_mean,
        "h1_is_winner": winner == "H1-gravity-TDR",
        "slot_decision": (
            "H1 validated as holdout best — MAY spend one weekly slot on an "
            "H1-derived submission"
            if winner == "H1-gravity-TDR" else
            f"TOP IS {winner}, NOT H1 — do NOT spend a slot on H1. "
            f"NOTE: {winner} is IN-SAMPLE (GBT trained on all labels) — "
            f"see holdout_proxy.py + holdout_struct.py for the honest "
            f"selection signals before spending any slot."),
    }
    print(f"\nverdict: {verdict['slot_decision']}")

    report = {
        "mode": "REAL DATA (training_features.tif + labels.tif)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_sha256": {k: sha256_file(data / k) for k in EXPECTED_SHA
                        if (data / k).exists()},
        "band_layout_asserted_from_tags": BANDS_NEEDED,
        "h5_redefinition": "ieq_n100a15(band16) ∩ shallow depth_to_base_surf("
                           "band15); no magnetic-source-depth band exists",
        "protocol": f"{args.folds}x{args.folds} spatial blocks, 3px buffer, "
                    f"top-{args.budget * 100:.1f}% emission",
        "caveats": [
            "Truth = catalogue faults in held-out geography (geographic-"
            "generalization proxy), NOT new-fault labels. Catalogue DTI "
            "ranks methods; it is not a leaderboard claim.",
            "H1..H5 + random are unsupervised (never saw the catalogue): "
            "out-of-sample by construction.",
            "Apex arms are IN-SAMPLE (GBT trained on all catalogue faults, "
            "no holdout): optimistic, memorization possible.",
            "Masking columns identical by construction when truth==known.",
            "No skeleton-thinning in this ranking pass; emission-policy "
            "(skeleton vs dense) is a separate follow-up experiment.",
        ],
        "summary": summary,
        "ranked_arms": [a for a, _ in ranked],
        "folds": rows,
        "degenerate_baselines_full_grid": base,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {rp} (elapsed {report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
