"""Selection experiment on the INDEPENDENT-proxy truth (SGMC proxy-only faults).

Truth = code-2 pixels of data/proxy/proxy_catalogue_sgmc.tif: real USGS SGMC
faults with NO training label within 300 m (61,664 px) — the "missing from the
catalogue" class, i.e. the closest stand-in available in this sandbox for the
scored new-fault population. known_mask = catalogue faults, so the masking
columns (exact vs dilated-3px) are informative here (unlike catalogue-truth
runs where they are identical by construction).

Same blocked protocol as holdout_real.py (4x4 blocks, 3px buffer, 3% budget,
exact-k). Arms: H1/H1b/H1c/H2/H3/H4/H5 + random + apex (soft/topk03) +
sibling full-grid audit fields are scored in reports/sibling_audit (catalogue)
and here on proxy (full-grid section at the end, when --siblings is given).

Decision rule: the arm with the best proxy-mean-DTI is the slot candidate —
BUT a slot additionally requires beating uniform-random by a clear margin
(skill, not budget geometry) and a documented reason the proxy resembles the
hidden new faults for THAT arm's physics.

Caveats (binding):
  * SGMC faults are mostly pre-Quaternary bedrock structure at state-map
    scale — a stand-in for "missing from the Quaternary catalogue", NOT for
    "expert-lidar new faults". The in-group SGMC-trained datapoint
    (GEMSDOE4, operator-reported LB 0.0343, UNVERIFIED) cautions that SGMC
    resemblance to hidden truth is limited.
  * The proxy raster is sibling-built (provenance in data/proxy/);
    rebuild-from-official on an unrestricted machine is pending.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rasterio  # noqa: E402

from gems import features as F  # noqa: E402
from gems.holdout import block_folds, score_on_fold, topk_field  # noqa: E402

from holdout_real import (BANDS_NEEDED, load_band_name_map, provided_tdr_score,  # noqa: E402
                          read_band_as_nan, read_labels, tile_apply)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Proxy-truth selection experiment")
    ap.add_argument("--report", default="reports/holdout_proxy.json")
    ap.add_argument("--budget", type=float, default=0.03)
    ap.add_argument("--siblings", default=None,
                    help="sibling clone root for full-grid proxy audit")
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
        assert (src.width, src.height) == (3292, 3730), "proxy grid mismatch"
        assert src.crs.to_epsg() == 32611, "proxy CRS mismatch"
        assert tuple(src.transform)[:6] == (
            100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0), "proxy tf mismatch"
        proxy = src.read(1)
    proxy_only = (proxy == 2) & footprint
    proxy_near = (proxy == 1) & footprint
    print(f"proxy-only truth px in footprint: {int(proxy_only.sum())} "
          f"(near-label: {int(proxy_near.sum())})")
    ptruth = proxy_only.astype(np.float64)

    # ---- score fields (same builders as holdout_real) ----
    scores: dict[str, np.ndarray] = {}
    print("== building score fields ==", flush=True)
    vg = read_band_as_nan(feat_path, mapping["iso_grav_anom_vg"], nodata)
    ghg = read_band_as_nan(feat_path, mapping["iso_grav_anom_hg"], nodata)
    scores["H1-gravity-TDR"] = tile_apply(
        provided_tdr_score, ptruth.shape, tile_rows=4, overlap=4, vg=vg,
        hg=ghg)
    scores["H1b-provided-grav-HG"] = np.where(
        np.isfinite(ghg), np.abs(np.nan_to_num(ghg)), np.nan).astype(np.float32)
    del vg, ghg
    grav = read_band_as_nan(feat_path, mapping["iso_grav_anom"], nodata)
    scores["H1c-recomputed-HGM"] = tile_apply(
        lambda grid: F.horizontal_gradient_magnitude(grid, method="gradient"),
        ptruth.shape, tile_rows=4, overlap=4, grid=grav)
    del grav
    dem = read_band_as_nan(feat_path, mapping["det_elev"], nodata)
    scarp = tile_apply(F.scarp_step, ptruth.shape, tile_rows=4, overlap=8,
                       dem=dem)
    scarp = scarp / (np.nanmax(np.asarray(scarp, dtype=np.float64)) + 1e-30)
    scores["H3-scarp-step"] = np.asarray(scarp, dtype=np.float32)
    del dem, scarp
    dilat = read_band_as_nan(feat_path, mapping["geod_dilaterate"], nodata)
    shear = read_band_as_nan(feat_path, mapping["geod_shearrate"], nodata)
    sec = read_band_as_nan(feat_path, mapping["geod_2ndinv"], nodata)
    sc = F.strain_corridors(dilat, shear, sec)
    dil_pos = np.clip(np.nan_to_num(sc["dilation_index"]), 0, None)
    valid_s = np.isfinite(dilat) & np.isfinite(shear) & np.isfinite(sec)
    scores["H2-dilation"] = np.where(valid_s, dil_pos, np.nan).astype(np.float32)
    del dilat, shear, sec, sc, dil_pos, valid_s
    cond = read_band_as_nan(feat_path, mapping["cond_surf"], nodata)
    depth = read_band_as_nan(feat_path, mapping["depth_to_base_surf"], nodata)
    scores["H4-cond-align"] = tile_apply(
        lambda conductivity, depth_to_base: F.conductivity_edges(
            conductivity, depth_to_base)["alignment"],
        ptruth.shape, tile_rows=4, overlap=4, conductivity=cond,
        depth_to_base=depth)
    del cond
    ieq = read_band_as_nan(feat_path, mapping["ieq_n100a15"], nodata)
    h5 = F.blind_fault_intersection(ieq, depth)
    h5score = np.where(np.nan_to_num(h5) > 0, np.nan_to_num(ieq), 0.0)
    scores["H5-blind-seismic"] = np.where(
        np.isfinite(ieq) & np.isfinite(depth), h5score, np.nan).astype(np.float32)
    del ieq, depth, h5, h5score
    rng = np.random.default_rng(8)
    rnd = rng.random(ptruth.shape)
    scores["Z-random"] = np.where(footprint, rnd, np.nan).astype(np.float32)
    del rnd
    apex_path = REPO_ROOT / "downloads" / "submission.tif"
    apex_native = None
    if apex_path.exists():
        with rasterio.open(apex_path) as src:
            apex_native = src.read(1).astype(np.float64)
        apex_native = np.where(footprint, np.nan_to_num(apex_native), np.nan)
        scores["A-apex-topk03"] = apex_native.astype(np.float32)

    folds = block_folds(ptruth.shape, footprint, n_blocks=4, buffer_px=3,
                        seed=8)
    rows = []
    for f in folds:
        tm = f["test_mask"]
        ys, xs = np.nonzero(tm)
        pad = 6
        y0 = max(0, ys.min() - pad)
        y1 = min(tm.shape[0], ys.max() + pad + 1)
        x0 = max(0, xs.min() - pad)
        x1 = min(tm.shape[1], xs.max() + pad + 1)
        g_c = ptruth[y0:y1, x0:x1]
        tm_c = tm[y0:y1, x0:x1]
        km_c = known[y0:y1, x0:x1]
        for arm, p_full in scores.items():
            field_c = topk_field(np.asarray(p_full, dtype=np.float64),
                                 footprint, args.budget)[y0:y1, x0:x1]
            r = score_on_fold(field_c, g_c, tm_c, known_mask=km_c)
            rows.append({"fold": f["fold"], "arm": arm,
                         "dti_unmasked": r["unmasked"]["DTI"],
                         "dti_masked_exact": r["known_masked_exact"]["DTI"],
                         "dti_masked_dil3": r["known_masked_dilated3"]["DTI"],
                         "TP_w": r["unmasked"]["TP_w"],
                         "FP_w": r["unmasked"]["FP_w"],
                         "FN_w": r["unmasked"]["FN_w"],
                         "n_truth_in_fold": r["n_truth_in_fold"]})
        if apex_native is not None:
            r = score_on_fold(apex_native[y0:y1, x0:x1], g_c, tm_c,
                              known_mask=km_c)
            rows.append({"fold": f["fold"], "arm": "A-apex-soft-native",
                         "dti_unmasked": r["unmasked"]["DTI"],
                         "dti_masked_exact": r["known_masked_exact"]["DTI"],
                         "dti_masked_dil3": r["known_masked_dilated3"]["DTI"],
                         "TP_w": r["unmasked"]["TP_w"],
                         "FP_w": r["unmasked"]["FP_w"],
                         "FN_w": r["unmasked"]["FN_w"],
                         "n_truth_in_fold": r["n_truth_in_fold"]})

    arms = sorted({r["arm"] for r in rows})
    summary = {}
    for arm in arms:
        for key in ("dti_unmasked", "dti_masked_exact", "dti_masked_dil3"):
            ds = [r[key] for r in rows if r["arm"] == arm]
            summary.setdefault(arm, {})[key] = {
                "mean": float(np.mean(ds)),
                "per_fold": [float(v) for v in ds]}
    ranked = sorted(arms, key=lambda a: summary[a]["dti_masked_exact"]["mean"],
                    reverse=True)
    print("\n== proxy results (mean DTI, masked-exact primary) ==")
    for arm in ranked:
        s = summary[arm]
        print(f"  {arm:>22} masked={s['dti_masked_exact']['mean']:.4f} "
              f"unmasked={s['dti_unmasked']['mean']:.4f} "
              f"dil3={s['dti_masked_dil3']['mean']:.4f}")

    rand_mean = summary["Z-random"]["dti_masked_exact"]["mean"]
    best, best_mean = ranked[0], summary[ranked[0]]["dti_masked_exact"]["mean"]
    h1_mean = summary["H1-gravity-TDR"]["dti_masked_exact"]["mean"]
    if best == "H1-gravity-TDR" and h1_mean > rand_mean + 0.02:
        decision = ("H1 validated on proxy (beats random by "
                    f"{h1_mean - rand_mean:.4f}) — slot candidate WITH "
                    "documented proxy limits")
    elif best_mean <= rand_mean + 0.02:
        decision = (f"NO arm beats random ({rand_mean:.4f}) on proxy — no "
                    "slot for any H-arm; proxy may not resemble hidden "
                    "faults for these physics")
    else:
        decision = (f"TOP IS {best} ({best_mean:.4f}), NOT H1 ({h1_mean:.4f}) "
                    "— no slot for H1")
    print(f"\nverdict: {decision}")

    # ---- sibling full-grid proxy audit (optional) ----
    sib_rows = []
    if args.siblings:
        from gems.metric import distance_weighted_tversky
        from audit_siblings import ARTIFACTS
        sibroot = Path(args.siblings)
        for name, rel in ARTIFACTS:
            p = sibroot / rel
            if not p.exists():
                continue
            with rasterio.open(p) as src:
                a = src.read(1).astype(np.float64)
            a = np.where(footprint, np.nan_to_num(a), 0.0)
            r = distance_weighted_tversky(a, ptruth, known_mask=known)
            sib_rows.append({"artifact": name,
                             "proxy_dti_masked": r["DTI"],
                             "TP_w": r["TP_w"], "FP_w": r["FP_w"],
                             "FN_w": r["FN_w"]})
            print(f"  sib {name:28} proxyDTI={r['DTI']:.4f}")
        if apex_native is not None:
            r = distance_weighted_tversky(np.nan_to_num(apex_native), ptruth,
                                          known_mask=known)
            sib_rows.append({"artifact": "8GEMSDOE-apex-soft",
                             "proxy_dti_masked": r["DTI"],
                             "TP_w": r["TP_w"], "FP_w": r["FP_w"],
                             "FN_w": r["FN_w"]})
            print(f"  sib {'8GEMSDOE-apex-soft':28} proxyDTI={r['DTI']:.4f}")

    report = {
        "mode": "REAL DATA, PROXY TRUTH (SGMC code-2, 61664px in-footprint subset)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "proxy_sha256": hashlib.sha256(proxy_path.read_bytes()).hexdigest(),
        "protocol": f"4x4 spatial blocks, 3px buffer, top-{args.budget * 100:.1f}% emission",
        "caveats": [
            "SGMC = pre-Quaternary bedrock structure at state-map scale: a "
            "stand-in for catalogue-missing faults, NOT for expert new faults.",
            "Proxy raster is sibling-built; rebuild-from-official pending.",
            "In-group SGMC-trained datapoint (GEMSDOE4 LB 0.0343) is "
            "operator-reported/UNVERIFIED and cautions limited resemblance.",
        ],
        "random_masked_mean": rand_mean,
        "summary": summary,
        "ranked_arms_masked": ranked,
        "folds": rows,
        "sibling_fullgrid_proxy": sib_rows,
        "verdict": decision,
        "elapsed_s": round(time.time() - t0, 1),
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {rp} (elapsed {report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
