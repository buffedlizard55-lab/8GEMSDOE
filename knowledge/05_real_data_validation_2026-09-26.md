# 8GEMSDOE Knowledge Base: Real-Data Validation Session 2026-09-26

All experiments below ran in this sandbox on the verified competition rasters
(hashes match `scripts/download_competition_data.sh` pins) and the exact
official DTI (`src/gems/metric.py`, tested against the problem description's
worked example TP=3.00/FP=1.89/FN=2.00 ⇒ 0.60). Reports: `reports/holdout_*.json`,
`reports/sibling_audit.json`. Nothing in this file is a leaderboard claim:
every DTI is against a visible truth population (catalogue, SGMC proxy, or
synthetic fixture), each labeled as such.

## 1. Data placement (blocker resolved in-sandbox)

- Dropbox TLS fails in this sandbox (verified `SSL_ERROR_SYSCALL`); DrivenData
  data tab redirects to login without auth (verified). GitHub egress works.
- Reassembled `training_features.tif` from the public sibling bridge
  (`github.com/buffedlizard55-lab/GEMSDOE data/bridge`, sparse clone) plus
  `existing_faults.tif` / `example_submission.tif`. All three SHA-256 match
  the pinned values (`4371c82e…`, `7ba308cc…`, `2176d08e…`).
- `scripts/prepare_data.py` passes: 19 bands, EPSG:32611, 100 m, 3292×3730,
  5,167,373 label/template footprint px, 60,988 fault px, 5,165,852 feature
  band-1 whole-grid valid px. Precise decomposition (verified): feature
  nodata is the finite float −3.4e38; 3,061 label-footprint px lack data on
  bands 1–5,7–19 (+12 extra on band 6 `tc`, union 3,073), while 1,540 finite
  feature px lie OUTSIDE the label footprint. Model code maps nodata→NaN.
- These hashes are repository pins, not DrivenData-authenticated checksums:
  independently compare against the authenticated data tab before prize
  submission. `data/*.tif` stays git-ignored; only reports + the 177 KB
  SGMC proxy raster (`data/proxy/`, with `PROVENANCE.json`) are committed.

## 2. Verified 19-band layout (from the file's own `band_name` tags)

| # | band_name | description (tag) | used by |
|---|---|---|---|
| 1 | mag_anom | magnetic anomaly | — |
| 2 | rtp | reduced-to-pole magnetics | — |
| 3 | tmi_hg | TMI horizontal gradient | Apex edge channel ✓ |
| 4 | geod_2ndinv | strain 2nd invariant | H2 ✓ |
| 5 | iso_grav_anom_slope | gravity slope | H1 family |
| 6 | tc | tilt/total curvature (magnetic edge detection) | Apex edge channel ✓ |
| 7 | geod_shearrate | shear strain rate | H2, Apex prior ✓ |
| 8 | geod_dilaterate | dilatation rate | H2, Apex prior ✓ |
| 9 | tmi_vg | TMI vertical gradient | — |
| 10 | deq_n100a15 | distance to earthquake | — |
| 11 | iso_grav_anom_vg | gravity vertical gradient | H1 real-data TDR ✓ |
| 12 | det_elev | detrended elevation | H3 ✓ |
| 13 | iso_grav_anom | isostatic gravity anomaly | H1 ✓ |
| 14 | tmi | total magnetic intensity | — |
| 15 | depth_to_base_surf | depth to basement / sediment thickness | H4, H5 ✓ |
| 16 | ieq_n100a15 | earthquake intensity/density | H5 ✓ |
| 17 | cond_surf | surface conductivity | H4, Apex prior ✓ |
| 18 | iso_grav_anom_hg | gravity horizontal gradient | H1, Apex edge ✓ |
| 19 | det_elev_slope | detrended elevation slope | Apex edge channel ✓ |

Consequences (all verified 2026-09-26, guarded by `tests/test_bandlayout.py`):
- The 7 band references in `scripts/train_apex_model.py` are all correct.
- **No magnetic-source-depth band exists** despite the problem description
  listing "top-of-crustal magnetic source depth estimate". H5 is redefined to
  `ieq_n100a15` (band 16) ∩ shallow `depth_to_base_surf` (band 15).
- The description's "depth to conductive base surface" has no exact tag
  match; band 15 ("depth to basement … sediment thickness") is the only
  depth band and is used for H4/H5 with this wording mismatch flagged.
- Band 6 (`tc`) has 12 fewer valid px (5,165,840) than the other bands.
- `sample_submission.tif` = 5,106,385 zeros + 60,988 ones, matching the
  catalogue exactly — contradicting the problem text's "predicts total fault
  absence". Both sides verified (text fetched, file read); the validator's
  finite-pixel footprint works under either reading.

## 3. H1–H5 catalogue-fold holdout (`scripts/holdout_real.py`)

Protocol: 4×4 blocks, 3 px buffer, exact 3% top-k budget, per-fold DTI
(unmasked; masking columns identical by construction when truth==known).
H1 real-data = provided-gradient TDR (bands 11+18; same signature as the
spectral fixture path, no 12 M-px FFT, contractor line-data gradients).

| arm | mean DTI | per-fold |
|---|---|---|
| A-apex-topk03 (IN-SAMPLE) | 0.8907 | 0.901/0.881/0.866/0.915 |
| A-apex-topk-massmatch 4.06% (IN-SAMPLE) | 0.8317 | 0.832/0.832/0.823/0.839 |
| A-apex-soft-native (IN-SAMPLE) | 0.7619 | 0.764/0.765/0.760/0.759 |
| Z-random | 0.1704 | 0.166/0.163/0.172/0.181 |
| H1b provided-grav-HG | 0.0437 | 0.073/0.024/0.050/0.028 |
| H1c recomputed-HGM | 0.0419 | 0.068/0.029/0.040/0.030 |
| H3 scarp-step | 0.0393 | 0.050/0.038/0.035/0.033 |
| H1 gravity-TDR | 0.0381 | 0.033/0.039/0.038/0.042 |
| H4 cond-align | 0.0326 | 0.025/0.019/0.039/0.047 |
| H5 blind-seismic (redefined) | 0.0255 | 0.055/0.005/0.013/0.029 |
| H2 dilation | 0.0114 | 0.042/0.000/0.004/0.000 |

Readings:
- Random@3% ≈ 0.17 is the calibrated budget-geometry floor (recall-heavy
  metric: β=0.8 rewards coverage even at chance). Sibling 6GEMSDOE's
  blocked-CV 0.1698 sits exactly on it — their trained model shows no
  held-out catalogue skill above chance.
- All unsupervised arms score BELOW random: their top edges systematically
  avoid catalogue faults. For H1/H4/H5 this is expected (they target the
  catalogue-complementary population by design); for H3 it is disappointing
  (Quaternary scarps should coincide) and supports H3's own premise that
  100 m is too coarse without 1 m calibration.
- Apex arms are IN-SAMPLE (GBT trained on all labels; top-3% covers every
  catalogue fault with FN=0). High numbers = memorization bound, not skill.
- Measured emission facts (replacing the false "strictly increasing"
  monotonicity claim, now corrected in code + docs): hardening at matched
  mass beats soft 0.8317 vs 0.7619 (+0.070); smaller budget beats larger
  0.8907 (3%) vs 0.8317 (4.06%). Binary top-k is a MEASURED-good policy at
  fixed budget, not a proven optimum.
- **Verdict: H1 NOT validated for a slot** (0.0381, below random). The
  catalogue proxy is the wrong population for catalogue-complementary
  hypotheses, so this neither validates nor falsifies H1 for new faults.

## 4. SGMC-proxy selection experiment (`scripts/holdout_proxy.py`)

Truth = sibling-built SGMC raster code 2 (61,664 px of real USGS SGMC
faults with no label within 300 m; grid + histogram re-verified locally;
rebuild-from-official pending — `services.arcgis.com` TLS fails here).
Masked-exact DTI (masking now informative), 3% budget folds:

- Z-random 0.1711 · H3 0.1271 · H5 0.0253 · H4 0.0178 · H1 0.0175 ·
  apex-soft 0.0152 · H2 0.0139 · H1c 0.0122 · apex-topk03 0.0113 · H1b 0.0095.
- **Verdict: no arm beats random on proxy — no H-arm slot.**
- Sibling full-grid proxy DTI (native budgets, masked): pindrop-ridge
  0.3018 · discovery 0.2396 · nodes 0.2343 · GEMSDOE4 0.2241 (in-sample:
  trained on this proxy) · ens12/union/recall ≈ 0.101 · extension 0.082 ·
  6GEMSDOE 0.035 · precision 0.025 · apex-soft 0.015.
- Structural caveat: proxy-only truth DEFINES AWAY near-catalogue signal,
  so proxy-DTI structurally favors catalogue-avoiding fields and punishes
  near-catalogue mass — possibly exactly backwards if hidden faults hug
  known traces (forum 11536 blesses "newly mapped geometry of an existing
  fault system"). The LB ordering hint (ens12 0.156 > pindrops 0.083–0.119
  despite lower proxy DTI) points the same way, but file→score mappings
  are operator-reported (only best+count are page-verifiable), so this is
  SUGGESTIVE, not conclusive. SGMC state-map traces may also be mislocated
  by ~100s m, penalizing precise detectors.

## 5. Hedge + structural tests (`holdout_hedge.py`, `holdout_struct.py`)

- H6 (2% apex-ranking + 0.5% H1 + 0.5% H3, 3% total): composite
  (mean of catalogue/proxy fold-means) 0.4286 vs apex-topk03 0.4510 vs
  random 0.1708 — no H6 slot; the in-sample catalogue half dominates.
- Train-only structural halo (envelope r=2 + 12 px extensions, honestly
  rebuilt per fold from train labels): catalogue-fold DTI ≈ 0.002 vs
  random 0.17 — no measurable cross-boundary skill UNDER THIS PROTOCOL
  (3 px buffer ≥ 2 px halo radius zeroes the measurement; fault-system
  folds would be the fairer test, not yet implemented).
- DISCARDED NUMBER (review catch, not reported as a result): a draft
  global-envelope run scored 0.58 via self-envelope (test faults covered
  by their own halo). Fixed to per-fold construction; the 0.58 is void.
- Incidental finding: apex-topk03 ≡ structural-topk03 to 4 decimals on
  all folds — at 3% budget the GBT contributes nothing to the ranking
  (structural prior saturates the budget); the GBT only matters at
  larger budgets.

## 6. Sibling audit (`scripts/audit_siblings.py`, direct byte reads)

- **Triple-verified duplicate**: GEMSDOE-ens12 == 5GEMSDOE-ens12 ==
  GEMSDOE2-recall (file sha `7f00890a…`, payload `cb2d2d5e…`, 172,974 px).
  Same bytes ⇒ same 0.1563/0.1560 is expected. Uniqueness of 8GEMSDOE apex
  (`b83ea0e7…`) re-verified against all 11 sibling artifacts (no match).
- Catalogue-monitor DTI (full grid, wrong population): apex-soft 0.7622 ·
  extension 0.4463 (292k px corridor) · 6GEMSDOE 0.4272 (in-sample) ·
  union 0.2382 · ens12 0.2298 · nodes 0.2068 · ridge 0.1957 ·
  discovery 0.1778 · precision 0.1387 (21k px) · GEMSDOE4 0.1114 (264k px,
  below random — SGMC-trained lineaments avoid catalogue by design).
- Pindrop fields place 0 px on catalogue faults (by construction);
  6GEMSDOE places 23,605 (15% of budget) on known faults (free under the
  verified masking rule, if exact-pixel).

## 7. Fixture (operator check, `holdout_validate.py --fixture`)

Synthetic buried fault (gravity step, no scarp): H1-spectral 0.84 vs
DEM-only 0.07 on the buried-trace fold — the H1 OPERATOR works when signal
is present. Pipeline check only, never a DTI claim.
