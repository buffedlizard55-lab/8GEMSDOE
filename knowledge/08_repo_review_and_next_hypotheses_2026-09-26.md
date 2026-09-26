# Repository review and next-hypothesis gate — 2026-09-26

## Executive finding

**No new strategy is authorized for a competition upload or implementation as a winning candidate in this session.** The working checkout does not contain the competition feature, label, or sample-submission rasters (`data/` contains only small reports and the SGMC proxy). Python dependencies are also absent (`numpy` import fails; `pytest` is not installed). Consequently, no independent rerun of a spatial holdout or actual-format raster validation was possible. Existing reports provide useful historical measurements, not independently reproduced measurements in this checkout. Do not treat their numbers as freshly verified.

This is an important mismatch with older README language saying data placement and validation were completed. Those statements describe a previous environment/session and are not evidence that the necessary data are present now. The submission `.tif` files are present, but without the official sample grid and dependencies this review cannot revalidate their CRS, transform, range, or scoring. There is no DrivenData score for the current raster recorded in the ledger.

## Results audit: duplicate score

The committed `reports/sibling_audit.json` reports that the GEMSDOE and 5GEMSDOE ens12 artifacts had the same file SHA-256 (`7f00890a…`) and payload SHA-256 (`cb2d2d5e…`), as did the GEMSDOE2 recall artifact. If those logged byte-level comparisons are accurate, identical predictions explain equal public scores; a score match alone would not establish duplication. The audit report paths refer to sibling checkout paths under `/tmp/sib/`, which are not present here, so this session can inspect the committed audit record but cannot repeat its source-byte comparison. This distinction should be retained in all public claims.

## Review of existing candidate evidence

`knowledge/07_new_hypotheses_H7-H10_2026-09-26.md` records an earlier experiment on four candidate arms. Its reported holdout results state H7 (magnetic TDR/analytic signal), H8 (tip linkage), H9 (valley coherence), and H10 (seismic ridge coherence) failed to beat the reported random baselines; the reported H8 test was the pre-ranked top candidate and was falsified on the component-holdout setup. Those values are quoted as historical report results only, not validated afresh today. Existing strategy says to require holdout improvement before a slot; this review does not override that gate.

## New candidates for a future data-enabled test (not implemented or slot-approved)

These are **research hypotheses**, not established geothermal/fault discoveries. They are distinct candidate transforms relative to the implementation descriptions in `src/`, `scripts/holdout_new.py`, and the H7–H10 knowledge note. Because current rasters are absent, they have not been validated on this checkout. Ranking is a qualitative pre-test prioritization, not a measured DTI ranking.

| Rank | Hypothesis / layers | Target signature and missed-catalogue rationale | Difference from this repo | Expected DTI upside / effort | Validation status |
|---|---|---|---|---|---|
| 1 | **H11: gravity–magnetic edge co-location and phase agreement** — `iso_grav_anom_hg` (18), `tmi_hg` (3), `tmi_vg` (9), optionally `rtp` (2) | Detect spatially coincident, similarly oriented gradients in independent density and magnetization fields, emphasizing coherent buried contacts potentially lacking surface scarps and thus absent from surface-evidence catalogues. | H1 is gravity-only TDR; H7 is magnetic TDR/analytic signal. This tests cross-survey co-location/gradient-direction agreement, not either individual edge family or a GBT feature alone. | **Medium-high / medium**; independent geophysical corroboration may suppress single-survey noise, but signal-to-fault specificity is uncertain. | **Blocked:** needs real bands and blocked holdout; no slot until it beats the best matched baseline on an explicitly stated hidden-truth-like population. |
| 2 | **H12: gravity curvature / ridge-valley differential** — `iso_grav_anom` (17), `iso_grav_anom_hg` (18), optionally `iso_grav_anom_vg` (19) | Multi-scale Laplacian/Hessian eigenvalue ridge-valley response, emphasizing elongated density boundaries or flexures rather than first-derivative edge magnitude. Buried faults may generate subtle, broad density curvature without a catalogued scarp. | Not the H1 first-derivative TDR, H3 DEM scarp step, H7 magnetic edge, or H9 DEM orientation-coherence operator; new operator and gravity channel. | **Medium / low-medium**; cheap raster transform, but gravity curvature can amplify noise and non-fault contacts. | **Blocked:** no current feature raster; compare multiple scales to same-fold random and existing arms. |
| 3 | **H13: cross-scale DEM lineament persistence** — `det_elev` (12), `det_elev_slope` (19) | Evaluate whether linear break-in-slope/curvature features persist across Gaussian scales and orientations; persistence could distinguish bedrock fault scarps from short-lived topographic noise and reveal subdued scarps below a fixed-resolution threshold. | H3 uses a profile step; H9 uses map-view structure coherence. H13 requires persistence/scale-space stability, not a single-scale scarp score or orientation-coherence score. | **Low-medium / medium**; likely limited by 100 m DEM resolution and overlap with H3/H9. | **Blocked:** feature rasters absent; spatial holdout required, scale parameters preregistered. |
| 4 | **H14: geophysical–seismic spatial concordance** — `ieq_n100a15` (16), `deq_n100a15` (10), `tmi_hg` (3), `iso_grav_anom_hg` (18) | Score linear seismic-density structures that lie near a coincident geophysical boundary; potential blind fault zones may align seismicity with buried contrasts. | H10 uses seismic ridge coherence and distance only; H11 uses gravity/magnetic co-location only. H14 tests the joint spatial conjunction; no current arm combines both in this manner. | **Low / medium-high**; conjunction may improve precision but could miss faults with sparse/no seismicity. | **Blocked:** no current feature raster; require ablation and blocked holdout against H10/H11 and matched random. |

### Candidate source/obtainability check

All proposed inputs are named as bands in the competition-derived layer layouts recorded in this repository; they require **no new external data source** if the competition `training_features.tif` can be made available. The current blocker is local data placement, not a need for third-party data. Official competition overview and data access links for manual verification:

- [DrivenData competition overview and rules](https://www.drivendata.org/competitions/306/competition-doe-gems/)
- [Official problem description](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
- [Competition data tab (authentication may be required)](https://www.drivendata.org/competitions/306/competition-doe-gems/data/)
- [Competition leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)
- [Official NREL/OSTI report PDF supplied in project materials](https://docs.nlr.gov/docs/fy26osti/96647.pdf)

No claim is made here that a particular geophysical pattern proves a fault or geothermal vent. The competition target is mapped faults, per the official problem description; geothermal-system interpretation is a separate scientific question.

## Validation gate / actionable next steps

1. In an authorized environment, put the official competition rasters in `data/` using the project fetch/verification procedure; verify the files against the authenticated data tab where possible. The script's SHA values are project pins, not proof of an official checksum.
2. Install `requirements.txt` (including the `rasterio` dependency used by holdout scripts; note `requirements.txt` and `pyproject.toml` currently specify slightly different dependency sets) and run `python scripts/prepare_data.py`.
3. Re-run current baseline holdouts and H11–H14 using spatially separated folds, buffered labels, fixed metric/budget, and per-fold results. Keep all candidate settings and random seeds preregistered; choose the top candidate only on the holdout procedure, not leaderboard feedback.
4. Do not touch a weekly submission slot unless a candidate beats the current holdout best on the agreed proxy/holdout and passes duplicate-byte, [0,1], and official grid conformance checks. An official competition result is only established after an actual authorized submission and displayed score.

## Verification log

- Repository branch was `arena/01a0df55-8gemsdoe`; working tree initially clean.
- `data/` inspection: no `training_features.tif`, `labels.tif`, or `sample_submission.tif` present.
- A repository-local `.venv` was created (ignored by Git) and dependencies installed there. `.venv/bin/pytest -q`: **29 passed, 3 skipped**.
- `.venv/bin/python scripts/validate_submission.py downloads/submission.tif`: all 13 local gates reported PASS, including [0,1] bounds, float32, shape 3292×3730, EPSG:32611, transform and counts. Important caveat: the official template is missing, so the validator explicitly used a self-footprint fallback for gates 11/13; this is **not strict template conformance**, and only DrivenData can confirm acceptance/score.
- The holdout scripts were not rerun: required source rasters are absent. Earlier holdout JSON reports parse correctly but remain historical evidence.
- JSON syntax checks for `reports/holdout_new.json`, `reports/sibling_audit.json`, and `reports/submissions_log.json` passed; `git diff --check` passed.
