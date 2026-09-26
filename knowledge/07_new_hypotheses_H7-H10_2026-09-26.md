# 8GEMSDOE Knowledge Base: New Hypotheses H7–H10 (2026-09-26, second round)

Four candidate geological hypotheses not previously implemented in this
repo (or the sibling programme), each naming its layers, physical
signature, why it should catch faults missing from the USGS/INGENIOUS
catalogue, and how it differs from implemented work. Ranked by expected
DTI gain vs implementation cost BEFORE measurement; the top candidate
(H8) was validated on a spatially-blocked holdout before any slot
consideration. All four validate WITHOUT new external data (every input
band is in `training_features.tif` + `labels.tif`), so no obtainability
check was needed.

## 1. The four candidates

### H7 — Magnetic basement lineaments (RTP/TMI tilt + analytic signal)
- **Layers:** `tmi_hg` (3), `tmi_vg` (9); reference `rtp` (2), `tmi` (14),
  `mag_anom` (1), `tc` (6). All tags asserted at runtime
  (`scripts/holdout_new.py`, `BANDS_NEEDED`).
- **Physical signature:** tilt derivative from the survey's own published
  gradients, TDR = atan2(vg, |hg|), zero-contour proximity × normalised
  analytic-signal amplitude AS = sqrt(hg² + vg²)
  (`src/gems/features.py:magnetic_tdr_as`). The TDR zero contour tracks
  lateral edges of magnetic units independent of anomaly amplitude.
- **Why unmapped:** "Gravity and magnetic surveys also help infer the
  presence of faults through density or magnetic anomalies" and "many
  [faults] are hidden below the surface, requiring geophysical data to
  detect" ([about page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/#about-the-task)).
  Buried basement faults juxtapose units of contrasting magnetization
  with no surface expression, while the Quaternary catalogue "contains
  information on faults … that demonstrate geological evidence of
  coseismic surface deformation" ([USGS faults page](https://www.usgs.gov/programs/earthquake-hazards/faults),
  re-fetched 2026-09-26) — so the buried population is systematically
  under-mapped.
- **Novelty:** H1 is gravity-only TDR; Apex uses Sobel(`tmi_hg`) only as
  one GBT input among 23, never as an emitted skeleton field; 6GEMSDOE
  computed magnetic HGM/tilt as blender inputs; GEMSDOE4's detectors are
  symmetric ridgeness, not TDR-zero-contour × AS. No line isolates the
  magnetic-only tilt/AS family as its primary field.

### H8 ★ TOP CANDIDATE — Tip-to-tip relay-linkage corridors
- **Layers:** catalogue geometry (`labels.tif`) + `geod_dilaterate` (8)
  as the extension weight.
- **Physical signature:** tips (fault pixels with exactly one
  8-neighbour) of DIFFERENT connected components, 3–20 px apart, whose
  outward strike vectors point at each other (mean cosine ≥ 0.5), are
  joined by a rasterised straight corridor weighted by gap decay ×
  collinearity × mean positive dilatation along the segment
  (`fault_endpoints`, `tip_linkage_corridors`).
- **Why unmapped:** relay/linkage faults between mapped segments are
  exactly "newly mapped geometry of an existing fault system", which the
  staff definition of "new fault" explicitly includes ([forum
  11516](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516),
  VERIFIED). Expert review for the Final Round cross-references
  submissions for candidate new faults ([problem description §
  structure](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure)),
  so geometrically-plausible linkages are reviewable discoveries.
- **Novelty:** Apex rays project outward from SINGLE tips and STOP at
  catalogue pixels (`src/geology.py` `break`) — no inter-tip corridor is
  ever emitted; its "relay" object is an intersection (deg≥3) buffer,
  not a tip-pair linkage; no dilatation/collinearity weighting exists
  anywhere in the repo or siblings.

### H9 — Valley-axis alignment lineaments (map-view coherence)
- **Layers:** `det_elev` (12) (+ `det_elev_slope` (19) context).
- **Physical signature:** structure-tensor orientation coherence of the
  detrended-elevation gradient field × normalised break-in-slope
  (`valley_axis_coherence`, `structure_coherence`). Long parallel
  contour bands (linear valleys / shutter ridges / deflected drainages)
  are coherent; flats and isotropic texture are forced to 0 by an
  energy floor. Sign-free: no curvature-sign convention asserted.
- **Why unmapped:** linear valleys and drainage deflections persist
  after the scarp step erodes below 100 m detectability; map-view
  alignment is invisible to cross-scarp profile operators and to
  compilers mapping offset layers ([about page: field detection via
  "fault scarps, offset rock layers"](https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/#about-the-task)).
- **Novelty:** H3 samples an antisymmetric step ALONG the gradient
  (cross-scarp profile); H9 measures orientation coherence IN MAP VIEW
  (along-valley alignment) — same bands, orthogonal operator. Sibling
  structure-tensor detectors (GEMSDOE4) run on edge bands generally,
  never coherence-gated break-in-slope on detrended elevation.

### H10 — Orientation-coherent microseismic ridges
- **Layers:** `ieq_n100a15` (16) + `deq_n100a15` (10).
- **Physical signature:** Gaussian-smoothed earthquake-density ridge ×
  structure-tensor coherence of the density gradients (linear ridges,
  not blobs) × proximity weight 1/(1+deq/median(deq))
  (`seismic_ridge_coherence`).
- **Why unmapped:** blind seismogenic faults lack surface expression and
  the catalogue requires surface-deformation evidence (USGS faults page,
  same sentence as H7); microseismic lineaments are the only
  surface-observable witness.
- **Novelty:** H5 is a binary quantile INTERSECTION of ieq with shallow
  basement (blob gating, no orientation); H10 is continuous,
  orientation-selective, and basement-free. Band 10 (`deq_n100a15`) is
  used nowhere else in this repo.

## 2. Pre-measurement ranking (expected DTI gain vs cost)

| rank | arm | expected gain | cost | rationale |
|---|---|---|---|---|
| 1 ★ | H8 linkage | highest | medium | serves the near-catalogue population (new geometry of known systems) that leaderboard ordering + forum intent suggest is rewarded; tip pairing + rasterisation is the only non-trivial code |
| 2 | H7 magnetic | high | low | largest plausible hidden population under cover; elementwise reuse of the H1 provided-gradient machinery |
| 3 | H10 seismic | medium | low | blind seismogenic faults are real but sparse; cheap structure-tensor code |
| 4 | H9 valley | medium-low | low-medium | 100 m DEM limits valley-axis resolution (same premise as H3's 1 m need) |

## 3. Validation (MEASURED 2026-09-26, `scripts/holdout_new.py`)

Protocol: whole-component 4-folds + 3 px buffer for H8 (truth = held-out
components ONLY; predictions admitted on test ∪ buffer so the official
300 m kernel credits near-hits; native mass vs matched-budget random +
padded-to-3% vs random@3%); 4×4 block folds + 3 px buffer for H7/H9/H10
at top-3% on catalogue monitor + SGMC-proxy selection (masked-exact).
Report: `reports/holdout_new.json` (elapsed 246.5 s, deterministic).

| arm | catalogue | proxy | verdict |
|---|---|---|---|
| H8-linkage-native (component folds) | 0.0001 vs matched random 0.0490, rays 0.0045 | — | FALSIFIED on gap-bridging: 9–29 corridor px enter eval zones per fold but earn TP≈0–5 vs random TP≈500–680. The corridors connect train tips through regions away from held-out components. NO SLOT. (Padded03 0.2281 < random03 0.2580: corridor mass displaces random mass.) |
| H7-mag-TDR-AS | 0.0465 | 0.0409 | below random both populations (same pattern as H1: edges avoid catalogue + proxy). Untested as buried-fault detector. NO SLOT. |
| H9-valley-coherence | 0.0753 (best unsupervised catalogue arm) | 0.1416 (best unsupervised arm on either population; stable folds 0.123–0.166) | still below random 0.171. New first far-field contrast candidate (replaces H3). NO SLOT. |
| H10-seismic-coherence | 0.0478 | 0.0234 | below random. NO SLOT. |

Random baselines re-measured identically (block cat 0.1704 / proxy
0.1711), confirming protocol comparability with the H1–H5 session.

Two review catches during this session (both fixed, see script
comments): (1) scoring component folds with predictions restricted to
the sparse test pixels zeroed every near-hit by construction — fixed to
test ∪ buffer eval masks with truth test-only, recording both DTI and
exact-hit DTI; (2) per-fold matched-random arm names split the summary
— fixed to stable names with per-row mass. The fair protocol confirms
the negative: H8 fails with near-hit credit too (TP≈0).

## 4. Consequences for the slot plan (updates knowledge/06)

- H8, the top-ranked new candidate, is FALSIFIED as a gap-bridging
  predictor on held-out components. No H8 slot. The honest residual:
  component folds test bridging between MAPPED components; truly
  unmapped relays have no held-out stand-in — same validation-ladder
  limit as before (only the leaderboard tests hidden truth).
- H9 becomes the first far-field contrast candidate (conditional second
  slot if the apex upload shows near-catalogue mass is not rewarded).
- Apex `b83ea0e7…` remains the first-slot candidate: still the only
  valid, unique, submission-shaped object testing the untested
  near-catalogue bet. No new DrivenData score is claimed or observed
  here.
