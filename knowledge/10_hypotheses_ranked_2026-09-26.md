# 10 — Five candidate hypotheses, ranked by (expected DTI gain) ÷ (cost)

Rules applied, from the founding prompt and the standing user constraints:

1. **No submission slot is spent on anything that has not beaten the current holdout best**
   under the *clean* protocol (`knowledge/09` §5): whole fault systems held out, model
   retrained inside the fold, exact kernel metric, budget-matched against a random control.
2. If a hypothesis needs data we do not have, the **specific free official source is named
   and its obtainability is confirmed** before the hypothesis is ranked (see
   `knowledge/09` §6 for the fetch/verification records).
3. Every hypothesis says which **layer(s)** it uses, the **physical signature** it detects,
   **why the trace is missing from USGS Quaternary / INGENIOUS**, and **how it differs from
   what is already implemented** in this repo.

Reference numbers used below (all measured over **3 folds**, `knowledge/09` §5.1,
`reports/lofso_external.json`, 3 % budget): random **0.0820**, in-file BASE **0.0760**,
BASE+radiometric **0.0785**, BASE+topo **0.0844**, BASE+topo+radiometric **0.0898**,
BASE+coherence+topo+radiometric **0.0933** (AUC 0.7714 → 0.7885 across the same arms).
Verified board pattern 0.1563 ⇒ ≈11× base-rate enrichment; the leader's 0.3049 ⇒ ≈22×
(`knowledge/09` §3).

---

## H-A — Hysteresis lineament detector on the 10 m scarp channel  ← **RANK 1, and the only one with positive measured evidence**

* **Layers.** The external planes `T_*` (9): `slope_p90, slope_max, steep_frac, slope_std,
  relief_local, curv_prof_absmax, aspect_coherence, hs_lineament, dem_mean` — already
  unpacked, on-grid and covering **100 %** of the scored footprint
  (`knowledge/09` §6.4; the "42 %" figure that appeared earlier the same day was a census
  bug in our own script, now fixed).
* **Physical signature.** A Quaternary normal or strike-slip fault in the Walker Lane
  produces a *slope break* with (a) high p90 slope and `steep_frac` on the fault-line
  scarp, (b) a **profile-curvature discontinuity** (`curv_prof_absmax`) at the range-front
  break, (c) **high aspect coherence** along-strike (facets and triangular facets keep one
  orientation for kilometres), (d) a continuous ridge in multi-azimuth hillshade
  (`hs_lineament`). Aggregating these from 10 m to 100 m keeps the *lineament*, which the
  competition's own band 12 (detrended elevation) has had removed by detrending and by the
  100 m grid itself.
* **Why it finds traces missing from the catalogue.** USGS Quaternary fault compilation and
  the INGENIOUS inventory are built from published 1:24 000–1:100 000 maps. Unmapped
  traces are overwhelmingly (i) low-slip-rate scarps with 1–5 m of offset buried in
  alluvial-fan sediment, and (ii) range-front faults whose expression is a *facet line*,
  not a mapped trace. Both are geomorphic, i.e. exactly what a 10 m DEM resolves and a
  100 m detrended grid does not.
* **Difference from what is implemented.** The channel exists
  (`data/external/topo_features_100m.tif`) and is wired into the plane stack
  (`planes_ext.f32`, planes 46–54), where it is consumed only as **per-pixel model
  features**. What is *not* implemented is a standalone **hysteresis lineament detector**:
  seed on `curv_prof_absmax` ∧ `aspect_coherence` above a high threshold, then grow the
  seed along the *local strike direction* (estimated from the structure tensor of
  `hs_lineament`) while a low threshold is satisfied, and thin the result to a 1-px trace.
  This matters because the metric pays for **covering truth pixels per emitted pixel**:
  thin continuous traces cover far more of a linear truth set than the blobby probability
  fields a per-pixel classifier emits at the same budget.
* **Obtainability (confirmed).** USGS 3DEP 1/3″ seamless DEM, free, official, public
  domain:
  `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt`.
  This sandbox cannot reach S3 (`curl: (35)`), but a GitHub-hosted runner can and the
  sibling repo already has the workflow (`build-aux-channels`, run 36201412901) plus the
  sha256-pinned git bridge back into the sandbox (`scripts/fetch_aux_bridge.sh`).
* **Expected gain.** Measured on the clean protocol over the *whole* footprint:
  BASE 0.0760 → BASE+T **0.0844** → BASE+T+R **0.0898** → BASE+C+T+R **0.0933**
  (**+22.8 %** over BASE, **+13.7 %** over the random control 0.0820), with the top arm
  beating random on **all three folds** (+10.1 %, +21.7 %, +9.3 %) — the first time anything
  in this project has beaten random for *unseen* traces. The sibling repo's independent paired
  4-fold test on catalogue labels measured **+10.5 %** (0.138 → 0.152, positive on *every*
  fold). Two protocols, two repositories, same sign and similar size. The hysteresis
  detector is the un-tested part and is the only mechanism here with a plausible route to
  the 2× enrichment the top of the board implies.
* **Cost.** **LOW**: no data acquisition left (the channel is already on-grid at 100 %
  coverage), no GPU, pure numpy/scipy on 2 CPUs. No new science risk — the channel is
  measured positive twice.

## H-B — 1 m LiDAR micro-scarp channel over a targeted subset  ← RANK 2 by gain, RANK 5 by cost

* **Layers.** New planes from 1 m DEM tiles: `slope_1m_p95`, `curv_plan_1m`,
  `lineament_1m` (multi-azimuth hillshade at 1 m, aggregated), `scarp_height` (max
  elevation difference across a 100 m cell perpendicular to local strike),
  `scarp_continuity` (directional variance of the strike estimate).
* **Physical signature.** Single-event and fault-line scarps of 0.5–3 m height and
  100–1000 m length. At 10 m these are 1–2 grey levels; at 1 m they are explicit
  topographic steps with a plan-curvature kink and a sharp break-of-slope pair
  (upslope-facing and downslope-facing facets).
* **Why missing from the catalogue.** This is the population the competition's experts are
  most likely to have mapped for the withheld truth: the release text for GeoDAWN itself
  says LiDAR was flown "coordinated with this effort … over a similar extent", and the
  withheld labels are described as *new* faults, including new geometry of mapped systems
  (forum 11536). A 1 m scarp is mappable; a 100 m detrended grid is not.
* **Difference from what is implemented.** Nothing 1 m exists in this repo. The
  competition offers a `1m_DEM_links.csv` (login-gated); we instead hold the **716 resolved
  tile URLs** (`knowledge/09` §6.3) and verified one of them live (185,344,605 B).
* **Obtainability (confirmed, with a hard constraint).** Anonymous `ListObjectsV2` on
  `prd-tnm` works and the tiles are `StorageClass STANDARD`; full coverage is ≈**130 GB**,
  so the realistic move is a **20-tile subset (≈4 GB)** over the highest-value sub-areas
  (Walker Lane transect, basin margins adjacent to known geothermal systems), chosen by the
  H-A detector's uncertainty.
* **Expected gain.** Unknown but plausibly the largest available; a 10× resolution increase
  on the one channel already measured positive. Honest caveat: **no measurement exists**,
  and the budget rule forbids spending a slot on it until the clean protocol says it wins.
* **Cost.** HIGH: needs a machine with normal egress and ≥50 GB disk, plus real compute to
  process 1 m rasters (each tile ≈9 600² px). Run **after** H-A.

## H-C — Radiometric ratio halos *intersected* with lineaments (fluid-pathway faults)  ← RANK 3

* **Layers.** `R_*` (7): `rad_k, rad_th, rad_u, rad_tc` and the ratios `rad_thk, rad_uk,
  rad_uth`, intersected with `T_hs_lineament` / `T_aspect_coherence`.
* **Physical signature.** Hydrothermal circulation along a permeable fault leaches U and K
  and precipitates Fe–Mn oxides/silica ⇒ a **linear U/Th or K/Th anomaly** aligned with a
  topographic lineament, often with a `rad_tc` (total count) low over leached ground. The
  GeoDAWN grids are Compton-scattering-, radon-, cosmic- and altitude-corrected (verified
  from the release abstract), so ratios are physically interpretable rather than artefacts.
* **Why missing from the catalogue.** Alteration-controlled structures frequently have *no*
  geomorphic expression, so they are not in a Quaternary-fault compilation, yet they are
  precisely what a geothermal expert maps as a hidden, permeable fault segment — and
  "new geometry of an existing system" is explicitly in scope (forum 11536).
* **Difference from what is implemented.** The 7 bands are on-grid and were used **only as
  raw model features** (measured +4.8 %: 0.0727 → 0.0762). Not implemented: the
  **intersection operator** (lineament ∧ ratio anomaly), the ratio *gradient* along-strike
  vs across-strike decomposition, or anomaly-axis thinning. The physics is in the
  coincidence, not in either channel alone.
* **Obtainability (confirmed).** <https://doi.org/10.5066/P93LGLVQ>, ScienceBase item
  `657e1d85d34e23d3533209f7`, `22103_area1_tiffs.zip` (45,685,879 B, md5
  `56d8450884ec737a10ad6031056c22e2` verified by the producer) /
  `22103_area2_tiffs.zip`; already fetched, sha256-verified and unpacked here.
  Coverage **99.975 %** of the footprint.
* **A useful coincidence, now measured rather than assumed.** The scored footprint is
  5,167,373 px × (100 m)² = **51,674 km²**, and the GeoDAWN release states its surveys span
  **51,857 km²** — the competition grid *is* the GeoDAWN survey area. That is why the
  organisers could ship radiometric bands and 1 m DEM links for exactly this extent, and it
  means there is no "outside the survey" region to worry about.
* **Bonus verified fact:** the producer tested whether competition **band 6** is what its
  tag claims. Tag says "Tilt angle or total curvature – magnetic field derivative"; a
  200,000-pixel test against the GeoDAWN total-count grid gives **Pearson 1.0, Spearman
  1.0, identical min/median/max** ⇒ *band 6 IS the GeoDAWN radiometric total count and the
  official tag description is wrong* (`data/evidence/external_build/radiometric/radiometric_channel.report.json`,
  field `band6_identity_test`). Any hypothesis that treats band 6 as a magnetic-derivative
  product is built on a false premise.
* **Expected gain.** SMALL and **only in combination**: measured −4.3 % versus random on its
  own (0.0785 vs 0.0820) but +0.0054 mean DTI on top of BASE+T (0.0844 → 0.0898). Treat it
  as an interaction/confirmation channel, never as a standalone detector.
* **Cost.** LOW (data already on-grid; pure numpy).

## H-D — Strain-corridor gap completion (link unmapped tips *inside* a geodetic corridor)  ← RANK 4

* **Layers.** Bands 4 (geodetic second invariant), 7 (shear rate), 8 (dilatation rate),
  plus the derived `H14_strain_corridor`, `H14_log_deq`, `H13_*` planes already in
  `planes.f32`.
* **Physical signature.** Interseismic strain localises on locked faults; **dilatation
  maxima** mark dilational jogs and stepovers, **shear-rate maxima** mark the locked
  segments. A corridor is the set of pixels within ~1 km of a strain-rate maximum that is
  collinear with two mapped trace tips.
* **Why missing from the catalogue.** The catalogue is geology/geomorphology-based. A
  locked, buried, low-slip fault inside a basin can show a strain-rate signature while
  having no mapped trace — and the *gap between two mapped tips of the same system* is the
  single most predictable place for withheld "new geometry".
* **Difference from what is implemented.** An earlier "H8 linkage" arm was **falsified**
  (0.0001 vs matched random 0.049) — but on the *contaminated* proxy that `knowledge/09`
  §4/§5 shows cannot rank anything, and without the strain-corridor restriction (it linked
  tips geometrically, ignoring whether the corridor is strain-loaded). The re-test is
  cheap and must be run on the clean protocol before the old verdict is trusted.
* **Obtainability.** No new data needed (all in-file).
* **Expected gain.** LOW: the in-file families are at/below random today; this is a
  *re-test of a verdict*, not a promising new channel.
* **Cost.** LOW (numpy + the existing harness).

## H-E — RTP/tilt-angle edges restricted to basin fill (buried-fault population)  ← RANK 5

* **Layers.** Bands 1 (magnetic anomaly), 2 (reduced-to-pole), 3 (TMI horizontal
  gradient), 6 (tilt angle / total curvature), 9 (TMI vertical gradient), 14 (TMI), and
  band 15 (depth to basement) as the *basin-fill mask*.
* **Physical signature.** A fault that juxtaposes magnetic basement against non-magnetic
  fill gives a **linear RTP gradient discontinuity**; the tilt-angle zero contour traces it
  independent of amplitude, and the vertical gradient sharpens the shallow part. Restricting
  to pixels where `depth_to_basement` is large (thick fill) selects the *buried* population.
* **Why missing from the catalogue.** Buried faults under basin fill have no surface
  expression ⇒ absent from a geomorphic compilation, yet they are mappable from geophysics
  and are the classic "hidden fault" target of a geothermal play.
* **Difference from what is implemented.** H1 (TDR zero contour + HGM on isostatic
  gravity) was measured at 0.038/0.018 — but as a detector of *catalogue* faults, which is
  the wrong population. The basin-fill-restricted magnetic variant has never been measured,
  and the restriction (band 15 as a mask, not a feature) is the actual novelty.
* **Obtainability.** No new data needed.
* **Expected gain.** LOW: the BASE arm that already contains these bands scores below
  random, so the prior is weak; the mask may recover some of it.
* **Cost.** LOW.

---

## Ranking table

| rank | hypothesis | new data needed? | source confirmed? | measured evidence | expected DTI effect | cost |
|---|---|---|---|---|---|---|
| 1 | **H-A** hysteresis lineament detector on the 10 m scarp channel | none — already on-grid, 100 % coverage | ✅ 3DEP 1/3″ VRT | BASE+C+T+R **+13.7 % over random, all 3 folds** (AUC 0.7885); BASE+T alone +2.8 %; **+10.5 %** in the sibling's paired 4-fold | largest plausible route to >2× enrichment | **LOW** (numpy/scipy, no GPU) |
| 2 | **H-B** 1 m LiDAR micro-scarp subset | yes, 20 tiles ≈4 GB | ✅ 716 URLs, one tile verified live (185 MB) | none yet | potentially largest, unquantified | HIGH |
| 3 | **H-C** radiometric ratios ∩ lineaments | no (already on-grid, 99.975 %) | ✅ DOI 10.5066/P93LGLVQ | **−4.3 % alone** but **+0.0054 mean on top of BASE+T** — interaction channel only | small, additive | LOW |
| 4 | **H-D** strain-corridor gap completion | no | n/a | previous falsification was on an unusable proxy | low | LOW |
| 5 | **H-E** RTP/tilt edges in basin fill | no | n/a | related arm 0.038/0.018 on the wrong population | low | LOW |

## Gate before any slot is spent (binding)

For each candidate, in order:

1. `python scripts/lofso_train_eval.py --planes-stem planes_ext --sets BASE,<CAND>
   --budgets 0.03 --regime 1,0,0 --n-truth 15000 --restrict-aux --folds 3` —
   the candidate must beat **both** `RAND` and `RAND_cov` on ≥2 of 3 folds.
2. Re-run with `--regime 0,1,0` (along-strike segments) and `--regime 0,0,1`
   (1 px misalignment ring) to prove the gain is not an artefact of one truth model.
3. Only then build a submission, and only if it is a **weakly dominant** modification of
   the verified 0.1563 pattern (as in `knowledge/09` §2) or has beaten it on the clean
   protocol at a matched budget.

## What would have to be true to reach 0.3049

`knowledge/09` §3: at a 3 % budget, 0.3049 needs ≈14,900 covered truth pixels vs the
verified pattern's ≈8,000 — **a factor of two in enrichment (11× → 22× base rate)**. No
in-file transform measured so far moves enrichment above 1×. The only channels with
measured positive effect are external topography (+11 %) and, weakly, radiometrics (+5 %),
both covering the whole grid. That is the honest state of the gap: the *data* is no longer
the binding constraint — the detector is.
