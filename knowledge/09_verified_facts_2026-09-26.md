# 09 — Verified facts, measurements and dead ends (2026-09-26 session)

Everything in this file was either **fetched from an official source this session**
(link given, quote given) or **measured in this sandbox** (command given, output given).
Nothing here is inferred from a model's own output without the measurement attached.
Where a claim from a previous session was found to be wrong, the correction is recorded
in §7 rather than silently dropped.

---

## 1. The metric, verbatim, and one algebraic consequence

Source: <https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/>
(fetched 2026-09-26, "Scoring" section, quoted exactly):

```
k(d) = max(1 − d/R, 0),   R = 300 m = 3 px at 100 m resolution

TP_w = Σ_{g∈G}  max_{x: d(x,g)≤R}  p(x)·k(d(x,g))
FP_w = Σ_{x: p(x)>0}       p(x)·[1 − max_{g∈G} k(d(x,g))]
FN_w = Σ_{g∈G}        [1 − max_{x}  p(x)·k(d(x,g))]

DTI  = TP_w / (TP_w + 0.2·FP_w + 0.8·FN_w + ε)
```

**Budget algebra (exact for binary p∈{0,1}, one-to-one matching).** If an emission
`E` (chargeable pixels, i.e. off-catalogue) is binary and every truth pixel is served by
at most one emitted pixel with `k≈1`, then `FP_w ≈ E − TP_w` and `FN_w ≈ G − TP_w`, so

```
DTI ≈ TP_w / (0.2·E + 0.8·G)          (TP_w coefficient cancels: 1 − 0.2 − 0.8 = 0)
```

Consequences that drive every decision in this repo:

* DTI is **linear in covered truth** at fixed budget — precision is the only lever.
* Adding one pixel with kernel-weighted hit probability `q` changes DTI by
  `(q − 0.2·DTI)/(0.2E + 0.8G)` ⇒ **emit iff `q > 0.2·DTI`**. At DTI = 0.1563 the
  break-even is `q > 3.13 %`; the base rate of truth in the footprint is ≈0.43 %
  (G≈22 k of 5.11 M px), so **random expansion always lowers the score**.
* Caveat, stated honestly: `FP_w = E − TP_w` is an approximation (an emitted pixel 1 px
  from truth pays 0.33, not 0). The sibling repository's 5-score fit using exactly this
  model class could **not** reproduce the board (see §4), so treat the algebra as a
  decision rule for *marginal* pixels, not as an absolute-score predictor.

## 2. The masking rule (the single most consequential verified fact)

Source: forum thread
<https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516>
(staff answers, fetched 2026-09-26 via the JSON endpoint `…/t/11516.json`):

* **post 2** — pixels of the supplied USGS/INGENIOUS catalogue are masked / excluded from
  evaluation, and they are also excluded from the *final round* label set.
* **post 4** — (a) the mask is **pixel-exact**: it is the training label raster itself;
  (b) a pixel that is near a known fault but far from real ground truth is **fully
  penalised**; (c) ground truth **may lie within 300 m of a known fault** ("corrections").

Source: <https://community.drivendata.org/t/11536> post 2 — a "new fault" is any fault
pixel **not** in USGS/INGENIOUS, and may include **new geometry of an already-mapped
system**.

**The weak-dominance theorem used for the recommended upload.** From (a) and the FP_w
definition, emitting `p>0` on a catalogue pixel adds **exactly 0** to `FP_w`. From (c) and
the linearity of `TP_w` in `p`, it can only **add** to `TP_w`. Hence for any emission `A`:

```
DTI(A ∪ catalogue) ≥ DTI(A)      (equality iff no truth pixel lies on the catalogue)
```

Measured implementation: `scripts/build_hedge_v2.py` → `reports/hedge_v2.json`.
Base = byte-verified `7f00890a…` (public **0.1563**), union = **227,507** px
(+54,533 catalogue px, chargeable px unchanged at **166,519**), 13/13 format gates PASS,
sha256 `052688eafbc7a55d90caab72252a50304fe5caa8ecf973f8e791b4ae910f0792`.
Independent cross-check: pixel-identical (0 differing px) to a candidate built by a
different code path in the sibling repo (`candidate_s5_catalogue_hedge.tif`, `132e23e1…`).

## 3. Verified leaderboard anchors (bytes ↔ public score)

All five files were re-hashed this session with rasterio (`tifffile` cannot read them:
LZW + floating-point predictor need `imagecodecs`, which is not installed here).

| sha256 (first 8) | file | public score | account | positive px | on catalogue px |
|---|---|---|---|---|---|
| `7f00890a` | `gemsdoe-ens12-adopted` | **0.1563** | extradr19 | 172,974 | 6,455 |
| `f68e590f` | `gemsdoe2-dual-family-union` | 0.1560 | smashi34 | 183,642 | 7,693 |
| `f347b70d` | `pindrop-v4-nodes` | 0.1193 | smrtdoog5 | 155,021 | 0 |
| `4e03fc97` | `pindrop-v4-ridge` | 0.1152 | SDCF9 | 155,021 | 0 |
| `37f9d5b8` | `pindrop-v4-discovery` | 0.0830 | wbg1 | 155,021 | 0 |
| `33cec71f` | `gems6_hgb88-topk03` | **0.0286** | — | 155,021 | 23,605 |
| `237f0063` | `GEMSDOE4 combined` | 0.0343 | — | 264,247 | 5,924 |

**Correction to a previous session's claim.** The 0.1563 artifact is **binary {0,1}
float32** (`np.unique` = `[0., 1.]`), *not* a graded 0.55/0.75/1.0 emission. The graded
uint8 file `downloads/8GEMSDOE-Apex-Geothermal-V1.tif` (`b83ea0e7…`, 427,862 px) is a
different, locally built artifact and was never on the board.

Board context (fetched 2026-09-26): #1 **DARD 0.3049**, #2 alexoktaba 0.2993,
#3 HardcoreTechGod 0.2854; top-5 cutoff 0.2589.

**What 0.3049 requires, quantitatively.** With `G ≈ 22,341` and `E = 155,021` (3 %),
`DTI = TP/(0.2E + 0.8G)` ⇒ `TP ≈ 14,900`, i.e. **67 % of all withheld truth pixels covered
at a 3 % budget — ≈22× the base rate**. The verified 0.1563 pattern covers ≈8,000 px
(≈36 %, ≈11× base rate). So the gap to the top of the board is *a factor of two in
enrichment*, not a threshold tweak.

## 4. Dead end, measured: the "hidden prior" fit cannot rank artifacts

The sibling repo (`5GEMSDOE`, `data/evidence/hidden_prior/fit.json`) fits a truth density
`w` over four covariates (uniform, `d_lab≤1`, `1<d_lab≤3`, salience) to the five public
scores, giving `w = (0, 0.1217, 0, 0)` and `G = 22,341` px — i.e. *all* withheld truth
within 100 m of the catalogue. Its own diagnostics refute it as an instrument:

* `partial_identification.smallest_possible_tol = 0.0278` — **no density in that model
  class reproduces the five scores it was fitted to** within ±0.028 (the scores themselves
  span 0.083–0.1563).
* The implied range for `37f9d5b8` is 0.1304–0.1330, but it actually scored **0.0830**.
* Leave-one-out errors 0.008–0.038.

Recorded conclusion: **five board scores cannot identify a truth-density field**, and a
model class that cannot fit its own five numbers cannot rank a sixth artifact. The
`G ≈ 22 k` figure is usable only as an order-of-magnitude prior in §3's algebra.

## 5. Dead end, measured: in-file features are at or below random for *new* traces

Protocol (built this session, `src/gems/pseudo_truth.py` + `scripts/lofso_train_eval.py`):
hold out **whole fault systems** (link distance 3 px, so no held-out pixel is within 3 px
of the visible catalogue), retrain a HistGradientBoosting classifier **inside each fold**,
emit the top 3 % of unmasked pixels, score with the exact kernel metric against 15,000
pseudo-truth pixels.

Fold 0, budget 3 % (`reports/lofso_train_eval.json` run was stopped after fold 0; the
external-channel run `reports/lofso_external.json` reproduces RAND/BASE identically):

| arm | DTI | hit rate |
|---|---|---|
| **RAND (random emission)** | **0.0813** | 0.0228 |
| BASE (19 bands + gradients + mean5) | 0.0727 | 0.0199 |
| BASE + H11 family | 0.0705 | 0.0193 |
| HALO r=1 / 2 / 3 / 5 around the catalogue | 0.0272 / 0.0369 / 0.0417 / 0.0437 | 0.008–0.010 |

### 5.1 The external channels change the answer (3 folds, `reports/lofso_external.json`)

Same protocol, same folds/seed/budget, 62-plane stack (46 in-file + 9 topo + 7 radiometric).
`AUC` is measured against each fold's held-out truth on a seeded 300,000-px sample.

| arm | fold 0 | fold 1 | fold 2 | **mean** | vs RAND | AUC |
|---|---|---|---|---|---|---|
| **BASE+C+T+R** | 0.0895 | **0.1000** | 0.0904 | **0.0933** | **+13.7 %** | **0.7885** |
| BASE+T+R | 0.0850 | 0.0947 | 0.0896 | 0.0898 | +9.4 % | 0.7843 |
| BASE+T | 0.0810 | 0.0905 | 0.0816 | 0.0844 | +2.8 % | 0.7833 |
| **RAND (control)** | 0.0813 | 0.0822 | 0.0827 | 0.0820 | — | 0.5 |
| BASE+R | 0.0762 | 0.0823 | 0.0770 | 0.0785 | −4.3 % | 0.7715 |
| BASE | 0.0727 | 0.0841 | 0.0713 | 0.0760 | −7.4 % | 0.7714 |
| HALO r=1…5 | 0.027–0.044 (all folds) | | | 0.0283–0.0438 | −65 %…−47 % | n/a |

Facts worth keeping:

* **BASE+C+T+R beats random on all three folds** (+10.1 %, +21.7 %, +9.3 %) — the first arm
  in this project to do so for *unseen* traces. `C` is the in-file structural-coherence
  family (7 planes), `T` the 10 m scarp channel, `R` the radiometric channel.
* **Radiometric alone hurts** (−4.3 %) but pays in combination with the scarp channel
  (+0.0054 mean over BASE+T). Do not use `R` by itself.
* **All model arms have real ranking skill** (AUC 0.771–0.789 ≫ 0.5), and the external
  channels raise AUC monotonically. The gap between AUC 0.79 and "+14 % DTI" is explained by
  the metric's own kernel: a *random* 3 % emission already lands within 3 px of ≈59 % of the
  pseudo-truth pixels (measured: RAND TP_w = 3,537 of 15,000), so ranking skill only buys the
  remaining margin. **Consequence: at a 3 % budget this task is near a coverage plateau, and
  the way to gain is a *sparser, more linear* emission (traces), not a better-ranked blob.**
* Fold truth composition: 15,000 px per fold, all from the "whole held-out systems" regime;
  truth pixels adjacent to the visible catalogue were 25 / 42 / 102 (0.2–0.7 %), i.e. the
  protocol is clean but not artificially sterilised.

Two more measured facts from the same harness:

* **`d_known` (distance-to-catalogue) as a feature is toxic** under this regime: every
  feature set collapses onto the same 1 px catalogue ring, DTI 0.0088. It is now opt-in
  (`--use-dknown`, default off).
* **`predict_proba` saturates** to 1.0 in float32, making top-k ties degenerate to raster
  order; the harness uses `decision_function` instead. (This silently invalidated an
  earlier comparison — re-measured.)

Interpretation, stated plainly: *the 19 supplied bands plus our 27 derived planes do not,
**on their own**, contain enough information to find unmapped fault traces better than
chance at a 3 % budget* (BASE mean 0.0760 vs random 0.0820). The correction-halo hypothesis
is falsified here too (halos score 2–3× **worse** than random). §5.1 shows what does move
the needle: the external channels, and the in-file coherence family combined with them.

## 6. Verified external data: what exists, where, how big, and what it covers

### 6.1 10 m topography (the channel that measurably helps)

* Source: USGS 3DEP 1/3 arc-second (≈10 m) seamless DEM, cloud-optimised VRT:
  `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt`
  (free, official, USGS public domain; nationwide coverage).
* 9 derived bands, aggregated to the exact competition grid:
  `slope_p90, slope_max, steep_frac, slope_std, relief_local, curv_prof_absmax,
  aspect_coherence, hs_lineament, dem_mean`.
* Transport: sha256-pinned git bridge (`data/aux_bridge/topo`, whole-file sha256
  `a6398d9950965dec6aae6ccecdaa6ced48645d133eab222cbdd11def9bdabfa4`, 32,523,329 B,
  uint8 254-level quantisation with per-band p0.5/p99.5 ranges recorded in the manifest).
  Fetch + verify + unpack: `bash scripts/fetch_aux_bridge.sh topo radiometric`.

### 6.2 Airborne radiometrics

* Source: **GeoDAWN** — Glen, J.M.G., and Earney, T.E., 2024, *GeoDAWN: Airborne magnetic
  and radiometric surveys of the northwestern Great Basin, Nevada and California*, U.S.
  Geological Survey data release, <https://doi.org/10.5066/P93LGLVQ>.
  ScienceBase item `657e1d85d34e23d3533209f7` (fetched 2026-09-26). Attached files
  include `22103_area1_tiffs.zip` (43.57 MB) and `22103_area2_tiffs.zip` (230.54 MB) —
  the release text says these contain "geoTIFF images of geophysical grids" — plus
  `GeoDAWN_ReadMe.pdf` and the contractor report
  `GeoDAWN_NV_WestCentral_Geophysical_2020_D21_Report.pdf` (7.28 MB).
* 7 bands on-grid: `rad_k, rad_th, rad_u, rad_tc, rad_thk, rad_uk, rad_uth`
  (bridge sha256 `6cb051f70f94…`, 25,475,158 B).
* Survey facts verified from the release abstract: 149,030 line-km over 51,857 km², flown
  Nov 2021–Nov 2022, Area 1 line spacing 200 m at ~100–150 m clearance, Area 2 spacing
  400 m at ~150–200 m clearance; radiometrics corrected for aircraft/cosmic background,
  radon, Compton scattering and altitude. **The release also states that airborne LiDAR was
  collected over a similar extent through the USGS 3DEP program.**

### 6.3 1 m LiDAR — obtainable, enumerated, heavy

* 716 tile URLs enumerated against the authoritative public bucket listing (lineage:
  OCR of the competition's `Digital-elevation-model-links-JSON.pdf` → resolved to
  `StagedProducts/Elevation/1m/Projects/…` keys; `n_unique_tiles_confirmed = 716`,
  projects: `NV_WestCentral_EarthMRI_2020_D20` 555, `NV_EastCentral_2021_D21` 130,
  `CA_SierraNevada_B22` 22, `NV_Humboldt_2021_D21` 9).
* **Verified live this session** by anonymous `ListObjectsV2` (no credentials):
  `…/1m/Projects/NV_WestCentral_EarthMRI_2020_D20/TIFF/USGS_1M_11_x49y451_NV_WestCentral_EarthMRI_2020_D20.tif`
  → `Size 185,344,605`, `LastModified 2026-02-14T02:55:56Z`, `StorageClass STANDARD`,
  ETag `50e10d3f9ed6eec10ddbe2d9e7b213d0`.
* ⇒ ≈**130 GB** for full coverage at ~180 MB/tile. Requires a machine with normal egress
  and a large disk; a 20-tile subset (≈4 GB) is the realistic first step.

### 6.4 Coverage of the aux channels — and an irregularity in our own census

**Measured coverage inside the true footprint** (re-measured with the corrected footprint
definition, `isfinite(labels) & labels >= 0`):

| channel | bands | coverage of the 5,167,373 px footprint |
|---|---|---|
| topo / scarp | 8 of 9 | **100.000 %** |
| `aspect_coherence` | 1 | 98.924 % |
| radiometric | all 7 | **99.975 %** |

This agrees with the producer's own runner reports
(`data/evidence/external_build/topo/topo_features_100m.report.json`:
`finite_px_band1 = 5,167,373`, `coverage_of_footprint = 1.0`, all nine bands
`finite_px = 5,167,373`; `…/radiometric/radiometric_channel.report.json`:
`coverage_of_footprint = 0.9998`, `area2_px_in_footprint = 5,166,085`).

**Irregularity, caught and fixed (our bug, not the data's).** The first version of
`scripts/build_aux_planes.py` defined the footprint as `np.isfinite(labels)` and therefore
reported "42.082 % coverage" for both channels. `data/labels.tif` is finite on **all
12,279,160** pixels and carries the sentinel `-1` outside the scored area, so that mask
selected the whole raster: 5,167,373 / 12,279,160 = **42.082 %** — the reported number was
the footprint's share of the raster, not the channel's share of the footprint. Fixed to
`isfinite & >= 0` (the definition `scripts/prepare_data.py` uses) with an assert on the
5,167,373 pixel count. **Any note, page or JSON in this repo written before the fix that
says "42 % coverage" is wrong and has been corrected.**

Consequence for strategy: the scarp channel needs **no** coverage rebuild. The remaining
work on it is the *detector*, not the data (see `knowledge/10` H-A).

Grid verification performed on both rasters: 3292×3730, EPSG:32611, transform
`(100, 0, 243350, 0, −100, 4508550)` — exact match to `data/training_features.tif`.
Footprint in WGS84 (computed with `rasterio.warp.transform_bounds`):
`[-120.0372, 37.3312, -116.1409, 40.7279]`.

## 7. Sandbox egress facts (measured, so nobody re-derives them)

| host | method | result |
|---|---|---|
| `github.com` | `git clone` | **works** (this is how the aux bridge travels) |
| `pypi.org` | `pip install` | **works** |
| `prd-tnm.s3.amazonaws.com` | `curl` | **fails**, `curl: (35) OpenSSL SSL_connect: SSL_ERROR_SYSCALL` |
| `www.sciencebase.gov`, `drivendata.org`, `openei.org` | `curl` | fail the same way |
| `drivendata.org`, `community.drivendata.org`, `www.sciencebase.gov`, `prd-tnm.s3.amazonaws.com` | the agent's page fetcher | **works** (different network path) — used to verify §6 |

Irregularity to flag: the USGS **TNM Access API**
(`https://tnmaccess.nationalmap.gov/api/v1/products?datasets=USGS%203DEP%201%20m&bbox=…`)
returned `"total": 0` for both the 1 m and 10 m datasets over this bbox, with the message
"The offset is greater than the total number of results for this query" at `offset=0`.
The same data is demonstrably present in the public S3 bucket (§6.3). **Do not use the TNM
API as an availability check; use `ListObjectsV2` on `prd-tnm`.**

Hardware: 2 CPUs, 3.9 GB RAM, 21 GB disk (14 GB free at session end), no GPU.
Performance facts measured: a per-arm random gather over the 2.26 GB plane memmap costs
≈10 min/arm-fold; one sequential pass building a 126k×62 sample matrix costs 11 s.
`scipy.ndimage.generic_filter` for windowed std is hours-slow; use
`sqrt(E[x²] − E[x]²)` box filters.

## 8. Corrections to earlier sessions' claims

| earlier claim | status | evidence |
|---|---|---|
| "hardening the emission strictly increases DTI" | **false** | §1 algebra: DTI is linear in covered truth; hardening only changes the chargeable set |
| "draft 0.58 structural skill" | **false (leakage)** | trained and scored on the same catalogue pixels |
| "magnetic source depth band exists" | **false** | band list is the 19 named bands; no depth-to-source product |
| "binary emission is always optimal" | **false as stated** | graded emission can be optimal when `q` varies; only the *marginal* rule in §1 is exact |
| "apex (b83ea0e7) is the first-slot candidate" | **retired** | §3 + §5: it puts all 60,988 catalogue px plus 366,874 chargeable px into the emission; the artifact family that hugs the catalogue hardest scored 0.0286 on the board. Expected ≈0.03–0.06 |
| "0.1563 = graded 0.55/0.75/1.0 pattern" | **false** | the verified file is binary {0,1} (§3) |
| "LOFSO on existing artifacts ranks submissions" | **false** | `hgb88` ranks first on LOFSO (0.1527) and last on the board (0.0286); the artifacts were trained on the catalogue that LOFSO uses as truth |

## 9. Reproduce everything in this file

```bash
bash scripts/fetch_aux_bridge.sh topo radiometric      # §6.1, §6.2 (sha256-verified)
python scripts/build_aux_planes.py                     # §6.4 coverage census
python scripts/build_hedge_v2.py                       # §2 artifact + cross-check
python scripts/validate_submission.py downloads/8GEMSDOE_Hedge-v2_submission.tif   # 13 gates
python scripts/lofso_train_eval.py --planes-stem planes_ext --folds 3 \
    --sets BASE,BASE+T,BASE+R,BASE+T+R,BASE+C+T+R --budgets 0.03 \
    --regime 1,0,0 --n-truth 15000 --restrict-aux --out reports/lofso_external.json  # §5
python scripts/calibrate_protocol.py --quick --folds 1  # §4-style contamination check
```

## Appendix — provenance discipline (added after PR #8 merge)

- **Sibling "identical bytes" is an inference, not an observation.** Verified: the artifacts
  published in GEMSDOE, 5GEMSDOE and GEMSDOE2's repos are byte-identical to each other
  (`7f00890a…`, payload `cb2d2d5e…`, 172,974 px, read with rasterio — `tifffile` cannot open
  them, flag F16). Not verified: that DrivenData received those bytes, because the platform
  does not expose uploaded files. The score identity 0.1563 = 0.1563 follows *if* the uploads
  matched; the shared published hash plus the identical board score make that the parsimonious
  reading, and it is labelled as such wherever it appears (README §duplicate,
  `leaderboard.html`, flag F10).
- **CI reproduces locally without the data.** `git archive HEAD | tar -x -C /tmp/ci_sim` then
  `pytest` gives exactly the CI file set (tracked files only) and reproduced PR #8's failure in
  7 s when the GitHub log host was unreachable. Use this whenever a CI failure cannot be read.
- **Only manifests are committed for `data/aux_bridge/*`; parts are gitignored.** Any integrity
  test over the parts must skip when they are absent and say how to fetch them.
