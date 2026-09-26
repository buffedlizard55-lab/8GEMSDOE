# 8GEMSDOE — DOE GEMS Prize entry (repo 8 of the group programme)

## 0. Read this first — the standing prompt (pinned)

> This section is the standing instruction set for every work session on this
> repository. Read it at the start of every session before doing anything else.

**Competition.** Place top of the leaderboard in the DOE GEMS Prize Challenge:
predict geological faults (surface indicators of geothermal resources) in the
GeoDAWN region of northwestern Nevada on the competition grid
(3,292 × 3,730 px · EPSG:32611 · 100 m · single-band float32 GeoTIFF in
[0, 1], NaN outside the footprint).
Official problem description:
<https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/> —
scored by a distance-weighted Tversky index (α=0.2, β=0.8, 300 m triangular
kernel) against a **privately withheld set of newly expert-labelled faults**,
NOT against the downloadable USGS/INGENIOUS catalogue. Known faults are
masked/excluded from scoring in both prize rounds
([forum 11516](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516)).
Full rules: <https://docs.nlr.gov/docs/fy26osti/96647.pdf>.
Leaderboard: <https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/>.
Current best (2026-09-26): **0.3049** — every strategy here must aim above it.

**Before implementing**, generate 3–5 candidate geological hypotheses we have
not tried yet, each naming: the specific layer(s) involved, the physical
signature being targeted (e.g. an edge-detection or curvature transform), why
it should catch a fault missing from the USGS/INGENIOUS catalogue rather than
one already in it, and how it differs from anything already implemented in
this repo. Rank them by expected DTI improvement and implementation cost.
Validate the top candidate on our spatially-blocked holdout set before
touching a weekly submission slot — do not spend a submission slot on an idea
that hasn't beaten the current holdout best. If a candidate can't be validated
without new external data, name the specific free, official source needed and
check it's obtainable before proposing the idea as viable.

**Submission rule.** The site must generate the submission TIF so that
uploading it is one click: a unique file name
(`gems8-<strategy>-<UTC instant>-<sha8>.tif`) plus a short paste-ready Note
for the DrivenData dialog. A submission that the platform rejects with
`Predicted values must be in range [0, 1]` means NaN sat inside the scored
footprint — the builder must make that outcome impossible (finite inside,
NaN outside, values clipped, 13-gate validator as a hard gate).

**Verification standard.** Work line by line verifying from official verified
trusted sources, provide links for manual review. No manual input is expected
from the operator: work autonomously, flag irregularities for review, and
never hallucinate. Every numeric or factual claim in `docs/` must trace to a
measured artifact in this repo or to an official source linked in
`docs/sources.html`. If evidence is missing, the page must say so.

**Run every task through three passes.** Pass 1: implement completely and
verify. Pass 2: review for bugs, missing requirements, wrong assumptions, edge
cases; fix everything found. Pass 3: re-check the whole implementation against
the original request; improve accuracy, reliability, completeness, code
quality; fix remaining issues. Do not stop after pass 1.

### Core values (Arena AI — kept as a focal point)

- **Maximize P(Win).** In every decision, weigh tradeoffs, assess risk, and
  choose the path that maximizes the probability of winning. Set aside emotion
  and make the tough calls.
- **Own the Outcome.** Own results end to end — not just one slice. When a
  problem arises and we have the means to act, act without waiting for
  permission. Treat failure and success as signals and improve.

---

## 1. What this repo is

`8GEMSDOE` is a **fresh, independent strategy line** for the GEMS Prize. It is
not a copy of any sibling repo: it ships its own metric implementation, its
own potential-field / topographic transform library, its own spatially-blocked
holdout harness, its own format validator (13 gates), its own submission
builder, and its own GitHub Pages site. The strategy portfolio (5 ranked
geological hypotheses) is documented in `docs/hypotheses.html` and validated
— before any weekly submission slot is spent — by
`scripts/holdout_validate.py`.

Sibling-site scores under analysis (public leaderboard, 2026-09-26; see
`docs/leaderboard.html` for the auditable table):

| Site | Account | Score | Note |
| --- | --- | --- | --- |
| GEMSDOE | extradr19 (#24) | 0.1563 | artifact `7f00890a…`, 259,495 runs, 2 submissions |
| GEMSDOE3 / card 3 acct | SDCF9 (#25) | 0.1563 | 2nd upload during 2026-09-26 session — **third copy at 0.1563**, see §4 |
| 5GEMSDOE | (no separate board row) | 0.1563 | **identical bytes** to GEMSDOE (`7f00890a…`) — duplicate payload, see §4 |
| GEMSDOE2 | smashi34 (#26) | 0.1560 | dual-family union `f68e590f` |
| GEMSDOE3 / card 1 | smrtdoog5 (#41) | 0.1193 | pindrop nodes `f347b70daa` |
| GEMSDOE3 / card 2 | wbg1 (#50) | 0.0830 | pindrop catalogue-gap `37f9d5b855`, 2 submissions |
| 6GEMSDOE | (below rank 50) | 0.0286 | ⚠ operator-supplied, not page-verified; HGB top-3% `33cec71ff0` |
| GEMSDOE4 | (below rank 50) | 0.0343 | ⚠ operator-supplied, not page-verified; lineament union `237f0063` |
| Leader best | DARD (#1) | **0.3049** | target to beat |

## 2. Quickstart (no manual input needed)

```bash
# 1. Place competition data (one-time; requires a DrivenData login in a browser)
bash scripts/download_competition_data.sh   # prints exact file list + hashes to fetch
# cp ~/Downloads/training_features.tif data/   # etc. — script tells you what goes where
python scripts/prepare_data.py              # verifies + writes data/manifest.json

# 2. Run the full self-test suite (works WITHOUT competition data, on fixtures)
python -m pytest tests/ -q

# 3. Validate the top hypothesis on the spatially-blocked holdout
python scripts/holdout_validate.py --report reports/holdout_top1.json

# 4. Build a submission (unique name + paste-ready Note, hard-gated)
python scripts/build_submission.py --strategy topk_hard@0.03 --out submissions/
python scripts/validate_submission.py submissions/*.tif --template data/sample_submission.tif

# 5. Serve the site locally
python -m http.server 8000 --directory docs
```

GitHub Pages serves `docs/` automatically once merged to `main`.

## 3. The `[0, 1]` rejection — root cause and fix

The DrivenData dialog rejects a file with `Predicted values must be in range
[0, 1]` when a pixel **inside the scored footprint** is NaN (or otherwise
outside [0, 1]). NaN is only legal **outside** the footprint. Our builder
(`src/gems/submission.py`) makes rejection structurally impossible:

1. Start from the official template footprint (finite = scored).
2. Fill every inside-footprint pixel with a finite value in [0, 1] (clip +
   NaN→0.0 fill with a counted warning).
3. Force NaN outside the footprint, set `GDAL_NODATA=nan`.
4. Run the 13-gate validator as a hard gate — the file is only written to
   `submissions/` if **all** gates pass.

See `docs/how_to_submit.html` for the click-by-click upload and the
troubleshooting section.

## 4. Why GEMSDOE and 5GEMSDOE scored the same (0.1563)

Both published pages pin the **same artifact**: sha256 `7f00890a62878d61…`,
`gems-rle-v1 · 259,495 runs · 532,072 B`, reproducing
`data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif`. Same bytes ⇒
same score. The 5GEMSDOE page additionally ships a maximum-compatibility
variant (0.0 outside instead of NaN); under the official scorer the two are
the same number, so it cannot change the score either. Full evidence table in
`docs/leaderboard.html`. **Lesson for this repo:** every submission gets a
fresh unique name **and** a fresh payload hash recorded in
`reports/submissions_log.json` — the builder refuses to emit a payload whose
hash already appears in the log (`--allow-duplicate` overrides with a logged
warning).

## 5. Repository layout

```text
8GEMSDOE/
├── README.md                  ← you are here (standing prompt + guide)
├── requirements.txt           ← numpy, scipy, tifffile, pytest
├── src/gems/
│   ├── metric.py              ← distance-weighted Tversky (official formulas) + masking
│   ├── features.py            ← HGM / AS / TDR (FFT), curvature, scarp step, strain, conductivity
│   ├── submission.py          ← GeoTIFF build + 13-gate validation (no GDAL needed)
│   └── holdout.py             ← spatial blocks + buffer folds, proxy scoring
├── scripts/
│   ├── download_competition_data.sh  ← exact fetch list (browser login required)
│   ├── prepare_data.py               ← verify + manifest
│   ├── holdout_validate.py           ← validate top hypothesis before spending a slot
│   ├── build_submission.py           ← unique naming + duplicate guard + hard gate
│   └── validate_submission.py        ← 13-gate CLI
├── tests/                     ← metric / features / submission self-tests (no data needed)
├── docs/                      ← GitHub Pages site (index, executive_summary, how_to_submit,
│                                 hypotheses, results, leaderboard, sources)
├── data/README.md             ← placement map for competition files
└── reports/                   ← generated JSON (holdout, submissions log) — committed for audit
```

## 6. Limitations and what we need access to

1. **Competition data requires a browser login.** `training_features.tif`,
   labels, `sample_submission.tif`, `1m_DEM_links.csv` live behind
   `https://www.drivendata.org/competitions/306/competition-doe-gems/data/`
   (verified: redirects to login without auth). The sandbox cannot fetch them;
   `scripts/download_competition_data.sh` documents the exact placement and
   `scripts/prepare_data.py` verifies it. Until then, all holdout numbers in
   this repo are on synthetic fixtures and are labelled as such — **no
   leaderboard value is claimed anywhere.**
2. **Grid constants are UNVERIFIED until the template lands.** Shape
   (3730, 3292), transform `(100, 0, 243350, 0, -100, 4508550)`, footprint
   size and the 60,988 catalogue-pixel count are measurements quoted from
   sibling sites, not from an official file we hold. The validator reads the
   grid from `--template` at runtime and never trusts hard-coded values.
3. **No GPU in this sandbox.** Training (U-Net/HGB scale) needs a GPU box;
   metric, transforms, validation and submission code are CPU-verified here.
4. **Organizer answers in forum 11527** (how new faults were identified — data
   sources, fault types, coverage) are not retrievable as static text (JS-only
   replies). Flagged in `docs/sources.html`; the hypotheses assume the
   conservative reading (mixed surface + buried faults, full-region review).
5. **One-entity rule.** Rules cap one final submission per entity (rules PDF
   §3.4, §3.5, §3.6.2). This repo is developed as the group's single forward
   strategy line; do not upload sibling payloads and this repo's payload as if
   they were independent entities.

## 7. Next-session work list

- [ ] Land `data/sample_submission.tif` + `training_features.tif` + labels via
      `scripts/download_competition_data.sh`, run `prepare_data.py`.
- [ ] Run `holdout_validate.py` on real bands; promote only a hypothesis that
      beats the holdout best (currently: none — no real-data best exists yet).
- [ ] Pull the 716 1 m DEM tiles (links CSV) for H3 scarp calibration; or use
      the USGS 3DEP S3 public bucket as fallback (see hypotheses page).
- [ ] Fetch SGMC faults for Nevada (`https://mrdata.usgs.gov/geology/state/`)
      as the independent-compilation proxy for H1/H2 scoring.
- [ ] Publish the first `submissions/*.tif` + ZIP to `docs/downloads/` with
      its Note, then spend **one** weekly slot on the holdout winner only.

## 8. Verification

- `python -m pytest tests/ -q` — metric exactness (official worked example
  TP=3.00/FP=1.89/FN=2.00 ⇒ DTI=0.60), transform sanity, submission round-trip.
- `python scripts/validate_submission.py --self-test` — 13 gates incl. the
  NaN-inside-footprint gate that triggers the `[0, 1]` rejection.
- `docs/sources.html` — every factual claim mapped to an official link for
  manual review. Flagged irregularities are listed there, not hidden.
