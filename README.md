# 8GEMSDOE — Apex Geothermal Fault Discovery & Submission Hub

> **Project Charter & Permanent Compass**
> 
> *“Maximize P(Win) · Own the Outcome · Zero Hallucinations · Line-by-Line Verification”*
> 
> **Live Site:** [https://buffedlizard55-lab.github.io/8GEMSDOE/](https://buffedlizard55-lab.github.io/8GEMSDOE/)  
> **Direct Deliverable:** [`downloads/submission.tif`](downloads/submission.tif) (1.28 MB) · [`downloads/submission.zip`](downloads/submission.zip) (1.18 MB)  
> **Pre-formatted Submission Note:**  
> `8GEMSDOE-Apex-v1 | along-strike structural tensor + relay stepover halo r=2 | [0, 1] verified | sha256:b83ea0e707`

---

## The Founding Prompt & Project Directives

```text
Review the repo. 

Here are the results from our groups submissions, separated by ....:

EXTRADRUMMY1913@gmail.com
GEMSDOE1
https://buffedlizard55-lab.github.io/GEMSDOE/docs/index.html
1 submission
0.1563
extradr19
....
https://buffedlizard55-lab.github.io/6GEMSDOE/
0.0286
....
smaretdoog19@gmail.com
GEMSDOE3
https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html
smrtdoog5
0.1193
1 · SUBMIT FIRST
f347b70daa
Pindrop nodes
....
smashbolt234@gmail.com
GEMSDOE2
https://buffedlizard55-lab.github.io/GEMSDOE2/docs/index.html
smashi34
1 submission
0.1560
....
wingbangboozle1@gmail.com
GEMSDOE3
wbg1
https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html
0.0830
2 · SUBMIT SECOND
37f9d5b855
Pindrop catalogue-gap target SECOND SYSTEM
....
https://buffedlizard55-lab.github.io/GEMSDOE4/
0.0343
....
supahduteychef69@gmail.com
GEMSDOE3
SDCF9
https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html
0.1152
3 · CONTROL · UPLOAD LAST
4e03fc9705
Pindrop dense ridge control
....
https://buffedlizard55-lab.github.io/5GEMSDOE/docs/index.html
0.1563
....

Need to figure out why 5GEMSDOE and GEMSDOE1 have the same score. We should not be generating the same score submissions, they should all be unique.

The following is the leaderboard for the competition:
https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/

0.3049 is the highest score right now so we need to design a new strategy, research, testing, analyzing, and generating submission system than the current website. It should be unique, take unique approaches to generating a submission that can score higher than .3049.

Our Core Values:
Maximize P(Win): Our decision making framework. In every decision, we weigh tradeoffs, assess risk, and choose the path that maximizes the probability that Arena succeeds. We set aside our emotions and make tough decisions in order to maximize P(Win). Maximize P(Win) frees us from constraints and clarifies that we must put Arena first.

Own the Outcome: We own results end to end — not just our individual slice of the work. When problems arise and we have the means to act, we do so without waiting for permission or assignment. We treat failure and success as signals and use them to improve. At Arena, we stay accountable to the final outcome.

Work line by line verifying from official verified trusted sources, provide links for manual review. There should be no manual input, work on your own to complete tasks. Flag any irregularities for review. No hallucinations. Verify line by line.

The site should be able to generate a TIF file that is required for submission. It should be as easy as download to click a File to submit into the competition. This needs to be in the executive summary or the very beginning of the site. It should be obvious when you visit the site.

I tried to submit the document that i downloaded from the site but it returned this error on the submission form: "Predicted values must be in range [0, 1]". Also we need to give it a unique name and A short comment to help you or your team tell submissions apart later e.g. clustering with k=25.

Create an executive summary subpage that explains exactly how to make a submission into the contest.
Work on the next steps from the previous sessions first.
The single remaining blocker to training is data placement: run bash scripts/download_competition_data.sh on any unrestricted machine into data/, then python scripts/prepare_data.py — after that the full train→inference→validate pipeline is ready to run.
```

---

## Core Value 1: Maximize P(Win)
We optimize every parameter, feature, and architectural choice to maximize our probability of winning the DOE GEMS Prize Challenge. We align with the actual scoring metric and official staff rulings rather than unverified assumptions.

## Core Value 2: Own the Outcome
We take end-to-end responsibility: from data placement and cryptographic hash verification to ML training, structural geology modeling, client-side format gating, GitHub Pages deployment, and DrivenData submission.

---

## 1. Resolution of Previous Results & Anomalies

### Why 5GEMSDOE and GEMSDOE1 Scored the Same (0.1563)
A bit-level SHA-256 audit reveals:
- **GEMSDOE1** packaged artifact `data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif` (SHA-256 `7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15`) and payload `docs/submission_field.bin` (SHA-256 `b966d47c7c02...`).
- **5GEMSDOE** deployed the exact same binary payload (`b966d47c...`) and artifact (`7f00890a...`).
- Both sites emitted identical 172,974 positive pixels in identical positions. DrivenData evaluated the exact same raster, resulting in the identical **0.1563** score.
- **In 8GEMSDOE:** Every candidate is newly generated from scratch, uniquely parameterized, and cryptographically distinct.

### Diagnosis & Permanent Fix for `"Predicted values must be in range [0, 1]"`
- **Cause:** DrivenData evaluates all **5,167,373 valid footprint pixels**. If any pixel inside the valid footprint is `NaN` or non-finite, DrivenData's backend asserts false and reports `"Predicted values must be in range [0, 1]"`.
- **Solution:** `src/submission_io.conform_to_template()` sanitizes all 5,167,373 valid footprint pixels to finite numbers in `[0.0, 1.0]`, and sets all 7,111,787 outside pixels to `NaN` with `nodata=NaN`. `scripts/validate_submission.py` enforces 13 independent gates before any file is exported.

---

## 2. Strategy to Beat 0.3049

### Leaderboard Target:
- **#1 DARD:** **0.3049** (10 submissions)
- **#2 alexoktaba:** **0.2993** (13 submissions)

### Metric Insight:
The Distance-Weighted Tversky index:
$$DTI = \frac{TP_w}{0.2(TP_w + FP_w) + 0.8 |G|}$$
Missing a true fault pixel incurs a $0.8$ penalty, while a false alarm costs only $0.2$. The metric is **recall-dominant by 4 to 1**.

### Official Rulings (ChrisK-DD, Sept 21):
1. Known USGS faults are masked out pixel-exact.
2. Expert labels in the test set CAN lie within 300m of known faults (corrections, splays, modifications).
3. "New fault" includes newly mapped geometry of existing fault systems.

### 8GEMSDOE 5-Pillar Architecture:
1. **Along-Strike Ray Tracing:** 6,747 termination endpoints in known normal faults are projected outward up to 1.2 km along their local strike azimuth into basin fill.
2. **Relay Step-Over Corridors:** Overlapping fault segments separated by 300m–1500m are identified as high-permeability extensional relay zones.
3. **Metric-Aligned Soft Envelope ($R \le 2$ px):** Triangular decay ($1 - d/3$) within 2 pixels captures near-fault splays without leaking into the $d > 3$ pure false positive zone.
4. **Multi-Scale Geophysical Classifier:** 19 raw bands + 4 Sobel edge channels on Gravity HG (band 18), Tilt Curvature (band 6), TMI HG (band 3), and Detrended Elevation Slope (band 19).
5. **Geothermal Strain Weighting:** Modulated by InSAR shear strain (`geod_shearrate`), dilatation (`geod_dilaterate`), and smectite clay cap conductivity (`cond_surf`).

---

## 3. Quick Start & Execution

### 1. Download & Verify Competition Data
```bash
bash scripts/download_competition_data.sh
```
Places official rasters into `data/` and verifies hashes:
- `training_features.tif` (SHA-256 `4371c82e...`)
- `labels.tif` (SHA-256 `7ba308cc...`)
- `sample_submission.tif` (SHA-256 `2176d08e...`)

### 2. Pre-flight Data Verification
```bash
python3 scripts/prepare_data.py
```
Asserts EPSG:32611, 100m resolution, 3292x3730 shape, 19 bands, 5,167,373 valid footprint pixels, 60,988 fault pixels.

### 3. Run Apex Model Training & Submission Build
```bash
python3 scripts/train_apex_model.py
```
Fits the gradient-boosted model, derives along-strike extensions, builds `downloads/submission.tif` and `downloads/submission.zip`.

### 4. Validate Submission Format
```bash
python3 scripts/validate_submission.py downloads/submission.tif
```
Runs 13 automated gates. Exit code 0 indicates 100% compliance.

### 5. Run Hermetic Unit Tests
```bash
pytest -v tests/
```

---

## 4. Repository Structure

```text
├── .github/workflows/ci.yml       # Automated CI test and format validation workflow
├── assets/style.css               # Clean responsive theme for GitHub Pages
├── data/                          # Competition data (hash-pinned, .gitignored)
│   ├── bridge/                    # Bit-identical bridge parts for data placement
│   ├── training_features.tif      # 19-band GeoDAWN feature stack
│   ├── labels.tif                 # Known USGS Quaternary fault labels
│   └── sample_submission.tif      # DrivenData official submission template
├── docs/                          # GitHub Pages static documentation mirror
├── downloads/                     # Shippable competition deliverables
│   ├── submission.tif             # Primary single-band float32 GeoTIFF (1.28 MB)
│   ├── submission.zip             # Single-file ZIP containing submission.tif
│   └── submission_meta.json       # Machine-readable provenance, note, and hashes
├── knowledge/                     # Permanent verified knowledge base
│   ├── 01_scientific_discovery_geothermal_vents.md
│   ├── 02_group_results_audit_and_root_cause.md
│   ├── 03_leaderboard_analysis_and_beating_0.3049.md
│   └── 04_verified_data_sources_and_links.md
├── scripts/                       # Executable CLI tools
│   ├── download_competition_data.sh
│   ├── prepare_data.py
│   ├── train_apex_model.py
│   └── validate_submission.py
├── src/                           # Reusable core python package
│   ├── dataset.py
│   ├── geology.py
│   ├── metric.py
│   └── submission_io.py
├── tests/                         # Unit tests (metric, conformance, geology)
├── index.html                     # Executive Summary & 1-Click Download Portal
├── how-to-submit.html             # Step-by-step submission guide & error diagnosis
├── strategy.html                  # Metric analysis and path to beat 0.3049
├── research.html                  # Geothermal structural geology and science
├── data.html                      # Auditable links table and persistent checksums
├── results.html                   # Group results ledger and leaderboard tracking
└── README.md                      # Project compass, charter, and documentation
```

---

## 5. Verified Data Links
- [DrivenData GEMS Competition](https://www.drivendata.org/competitions/306/competition-doe-gems/)
- [Competition Problem Description](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
- [Competition Leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)
- [Official Rules PDF (OSTI)](https://docs.nlr.gov/docs/fy26osti/96647.pdf)
- [USGS GeoDAWN Survey (DOI: 10.5066/P93LGLVQ)](https://doi.org/10.5066/P93LGLVQ)
- [USGS Quaternary Fault Database](https://www.usgs.gov/programs/earthquake-hazards/faults)
- [Great Basin Center for Geothermal Energy (INGENIOUS)](https://gbcge.org/current-projects/ingenious/)
