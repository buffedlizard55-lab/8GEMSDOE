# 8GEMSDOE Knowledge Base: Slot Strategy After Validation 2026-09-26

Decision framework: Maximize P(Win), Own the Outcome. Binding constraint:
no slot on an idea that has not beaten the current spatially-blocked
holdout best ON A POPULATION RESEMBLING THE HIDDEN TRUTH.

## 1. What the session proved about our validation ladder

No available truth population can validate a new-fault submission:

| population | rewards | blind to | verdict use |
|---|---|---|---|
| catalogue folds | memorization / near-catalogue mass | far-field new faults | kills nothing; bounds fitting |
| SGMC proxy (code 2) | catalogue-avoiding fields (structurally) | near-catalogue new geometry | kills nothing; measures far-field ranking under a hostile, possibly backwards proxy |
| synthetic fixture | operator correctness | everything real | pipeline check only |

The only decision-grade signal left is the leaderboard itself (3 slots/week
per account). Slots must therefore be spent as maximally-informative
EXPERIMENTS, not lottery tickets: one canonical account, each upload a
population hypothesis with a pre-registered interpretation, never duplicates.

## 2. Current slot plan (canonical account)

1. **First slot — apex `b83ea0e7…` (BUILT, UNIQUE, UNUPLOADED).**
   The only submission-shaped object this session that (a) passes all
   format gates ([0,1] verified on disk: min 0, max 0.95), (b) has no byte
   twin anywhere in the sibling set, (c) bets on the near-catalogue
   population (envelope + along-strike extensions + GBT anomalies) that no
   sibling has uploaded pure. It tests: does near-catalogue soft mass beat
   the 0.156x dense-detector plateau on hidden truth? Highest-EV information
   purchase available. Paste-ready note lives in `reports/submissions_log.json`.
2. **Second slot — conditional on the first score** (decide after observing;
   one slot per week max):
   - apex ≈ 0.156x (near mass adds nothing) → upload the FAR-FIELD
     contrast: the best sparse label-free arm (H3-12px-skeleton or
     H1-provided-TDR top-k) to test the complementary population.
   - apex ≳ 0.20 (near mass rewarded) → double down: budget-tuned
     near-catalogue emission (top-k grid 2–5% on train-only structural +
     extensions; H6b-style hedge with FAR less than 1% far-field).
   - apex ≲ 0.08 (near mass punished) → hidden faults avoid catalogue;
     pivot to far-field arms + NBMG-proxy selection (§4).
3. **Never**: duplicates (code-enforced), parallel same-payload uploads
   across accounts (one entity ⇒ one final submission; §3.4/A.3), more than
   one slot/week, or H-arm slots ahead of the evidence above.

## 3. Hypothesis ranking after measurement (see `docs/hypotheses.html`)

- H1 (gravity edges): operator verified on fixture; real-data TDR variants
  0.038–0.044 vs random 0.17 on catalogue, 0.010–0.018 on proxy. Falsified
  as a CATALOGUE detector; untested as a buried-fault detector (no buried
  truth available). Slot-readiness: NO — second-slot contrast candidate.
- H2 (strain corridors): 0.011 catalogue / 0.014 proxy; concentrated mass
  is structurally punished by fold-DTI. Needs fault-system folds. NO slot.
- H3 (scarp steps): best proxy arm (0.127 vs random 0.171); 100 m-only
  catalogue 0.039 supports its own 1 m-calibration premise. First far-field
  contrast candidate. NO slot yet.
- H4 (conductivity): 0.033 / 0.018. NO slot.
- H5 (blind seismic ∩ shallow basement): REDEFINED (no mag-depth band);
  0.026 / 0.025, highest fold variance (0.055 vs 0.005) — unstable. NO slot.
- H6 (hedge composite): composite 0.429 < apex 0.451 — no slot on composite
  evidence; concept retained for the decision tree above.

## 4. Best next proxy: NBMG Quaternary faults (TO VERIFY, then select)

The SGMC bedrock proxy is structurally hostile and positionally coarse.
The specific next source to attempt: **Nevada Bureau of Mines and Geology
(NBMG) Quaternary-fault mapping** (county-scale, more precise than SGMC
bedrock, may include faults absent from QFFDB/INGENIOUS). Steps on an
unrestricted machine: (1) verify a free official NBMG Quaternary-fault
download exists (candidate: the NBMG open-data portal / Map 167 series —
URL NOT yet verified, do not assert); (2) rasterize to the 32611 grid with
the `build_proxy_catalogue.py` method (R=300 m code split); (3) rerun
`holdout_proxy.py` with `--proxy` pointed at it; (4) only then re-rank
H-arms. Until then: NO far-field slot claims.

## 5. Two forum questions worth more than a slot (draft, unposted)

- 11516 follow-up: "Is prediction within 300 m of EXISTING catalogue
  traces eligible for credit under the private scoring labels (surely yes
  for new geometry of existing systems), and is exact-pixel known-fault
  mass excluded or free?" (posts 3–4 could not be rendered; near-fault
  ruling UNVERIFIED — never assert it.)
- New-vs-far prior: "Roughly what share of scoring faults are expected to
  be (a) new geometry of known systems vs (b) entirely concealed faults?"
  The answer picks the population our budget should serve.
