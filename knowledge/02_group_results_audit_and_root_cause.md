# 8GEMSDOE Knowledge Base: Group Submission Audit & Root Cause Analysis

## 1. Group Submission Results Ledger (Verified 2026-09-26)

Page-verifiable facts are (account, best score, submission count) from the
official top-50 leaderboard. File→score mappings are OPERATOR-REPORTED
(cannot be verified from the board) and flagged with † below.

| Account | Best | #subs | Rank | † Claimed file → methodology |
|---|---|---|---|---|
| `extradr19` | **0.1563** | 2 | #24 | † `7f00890a…` ens12 U-Net ensemble (172,974 px @1.0). Second upload unknown. |
| `SDCF9` | **0.1563** | 2 | #25 | † Second upload also 0.1563 (observed live 2026-09-26): extends the duplication to a third board row. Prior best was 0.1152 († ridge `4e03fc9705`). |
| `smashi34` | **0.1560** | 1 | #26 | † `f68e590f` dual-family union (183,642 px) — OR the shared `7f00890a…` recall core; 1 submission but mapping unverified. |
| `smrtdoog5` | **0.1193** | 1 | #41 | † `f347b70daa` pindrop nodes (155,021 px, 0 px on catalogue) — cleanest single-file mapping claim. |
| `wbg1` | **0.0830** | 2 | #50 | † `37f9d5b855` catalogue-gap arm. Second upload unknown. |
| *(below top-50)* | 0.0343† | ? | ? | † GEMSDOE4 combined `237f0063…` (264,247 px). Score operator-reported, NOT page-verifiable. |
| *(below top-50)* | 0.0286† | ? | ? | † 6GEMSDOE HGB `33cec71ff0` (155,021 px). Score operator-reported, NOT page-verifiable. |

5GEMSDOE has no separate leaderboard account — consistent with it being the
same upload as GEMSDOE (§2).

---

## 2. Root Cause: Why 5GEMSDOE and GEMSDOE1 Have the Exact Same Score (0.1563)

### The Question:
*"Need to figure out why 5GEMSDOE and GEMSDOE1 have the same score. We should not be generating the same score submissions, they should all be unique."*

### Mathematical & Forensic Proof:
1. In `GEMSDOE`, the published submission file was built from artifact:
   - File: `data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif`
   - File SHA-256: `7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15`
   - Pre-packaged binary payload: `docs/submission_field.bin` (SHA-256: `b966d47c7c02b1c2de0d06553363bc8cccd887c697f760e63f49db9fd99a0351`)
   - Emitted values: Exactly 172,974 pixels set to `1.0`, exactly 4,994,399 pixels set to `0.0`, exactly 7,111,787 pixels set to `NaN`.
2. In `5GEMSDOE`, the client-side download generator `docs/geotiff_writer.js` read the exact same payload:
   - File SHA-256: `7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15`
   - Payload SHA-256: `b966d47c7c02b1c2de0d06553363bc8cccd887c697f760e63f49db9fd99a0351`
3. Because the raster pixels submitted from both repositories were **bit-for-bit identical**, DrivenData evaluated the exact same predictions against the public test set, returning the exact same score of **0.1563**.

### Direct-byte re-verification (2026-09-26, `scripts/audit_siblings.py`):

The six sibling repos were cloned and 11 artifacts read byte-for-byte
(`reports/sibling_audit.json`):
- GEMSDOE-ens12 == 5GEMSDOE-ens12 == GEMSDOE2-recall (`gemsdoe2-recall-arm-7f00890a.tif`):
  file sha `7f00890a6287…`, payload stream sha `cb2d2d5e0fed…`, 172,974 px @1.0.
  A TRIPLE duplicate across three repos — same bytes ⇒ same 0.1563/0.1560 expected.
- 8GEMSDOE apex (`downloads/submission.tif`, file sha `b83ea0e70748…`) matches
  NONE of the 11 sibling artifacts: uniqueness re-verified this session.

### Remediation in 8GEMSDOE:
In `8GEMSDOE`, every submission candidate is generated independently with unique structural hypotheses, along-strike vector expansions, and new trained model weights. No legacy binaries are reused. Uniqueness is enforced in code: `record_submission()` refuses a payload whose hash already appears (override needs `--allow-duplicate` + logged warning), and every build carries a unique timestamped filename plus a paste-ready Note.

---

## 3. Root Cause & Permanent Fix: "Predicted values must be in range [0, 1]"

### The Problem:
When submitting a generated GeoTIFF to DrivenData, the platform rejected the file with the exact error:
`"Predicted values must be in range [0, 1]"`

### The Forensic Diagnosis:
1. Every *finite* number in the file was in `[0, 1]`.
2. However, the official test set is scored on the **5,167,373 valid footprint pixels** defined by `sample_submission.tif` (the GeoDAWN active survey area).
3. In earlier site builders or raw model exports, **3,061 pixels inside the valid footprint** contained `NaN` or un-imputed nodata values (count operator-reported from the original incident; all 12 current repo/sibling artifacts re-checked 2026-09-26 show **0** NaN inside — the incident predates them).
4. On DrivenData's backend ingestion validator, `NaN` inside the valid evaluation footprint fails the `[0, 1]` assertion, throwing `"Predicted values must be in range [0, 1]"`.
5. Additionally, any finite predictions placed outside the 5,167,373 footprint violate the boundary mask.

### The Permanent Solution Implemented in 8GEMSDOE:
1. `src/gems/submission.py:conform_to_template(raw_pred)`:
   - Strictly enforces that **every single pixel inside the 5,167,373 valid footprint is finite and clipped to `[0.0, 1.0]`**.
   - Strictly enforces that **every single pixel outside the valid footprint is `NaN`** with `nodata = NaN`.
2. `src/gems/submission.py:validate_submission()` — 13 gates; the
   `NAN-INSIDE-FOOTPRINT` gate is the exact rejection condition:
   - 2026-09-26: apex `downloads/submission.tif` passes **13/13** against the
     official template — finite range [0, 0.95], 0 NaN inside, 0 finite outside.
   - Fixed this session: the reader prefers `rasterio`, so LZW-compressed
     official files (e.g. `example_submission.tif`) validate instead of raising
     `ValueError: <COMPRESSION.LZW> requires 'imagecodecs'`.
