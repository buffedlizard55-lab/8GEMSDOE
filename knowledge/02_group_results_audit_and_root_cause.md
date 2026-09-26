# 8GEMSDOE Knowledge Base: Group Submission Audit & Root Cause Analysis

## 1. Group Submission Results Ledger (Verified 2026-09-26)

| Submission / Account | Site URL | Public Score | Key Methodology | Underlying Root Cause |
|---|---|---|---|---|
| **GEMSDOE1** (`extradr19`) | `buffedlizard55-lab.github.io/GEMSDOE/docs/index.html` | **0.1563** | 11-model deep ensemble (U-Net, DeepLabV3+, ResNet34 backbones), floor 0.1, thinning. | Primary baseline ensemble. Emitted 172,974 pixels. |
| **5GEMSDOE** | `buffedlizard55-lab.github.io/5GEMSDOE/docs/index.html` | **0.1563** | Exported `submission_field.bin` from `data/evidence/runs/ens12-adopted-floor0.1-w0/`. | **IDENTICAL BYTES**: Packaged the exact same raster artifact (sha256 `7f00890a...`) as GEMSDOE1! |
| **GEMSDOE2** (`smashi34`) | `buffedlizard55-lab.github.io/GEMSDOE2/docs/index.html` | **0.1560** | Dual-family recall-union system. | Dominated by the same `7f00890a...` recall core; marginal edge difference. |
| **GEMSDOE3** (`smrtdoog5`) | `buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html` | **0.1193** | Pindrop v4: sparse node sampling (`k=4`, `s=3%`). | Severely penalized by false negative weight ($\beta=0.8$) due to omitting continuous fault paths. |
| **GEMSDOE3** (`SDCF9`) | `buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html` | **0.1152** | Pindrop dense ridge control (`4e03fc9705`). | Concentrated mass along known ridges without adequate strike-continuation. |
| **GEMSDOE3** (`wbg1`) | `buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html` | **0.0830** | Pindrop catalogue-gap target (`37f9d5b855`). | Gap-only emission without near-fault halo; missed near-fault modifications. |
| **GEMSDOE4** | `buffedlizard55-lab.github.io/GEMSDOE4/` | **0.0343** | Experimental spatial masking. | Over-pruning caused massive false negative penalty. |
| **6GEMSDOE** | `buffedlizard55-lab.github.io/6GEMSDOE/` | **0.0286** | Ultra-sparse candidate. | Minimal emitted mass; denominator $0.8 \times |G|$ dominated the metric. |

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

### Remediation in 8GEMSDOE:
In `8GEMSDOE`, every submission candidate is generated independently with unique structural hypotheses, along-strike vector expansions, and new trained model weights. No legacy binaries are reused.

---

## 3. Root Cause & Permanent Fix: "Predicted values must be in range [0, 1]"

### The Problem:
When submitting a generated GeoTIFF to DrivenData, the platform rejected the file with the exact error:
`"Predicted values must be in range [0, 1]"`

### The Forensic Diagnosis:
1. Every *finite* number in the file was in `[0, 1]`.
2. However, the official test set is scored on the **5,167,373 valid footprint pixels** defined by `sample_submission.tif` (the GeoDAWN active survey area).
3. In earlier site builders or raw model exports, **3,061 pixels inside the valid footprint** contained `NaN` or un-imputed nodata values.
4. On DrivenData's backend ingestion validator, `NaN` inside the valid evaluation footprint fails the `[0, 1]` assertion, throwing `"Predicted values must be in range [0, 1]"`.
5. Additionally, any finite predictions placed outside the 5,167,373 footprint violate the boundary mask.

### The Permanent Solution Implemented in 8GEMSDOE:
1. `src/submission_io.conform_to_template(raw_pred)`:
   - Strictly enforces that **every single pixel inside the 5,167,373 valid footprint is finite and clipped to `[0.0, 1.0]`**.
   - Strictly enforces that **every single pixel outside the valid footprint is `NaN`** with `nodata = NaN`.
2. `scripts/validate_submission.py`:
   - Enforces 13 independent gates before any file is packaged or presented. Gate 11 verifies zero NaNs inside the valid footprint.
