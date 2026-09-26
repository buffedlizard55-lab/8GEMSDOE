# 8GEMSDOE Knowledge Base: Leaderboard Analysis & Strategy to Exceed 0.3049

## 1. Current Competition Landscape (Leaderboard Snapshot 2026-09-26)
- **Top Score on Leaderboard**: **0.3049** (Participant: `DARD`, 10 submissions)
- **#2 Participant**: `alexoktaba` (0.2993, 13 submissions)
- **#3 Participant**: `HardcoreTechGod` (0.2854, 6 submissions)
- **#4 Participant**: `mzoorob` (0.2843, 15 submissions)
- **Our Group Baseline**: `extradr19` (0.1563, rank #24)
- **Gap to First Place**: **+0.1486** (an ~95% relative increase required).

---

## 2. Mathematical Dissection of the Competition Metric
The performance metric is the **Distance-Weighted Tversky Index (DW-Tversky)**:

$$DTI(\alpha=0.2, \beta=0.8) = \frac{TP_w}{TP_w + 0.2 FP_w + 0.8 FN_w}$$

Using the identity $FN_w = |G| - TP_w$ where $|G|$ is the total number of ground truth fault pixels:

$$\text{Denominator} = TP_w + 0.2 FP_w + 0.8(|G| - TP_w) = 0.2(TP_w + FP_w) + 0.8 |G|$$

$$DTI = \frac{TP_w}{0.2(TP_w + FP_w) + 0.8 |G|}$$

### Critical Strategic Insights:
1. **Asymmetric Penalty ($\beta = 0.8$ vs $\alpha = 0.2$)**:
   - The penalty for missing a fault pixel ($\beta = 0.8$) is **4 times higher** than the penalty for a false alarm ($\alpha = 0.2$).
   - A conservative model that predicts very few pixels (such as Pindrop or aggressive thresholding) suffers devastating penalties because $0.8 |G|$ remains in the denominator while $TP_w$ stays tiny.
2. **The 300m Support Kernel ($R = 3$ pixels)**:
   - $k(d) = \max(1 - d/3, 0)$.
   - Distance 0: $k = 1.00$
   - Distance 1: $k = 0.67$
   - Distance 2: $k = 0.33$
   - Distance $\ge 3$: $k = 0.00$
   - Predicting a soft corridor of width 2 pixels around high-probability structures captures near-misses ($k=0.67, 0.33$) that would otherwise become pure false negatives!

---

## 3. Official Staff Clarification (Sept 21, ChrisK-DD): The Secret to 0.3049
On the DrivenData community forum (Topic 11516 & 11536), competition staff explicitly clarified:
1. **Known USGS Faults Are Masked Out**:
   - "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation, so they do not count towards penalty terms."
2. **Near-Fault Corrections Are Evaluated**:
   - "A new-fault ground truth pixel can indeed lie within 300m of a known fault trace. Such pixels would constitute corrections or modifications to existing fault traces. Identifying these corrections is one outcome we are aiming for as part of this competition."
3. **Definition of New Fault**:
   - "'new fault' means 'any fault pixel not already captured by USGS/INGENIOUS' and can include newly mapped geometry of an existing fault system."

### Why Previous Submissions Plateaued at 0.1563:
- GEMSDOE1 and 5GEMSDOE thinned the predictions into a 1-pixel skeleton. If the expert ground truth lay 100m (1 pixel) to the side (due to high-resolution lidar mapping), a 1-pixel skeleton scored only $k(1) = 0.67$ or missed entirely.
- By contrast, `7GEMSDOE` tried a wide 15-pixel halo ($r=15$). But the metric kernel only extends to $R=3$ pixels! Pixels at distance 4–15 pixels from a fault contributed massive false positive mass ($FP_w$) without any possibility of contributing to $TP_w$, degrading performance.

---

## 4. The 8GEMSDOE Apex Discovery Strategy (Path to >0.3049)
To beat 0.3049, our system unites five distinct geological and mathematical pillars:

1. **Along-Strike Ray Tracing (Strike Azimuth Continuation)**:
   - Normal faults in the Basin & Range do not stop at bedrock-alluvium contacts; their tip damage zones continue 0.5–1.5 km into the basin.
   - We extract 6,747 termination endpoints from known faults, compute their local strike vectors, and project directional continuation rays into basin fill.
2. **Relay Step-Over Corridors**:
   - Between overlapping fault segments separated by 300m–1500m, high-strain extensional fracture meshes develop. We detect and score these relay zones.
3. **Multi-Scale Geophysical Classifier**:
   - 19 GeoDAWN bands + Sobel edge derivatives on Gravity HG (band 18), Tilt Curvature (band 6), TMI HG (band 3), and Detrended Elevation Slope (band 19).
   - High-capacity HistGradientBoosting model trained with hard negatives.
4. **Geothermal Strain & Alteration Weighting**:
   - Up-weight structural candidates in areas of elevated GPS/InSAR shear strain (`geod_shearrate`), dilatation (`geod_dilaterate`), and high subsurface conductivity (`cond_surf` hydrothermal smectite clay caps).
5. **Exact Metric-Aligned Soft Envelope ($R \le 2$ pixels)**:
   - A triangular decay envelope ($k(d) = 1 - d/3$) strictly capped at radius 2 pixels ($d \le 200$m).
   - Captures expert near-fault corrections and splays while generating zero excess false positives beyond the 300m support radius.
