#!/usr/bin/env bash
# 8GEMSDOE — competition data placement guide (no credentials in this repo).
#
# BLOCKER (verified 2026-09-26): the DrivenData data tab
#   https://www.drivendata.org/competitions/306/competition-doe-gems/data/
# redirects to the login page without authentication, so no script running here
# can fetch the files. They must be downloaded in a browser (one-time) and
# placed into data/ as mapped below. After placement, run:
#   python scripts/prepare_data.py
#
# Expected files (names per the problem description + rules §3.3):
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

cat <<'EOF'
====================================================================
8GEMSDOE data placement — do this once in a browser (DrivenData login)
--------------------------------------------------------------------
1. Open: https://www.drivendata.org/competitions/306/competition-doe-gems/data/
2. Download every file on the data tab. Expected (verify against the tab):

   data/training_features.tif     multiband 100 m GeoTIFF (GeoDAWN + geospatial bands)
   data/training_labels.*         fault labels, raster and/or vector as provided
   data/sample_submission.tif     THE format template (total fault absence)
   data/1m_DEM_links.csv          URLs for the 1 m 3DEP DEM tiles

3. Record SHA256 hashes:
     sha256sum data/* > data/SHA256SUMS.txt
4. Verify placement:
     python scripts/prepare_data.py

Free official external data (no login) used by hypotheses H1..H5:
  * INGENIOUS Great Basin Regional Dataset Compilation, DOI 10.15121/1881483
      https://gdr.openei.org/submissions/1391
      (also indexed at https://catalog.data.gov/dataset/ingenious-great-basin-regional-dataset-compilation)
  * USGS Quaternary Fault & Fold Database (KML + GIS shapefiles, public domain)
      https://www.usgs.gov/programs/earthquake-hazards/faults
  * USGS State Geologic Map Compilation (SGMC) — faults layer StatewideFaults.shp
      https://mrdata.usgs.gov/geology/state/  (DOI 10.5066/F7WH2N65)
  * USGS 3DEP 1 m DEM — staged products + AWS public dataset:
      https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation/1m/Projects/
      s3://usgs-lidar-public (EPT, public) / s3://usgs-lidar (Requester Pays, full LASzip)
  * GeoDAWN source grids (magnetic + radiometric), DOI 10.5066/P93LGLVQ
      https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and
  * Nevada isostatic gravity grids (DS 234)
      https://pubs.usgs.gov/ds/2006/234/
  * NBMG open data / web apps (Quaternary faults, geothermal)
      https://data-nbmg.opendata.arcgis.com/pages/web-applications
====================================================================
EOF

echo "--- current data/ contents ---"
ls -la "$DATA_DIR" || true
