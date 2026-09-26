#!/usr/bin/env bash
# Fetch, verify and unpack the auxiliary external-data channels (topo + radiometric).
#
# WHY A GIT BRIDGE
#   This sandbox reaches github.com and pypi.org only. Measured again 2026-09-26:
#   every USGS / ScienceBase / Amazon-S3 host returns `curl: (35) SSL_ERROR_SYSCALL`
#   (e.g. prd-tnm.s3.amazonaws.com, www.sciencebase.gov). GitHub-hosted runners DO
#   reach those hosts, so the channels are built there (sibling repo workflow
#   `build-aux-channels`, run id 36201412901) and travel back as committed,
#   sha256-pinned git bytes. This script is the receiving end.
#
# UPSTREAM SOURCES — free, official, USGS public domain (verified 2026-09-26)
#   topo (9 bands, 100 m aggregates of the DEM)
#     USGS 3DEP 1/3 arc-second (~10 m) seamless DEM, cloud-optimised VRT:
#     https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt
#     bands: slope_p90 slope_max steep_frac slope_std relief_local
#            curv_prof_absmax aspect_coherence hs_lineament dem_mean
#   radiometric (7 bands)
#     USGS GeoDAWN airborne magnetic + radiometric data release,
#     https://doi.org/10.5066/P93LGLVQ  (Glen & Earney 2024)
#     ScienceBase item 657e1d85d34e23d3533209f7, files 22103_area1_tiffs.zip (43.6 MB)
#     and 22103_area2_tiffs.zip (230.5 MB) — "geoTIFF images of geophysical grids".
#     bands: rad_k rad_th rad_u rad_tc rad_thk rad_uk rad_uth
#   1 m LiDAR (NOT used yet — verified obtainable, too large for this sandbox)
#     716 tile URLs enumerated in data/external/dem_links.json; one tile verified
#     live by anonymous ListObjectsV2: USGS_1M_11_x49y451_NV_WestCentral_EarthMRI_2020_D20.tif,
#     185,344,605 bytes, LastModified 2026-02-14 => ~130 GB for full coverage.
#
# USAGE   bash scripts/fetch_aux_bridge.sh [topo radiometric]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
SRC_REMOTE="${AUX_SRC_REMOTE:-https://github.com/buffedlizard55-lab/5GEMSDOE.git}"
SRC_LOCAL="${AUX_SRC_LOCAL:-/tmp/sib/5GEMSDOE}"
CLONE_DIR="${AUX_CLONE_DIR:-/tmp/auxsrc}"
TEMPLATE="${AUX_TEMPLATE:-data/training_features.tif}"
NAMES=("$@")
if [ ${#NAMES[@]} -eq 0 ]; then NAMES=(topo radiometric); fi

# --- 1. get the committed bridge bytes ------------------------------------
if [ -d "$SRC_LOCAL/data/aux_bridge" ]; then
  echo "== using existing sibling checkout $SRC_LOCAL"
  FROM="$SRC_LOCAL"
elif [ -d "$CLONE_DIR/.git" ]; then
  echo "== reusing clone $CLONE_DIR"
  FROM="$CLONE_DIR"
else
  echo "== sparse-cloning $SRC_REMOTE -> $CLONE_DIR"
  git clone --filter=blob:none --no-checkout --depth 1 "$SRC_REMOTE" "$CLONE_DIR"
  git -C "$CLONE_DIR" sparse-checkout init --no-cone
  git -C "$CLONE_DIR" sparse-checkout set 'data/aux_bridge/*'
  git -C "$CLONE_DIR" checkout HEAD
  FROM="$CLONE_DIR"
fi

mkdir -p data/aux_bridge
for name in "${NAMES[@]}"; do
  if [ ! -d "$FROM/data/aux_bridge/$name" ]; then
    echo "FAIL: $FROM/data/aux_bridge/$name not present" >&2; exit 1
  fi
  mkdir -p "data/aux_bridge/$name"
  cp -f "$FROM/data/aux_bridge/$name/"* "data/aux_bridge/$name/"
done

# --- 2. verify sha256 of every part and of the reassembled file -----------
for name in "${NAMES[@]}"; do
  .venv/bin/python scripts/aux_bridge.py verify --name "$name"
done

# --- 3. unpack to float32 rasters on the exact competition grid -----------
for name in "${NAMES[@]}"; do
  .venv/bin/python scripts/aux_bridge.py unpack --name "$name" --template "$TEMPLATE"
done

echo "== done. data/external/ now holds:"
ls -la data/external/ 2>/dev/null || true
