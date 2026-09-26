#!/usr/bin/env bash
# scripts/download_competition_data.sh
# Download and verify the official GEMS Prize competition rasters into $DEST
# (default: ./data or $GEMS_DATA_DIR).
#
# Every file is sha256-verified against the official DrivenData/USGS pins:
#   example_submission.tif: 2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc
#   existing_faults.tif:    7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093
#   training_features.tif:  4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5
#
# Data source hierarchy:
#   1. Local git bridge in data/bridge (fast, offline, hash-pinned)
#   2. Official Dropbox mirrors (requires outbound egress)
#   3. Fallback sparse clone from sibling public repositories

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${GEMS_DATA_DIR:-$REPO_ROOT/data}"
BRIDGE="${REPO_ROOT}/data/bridge"
TMP_CLONE=""
cleanup() { [[ -z "$TMP_CLONE" ]] || rm -rf "$TMP_CLONE"; }
trap cleanup EXIT
fetch_bridge() {
  if [[ -z "$TMP_CLONE" ]]; then
    TMP_CLONE=$(mktemp -d)
    git clone --depth 1 --filter=blob:none --sparse https://github.com/buffedlizard55-lab/GEMSDOE "$TMP_CLONE"
    (cd "$TMP_CLONE" && git sparse-checkout set data/bridge)
  fi
  BRIDGE="$TMP_CLONE/data/bridge"
}


mkdir -p "$DEST"

declare -A SHA=(
  ["example_submission.tif"]="2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc"
  ["existing_faults.tif"]="7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093"
  ["training_features.tif"]="4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5"
)

declare -A URL=(
  ["example_submission.tif"]="https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=1"
  ["existing_faults.tif"]="https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=1"
  ["training_features.tif"]="https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=1"
)

verify_sha() {
  local target="$1"
  local expected="$2"
  if [[ ! -f "$target" ]]; then
    return 1
  fi
  local actual
  actual=$(sha256sum "$target" | cut -d' ' -f1)
  if [[ "$actual" == "$expected" ]]; then
    echo "PASS [sha256] $target ($actual)"
    return 0
  else
    echo "FAIL [sha256] $target: expected $expected, got $actual"
    return 1
  fi
}

echo "=== GEMS Prize Data Placement Pipeline ==="
echo "Destination directory: $DEST"

# 1. Assemble training_features.tif
if ! verify_sha "$DEST/training_features.tif" "${SHA["training_features.tif"]}"; then
  echo "Reassembling training_features.tif from bridge parts..."
  if [[ -f "$BRIDGE/gems-geodawn-numerical-features.tif.part-000" ]]; then
    cat "$BRIDGE/gems-geodawn-numerical-features.tif.part-"* > "$DEST/training_features.tif.tmp"
    mv "$DEST/training_features.tif.tmp" "$DEST/training_features.tif"
  elif curl -fL --retry 3 --connect-timeout 10 -o "$DEST/training_features.tif.tmp" "${URL["training_features.tif"]}"; then
    mv "$DEST/training_features.tif.tmp" "$DEST/training_features.tif"
  else
    echo "Attempting fallback clone from sibling repository..."
    fetch_bridge
    cat "$BRIDGE/gems-geodawn-numerical-features.tif.part-"* > "$DEST/training_features.tif"
  fi
  verify_sha "$DEST/training_features.tif" "${SHA["training_features.tif"]}"
fi

# 2. Existing faults (labels.tif and existing_faults.tif)
if ! verify_sha "$DEST/existing_faults.tif" "${SHA["existing_faults.tif"]}"; then
  if [[ -f "$BRIDGE/existing_faults.tif" ]]; then
    cp "$BRIDGE/existing_faults.tif" "$DEST/existing_faults.tif"
  else
    if ! curl -fL --retry 3 --connect-timeout 10 -o "$DEST/existing_faults.tif.tmp" "${URL["existing_faults.tif"]}"; then
      fetch_bridge
      cp "$BRIDGE/existing_faults.tif" "$DEST/existing_faults.tif.tmp"
    fi
    mv "$DEST/existing_faults.tif.tmp" "$DEST/existing_faults.tif"
  fi
  verify_sha "$DEST/existing_faults.tif" "${SHA["existing_faults.tif"]}"
fi
cp -f "$DEST/existing_faults.tif" "$DEST/labels.tif"

# 3. Example submission (sample_submission.tif and example_submission.tif)
if ! verify_sha "$DEST/example_submission.tif" "${SHA["example_submission.tif"]}"; then
  if [[ -f "$BRIDGE/example_submission.tif" ]]; then
    cp "$BRIDGE/example_submission.tif" "$DEST/example_submission.tif"
  else
    if ! curl -fL --retry 3 --connect-timeout 10 -o "$DEST/example_submission.tif.tmp" "${URL["example_submission.tif"]}"; then
      fetch_bridge
      cp "$BRIDGE/example_submission.tif" "$DEST/example_submission.tif.tmp"
    fi
    mv "$DEST/example_submission.tif.tmp" "$DEST/example_submission.tif"
  fi
  verify_sha "$DEST/example_submission.tif" "${SHA["example_submission.tif"]}"
fi
cp -f "$DEST/example_submission.tif" "$DEST/sample_submission.tif"

echo ""
echo "=== All official rasters placed and verified successfully! ==="
ls -lh "$DEST"/*.tif
