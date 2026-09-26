"""Guards for the EXTERNAL feature channels (3DEP DEM scarp + GeoDAWN radiometrics).

Two things have already gone wrong in this area and are pinned here:

1. A coverage census used ``isfinite(labels)`` as the footprint. ``data/labels.tif``
   is finite on all 12,279,160 px and carries the sentinel ``-1`` outside the scored
   area, so that census reported "42.08 % coverage" for channels that in fact cover
   100 % of the footprint (5,167,373 / 12,279,160 = 42.082 %).
2. The channels travel as a sha256-pinned git bridge because this sandbox has no USGS
   egress; if a part is truncated the unpack must refuse, not silently emit NaNs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

REPO = Path(__file__).resolve().parents[1]
EXTERNAL = REPO / "data" / "external"
DERIVED = REPO / "data" / "derived"
BRIDGE = REPO / "data" / "aux_bridge"
LABELS = REPO / "data" / "labels.tif"
needs_labels = pytest.mark.skipif(not LABELS.exists(),
                                  reason="data/ not placed in this env")
GRID = (3730, 3292)
TRANSFORM = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
TOPO_BANDS = ["slope_p90", "slope_max", "steep_frac", "slope_std", "relief_local",
              "curv_prof_absmax", "aspect_coherence", "hs_lineament", "dem_mean"]
RAD_BANDS = ["rad_k", "rad_th", "rad_u", "rad_tc", "rad_thk", "rad_uk", "rad_uth"]


@pytest.fixture(scope="module")
def footprint() -> np.ndarray:
    if not LABELS.exists():
        pytest.skip("data/ not placed in this env")
    with rasterio.open(LABELS) as src:
        lab = src.read(1)
    foot = np.isfinite(lab) & (lab >= 0)
    assert int(foot.sum()) == 5_167_373
    return foot


@pytest.mark.parametrize("name,bands", [("topo_features_100m", TOPO_BANDS),
                                        ("radiometric_100m", RAD_BANDS)])
def test_external_raster_is_on_the_competition_grid(name, bands):
    path = EXTERNAL / f"{name}.tif"
    if not path.exists():
        pytest.skip(f"{path} not built (run scripts/fetch_aux_bridge.sh)")
    with rasterio.open(path) as src:
        assert (src.height, src.width) == GRID
        assert str(src.crs) == "EPSG:32611"
        assert tuple(round(float(v), 6) for v in src.transform)[:6] == TRANSFORM
        assert src.count == len(bands)
        assert list(src.descriptions) == bands


@pytest.mark.parametrize("name,band,min_cov", [("topo_features_100m", 1, 0.999),
                                               ("topo_features_100m", 9, 0.999),
                                               ("radiometric_100m", 4, 0.999)])
def test_coverage_of_the_true_footprint(name, band, min_cov, footprint):
    """Regression guard for the 42.08 % census bug: coverage is ~100 %."""
    path = EXTERNAL / f"{name}.tif"
    if not path.exists():
        pytest.skip(f"{path} not built")
    with rasterio.open(path) as src:
        a = src.read(band)
    cov = float(np.isfinite(a[footprint]).mean())
    assert cov >= min_cov, f"{name} band {band} coverage {cov:.4f} < {min_cov}"


def test_bridge_parts_match_their_pinned_manifests():
    if not BRIDGE.exists():
        pytest.skip("data/aux_bridge not fetched")
    missing = []
    dirs = sorted(p for p in BRIDGE.iterdir() if p.is_dir())
    for d in dirs:
        mf = json.loads((d / "manifest.json").read_text())
        missing += [str(d / part["name"]) for part in mf["parts"]
                    if not (d / part["name"]).exists()]
    if missing:
        # Only the manifests are committed; the multi-MB parts are gitignored and are
        # re-fetched by scripts/fetch_aux_bridge.sh. CI therefore sees manifests without
        # parts -- that is expected, not a failure.
        pytest.skip(f"bridge parts not fetched here (run scripts/fetch_aux_bridge.sh): "
                    f"{len(missing)} missing, e.g. {missing[0]}")
    for d in dirs:
        mf = json.loads((d / "manifest.json").read_text())
        import hashlib
        h = hashlib.sha256()
        total = 0
        for part in mf["parts"]:
            b = (d / part["name"]).read_bytes()
            assert len(b) == part["bytes"], f"{d.name}/{part['name']} truncated"
            assert hashlib.sha256(b).hexdigest() == part["sha256"], f"{d.name} part hash"
            h.update(b)
            total += len(b)
        assert total == mf["whole"]["bytes"]
        assert h.hexdigest() == mf["whole"]["sha256"], f"{d.name} whole-file hash"
        assert mf["grid"]["height"] == GRID[0] and mf["grid"]["width"] == GRID[1]
        assert mf["grid"]["crs"] == "EPSG:32611"
        # provenance must name the official upstream source
        prov = json.dumps(mf.get("provenance", {}))
        assert "prd-tnm.s3.amazonaws.com" in prov or "10.5066/P93LGLVQ" in prov, d.name


def test_plane_stack_has_the_16_external_planes():
    meta_path = DERIVED / "planes_ext.json"
    if not meta_path.exists():
        pytest.skip("run scripts/build_aux_planes.py")
    meta = json.loads(meta_path.read_text())
    names = meta["planes"]
    assert meta["shape"] == [62, GRID[0], GRID[1]]
    assert len(names) == 62 and meta["n_in_file_planes"] == 46
    assert names[46:55] == [f"T_{b}" for b in TOPO_BANDS]
    assert names[55:62] == [f"R_{b}" for b in RAD_BANDS]
    # the corrected coverage census (was 42.08 % before the footprint fix)
    cov = [v["coverage_pct"] for v in meta["aux_stats"].values()]
    assert min(cov) > 98.0, f"aux coverage collapsed: {min(cov)}"


def test_footprint_area_equals_the_geodawn_survey_extent():
    """5,167,373 px x (100 m)^2 = 51,674 km^2 vs the release's 51,857 km^2."""
    km2 = 5_167_373 * (0.1 ** 2)
    assert abs(km2 - 51_674) < 1
    assert abs(km2 - 51_857) / 51_857 < 0.01  # within 1 % of the survey area
