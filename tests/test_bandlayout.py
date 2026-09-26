"""Guard the verified 19-band layout against index hallucinations.

Band identities come from the file's own `band_name` tags (verified
2026-09-26 against training_features.tif sha256 4371c82e...). Any code that
consumes bands by position must agree with this table. Skipped when data/
is absent (CI without the 400 MB download).
"""
from pathlib import Path

import pytest

rasterio = pytest.importorskip("rasterio")

EXPECTED = {
    1: "mag_anom",
    2: "rtp",
    3: "tmi_hg",
    4: "geod_2ndinv",
    5: "iso_grav_anom_slope",
    6: "tc",
    7: "geod_shearrate",
    8: "geod_dilaterate",
    9: "tmi_vg",
    10: "deq_n100a15",
    11: "iso_grav_anom_vg",
    12: "det_elev",
    13: "iso_grav_anom",
    14: "tmi",
    15: "depth_to_base_surf",
    16: "ieq_n100a15",
    17: "cond_surf",
    18: "iso_grav_anom_hg",
    19: "det_elev_slope",
}

FEAT = Path(__file__).resolve().parents[1] / "data" / "training_features.tif"


@pytest.mark.skipif(not FEAT.exists(), reason="data/ not placed in this env")
def test_band_tags_match_verified_layout():
    with rasterio.open(FEAT) as src:
        assert src.count == 19
        for idx, name in EXPECTED.items():
            assert src.tags(idx).get("band_name") == name, idx


@pytest.mark.skipif(not FEAT.exists(), reason="data/ not placed in this env")
def test_no_magnetic_source_depth_band():
    # The problem description lists a "top-of-crustal magnetic source depth
    # estimate" band; the file has none (verified 2026-09-26). If a future
    # data revision adds one, H5 must be revisited — this test fails loudly.
    with rasterio.open(FEAT) as src:
        names = [src.tags(i).get("band_name", "") for i in range(1, 20)]
    assert not any("depth" in n and "mag" in n for n in names), names
    assert not any("source" in n for n in names), names
