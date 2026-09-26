"""Guards for the recommended upload and for the facts it rests on.

These tests are deliberately *artifact* tests: they re-read the file a submitter
would download and re-check the claims made about it on the site, so a stale page
or a silently rebuilt raster fails CI instead of reaching DrivenData.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports" / "hedge_v2.json"
TIF = REPO / "downloads" / "8GEMSDOE_Hedge-v2_submission.tif"
LABELS = REPO / "data" / "labels.tif"
needs_labels = pytest.mark.skipif(not LABELS.exists(),
                                  reason="data/ not placed in this env")
ZIP = REPO / "downloads" / "8GEMSDOE_Hedge-v2_submission.zip"
BASE_SHA256 = "7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15"

pytestmark = pytest.mark.skipif(
    not (REPORT.exists() and TIF.exists()),
    reason="run `python scripts/build_hedge_v2.py` first",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text())


@pytest.fixture(scope="module")
def raster():
    with rasterio.open(TIF) as src:
        return src.read(1), src


@needs_labels
def test_footprint_definition_is_the_sentinel_aware_one():
    """The bug this repo already paid for once: labels.tif is finite everywhere."""
    with rasterio.open(REPO / "data" / "labels.tif") as src:
        lab = src.read(1)
    assert np.isfinite(lab).sum() == lab.size, "labels.tif changed: sentinel no longer finite"
    foot = np.isfinite(lab) & (lab >= 0)
    assert int(foot.sum()) == 5_167_373
    assert int((lab[~foot] == -1).sum()) == lab.size - 5_167_373


def test_report_hashes_match_the_bytes_on_disk(report):
    assert sha256(TIF) == report["sha256_tif"]
    assert ZIP.exists() and sha256(ZIP) == report["sha256_zip"]
    assert report["base_sha256"] == BASE_SHA256


@needs_labels
def test_emission_is_the_verified_base_union_the_catalogue(report, raster):
    out, _ = raster
    with rasterio.open(REPO / "data" / "labels.tif") as src:
        lab = src.read(1)
    footprint = np.isfinite(lab) & (lab >= 0)
    catalogue = lab == 1
    pos = np.nan_to_num(out, nan=0.0) > 0
    assert int(pos.sum()) == report["positive_px"] == 227_507
    assert int((pos & footprint).sum()) == int(pos.sum()), "positive pixel outside footprint"
    assert pos[catalogue].all(), "every catalogue pixel must be emitted (weak dominance)"
    assert report["chargeable_px_off_catalogue"] == 166_519
    # the base pattern is a strict subset: nothing was removed from the 0.1563 file
    assert int((pos & ~catalogue).sum()) == 166_519


@needs_labels
def test_conformance_the_platform_rejects_otherwise(raster):
    """'Predicted values must be in range [0, 1]' — the error already hit once."""
    out, src = raster
    with rasterio.open(REPO / "data" / "labels.tif") as lr:
        lab = lr.read(1)
    footprint = np.isfinite(lab) & (lab >= 0)
    assert src.count == 1 and src.dtypes[0] == "float32"
    assert str(src.crs) == "EPSG:32611"
    assert (src.height, src.width) == (3730, 3292)
    assert tuple(round(v, 6) for v in src.transform)[:6] == (
        100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
    assert np.isfinite(out[footprint]).all(), "NaN/Inf inside the scored footprint"
    assert out[footprint].min() >= 0.0 and out[footprint].max() <= 1.0
    assert (~np.isfinite(out[~footprint])).all(), "finite value outside the footprint"


def test_crosscheck_against_independent_build(report):
    cc = report["crosscheck_vs_sibling_candidate"]
    if not cc.get("available"):
        pytest.skip("sibling checkout not present")
    assert cc["sha256_8"] == cc["pinned_sha8_expected"] == "132e23e1"
    assert cc["pixel_identical_to_ours"] is True and cc["differing_px"] == 0


def test_site_hero_points_at_this_file_and_quotes_this_note(report):
    note = report["drivendata_submission_note"]
    for page in ("index.html", "docs/index.html"):
        html = (REPO / page).read_text()
        assert "downloads/8GEMSDOE_Hedge-v2_submission.tif" in html, page
        assert note in html, f"{page}: paste-ready note is stale"
        assert report["sha256_tif"][:10] in html, f"{page}: sha256 is stale"
    # the retired apex artifact must no longer be the primary download
    for page in ("index.html", "docs/index.html"):
        html = (REPO / page).read_text()
        assert 'href="downloads/submission.tif"' not in html, page


def test_site_mirrors_are_byte_identical():
    for name in ("index.html", "executive_summary.html"):
        assert (REPO / name).read_bytes() == (REPO / "docs" / name).read_bytes(), name


def test_docs_mirror_also_serves_the_file():
    mirror = REPO / "docs" / "downloads" / TIF.name
    assert mirror.exists() and sha256(mirror) == sha256(TIF)
