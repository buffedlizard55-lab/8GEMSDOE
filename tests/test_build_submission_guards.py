"""Fail-closed guards on the submission builder (carried forward from PR #3's audit).

These tests need no competition data: they build a 60x50 template in a temp dir, so
they run in CI as well as locally. What they pin is *behaviour under missing inputs*
-- the failure mode that produced the GEMSDOE/5GEMSDOE duplicate-upload lesson and
that would let a synthetic fixture reach the weekly upload slot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems.submission import write_geotiff  # noqa: E402

pytest.importorskip("tifffile")

import importlib  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
build_submission = importlib.import_module("build_submission")

TF = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)


def _tiny_template(path: Path, h: int = 60, w: int = 50) -> np.ndarray:
    """Official-template-shaped fixture: finite inside the footprint, NaN outside."""
    t = np.full((h, w), np.nan, dtype=np.float32)
    t[5:h - 3, 4:w - 4] = 0.0
    write_geotiff(path, t, TF)
    return t


def _argv(tmp: Path, template: Path | None, *, scores: Path | None = None,
          strategy: str = "topk_hard@0.03", demo: bool = False,
          allow_duplicate: bool = False) -> list[str]:
    out = tmp / "out"
    a = ["--strategy", strategy, "--out", str(out), "--log", str(tmp / "ledger.json")]
    if template is not None:
        a += ["--template", str(template)]
    else:
        a += ["--template", str(tmp / "does-not-exist.tif")]
    if scores is not None:
        a += ["--scores", str(scores)]
    if demo:
        a += ["--demo"]
    if allow_duplicate:
        a += ["--allow-duplicate"]
    return a


def _tifs(tmp: Path) -> list[Path]:
    return sorted((tmp / "out").glob("*.tif"))


def test_missing_template_fails_closed_instead_of_falling_back_to_a_fixture(tmp_path):
    """Before the guard this silently built a DEMO raster and logged it."""
    with pytest.raises(SystemExit) as exc:
        build_submission.main(_argv(tmp_path, None))
    assert exc.value.code == 2                      # argparse error
    assert _tifs(tmp_path) == []                    # nothing was written
    assert not (tmp_path / "ledger.json").exists()  # nothing was logged


def test_real_build_without_scores_is_refused(tmp_path):
    tpl = tmp_path / "template.tif"
    _tiny_template(tpl)
    with pytest.raises(SystemExit) as exc:
        build_submission.main(_argv(tmp_path, tpl))
    assert exc.value.code == 2
    assert _tifs(tmp_path) == []
    assert not (tmp_path / "ledger.json").exists()


def test_zeros_strategy_needs_no_scores_but_still_needs_the_template(tmp_path):
    tpl = tmp_path / "template.tif"
    _tiny_template(tpl)
    rc = build_submission.main(_argv(tmp_path, tpl, strategy="zeros"))
    assert rc == 0
    assert len(_tifs(tmp_path)) == 1
    log = json.loads((tmp_path / "ledger.json").read_text())["submissions"]
    assert len(log) == 1 and log[0]["demo"] is False


def test_demo_path_still_works_and_is_stamped_do_not_upload(tmp_path, capsys):
    rc = build_submission.main(_argv(tmp_path, None, demo=True))
    assert rc == 0
    out = _tifs(tmp_path)
    assert len(out) == 1 and "demo" in out[0].name
    printed = capsys.readouterr().out
    assert "DEMO — do not upload" in printed
    log = json.loads((tmp_path / "ledger.json").read_text())["submissions"]
    assert log[0]["demo"] is True


def test_real_build_from_a_score_field_succeeds(tmp_path):
    tpl = tmp_path / "template.tif"
    t = _tiny_template(tpl)
    scores = tmp_path / "scores.npy"
    rng = np.random.default_rng(3)
    np.save(scores, rng.random(t.shape).astype(np.float32))
    rc = build_submission.main(_argv(tmp_path, tpl, scores=scores))
    assert rc == 0
    out = _tifs(tmp_path)
    assert len(out) == 1 and "real" in out[0].name
    log = json.loads((tmp_path / "ledger.json").read_text())["submissions"]
    assert log[0]["demo"] is False
    assert log[0]["payload_sha256"] and len(log[0]["payload_sha256"]) == 64


def test_rebuilding_identical_bytes_is_refused_and_leaves_no_stray_raster(tmp_path):
    """Duplicate-payload guard: the second file must be rolled back, not left behind."""
    tpl = tmp_path / "template.tif"
    t = _tiny_template(tpl)
    scores = tmp_path / "scores.npy"
    np.save(scores, np.full(t.shape, 0.5, dtype=np.float32))
    assert build_submission.main(_argv(tmp_path, tpl, scores=scores)) == 0
    first = _tifs(tmp_path)
    assert len(first) == 1

    with pytest.raises((ValueError, SystemExit)):
        build_submission.main(_argv(tmp_path, tpl, scores=scores))

    assert _tifs(tmp_path) == first          # exactly one raster on disk
    log = json.loads((tmp_path / "ledger.json").read_text())["submissions"]
    assert len(log) == 1                     # and exactly one ledger entry


def test_duplicate_can_be_overridden_explicitly(tmp_path):
    tpl = tmp_path / "template.tif"
    t = _tiny_template(tpl)
    scores = tmp_path / "scores.npy"
    np.save(scores, np.full(t.shape, 0.25, dtype=np.float32))
    assert build_submission.main(_argv(tmp_path, tpl, scores=scores)) == 0
    # Names are second-granularity, so the override may be refused either by the
    # ledger guard (rc 0 path is impossible then) or by the overwrite guard -- both
    # are correct fail-closed outcomes; what must never happen is a silent second
    # copy of the same bytes.
    try:
        rc = build_submission.main(
            _argv(tmp_path, tpl, scores=scores, allow_duplicate=True))
    except SystemExit as exc:
        rc = exc.code
    assert rc in (0, 2)
    log = json.loads((tmp_path / "ledger.json").read_text())["submissions"]
    if rc == 0:
        assert len(log) == 2 and "duplicate_warning" in log[1]
        assert len(_tifs(tmp_path)) == 2
    else:
        assert len(log) == 1 and len(_tifs(tmp_path)) == 1
