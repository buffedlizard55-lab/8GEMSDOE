"""Submission builder/validator tests (synthetic grids, no data needed)."""
import tempfile
from pathlib import Path

import numpy as np

from gems.submission import (conform_to_template, footprint_of_template,
                             read_geotiff, validate_submission, write_geotiff)

TF = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)


def _template(h=40, w=34):
    t = np.zeros((h, w), dtype=np.float32)
    t[0:4, :] = np.nan
    return t


def test_roundtrip_and_gates_pass():
    t = _template()
    raw = np.random.default_rng(3).random(t.shape)
    good, rep = conform_to_template(raw, t)
    assert rep["nan_outside"] == 4 * t.shape[1]
    with tempfile.TemporaryDirectory() as td:
        tp, gp = Path(td) / "t.tif", Path(td) / "g.tif"
        write_geotiff(tp, t, TF)
        write_geotiff(gp, good, TF)
        tinfo = read_geotiff(tp)
        assert tinfo["epsg"] == 32611
        assert tinfo["transform"] == TF
        res = validate_submission(gp, tinfo)
        assert res["pass"], res["gates"]
        assert footprint_of_template(t).sum() == (t.shape[0] - 4) * t.shape[1]


def test_conform_fixes_rejection_conditions():
    t = _template()
    bad = np.ones(t.shape) * 0.7
    bad[10, 10] = np.nan   # inside -> would trigger [0,1] rejection
    bad[1, 1] = 0.9        # outside -> must become NaN
    bad[11, 11] = 2.5      # >1 -> must clip
    good, rep = conform_to_template(bad, t)
    fp = footprint_of_template(t)
    assert np.isfinite(good[fp]).all()
    assert np.isnan(good[~fp]).all()
    assert np.nanmin(good) >= 0.0 and np.nanmax(good) <= 1.0
    assert rep["nan_inside_filled_with_0"] == 1
