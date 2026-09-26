"""Regression tests for the blocked-holdout harness (exact budgets, no leakage)."""
import numpy as np
import pytest

from gems.holdout import block_folds, score_on_fold, topk_field


def test_topk_exact_budget_on_plateau():
    # Plateau field: 90% zeros. A >=threshold rule would over-select massively;
    # exact-k must emit EXACTLY round(frac * n_footprint) pixels.
    rng = np.random.default_rng(0)
    score = np.zeros((50, 40))
    score[rng.random((50, 40)) < 0.1] = 1.0  # 10% ones on a 90% zero plateau
    fp = np.ones((50, 40), dtype=bool)
    fp[0:5, :] = False
    field = topk_field(score, fp, 0.03)
    n_fp = int(fp.sum())
    assert int(field.sum()) == int(round(0.03 * n_fp))
    assert set(np.unique(field)).issubset({0.0, 1.0})


def test_topk_deterministic_tiebreak():
    score = np.zeros((30, 30))  # total tie: every pixel equal
    fp = np.ones((30, 30), dtype=bool)
    a = topk_field(score, fp, 0.05)
    b = topk_field(score, fp, 0.05)
    assert np.array_equal(a, b)
    assert int(a.sum()) == int(round(0.05 * fp.sum()))


def test_topk_respects_footprint():
    rng = np.random.default_rng(1)
    score = rng.random((20, 20))
    fp = np.zeros((20, 20), dtype=bool)
    fp[5:15, 5:15] = True
    field = topk_field(score, fp, 0.10)
    assert (field[~fp] == 0.0).all()
    assert int(field.sum()) == int(round(0.10 * fp.sum()))


def test_block_folds_buffer_excludes_kernel():
    # No train pixel may sit inside the metric kernel (R=3) of a test pixel:
    # buffer ring must cover the 3px neighbourhood of every test block.
    fp = np.ones((40, 40), dtype=bool)
    truth = np.zeros((40, 40))
    folds = block_folds((40, 40), fp, n_blocks=4, buffer_px=3, seed=8)
    from scipy import ndimage
    for f in folds:
        dil = ndimage.binary_dilation(
            f["test_mask"], structure=np.ones((7, 7), dtype=bool))
        leaked = dil & f["train_mask"]
        assert not leaked.any(), f"fold {f['fold']}: train inside kernel"
        # partition covers footprint exactly once
        cover = (f["train_mask"].astype(int) + f["test_mask"].astype(int)
                 + f["buffer_mask"].astype(int))
        assert (cover[fp] == 1).all()


def test_score_on_fold_known_masking():
    # Masking known pixels must not change TP/FN, only (weakly) FP.
    rng = np.random.default_rng(2)
    pred = (rng.random((30, 30)) < 0.1).astype(float)
    truth = np.zeros((30, 30))
    truth[10:20, 14:16] = 1.0
    tm = np.ones((30, 30), dtype=bool)
    known = truth > 0
    r = score_on_fold(pred, truth, tm, known_mask=known)
    assert r["unmasked"]["DTI"] == pytest.approx(
        r["known_masked_exact"]["DTI"], abs=1e-9)  # truth==known: identical
    assert r["n_truth_in_fold"] == 20
