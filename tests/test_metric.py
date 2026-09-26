"""Tests for Distance-Weighted Tversky Index metric."""

import numpy as np
import pytest
from src.metric import dtvi, dtvi_components


def test_metric_perfect_match():
    # 10x10 grid with a single fault line
    truth = np.zeros((10, 10), dtype=np.int8)
    truth[2:7, 5] = 1
    pred = truth.astype(np.float64)

    c = dtvi_components(pred, truth)
    score = dtvi(pred, truth)

    assert c["n_truth"] == 5
    assert c["TP_w"] == pytest.approx(5.0)
    assert c["FP_w"] == pytest.approx(0.0)
    assert c["FN_w"] == pytest.approx(0.0)
    assert score == pytest.approx(1.0)


def test_metric_zero_predictions():
    truth = np.zeros((10, 10), dtype=np.int8)
    truth[2:7, 5] = 1
    pred = np.zeros((10, 10), dtype=np.float64)

    c = dtvi_components(pred, truth)
    score = dtvi(pred, truth)

    assert c["n_truth"] == 5
    assert c["TP_w"] == pytest.approx(0.0)
    assert c["FP_w"] == pytest.approx(0.0)
    assert c["FN_w"] == pytest.approx(5.0)
    assert score == pytest.approx(0.0)


def test_metric_tp_fn_identity():
    truth = np.zeros((15, 15), dtype=np.int8)
    truth[3:12, 7] = 1
    # Random predictions in [0, 1]
    rng = np.random.RandomState(42)
    pred = rng.rand(15, 15)

    c = dtvi_components(pred, truth)
    # The mathematical identity TP_w + FN_w == |G| must hold exactly
    assert c["TP_w"] + c["FN_w"] == pytest.approx(c["n_truth"])


def test_metric_kernel_distance_decay():
    # Single point ground truth at (5, 5)
    truth = np.zeros((11, 11), dtype=np.int8)
    truth[5, 5] = 1

    # Prediction 1 pixel away: d=1 -> k(1) = 1 - 1/3 = 2/3
    pred_1 = np.zeros((11, 11), dtype=np.float64)
    pred_1[5, 6] = 1.0
    c1 = dtvi_components(pred_1, truth)
    assert c1["TP_w"] == pytest.approx(2.0 / 3.0)

    # Prediction 2 pixels away: d=2 -> k(2) = 1 - 2/3 = 1/3
    pred_2 = np.zeros((11, 11), dtype=np.float64)
    pred_2[5, 7] = 1.0
    c2 = dtvi_components(pred_2, truth)
    assert c2["TP_w"] == pytest.approx(1.0 / 3.0)

    # Prediction 3 pixels away: d=3 -> k(3) = 0
    pred_3 = np.zeros((11, 11), dtype=np.float64)
    pred_3[5, 8] = 1.0
    c3 = dtvi_components(pred_3, truth)
    assert c3["TP_w"] == pytest.approx(0.0)
