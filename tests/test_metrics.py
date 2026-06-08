"""Metric correctness on known inputs: accuracy (Eq.9), RMSE (Eq.10), E/N (Eq.11)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnss_fcnn.evaluation.metrics import accuracy, baseline_rmse, mae, rmse  # noqa: E402
from gnss_fcnn.evaluation.positioning import east_north_rmse  # noqa: E402


def test_rmse_and_mae_known():
    y = np.array([0.0, 0.0, 0.0]); p = np.array([3.0, 0.0, 4.0])
    assert np.isclose(rmse(y, p), np.sqrt((9 + 0 + 16) / 3))
    assert np.isclose(mae(y, p), (3 + 0 + 4) / 3)


def test_baseline_rmse_is_quadratic_mean():
    e = np.array([3.0, 4.0])
    assert np.isclose(baseline_rmse(e), np.sqrt((9 + 16) / 2))


def test_accuracy_known():
    assert np.isclose(accuracy(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])), 0.75)


def test_east_north_rmse_eq11():
    e = np.array([3.0, 0.0]); n = np.array([0.0, 4.0])
    # sqrt( mean( (e^2+n^2)/2 ) ) = sqrt( ((9/2)+(16/2))/2 )
    assert np.isclose(east_north_rmse(e, n), np.sqrt(((9 / 2) + (16 / 2)) / 2))
