"""Feature assembly + scaler: train-only fit, correct range, invertibility."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_fcnn.data.scalers import MinMaxScaler  # noqa: E402


def test_minmax_classification_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3)).astype("float32")
    Xs = MinMaxScaler((0.0, 1.0)).fit_transform(X)
    assert np.isclose(Xs.min(), 0.0) and np.isclose(Xs.max(), 1.0)


def test_minmax_regression_range():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3)).astype("float32")
    Xs = MinMaxScaler((-1.0, 1.0)).fit_transform(X)
    assert np.isclose(Xs.min(), -1.0) and np.isclose(Xs.max(), 1.0)


def test_scaler_fit_on_train_only_no_leakage():
    """A val point outside the train range may exceed [0,1] -> proves no refit."""
    X_tr = np.array([[0.0], [1.0], [2.0]], dtype="float32")
    X_val = np.array([[4.0]], dtype="float32")  # beyond train max
    s = MinMaxScaler((0.0, 1.0)).fit(X_tr)
    assert s.transform(X_val)[0, 0] > 1.0


def test_scaler_json_roundtrip():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 3)).astype("float32")
    s = MinMaxScaler((-1.0, 1.0)).fit(X)
    s2 = MinMaxScaler.from_dict(s.to_dict())
    assert np.allclose(s.transform(X), s2.transform(X))
