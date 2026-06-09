"""Visualisation: palette constants present + figure generators write files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_fcnn.viz import figures, palette  # noqa: E402


def test_palette_has_paper_hex():
    assert palette.LOS == "#4472C4" and palette.NLOS == "#ED7D31"
    assert palette.GRAY == "#A5A5A5"
    assert palette.TABLE_FILL == "#A9D08E"


def test_figure13_writes_nonempty(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    elev = rng.uniform(5, 90, n); cnr = rng.uniform(10, 50, n)
    lab = rng.integers(0, 2, n)
    paths = figures.figure13(elev, cnr, lab, elev, cnr, lab, tmp_path)
    assert len(paths) == 2 and all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_training_curves_writes_nonempty(tmp_path):
    hist = [{"epoch": e, "train_loss": 1.0 / e, "val_accuracy": 0.5 + e / 100,
             "val_rmse": 20.0 - e} for e in range(1, 11)]
    p_clf = figures.training_curves(hist, "classification", "M2", tmp_path)
    p_reg = figures.training_curves(hist, "regression", "M4", tmp_path)
    assert all(p.exists() and p.stat().st_size > 0 for p in p_clf + p_reg)
