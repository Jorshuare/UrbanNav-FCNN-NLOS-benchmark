"""Training layer: loss/scheduler construction + that the loop actually learns."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_fcnn.config import load_config  # noqa: E402
from gnss_fcnn.models.fcnn import build_fcnn  # noqa: E402
from gnss_fcnn.training.losses import build_loss  # noqa: E402
from gnss_fcnn.training.schedulers import build_scheduler  # noqa: E402
from gnss_fcnn.training.seed import seed_everything  # noqa: E402
from gnss_fcnn.training.trainer import Trainer  # noqa: E402


def test_build_loss_per_task():
    assert isinstance(build_loss(load_config(train="classification", features="classification")),
                      torch.nn.CrossEntropyLoss)
    assert isinstance(build_loss(load_config(train="regression", features="regression")),
                      torch.nn.L1Loss)


def test_scheduler_decays_099_per_epoch():
    cfg = load_config()
    net = build_fcnn(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    sched = build_scheduler(opt, cfg)
    lr0 = opt.param_groups[0]["lr"]
    sched.step()
    assert np.isclose(opt.param_groups[0]["lr"], lr0 * 0.99)


def test_classification_loop_learns_separable_data():
    """On linearly separable toy data the loop should reach high accuracy."""
    seed_everything(0)
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, (300, 3)), rng.normal(2, 0.5, (300, 3))]).astype("float32")
    y = np.array([0] * 300 + [1] * 300)
    cfg = load_config(model="fcnn_m1", train="classification", features="classification",
                      overrides={"train": {"epochs": 30, "early_stopping": {"patience": 30}}})
    res = Trainer(build_fcnn(cfg), cfg).fit(X, y, X, y)
    assert res["best_metric"] > 0.95


def test_regression_loop_reduces_rmse():
    """Train loss/RMSE should drop substantially on a learnable linear target."""
    seed_everything(0)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 3)).astype("float32")
    y = (3.0 * X[:, 0] - 2.0 * X[:, 1]).astype("float32")
    cfg = load_config(model="fcnn_m1", train="regression", features="regression",
                      overrides={"train": {"epochs": 40, "early_stopping": {"patience": 40}}})
    res = Trainer(build_fcnn(cfg), cfg).fit(X, y, X, y)
    assert res["history"][-1]["val_rmse"] < res["history"][0]["val_rmse"] * 0.5


def test_seed_reproducibility_end_to_end():
    """Same seed -> identical validation metric."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3)).astype("float32")
    y = (X[:, 0] > 0).astype("int64")
    cfg = load_config(model="fcnn_m1", train="classification", features="classification",
                      overrides={"train": {"epochs": 10, "early_stopping": {"patience": 10}}})
    seed_everything(123); a = Trainer(build_fcnn(cfg), cfg).fit(X, y, X, y)["best_metric"]
    seed_everything(123); b = Trainer(build_fcnn(cfg), cfg).fit(X, y, X, y)["best_metric"]
    assert a == b
