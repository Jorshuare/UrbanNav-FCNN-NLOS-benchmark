"""FCNN factory: each Table-8 variant has correct shape, depth, and head dims."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_fcnn.config import load_config  # noqa: E402
from gnss_fcnn.models import heads  # noqa: E402
from gnss_fcnn.models.fcnn import build_fcnn  # noqa: E402

VARIANTS = {
    "fcnn_m1": [64, 128, 64],
    "fcnn_m2": [64, 256, 64],
    "fcnn_m3": [64, 128, 32],
    "fcnn_m4": [128, 128, 64],
    "fcnn_m5": [64, 128, 64, 64],
}


def _linear_widths(net) -> list[int]:
    return [m.out_features for m in net.net if isinstance(m, torch.nn.Linear)]


@pytest.mark.parametrize("model,hidden", VARIANTS.items())
def test_classification_architecture(model, hidden):
    net = build_fcnn(load_config(model=model, features="classification", train="classification"))
    assert _linear_widths(net) == hidden + [2]          # hidden + 2-node head
    assert net(torch.randn(16, 3)).shape == (16, 2)
    assert net.n_parameters > 0


@pytest.mark.parametrize("model,hidden", VARIANTS.items())
def test_regression_architecture(model, hidden):
    net = build_fcnn(load_config(model=model, features="regression", train="regression"))
    assert _linear_widths(net) == hidden + [1]          # hidden + 1-node head
    assert net(torch.randn(16, 3)).shape == (16, 1)


def test_relu_count_matches_hidden_depth():
    net = build_fcnn(load_config(model="fcnn_m5"))       # 4 hidden layers
    n_relu = sum(1 for m in net.net if isinstance(m, torch.nn.ReLU))
    assert n_relu == 4


def test_classification_head_semantics():
    net = build_fcnn(load_config(model="fcnn_m2"))
    logits = net(torch.randn(32, 3))
    probs = heads.class_probabilities(logits)
    assert torch.allclose(probs.sum(dim=1), torch.ones(32), atol=1e-5)
    preds = heads.predicted_class(logits)
    assert preds.min() >= 0 and preds.max() <= 1          # classes {0=NLOS, 1=LOS}


def test_seed_determinism_in_init():
    from gnss_fcnn.training.seed import seed_everything
    seed_everything(42)
    a = build_fcnn(load_config(model="fcnn_m4"))
    seed_everything(42)
    b = build_fcnn(load_config(model="fcnn_m4"))
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)                        # same seed -> same init
