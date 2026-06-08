"""Loss functions (Table 7).

- classification: ``CrossEntropyLoss`` — softmax + negative-log-likelihood in one
  numerically stable op (faithful to the paper's "negative log-likelihood loss"
  with a softmax output layer; expects raw logits).
- regression: ``L1Loss`` (MAE) — robust to the heavy NLOS pseudorange outliers.
"""

from __future__ import annotations

from omegaconf import DictConfig
from torch import nn


def build_loss(cfg: DictConfig) -> nn.Module:
    name = cfg.train.loss.lower()
    if name == "nll":
        return nn.CrossEntropyLoss()
    if name == "mae":
        return nn.L1Loss()
    raise ValueError(f"unsupported loss '{cfg.train.loss}'")
