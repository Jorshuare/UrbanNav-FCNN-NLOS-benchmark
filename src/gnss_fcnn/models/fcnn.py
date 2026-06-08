"""Parametrised feed-forward network — one factory drives all 5 variants (DRY).

Architecture (Figure 12, Table 8): 3 inputs -> hidden layers (Linear + ReLU) ->
task head (a final Linear). The 5 variants M1..M5 and the 2 task heads are pure
configuration differences. The model returns raw logits/values; softmax (clf) is
applied at the loss / inference (see models/heads.py), regression is linear.

Weight init: PyTorch default (Kaiming-uniform for Linear) — the paper does not
specify one; it is held fixed by the global seed (training/seed.py).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omegaconf import DictConfig
from torch import nn

from gnss_fcnn.models.heads import INPUT_DIM, build_activation


class FCNN(nn.Module):
    """Fully connected network: in_dim -> [hidden (act)]* -> out_dim (linear)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.hidden_dims = list(hidden_dims)

        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden_dims:
            layers += [nn.Linear(prev, width), build_activation(activation)]
            prev = width
        layers.append(nn.Linear(prev, out_dim))  # task head: no output activation
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_fcnn(cfg: DictConfig) -> FCNN:
    """Construct the FCNN for the composed config (model variant + task head)."""
    return FCNN(
        in_dim=INPUT_DIM,
        hidden_dims=cfg.model.hidden_dims,
        out_dim=cfg.train.head.out_dim,
        activation=cfg.model.activation,
    )
