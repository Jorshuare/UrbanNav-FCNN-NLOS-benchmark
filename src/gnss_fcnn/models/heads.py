"""Task-specific output semantics for the FCNN (Figure 12).

The network itself emits raw logits; this module holds the head conventions so
``fcnn.py`` stays a generic MLP (SOC):

- classification: 2 logits (P_NLOS, P_LOS); softmax -> probabilities, argmax ->
  class. Output-node order is (NLOS, LOS) so class index 0 = NLOS, 1 = LOS,
  matching the pinned dataset label (CLAUDE.md §8).
- regression: 1 linear output = pseudorange correction in metres (identity).
"""

from __future__ import annotations

import torch
from torch import nn

INPUT_DIM = 3  # CNR, Elevation, Pr_Residual


def build_activation(name: str) -> nn.Module:
    """Map a config activation name to a module (ReLU for all hidden layers)."""
    activations = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    key = name.lower()
    if key not in activations:
        raise ValueError(f"unsupported activation '{name}'")
    return activations[key]()


def class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over the 2 class logits -> (P_NLOS, P_LOS), rows sum to 1."""
    return torch.softmax(logits, dim=1)


def predicted_class(logits: torch.Tensor) -> torch.Tensor:
    """Argmax decision -> class index (0 = NLOS, 1 = LOS)."""
    return logits.argmax(dim=1)
