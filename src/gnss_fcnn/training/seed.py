"""Reproducibility: seed Python, NumPy, and torch; enable deterministic algorithms."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed all RNGs and (optionally) force deterministic algorithms.

    Called once at the start of every run so results are bit-reproducible given
    the same environment (CLAUDE.md §10).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
