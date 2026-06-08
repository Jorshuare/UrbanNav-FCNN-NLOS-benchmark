"""Learning-rate schedule (Table 7): multiplicative ×0.99 decay per epoch."""

from __future__ import annotations

from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler


def build_scheduler(optimizer: Optimizer, cfg: DictConfig) -> LRScheduler:
    sched = cfg.train.lr_schedule
    if sched.name.lower() == "exponential":
        return ExponentialLR(optimizer, gamma=float(sched.gamma))
    raise ValueError(f"unsupported lr_schedule '{sched.name}'")
