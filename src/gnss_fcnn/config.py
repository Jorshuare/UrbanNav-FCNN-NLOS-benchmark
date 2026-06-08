"""Configuration loading.

Single source of truth for run configuration; composes the YAML files in
``configs/`` (mirrors a Hydra defaults list) and resolves ``${...}``
interpolations. No hyperparameter should live outside ``configs/`` + this module.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Repository root = two levels up from this file (src/gnss_fcnn/config.py).
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


def load_config(
    *,
    model: str = "fcnn_m2",
    features: str = "classification",
    train: str = "classification",
    data: str = "default",
    eval: str = "default",
    overrides: dict | None = None,
) -> DictConfig:
    """Compose the run config from the group YAML files.

    Mirrors ``configs/config.yaml``'s defaults list but is explicit so scripts
    can select the task without Hydra's working-directory side effects.
    """
    base = OmegaConf.load(CONFIGS / "config.yaml")
    base.pop("defaults", None)  # we resolve the group selection ourselves

    groups = {
        "data": OmegaConf.load(CONFIGS / "data" / f"{data}.yaml"),
        "features": OmegaConf.load(CONFIGS / "features" / f"{features}.yaml"),
        "model": OmegaConf.load(CONFIGS / "model" / f"{model}.yaml"),
        "train": OmegaConf.load(CONFIGS / "train" / f"{train}.yaml"),
        "eval": OmegaConf.load(CONFIGS / "eval" / f"{eval}.yaml"),
    }
    cfg = OmegaConf.merge(base, groups)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    OmegaConf.resolve(cfg)
    cfg.root = str(ROOT)
    return cfg


def abspath(cfg: DictConfig, p: str) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(p)
    return path if path.is_absolute() else ROOT / path
