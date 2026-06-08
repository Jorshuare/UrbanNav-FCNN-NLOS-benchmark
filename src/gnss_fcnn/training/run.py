"""Training runs: load processed data -> build FCNN -> train -> (save).

``fit_model`` returns the trained pieces so evaluation can predict on the test
set; ``train_single`` wraps it to persist a checkpoint + history. Shared by
scripts/02, scripts/03, and scripts/04 (DRY).
"""

from __future__ import annotations

import torch

from gnss_fcnn.config import abspath, load_config
from gnss_fcnn.models.fcnn import build_fcnn
from gnss_fcnn.training.seed import seed_everything
from gnss_fcnn.training.trainer import Trainer
from gnss_fcnn.utils.io import load_arrays, save_json


def fit_model(*, model: str, task: str, seed: int | None = None, device: str = "cpu"):
    """Train one FCNN; return (cfg, trainer, data, summary)."""
    cfg = load_config(model=model, features=task, train=task)
    if seed is not None:
        cfg.seed = int(seed)
    seed_everything(int(cfg.seed), deterministic=bool(cfg.deterministic))

    proc = abspath(cfg, cfg.paths.processed_dir) / task
    data = {k: load_arrays(proc / f"{k}.npz") for k in ("train", "val", "test")}

    net = build_fcnn(cfg)
    trainer = Trainer(net, cfg, device=device)
    result = trainer.fit(data["train"]["X"], data["train"]["y"],
                         data["val"]["X"], data["val"]["y"])

    train_eval = trainer.evaluate(data["train"]["X"], data["train"]["y"])
    val_eval = trainer.evaluate(data["val"]["X"], data["val"]["y"])
    summary = {
        "task": task, "model": cfg.model.name, "seed": int(cfg.seed),
        "best_epoch": result["best_epoch"], "stopped_epoch": result["stopped_epoch"],
        "n_parameters": net.n_parameters,
        "train_metric": train_eval["metric"], "val_metric": val_eval["metric"],
        **{f"val_{k.split('_')[-1]}": v for k, v in val_eval.items() if k.startswith("val_")},
    }
    return cfg, trainer, data, summary, result


def train_single(*, model: str, task: str, seed: int | None = None,
                  save: bool = True, device: str = "cpu") -> dict:
    cfg, trainer, _, summary, result = fit_model(model=model, task=task, seed=seed, device=device)
    if save:
        results = abspath(cfg, cfg.paths.results_dir)
        tag = f"{task}_{cfg.model.name}_seed{cfg.seed}"
        (results / "models").mkdir(parents=True, exist_ok=True)
        torch.save(trainer.model.state_dict(), results / "models" / f"{tag}.pt")
        save_json({"summary": summary, "history": result["history"]},
                  results / "metrics" / f"{tag}.json")
    return summary
