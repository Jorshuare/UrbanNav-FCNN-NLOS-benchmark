"""Generic train/validate loop shared by both tasks (DRY).

Implements the paper's procedure (Table 7): Adam (lr 0.01), ExponentialLR(0.99)
stepped per epoch, CrossEntropy/MAE loss, with validation-metric early stopping
(epoch count is unspecified in the paper). The best-validation state is restored
at the end; a per-epoch history is returned for the training-curve figure.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from gnss_fcnn.evaluation.metrics import rmse
from gnss_fcnn.training.losses import build_loss
from gnss_fcnn.training.schedulers import build_scheduler


def pick_device(prefer: str = "cpu") -> torch.device:
    """Select a device. Default CPU: the model is tiny (≤33k params), so CPU is
    fast *and* fully deterministic; MPS/CUDA available on request."""
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Trainer:
    """Train one FCNN for one task according to ``cfg``."""

    def __init__(self, model: torch.nn.Module, cfg: DictConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.task = cfg.train.task
        self.device = pick_device(device)
        self.model = model.to(self.device)
        self.loss_fn = build_loss(cfg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg.train.optimizer.lr))
        self.scheduler = build_scheduler(self.optimizer, cfg)

        es = cfg.train.early_stopping
        self.monitor_mode = es.mode                 # 'max' (accuracy) / 'min' (rmse)
        self.patience = int(es.patience)
        self.max_epochs = int(cfg.train.epochs)
        self.batch_size = int(cfg.train.batch_size)
        self.seed = int(cfg.seed)

    # ---- data ------------------------------------------------------------
    def _targets(self, y: np.ndarray) -> torch.Tensor:
        if self.task == "classification":
            return torch.as_tensor(y, dtype=torch.long)
        return torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)

    def _train_loader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), self._targets(y))
        g = torch.Generator().manual_seed(self.seed)   # reproducible shuffling
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, generator=g)

    # ---- evaluation ------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        self.model.eval()
        out = self.model(torch.as_tensor(X, dtype=torch.float32, device=self.device)).cpu()
        if self.task == "classification":
            preds = out.argmax(dim=1).numpy()
            acc = float((preds == y).mean())
            loss = float(self.loss_fn(out, torch.as_tensor(y, dtype=torch.long)))
            return {"val_loss": loss, "val_accuracy": acc, "metric": acc}
        preds = out.squeeze(1).numpy()
        loss = float(self.loss_fn(out.squeeze(1), torch.as_tensor(y, dtype=torch.float32)))
        return {"val_loss": loss, "val_rmse": rmse(y, preds), "metric": rmse(y, preds)}

    # ---- fit -------------------------------------------------------------
    def fit(self, X_tr, y_tr, X_val, y_val) -> dict:
        loader = self._train_loader(X_tr, y_tr)
        improved = (lambda new, best: new > best) if self.monitor_mode == "max" else (lambda new, best: new < best)

        history: list[dict] = []
        best_metric, best_state, best_epoch, wait = None, None, 0, 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            total, n = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                total += loss.item() * len(xb)
                n += len(xb)
            self.scheduler.step()                       # per-epoch LR decay

            val = self.evaluate(X_val, y_val)
            lr = self.optimizer.param_groups[0]["lr"]
            history.append({"epoch": epoch, "train_loss": total / n, **val, "lr": lr})

            if best_metric is None or improved(val["metric"], best_metric):
                best_metric, best_epoch = val["metric"], epoch
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        self.model.load_state_dict(best_state)          # restore best-val weights
        return {"history": history, "best_metric": best_metric, "best_epoch": best_epoch,
                "stopped_epoch": history[-1]["epoch"]}

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Class index (clf) or pseudorange correction in metres (reg)."""
        self.model.eval()
        out = self.model(torch.as_tensor(X, dtype=torch.float32, device=self.device)).cpu()
        return out.argmax(dim=1).numpy() if self.task == "classification" else out.squeeze(1).numpy()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities (N, 2) = (P_NLOS, P_LOS). Classification only."""
        self.model.eval()
        logits = self.model(torch.as_tensor(X, dtype=torch.float32, device=self.device)).cpu()
        return torch.softmax(logits, dim=1).numpy()
