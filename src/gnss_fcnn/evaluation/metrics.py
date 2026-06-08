"""Metrics for the benchmark tasks (paper Eqs. 9-11).

Classification: accuracy (Eq. 9, the paper's metric) plus precision/recall/F1/
ROC-AUC for completeness, with the **positive class = NLOS** (label 0), matching
the paper's TP = correctly identified NLOS. Regression: RMSE (Eq. 10) and MAE.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

NLOS = 0  # positive class for precision/recall/F1 (CLAUDE.md §8)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def baseline_rmse(pr_error: np.ndarray) -> float:
    """Uncorrected RMSE of the pseudorange error (prediction = 0)."""
    return rmse(pr_error, np.zeros_like(pr_error))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prob_nlos: np.ndarray | None = None
) -> dict:
    """Accuracy (primary) + NLOS-positive precision/recall/F1 (+ ROC-AUC)."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_nlos": float(precision_score(y_true, y_pred, pos_label=NLOS, zero_division=0)),
        "recall_nlos": float(recall_score(y_true, y_pred, pos_label=NLOS, zero_division=0)),
        "f1_nlos": float(f1_score(y_true, y_pred, pos_label=NLOS, zero_division=0)),
    }
    if prob_nlos is not None:
        # roc_auc_score expects score of the positive class; (y_true==NLOS) -> 1
        out["roc_auc"] = float(roc_auc_score((np.asarray(y_true) == NLOS).astype(int), prob_nlos))
    return out
