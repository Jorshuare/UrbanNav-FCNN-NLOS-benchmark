"""Figure generators for the reproduction (paper palette).

Reproduces Figure 13 (C/N0 vs elevation, LOS=blue / NLOS=orange) and the
training/validation curves for the selected models. Uses the Agg backend so it
runs headless; writes PNG + PDF at 300 dpi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gnss_fcnn.viz import palette  # noqa: E402


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def figure13(
    elev_train: np.ndarray, cnr_train: np.ndarray, label_train: np.ndarray,
    elev_test: np.ndarray, cnr_test: np.ndarray, pred_test: np.ndarray,
    out_dir: Path,
) -> list[Path]:
    """Two-panel C/N0-vs-elevation scatter (paper Fig. 13).

    label/pred == 1 -> LOS (blue), == 0 -> NLOS (orange). The test panel is
    coloured by the model's predicted class (the held-out labels are not public).
    """
    palette.apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)

    panels = [
        (axes[0], elev_train, cnr_train, label_train, "(a) Training set — true label"),
        (axes[1], elev_test, cnr_test, pred_test, "(b) Test set — predicted label"),
    ]
    for ax, elev, cnr, cls, title in panels:
        los, nlos = cls == 1, cls == 0
        ax.scatter(elev[los], cnr[los], s=3, c=palette.LOS, alpha=0.35, label="LOS", rasterized=True)
        ax.scatter(elev[nlos], cnr[nlos], s=3, c=palette.NLOS, alpha=0.35, label="NLOS", rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Elevation Angle (degree)")
        ax.set_xlim(0, 90)
        leg = ax.legend(markerscale=4, framealpha=0.9)
        for h in leg.legend_handles:
            h.set_alpha(1.0)
    axes[0].set_ylabel(r"C/N$_0$ (dB-Hz)")
    fig.suptitle("Figure 13 — LOS/NLOS distribution in C/N$_0$ vs elevation", fontsize=11)
    return _save(fig, out_dir, "fig13_cnr_vs_elevation")


def training_curves(history: list[dict], task: str, model: str, out_dir: Path) -> list[Path]:
    """Loss + validation metric per epoch for one trained model."""
    palette.apply_paper_style()
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    metric_key = "val_accuracy" if task == "classification" else "val_rmse"
    metric = [h[metric_key] for h in history]
    metric_label = "Val accuracy" if task == "classification" else "Val RMSE (m)"

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, train_loss, color=palette.GRAY, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color=palette.GRAY)
    ax1.tick_params(axis="y", labelcolor=palette.GRAY)

    colour = palette.CLASSIFICATION if task == "classification" else palette.REGRESSION
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(epochs, metric, color=colour, label=metric_label)
    ax2.set_ylabel(metric_label, color=colour)
    ax2.tick_params(axis="y", labelcolor=colour)

    fig.suptitle(f"{model} ({task}) — training curve", fontsize=11)
    return _save(fig, out_dir, f"training_curve_{task}_{model}")
