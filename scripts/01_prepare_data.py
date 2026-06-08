"""Build model-ready features from data/raw/*.xlsx into data/processed/.

Reproduction steps (CLAUDE.md §4, §6; guide steps 1-4), per task:
    load xlsx -> assert invariants -> (regression: drop master sats)
    -> assemble [CNR, Elevation, Pr_Residual] -> 80/20 split (seeded; stratified
    for classification) -> fit min-max scaler on TRAIN only -> transform
    train/val + the held-out test features.

Outputs per task in data/processed/<task>/:
    train.npz (X, y), val.npz (X, y), test.npz (X), scaler.json, meta.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_fcnn.config import abspath, load_config  # noqa: E402
from gnss_fcnn.data import clean, features, loader, splits  # noqa: E402
from gnss_fcnn.data.scalers import MinMaxScaler  # noqa: E402
from gnss_fcnn.utils.io import save_arrays, save_json  # noqa: E402

TASKS = [
    ("classification", "fcnn_m2"),
    ("regression", "fcnn_m4"),
]


def prepare_task(task: str, model: str) -> dict:
    cfg = load_config(model=model, features=task, train=task)
    cols = cfg.data.columns

    # --- load + invariants (on the full, unfiltered training table) ----------
    df = loader.load_table(abspath(cfg, cfg.data.train_xlsx))
    clean.assert_train_invariants(df, cols.label)

    required = [cols.cnr, cols.elevation, cols.pr_residual, cols.label, cols.pr_error]
    df = clean.drop_invalid(df, required)

    # --- regression: drop differencing master satellites (Pr_Error == 0) -----
    if task == "regression" and cfg.data.cleaning.drop_master_satellites_for_regression:
        df = clean.drop_master_satellites(df, cols.pr_error)

    # --- features + target ---------------------------------------------------
    X = features.feature_matrix(df, cfg)
    y = features.target_vector(df, cfg)

    # --- 80/20 split (seeded; stratify for classification) -------------------
    stratify = bool(cfg.data.split.stratify_classification) and task == "classification"
    X_tr, X_val, y_tr, y_val = splits.train_val_split(
        X, y,
        val_fraction=cfg.data.split.val_fraction,
        seed=int(cfg.data.split.seed),
        stratify=stratify,
    )

    # --- min-max scaler fit on TRAIN only (no leakage) -----------------------
    lo, hi = (float(v) for v in cfg.features.normalisation.range)
    scaler = MinMaxScaler((lo, hi)).fit(X_tr)
    X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)

    # --- transform the held-out test features (no labels in the file) --------
    df_test = loader.load_table(abspath(cfg, cfg.data.test_xlsx))
    X_test = features.feature_matrix(df_test, cfg)
    X_test_s = scaler.transform(X_test)

    # --- persist -------------------------------------------------------------
    out = abspath(cfg, cfg.paths.processed_dir) / task
    save_arrays(out / "train.npz", X=X_tr_s, y=y_tr)
    save_arrays(out / "val.npz", X=X_val_s, y=y_val)
    save_arrays(out / "test.npz", X=X_test_s)
    save_json(scaler.to_dict(), out / "scaler.json")

    meta = {
        "task": task,
        "feature_order": ["CNR", "Elevation", "Pr_Residual"],
        "normalisation_range": [lo, hi],
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "stratified": stratify,
        "seed": int(cfg.data.split.seed),
        "scaled_train_min": [round(float(v), 4) for v in X_tr_s.min(0)],
        "scaled_train_max": [round(float(v), 4) for v in X_tr_s.max(0)],
    }
    if task == "classification":
        meta["train_label_counts"] = {int(k): int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))}
        meta["val_label_counts"] = {int(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))}
    else:
        meta["val_gt_rmse_uncorrected"] = round(float(np.sqrt(np.mean(y_val ** 2))), 2)
    save_json(meta, out / "meta.json")
    return meta


def main() -> None:
    for task, model in TASKS:
        meta = prepare_task(task, model)
        print(f"[{task}] train={meta['n_train']} val={meta['n_val']} test={meta['n_test']} "
              f"range={meta['normalisation_range']} stratified={meta['stratified']}")
        print(f"          scaled train min={meta['scaled_train_min']} max={meta['scaled_train_max']}")
        if task == "regression":
            print(f"          val uncorrected GT RMSE = {meta['val_gt_rmse_uncorrected']} m (paper: 23.87)")
    print("Data prep complete -> data/processed/")


if __name__ == "__main__":
    main()
