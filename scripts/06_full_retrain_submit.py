"""Improvement #1 + #2: retrain the selected classifier (M2) on 100% of the
labeled data and build a 10-seed probability ensemble for the Kaggle test set.

Rationale (Report-04 analysis): we were higher than the paper on validation but
lower on the Kaggle test, suggesting the benchmark model used all labeled data.
Recipe: epoch budget is selected from the 80/20 early-stopping run (~90), then we
retrain on the full set for that many epochs (no held-out val), across seeds
42..51, and average class probabilities (ensembles beat single seeds on test).

Outputs:
  reports/submission_classification_full_seed42.csv   (single full-data model)
  reports/submission_classification_full_ensemble.csv (10-seed mean-prob ensemble)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_fcnn.config import abspath, load_config  # noqa: E402
from gnss_fcnn.data import clean, features, loader  # noqa: E402
from gnss_fcnn.data.scalers import MinMaxScaler  # noqa: E402
from gnss_fcnn.models.fcnn import build_fcnn  # noqa: E402
from gnss_fcnn.training.seed import seed_everything  # noqa: E402
from gnss_fcnn.training.trainer import Trainer  # noqa: E402

EPOCH_BUDGET = 90      # selected from 80/20 validation early-stopping (best ~91)
SEEDS = list(range(42, 52))
REPORTS = ROOT / "reports"


def build_full_data():
    cfg = load_config(model="fcnn_m2", features="classification", train="classification")
    cols = cfg.data.columns
    df = loader.load_table(abspath(cfg, cfg.data.train_xlsx))
    clean.assert_train_invariants(df, cols.label)
    df = clean.drop_invalid(df, [cols.cnr, cols.elevation, cols.pr_residual, cols.label])
    X_full = features.feature_matrix(df, cfg)
    y_full = features.target_vector(df, cfg)
    scaler = MinMaxScaler((0.0, 1.0)).fit(X_full)        # fit on ALL labeled data
    df_test = loader.load_table(abspath(cfg, cfg.data.test_xlsx))
    X_test = scaler.transform(features.feature_matrix(df_test, cfg))
    return cfg, scaler.transform(X_full), y_full, X_test


def main() -> None:
    cfg, X_full, y_full, X_test = build_full_data()
    cfg.train.epochs = EPOCH_BUDGET
    cfg.train.early_stopping.patience = EPOCH_BUDGET     # disable early stop (full-data)
    print(f"full labeled rows={len(y_full)} | test rows={len(X_test)} | "
          f"epochs={EPOCH_BUDGET} | seeds={SEEDS[0]}..{SEEDS[-1]}")

    probs_sum = np.zeros((len(X_test), 2))
    seed42_pred = None
    for s in SEEDS:
        seed_everything(s, deterministic=bool(cfg.deterministic))
        trainer = Trainer(build_fcnn(cfg), cfg)
        trainer.fit(X_full, y_full, X_full, y_full)      # val=train (history only)
        train_acc = trainer.evaluate(X_full, y_full)["metric"]
        p = trainer.predict_proba(X_test)                # (N,2) = (P_NLOS, P_LOS)
        probs_sum += p
        if s == 42:
            seed42_pred = p.argmax(axis=1).astype(int)
        print(f"  seed {s}: full-data train acc = {train_acc:.4f}", flush=True)

    ens_pred = (probs_sum / len(SEEDS)).argmax(axis=1).astype(int)

    ids = np.arange(len(X_test))
    pd.DataFrame({"ID": ids, "Predict": seed42_pred}).to_csv(
        REPORTS / "submission_classification_full_seed42.csv", index=False)
    pd.DataFrame({"ID": ids, "Predict": ens_pred}).to_csv(
        REPORTS / "submission_classification_full_ensemble.csv", index=False)

    # compare to the original 80%-trained submission (scored 0.812)
    prev = pd.read_csv(REPORTS / "submission_classification.csv")["Predict"].to_numpy()
    print(f"ensemble class dist: NLOS={int((ens_pred==0).sum())} LOS={int((ens_pred==1).sum())}")
    print(f"changed vs original 80% submission: "
          f"seed42_full={int((seed42_pred!=prev).sum())} "
          f"ensemble={int((ens_pred!=prev).sum())} of {len(prev)} rows")
    print("\nWrote: submission_classification_full_{seed42,ensemble}.csv  -> resubmit to Kaggle")


if __name__ == "__main__":
    main()
