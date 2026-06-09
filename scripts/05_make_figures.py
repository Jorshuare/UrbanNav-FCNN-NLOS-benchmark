"""Stage 05 — regenerate paper-style figures into reports/figures/.

- Figure 13: C/N0 vs elevation scatter (LOS=blue, NLOS=orange), train (true label)
  + test (predicted label, from reports/submission_classification.csv).
- Training curves: M2 (classification) and M4 (regression) from results/metrics/*.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_fcnn.config import abspath, load_config  # noqa: E402
from gnss_fcnn.data import loader  # noqa: E402
from gnss_fcnn.utils.io import load_json  # noqa: E402
from gnss_fcnn.viz import figures  # noqa: E402


def main() -> None:
    cfg = load_config()
    cols = cfg.data.columns
    out_dir = abspath(cfg, cfg.eval.report.figures_dir)

    # --- Figure 13 ---
    train = loader.load_table(abspath(cfg, cfg.data.train_xlsx))
    test = loader.load_table(abspath(cfg, cfg.data.test_xlsx))
    sub = pd.read_csv(ROOT / "reports" / "submission_classification.csv")
    assert len(sub) == len(test), "submission/test row mismatch"

    paths = figures.figure13(
        train[cols.elevation].to_numpy(), train[cols.cnr].to_numpy(), train[cols.label].to_numpy(),
        test[cols.elevation].to_numpy(), test[cols.cnr].to_numpy(), sub["Predict"].to_numpy(),
        out_dir,
    )
    print("Figure 13 ->", ", ".join(p.name for p in paths))

    # --- training curves ---
    for task, model in [("classification", "M2"), ("regression", "M4")]:
        hist = load_json(ROOT / "results" / "metrics" / f"{task}_{model}_seed42.json")["history"]
        paths = figures.training_curves(hist, task, model, out_dir)
        print(f"Training curve {model} ({task}) ->", ", ".join(p.name for p in paths))

    print(f"\nFigures written -> {out_dir}")


if __name__ == "__main__":
    main()
