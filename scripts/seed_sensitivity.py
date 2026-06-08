"""Stage-01 diagnostic: sensitivity of the *uncorrected* validation RMSE to the
train/val split seed.

No model is involved — this measures only how the heavy-tailed Pr_Error baseline
(paper Table 8 reference = 23.87 m) moves with the split. Used to justify keeping
the canonical seed 42 rather than seed-picking. Reuses the real pipeline split
(data.splits.train_val_split) so the numbers match what training would see.

Usage:
    python scripts/seed_sensitivity.py            # seeds 0..199
    python scripts/seed_sensitivity.py 50         # seeds 0..49
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_fcnn.config import abspath, load_config  # noqa: E402
from gnss_fcnn.data import clean, features, loader  # noqa: E402
from gnss_fcnn.data.splits import train_val_split  # noqa: E402
from gnss_fcnn.evaluation.metrics import baseline_rmse  # noqa: E402
from gnss_fcnn.utils.io import save_json  # noqa: E402

PAPER_REFERENCE = 23.87  # Table 8 uncorrected GT RMSE (m)
CANONICAL_SEED = 42


def main(n_seeds: int = 200) -> None:
    cfg = load_config(features="regression", train="regression")
    cols = cfg.data.columns

    df = loader.load_table(abspath(cfg, cfg.data.train_xlsx))
    df = clean.drop_invalid(df, [cols.pr_error])
    df = clean.drop_master_satellites(df, cols.pr_error)  # regression set
    pr_error = features.target_vector(df, cfg)

    val_fraction = float(cfg.data.split.val_fraction)
    results: list[tuple[int, float]] = []
    for seed in range(n_seeds):
        # split the Pr_Error vector exactly as the pipeline splits features
        _, _, _, y_val = train_val_split(
            pr_error.reshape(-1, 1), pr_error,
            val_fraction=val_fraction, seed=seed, stratify=False,
        )
        results.append((seed, baseline_rmse(y_val)))

    vals = np.array([r[1] for r in results])
    seed42 = dict(results)[CANONICAL_SEED]
    order = np.argsort(np.abs(vals - PAPER_REFERENCE))
    closest = [(int(results[i][0]), round(float(results[i][1]), 3)) for i in order[:8]]

    summary = {
        "n_seeds": n_seeds,
        "paper_reference_rmse": PAPER_REFERENCE,
        "canonical_seed": CANONICAL_SEED,
        "canonical_seed_rmse": round(float(seed42), 3),
        "mean": round(float(vals.mean()), 3),
        "std": round(float(vals.std()), 3),
        "min": round(float(vals.min()), 3),
        "max": round(float(vals.max()), 3),
        "n_at_or_below_paper": int((vals <= PAPER_REFERENCE).sum()),
        "closest_seeds_to_paper": closest,
    }

    tables = abspath(cfg, cfg.eval.report.tables_dir)
    save_json(summary, tables / "seed_sensitivity_baseline.json")
    with open(tables / "seed_sensitivity_baseline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "val_baseline_rmse_m"])
        w.writerows([(s, round(v, 4)) for s, v in results])

    print(f"seeds={n_seeds} | mean={summary['mean']} std={summary['std']} "
          f"min={summary['min']} max={summary['max']}")
    print(f"canonical seed {CANONICAL_SEED} -> {summary['canonical_seed_rmse']} m "
          f"(paper {PAPER_REFERENCE}) | seeds <= paper: {summary['n_at_or_below_paper']}/{n_seeds}")
    print(f"closest seeds to paper: {closest[:3]}")
    print(f"-> reports/tables/seed_sensitivity_baseline.{{csv,json}}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(n)
