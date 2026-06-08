"""Train an FCNN variant for pseudorange error prediction (default M4).

Usage: python scripts/03_train_regression.py [fcnn_m4] [seed]
Stage-03 smoke run trains the canonical M4; the full 5-variant sweep is Stage 04.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnss_fcnn.training.run import train_single  # noqa: E402

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "fcnn_m4"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    s = train_single(model=model, task="regression", seed=seed)
    print(f"[regression] {s['model']} seed={s['seed']} "
          f"val_rmse={s['val_rmse']:.2f} m (paper M4 val=15.14) "
          f"best_epoch={s['best_epoch']}/{s['stopped_epoch']}")
