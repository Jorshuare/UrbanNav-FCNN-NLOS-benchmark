"""Train an FCNN variant for LOS/NLOS classification (default M2).

Usage: python scripts/02_train_classification.py [fcnn_m2] [seed]
Stage-03 smoke run trains the canonical M2; the full 5-variant sweep is Stage 04.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnss_fcnn.training.run import train_single  # noqa: E402

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "fcnn_m2"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    s = train_single(model=model, task="classification", seed=seed)
    print(f"[classification] {s['model']} seed={s['seed']} "
          f"val_accuracy={s['val_accuracy']:.4f} (paper M2 val=0.771) "
          f"best_epoch={s['best_epoch']}/{s['stopped_epoch']}")
