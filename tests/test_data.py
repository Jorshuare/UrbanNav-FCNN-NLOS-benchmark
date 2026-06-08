"""Pin the LOS/NLOS label polarity and cleaning invariants (CLAUDE.md §8).

These are the guardrails that protect the whole reproduction: if the input data
is ever swapped or the label convention flips, the FCNN accuracy would silently
become 1-accuracy. We pin the mapping by both count and physics (NLOS has the
larger pseudorange error).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_fcnn.config import abspath, load_config  # noqa: E402
from gnss_fcnn.data import clean, loader  # noqa: E402

cfg = load_config()
df = loader.load_table(abspath(cfg, cfg.data.train_xlsx))
cols = cfg.data.columns


def test_row_count_and_no_nan():
    assert len(df) == clean.EXPECTED_ROWS
    required = [cols.cnr, cols.elevation, cols.pr_residual, cols.label, cols.pr_error]
    assert df[required].isna().sum().sum() == 0


def test_label_counts_match_paper():
    counts = df[cols.label].value_counts().to_dict()
    assert counts == clean.EXPECTED_LABEL_COUNTS  # {0: 29949 NLOS, 1: 44136 LOS}


def test_label_polarity_pinned_by_physics():
    """label 0 must be NLOS (larger |Pr_Error|) and label 1 must be LOS."""
    mae0 = df.loc[df[cols.label] == 0, cols.pr_error].abs().mean()
    mae1 = df.loc[df[cols.label] == 1, cols.pr_error].abs().mean()
    assert mae0 > mae1, "label 0 should be NLOS with larger pseudorange error"
    assert round(mae0, 2) == 19.07 and round(mae1, 2) == 2.76  # paper Table 6 TOTAL


def test_master_satellite_count():
    n_master = int((df[cols.pr_error] == 0).sum())
    assert n_master == clean.N_MASTER_SATELLITES
    filtered = clean.drop_master_satellites(df, cols.pr_error)
    assert len(filtered) == clean.EXPECTED_ROWS - clean.N_MASTER_SATELLITES


def test_unique_epochs():
    assert df[cols.time].nunique() == 4471  # paper: 4,471 train/val epochs
