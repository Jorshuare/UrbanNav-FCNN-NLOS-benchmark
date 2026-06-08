"""Positioning-error baseline (Task 3): OLS least-squares, Eq. 11.

The benchmark baseline is a conventional ordinary-least-squares position solution
(no FCNN). Its per-epoch East/North error equals (baseline position - ground
truth) = the no-suffix ``East_error``/``North_error`` columns, which we verify
equals ``*_base - *_gt``. The score is the combined East/North RMSE (Eq. 11):

    Score = sqrt( (1/n) * sum_i [ (E_gt - E_pre)^2 + (N_gt - N_pre)^2 ] / 2 )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def east_north_rmse(east_err: np.ndarray, north_err: np.ndarray) -> float:
    """Combined East/North RMSE (Eq. 11)."""
    east_err = np.asarray(east_err, dtype=np.float64)
    north_err = np.asarray(north_err, dtype=np.float64)
    return float(np.sqrt(np.mean((east_err ** 2 + north_err ** 2) / 2.0)))


def epoch_errors(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One (East_error, North_error) per epoch; validates error == base - gt."""
    ep = df.drop_duplicates("GPS_Time(s)")
    e, n = ep["East_error"].to_numpy(), ep["North_error"].to_numpy()
    assert np.allclose(e, ep["East_error_base"] - ep["East_error_gt"], atol=0.05)
    assert np.allclose(n, ep["North_error_base"] - ep["North_error_gt"], atol=0.05)
    return e, n


def ols_validation_rmse(df: pd.DataFrame, *, val_fraction: float, seed: int) -> dict:
    """OLS East/North RMSE on the held-out validation epochs (Eq. 11)."""
    e, n = epoch_errors(df)
    _, e_val, _, n_val = train_test_split(e, n, test_size=val_fraction, random_state=seed)
    return {
        "n_epochs": int(len(e)),
        "n_val_epochs": int(len(e_val)),
        "rmse_all": east_north_rmse(e, n),
        "rmse_val": east_north_rmse(e_val, n_val),
    }
