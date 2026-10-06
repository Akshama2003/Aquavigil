"""Leak detection: standardized residuals, smoothed, with a persistence rule."""

import numpy as np

from .model import Baseline
from .simulate import STEPS

SMOOTH = 4  # steps (1 hour) of smoothing
PERSIST = 4  # consecutive steps above threshold before alarming


def residuals(base: Baseline, X: np.ndarray, start_step: int = 0) -> np.ndarray:
    """Observed minus expected pressure. `start_step` is the step-of-day of column 0."""
    idx = (start_step + np.arange(X.shape[1])) % STEPS
    return X - base.median[:, idx]


def statistic(base: Baseline, res: np.ndarray) -> np.ndarray:
    z = res / base.sigma[:, None]
    c = np.cumsum(np.pad(z, ((0, 0), (1, 0))), axis=1)
    zs = (c[:, SMOOTH:] - c[:, :-SMOOTH]) / SMOOTH
    stat = (zs**2).mean(axis=0)
    return np.concatenate([np.zeros(SMOOTH - 1), stat])


def find_alarm(stat: np.ndarray, threshold: float, search_from: int = 0):
    run = 0
    for t in range(search_from, len(stat)):
        run = run + 1 if stat[t] > threshold else 0
        if run >= PERSIST:
            return t
    return None


def calibrate(base: Baseline, X_clean: np.ndarray, margin: float = 1.25) -> float:
    """Threshold just above the largest *sustained* statistic seen on clean data.

    An alarm needs PERSIST consecutive exceedances, so the quantity that matters
    is the minimum of the statistic over each PERSIST-step window, not its peak.
    """
    stat = statistic(base, residuals(base, X_clean))
    windows = np.lib.stride_tricks.sliding_window_view(stat, PERSIST)
    return float(windows.min(axis=1).max() * margin)
