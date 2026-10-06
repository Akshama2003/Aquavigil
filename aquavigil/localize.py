"""Leak localization: which pipes best explain the pressure pattern after the alarm?"""

import numpy as np

from .detect import PERSIST
from .model import Baseline
from .network import Network


def localize(
    net: Network, base: Baseline, res: np.ndarray, alarm: int, window: int = 12
) -> list:
    """Rank every pipe by how well a single leak on it explains the residuals.

    For each pipe we fit a non-negative leak size by least squares against the
    (sigma-weighted) post-alarm residuals and rank pipes by remaining error.
    """
    lo = max(0, alarm - PERSIST + 1)
    hi = min(res.shape[1], alarm + window)
    d = -res[:, lo:hi].mean(axis=1) / base.sigma  # observed drop, in sigmas
    Sw = net.S / base.sigma[:, None]
    size = np.maximum(0.0, (Sw * d[:, None]).sum(axis=0) / (Sw**2).sum(axis=0))
    err = ((d[:, None] - Sw * size[None, :]) ** 2).sum(axis=0)
    order = np.argsort(err, kind="stable")
    return [
        {"pipe": int(p), "size": float(size[p]), "score": float(err[p])}
        for p in order
    ]
