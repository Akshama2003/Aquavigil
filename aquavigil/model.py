"""Learned "normal" behaviour: a per-sensor, per-time-of-day pressure baseline."""

import json

import numpy as np

from .network import Network
from .simulate import STEPS, simulate


class Baseline:
    def __init__(self, median, sigma, threshold=None, meta=None):
        self.median = np.asarray(median, float)  # (n_sensors, STEPS)
        self.sigma = np.asarray(sigma, float)  # (n_sensors,)
        self.threshold = threshold
        self.meta = meta or {}

    @classmethod
    def fit(cls, X: np.ndarray) -> "Baseline":
        ns, T = X.shape
        days = T // STEPS
        R = X[:, : days * STEPS].reshape(ns, days, STEPS)
        med = np.median(R, axis=1)
        res = (R - med[:, None, :]).reshape(ns, -1)
        centre = np.median(res, axis=1, keepdims=True)
        sigma = 1.4826 * np.median(np.abs(res - centre), axis=1)
        return cls(med, sigma)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "median": self.median.tolist(),
                    "sigma": self.sigma.tolist(),
                    "threshold": self.threshold,
                    "meta": self.meta,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "Baseline":
        with open(path) as f:
            d = json.load(f)
        return cls(d["median"], d["sigma"], d["threshold"], d.get("meta"))


def train(net: Network, seed: int = 42, train_days: int = 14, val_days: int = 90):
    """Fit the baseline on clean data and calibrate the alarm threshold on held-out clean days."""
    from .detect import calibrate

    rng = np.random.default_rng(seed)
    base = Baseline.fit(simulate(net, train_days, rng))
    base.threshold = calibrate(base, simulate(net, val_days, rng))
    base.meta = {"n": net.n, "n_sensors": net.n_sensors, "decay": net.decay}
    return base
