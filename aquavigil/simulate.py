"""Synthetic pressure time series: daily demand cycle, noise, and injected leaks.

Pressures are in metres of head, sampled every 15 minutes (96 steps per day).
A leak is a tuple (pipe_index, size, start_step). `size` is the pressure drop,
in metres, that the leak causes at a sensor half a pipe-length away.

To avoid a backtest that is trivially optimistic, the *true* sensitivity used to
inject a leak is the nominal one times a log-normal perturbation (model
mismatch), while the localizer only ever sees the nominal matrix.
"""

import numpy as np

from .network import Network

STEPS = 96  # 15-minute steps per day
STEP_HOURS = 24 / STEPS


def demand_pattern() -> np.ndarray:
    h = np.arange(STEPS) / 4.0
    d = (
        0.25
        + 0.8 * np.exp(-(((h - 7) / 1.6) ** 2))
        + 0.9 * np.exp(-(((h - 19) / 2.2) ** 2))
    )
    return d / d.max()


def simulate(
    net: Network,
    n_days: int,
    rng: np.random.Generator,
    leak=None,
    mismatch: float = 0.25,
    noise: float = 0.15,
    day_var: float = 0.04,
) -> np.ndarray:
    """Return pressures with shape (n_sensors, n_days * STEPS)."""
    T = n_days * STEPS
    dist0 = np.array([net.dist_to_reservoir(s) for s in net.sensors])
    p0 = 32.0 - 0.2 * dist0
    a = 1.5 + 0.1 * dist0

    dem = np.tile(demand_pattern(), n_days)
    day_factor = np.repeat(1.0 + day_var * rng.standard_normal(n_days), STEPS)

    P = (
        p0[:, None]
        - a[:, None] * (dem * day_factor)[None, :]
        + noise * rng.standard_normal((net.n_sensors, T))
    )

    if leak is not None:
        pipe, size, start = leak
        col = net.S[:, pipe] * rng.lognormal(0.0, mismatch, net.n_sensors)
        P[:, start:] -= size * col[:, None]
    return P
