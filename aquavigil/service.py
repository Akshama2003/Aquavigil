"""Shared scan logic used by both the Lambda handler and the CLI."""

import numpy as np

from .agent import build_work_order
from .detect import PERSIST, SMOOTH, find_alarm, residuals, statistic
from .localize import localize
from .model import Baseline
from .network import Network, build_grid
from .simulate import STEPS


def load_assets(baseline_path: str):
    base = Baseline.load(baseline_path)
    m = base.meta
    net = build_grid(m["n"], m["n_sensors"], m["decay"])
    return net, base


def scan(payload: dict, net: Network, base: Baseline, top_k: int = 5) -> dict:
    """payload: {"pressures": [[...] one row per sensor], "start_step_of_day": int}"""
    try:
        X = np.asarray(payload["pressures"], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ValueError("payload needs a numeric 'pressures' matrix")
    if X.ndim != 2 or X.shape[0] != net.n_sensors:
        raise ValueError(f"'pressures' must have {net.n_sensors} rows (one per sensor)")
    if X.shape[1] < SMOOTH + PERSIST:
        raise ValueError("not enough time steps to scan")
    if not np.isfinite(X).all():
        raise ValueError("'pressures' contains missing or non-finite values")

    start = int(payload.get("start_step_of_day", 0)) % STEPS
    res = residuals(base, X, start)
    stat = statistic(base, res)
    alarm = find_alarm(stat, base.threshold)
    out = {
        "threshold": base.threshold,
        "max_statistic": float(stat.max()),
    }
    if alarm is None:
        out["status"] = "normal"
        return out

    ranking = localize(net, base, res, alarm)
    order = build_work_order(net, (start + alarm) % STEPS, ranking, top_k)
    out.update(status="leak_suspected", alarm_index=int(alarm), work_order=order)
    return out
