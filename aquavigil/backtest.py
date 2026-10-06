"""Backtest: inject leaks, measure detection and localization against baselines."""

import numpy as np

from .detect import find_alarm, residuals, statistic
from .localize import localize
from .model import train
from .network import build_grid
from .simulate import STEP_HOURS, STEPS, simulate

BUCKETS = [("small (<0.6)", 0.0, 0.6), ("medium (0.6-1.5)", 0.6, 1.5), ("large (>1.5)", 1.5, 99.0)]
KS = (1, 3, 5)


def run(trials: int = 300, seed: int = 42, clean_trials: int = 200) -> dict:
    net = build_grid()
    base = train(net, seed)
    rng = np.random.default_rng(seed + 1)
    sizes = np.exp(rng.uniform(np.log(0.3), np.log(3.0), trials))

    rows = []
    for size in sizes:
        pipe = int(rng.integers(net.n_pipes))
        start = int(STEPS + rng.integers(0, STEPS))
        X = simulate(net, 3, rng, leak=(pipe, float(size), start))
        res = residuals(base, X)
        stat = statistic(base, res)
        alarm = find_alarm(stat, base.threshold, search_from=start)
        row = {"size": float(size), "detected": alarm is not None}
        if alarm is not None:
            row["delay_h"] = (alarm - start) * STEP_HOURS
            order = [r["pipe"] for r in localize(net, base, res, alarm)]
            row["rank"] = order.index(pipe) + 1
            row["err"] = net.pipe_distance(order[0], pipe)
            # Naive baseline: go to the sensor with the biggest drop, nearest pipes first.
            drop = -res[:, max(0, alarm - 3) : alarm + 12].mean(axis=1)
            s = net.sensors[int(np.argmax(drop))]
            dist = np.abs(net.pipe_mid - net.coords[s]).sum(axis=1)
            naive = np.argsort(dist + 1e-6 * rng.random(net.n_pipes), kind="stable")
            row["naive_rank"] = int(np.where(naive == pipe)[0][0]) + 1
        rows.append(row)

    # False alarms on clean 3-day scenarios.
    fa = 0
    for _ in range(clean_trials):
        X = simulate(net, 3, rng)
        stat = statistic(base, residuals(base, X))
        fa += find_alarm(stat, base.threshold) is not None

    def summarize(sel):
        det = [r for r in sel if r["detected"]]
        out = {"n": len(sel), "detection_rate": len(det) / len(sel) if sel else 0.0}
        if det:
            out["median_delay_h"] = float(np.median([r["delay_h"] for r in det]))
            out["median_top1_error_pipes"] = float(np.median([r["err"] for r in det]))
            for k in KS:
                out[f"top{k}"] = float(np.mean([r["rank"] <= k for r in det]))
                out[f"naive_top{k}"] = float(np.mean([r["naive_rank"] <= k for r in det]))
        for k in KS:
            out[f"random_top{k}"] = k / net.n_pipes
        return out

    result = {
        "config": {
            "trials": trials,
            "seed": seed,
            "n_pipes": net.n_pipes,
            "n_sensors": net.n_sensors,
            "threshold": base.threshold,
        },
        "overall": summarize(rows),
        "by_size": {
            name: summarize([r for r in rows if lo <= r["size"] < hi])
            for name, lo, hi in BUCKETS
        },
        "false_alarms": {"scenarios": clean_trials, "scenario_days": 3, "with_alarm": int(fa)},
    }
    return result


def to_markdown(res: dict) -> str:
    c = res["config"]
    lines = [
        "# Backtest results (synthetic network)",
        "",
        f"{c['trials']} injected leaks, {c['n_pipes']} pipes, {c['n_sensors']} sensors, seed {c['seed']}.",
        "Localization columns are measured on detected leaks only.",
        "",
        "| Leak size | n | Detected | Median delay (h) | Top-1 | Top-3 | Top-5 | Naive top-5 | Random top-5 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def row(name, s):
        if "top1" not in s:
            return f"| {name} | {s['n']} | {s['detection_rate']:.0%} | - | - | - | - | - | {s['random_top5']:.0%} |"
        return (
            f"| {name} | {s['n']} | {s['detection_rate']:.0%} | {s['median_delay_h']:.2f} | "
            f"{s['top1']:.0%} | {s['top3']:.0%} | {s['top5']:.0%} | "
            f"{s['naive_top5']:.0%} | {s['random_top5']:.0%} |"
        )

    for name, s in res["by_size"].items():
        lines.append(row(name, s))
    lines.append(row("**all**", res["overall"]))
    fa = res["false_alarms"]
    lines += [
        "",
        f"False alarms: {fa['with_alarm']} of {fa['scenarios']} clean {fa['scenario_days']}-day scenarios raised an alarm.",
        "",
        "Caveat: pressures come from a linearized surrogate of a 6x6 grid network with log-normal",
        "model mismatch, not from real utility data. These numbers validate the method, not a deployment.",
    ]
    return "\n".join(lines) + "\n"
