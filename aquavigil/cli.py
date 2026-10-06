"""Command line: python -m aquavigil.cli {train,backtest,payload,scan}"""

import argparse
import json
import os

import numpy as np

from .backtest import run, to_markdown
from .model import train
from .network import build_grid
from .service import load_assets, scan
from .simulate import STEPS, simulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "data", "baseline.json")


def main(argv=None):
    p = argparse.ArgumentParser(prog="aquavigil")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("train", help="fit the baseline and write data/baseline.json")
    s.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("backtest", help="run the injected-leak backtest")
    s.add_argument("--trials", type=int, default=300)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default=os.path.join(ROOT, "results"))

    s = sub.add_parser("payload", help="write a demo scan payload with an injected leak")
    s.add_argument("--pipe", type=int, default=17)
    s.add_argument("--size", type=float, default=1.5)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default=os.path.join(ROOT, "data", "demo_payload.json"))

    s = sub.add_parser("scan", help="scan a payload JSON file")
    s.add_argument("file")

    a = p.parse_args(argv)

    if a.cmd == "train":
        net = build_grid()
        base = train(net, a.seed)
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        base.save(BASELINE)
        print(f"wrote {BASELINE} (threshold {base.threshold:.3f})")

    elif a.cmd == "backtest":
        res = run(a.trials, a.seed)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "backtest.json"), "w") as f:
            json.dump(res, f, indent=2)
        md = to_markdown(res)
        with open(os.path.join(a.out, "backtest.md"), "w") as f:
            f.write(md)
        print(md)

    elif a.cmd == "payload":
        net = build_grid()
        rng = np.random.default_rng(a.seed)
        start = STEPS + 40  # leak begins on day 2 at 10:00
        X = simulate(net, 2, rng, leak=(a.pipe, a.size, start))
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"pressures": X.round(3).tolist(), "start_step_of_day": 0}, f)
        print(f"wrote {a.out}; true leak on {net.pipe_label(a.pipe)}")

    elif a.cmd == "scan":
        net, base = load_assets(BASELINE)
        with open(a.file) as f:
            out = scan(json.load(f), net, base)
        print(out["work_order"]["markdown"] if "work_order" in out else json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
