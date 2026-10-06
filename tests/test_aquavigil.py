import json

import numpy as np
import pytest

from aquavigil import backtest
from aquavigil.detect import find_alarm, residuals, statistic
from aquavigil.model import train
from aquavigil.network import build_grid
from aquavigil.service import scan
from aquavigil.simulate import STEPS, simulate
from lambda_fn import handler as lambda_handler


@pytest.fixture(scope="module")
def assets():
    net = build_grid()
    return net, train(net, seed=42)


def test_network_shapes(assets):
    net, _ = assets
    assert net.n_pipes == 60
    assert net.S.shape == (8, 60)
    assert net.S.max() == pytest.approx(1.0)


def test_clean_series_has_no_alarm(assets):
    net, base = assets
    X = simulate(net, 1, np.random.default_rng(123))
    assert scan({"pressures": X.tolist()}, net, base)["status"] == "normal"


def test_large_leak_is_detected_and_ranked(assets):
    net, base = assets
    rng = np.random.default_rng(5)
    hits = 0
    for pipe in (3, 17, 30, 44, 55):
        X = simulate(net, 2, rng, leak=(pipe, 2.5, STEPS + 40))
        out = scan({"pressures": X.tolist()}, net, base)
        assert out["status"] == "leak_suspected"
        hits += pipe in [s["pipe"] for s in out["work_order"]["suspects"]]
    assert hits >= 4


def test_bad_payload_rejected(assets):
    net, base = assets
    with pytest.raises(ValueError):
        scan({"pressures": [[1, 2, 3]]}, net, base)
    with pytest.raises(ValueError):
        scan({}, net, base)


def test_handler_roundtrip(assets, monkeypatch):
    net, base = assets
    monkeypatch.setattr(lambda_handler, "_ASSETS", (net, base))
    X = simulate(net, 2, np.random.default_rng(9), leak=(20, 2.0, STEPS + 30))
    r = lambda_handler.handler({"body": json.dumps({"pressures": X.tolist()})})
    assert r["statusCode"] == 200
    assert json.loads(r["body"])["status"] == "leak_suspected"
    assert lambda_handler.handler({"body": "{}"})["statusCode"] == 400


def test_backtest_beats_baselines():
    res = backtest.run(trials=80, seed=7, clean_trials=20)
    large = res["by_size"]["large (>1.5)"]
    assert large["detection_rate"] >= 0.9
    assert large["top5"] > 3 * large["random_top5"]
    assert large["top5"] >= large["naive_top5"]
