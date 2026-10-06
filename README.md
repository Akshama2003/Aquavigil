# AquaVigil

**Finds where a water network is leaking and tells a crew which pipes to dig up first.**

Built for the Bharat Builds Tour *Environmental Hacks* (Track 2: Heat and Water, "Leaks").

Cities lose a large share of treated water to leaks, and crews usually find them
after water is already surfacing. AquaVigil watches pressure sensors, flags a leak
within hours, and turns "something is wrong somewhere" into a ranked shortlist of
pipes and an inspection work order.

```
pressure sensors -> API Gateway -> Lambda (detect + localize) -> work order -> DynamoDB
                                          |
                                  Strands agent (optional, Bedrock)
                                  plain-language message for the crew
```

## Results

Backtest on a synthetic 6x6 grid network (60 pipes, 8 sensors), 300 injected leaks,
200 clean scenarios, seed 42. Reproduce with `python -m aquavigil.cli backtest`.

| Leak size | n | Detected | Median delay (h) | Top-1 | Top-3 | Top-5 | Naive top-5 | Random top-5 |
|---|---|---|---|---|---|---|---|---|
| small (<0.6) | 87 | 11% | 12.12 | 60% | 70% | 80% | 90% | 8% |
| medium (0.6-1.5) | 129 | 67% | 2.50 | 49% | 85% | 93% | 64% | 8% |
| large (>1.5) | 84 | 98% | 1.25 | 74% | 96% | 99% | 55% | 8% |
| **all** | 300 | 60% | 1.50 | 61% | 89% | 95% | 61% | 8% |

False alarms: 5 of 200 clean 3-day scenarios (2.5%) raised an alarm.

How to read this honestly:

- Localization columns are measured **only on leaks that were detected**.
- **Small leaks are mostly missed** (11% detected). They sit near the sensor noise floor. The small-leak
  localization numbers rest on about 10 leaks and are not meaningful; the naive baseline beating us
  there is within noise at that sample size.
- On medium and large leaks, the method beats both a random guess and a naive
  "go to the sensor with the biggest drop" baseline at top-5.
- The alarm threshold is set by rare high-demand-drift days in clean data. A demand-aware
  baseline would raise detection of medium leaks without more false alarms (see Roadmap).

## What is simulated (read this before quoting numbers)

There is **no real utility data** in this repo. Pressures come from a linearized
surrogate, not a hydraulic solver:

- A 6x6 grid of junctions fed from a corner reservoir. Sensors are spread by farthest-point selection.
- A sensitivity matrix `S` says how much a leak on pipe *p* drops pressure at sensor *s*
  (decays with distance, larger far from the reservoir).
- Daily demand cycle, day-to-day demand variation, and sensor noise.
- To avoid a trivially optimistic test, leaks are injected with a **log-normal perturbation of `S`**
  (model mismatch). The localizer only sees the nominal matrix.

So the backtest validates the *method*. It does not show performance on a real network.
Validating on a real or EPANET/WNTR-simulated network is the first roadmap item.

"Estimated loss" in work orders uses a placeholder constant (`LOSS_M3_PER_DAY_PER_UNIT`
in `aquavigil/agent.py`). It is not calibrated; do not quote it as a measurement.

## Quick start

```bash
pip install -r requirements-dev.txt
python -m aquavigil.cli train                 # fit baseline -> data/baseline.json
python -m aquavigil.cli backtest              # results/backtest.md + .json
python -m aquavigil.cli payload               # data/demo_payload.json with an injected leak
python -m aquavigil.cli scan data/demo_payload.json   # prints the work order
pytest
```

## How it works

| Step | File | What it does |
|---|---|---|
| Network | `aquavigil/network.py` | Grid network, sensor placement, sensitivity matrix |
| Simulate | `aquavigil/simulate.py` | Demand cycle, noise, leak injection with model mismatch |
| Baseline | `aquavigil/model.py` | Per-sensor, per-time-of-day median pressure, trained on 14 clean days |
| Detect | `aquavigil/detect.py` | Standardized residuals, 1-hour smoothing, alarm after 4 consecutive exceedances; threshold calibrated on 90 held-out clean days |
| Localize | `aquavigil/localize.py` | Fits a non-negative leak size per pipe, ranks pipes by remaining error |
| Work order | `aquavigil/agent.py` | Ranked shortlist + optional Strands/Bedrock plain-language rewrite |
| API | `lambda_fn/handler.py`, `template.yaml` | `POST /scan`, stores work orders in DynamoDB |

## Deploy to AWS

```bash
sam build
sam deploy --guided
curl -X POST "$SCAN_API_URL" -H 'Content-Type: application/json' -d @data/demo_payload.json
```

For the optional agent text: `pip install -r requirements-agent.txt` and enable a model in Amazon Bedrock.

**Tested here:** the Python package, the CLI, the backtest, and the Lambda handler logic
(called directly in `pytest`).
**Not tested here:** `sam build` / `sam deploy`, the live API Gateway and DynamoDB path, and the
Strands/Bedrock narration. These need your AWS account, so run them once before the event.
If `sam build` complains about numpy for the Lambda runtime, use `sam build --use-container`.

## Roadmap

1. Validate on a real network or WNTR/EPANET simulations (and the public BattLeDIM dataset, after checking it is still available).
2. Demand-aware baseline to cut false alarms and catch smaller leaks.
3. Scheduled scans (EventBridge) and a simple map UI of ranked pipes.
4. Calibrate the loss estimate against real flow data.
5. Floods: out of scope for this entry on purpose.

## License

MIT (add a LICENSE file before publishing).
