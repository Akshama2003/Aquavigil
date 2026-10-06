# Backtest results (synthetic network)

300 injected leaks, 60 pipes, 8 sensors, seed 42.
Localization columns are measured on detected leaks only.

| Leak size | n | Detected | Median delay (h) | Top-1 | Top-3 | Top-5 | Naive top-5 | Random top-5 |
|---|---|---|---|---|---|---|---|---|
| small (<0.6) | 87 | 11% | 12.12 | 60% | 70% | 80% | 90% | 8% |
| medium (0.6-1.5) | 129 | 67% | 2.50 | 49% | 85% | 93% | 64% | 8% |
| large (>1.5) | 84 | 98% | 1.25 | 74% | 96% | 99% | 55% | 8% |
| **all** | 300 | 60% | 1.50 | 61% | 89% | 95% | 61% | 8% |

False alarms: 5 of 200 clean 3-day scenarios raised an alarm.

Caveat: pressures come from a linearized surrogate of a 6x6 grid network with log-normal
model mismatch, not from real utility data. These numbers validate the method, not a deployment.
