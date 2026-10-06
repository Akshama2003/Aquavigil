# 3-minute demo video outline

The judges only see the video, so show a working loop and a real number.

| Time | Show | Say |
|---|---|---|
| 0:00-0:25 | A leaking pipe / water loss headline you have verified | "Leaks are found after the water is already on the street. We find them from pressure data." |
| 0:25-1:15 | `payload` then `scan` in the terminal, then the ranked list on a map | "Eight sensors, one anomaly, five pipes to check, in order." |
| 1:15-2:00 | `results/backtest.md` | State the table honestly: strong on medium/large leaks, weak on small ones, 2.5% false-alarm scenarios. |
| 2:00-2:30 | Architecture: API Gateway, Lambda, DynamoDB, Strands agent on Bedrock | "Open-source Strands agent writes the crew message; everything is serverless." |
| 2:30-3:00 | Limits and next step | "Validated on a simulated network. Next: real network data and a demand-aware baseline." |

Before recording: run `sam deploy` once, hit the live URL with `data/demo_payload.json`, and
screen-record that call so the AWS part is real, not just a slide.
