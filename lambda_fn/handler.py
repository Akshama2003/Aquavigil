"""API Gateway handler: POST /scan with {"pressures": [[...]], "start_step_of_day": 0}."""

import json
import os
import time

from aquavigil.service import load_assets, scan

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS = None


def _assets():
    global _ASSETS
    if _ASSETS is None:
        _ASSETS = load_assets(os.path.join(_ROOT, "data", "baseline.json"))
    return _ASSETS


def _resp(code: int, body: dict) -> dict:
    return {
        "statusCode": code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _save(result: dict) -> None:
    table = os.environ.get("ORDERS_TABLE")
    if not table or result.get("status") != "leak_suspected":
        return
    import boto3

    boto3.resource("dynamodb").Table(table).put_item(
        Item={
            "id": str(int(time.time() * 1000)),
            "markdown": result["work_order"]["markdown"],
            "top_pipe": result["work_order"]["suspects"][0]["label"],
        }
    )


def handler(event, context=None):
    try:
        raw = event.get("body") if isinstance(event, dict) and "body" in event else event
        payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
        net, base = _assets()
        result = scan(payload, net, base)
    except (ValueError, json.JSONDecodeError) as e:
        return _resp(400, {"error": str(e)})
    _save(result)
    return _resp(200, result)
