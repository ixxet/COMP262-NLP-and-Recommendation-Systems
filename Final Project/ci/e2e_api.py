"""Minimal API smoke test for local or cluster endpoints."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("GIFT_CARD_API_URL", "http://localhost:8000").rstrip("/")


def request(path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"content-type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    try:
        health = request("/healthz")
        summary = request("/v1/summary")
        sentiment = request(
            "/v1/sentiment/predict",
            {"text": "The card was easy and perfect.", "rating": 5},
        )
        ask = request(
            "/v1/assistant/ask",
            {"question": "Why is macro-F1 better than accuracy?", "max_evidence": 2},
        )
    except urllib.error.URLError as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        return 1

    assert health["status"] == "ok"
    assert summary["project"]["team"] == "Group 5"
    assert sentiment["label"] == "positive"
    assert ask["evidence"]
    print("Gift Cards sentiment API smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
