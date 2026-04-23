from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from giftcard_sentiment.api.main import app


class ApiContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_and_summary(self) -> None:
        health = self.client.get("/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        summary = self.client.get("/v1/summary")
        self.assertEqual(summary.status_code, 200)
        payload = summary.json()
        self.assertEqual(payload["project"]["team"], "Group 5")
        self.assertGreaterEqual(len(payload["model_metrics"]), 4)

    def test_sentiment_prediction(self) -> None:
        response = self.client.post(
            "/v1/sentiment/predict",
            json={"text": "This gift card was easy, fast, and perfect!", "rating": 5},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["label"], "positive")
        self.assertEqual(payload["rating_label"], "positive")

    def test_recommender_enhancement(self) -> None:
        response = self.client.post(
            "/v1/recommend/enhance",
            json={"actual_rating": 2, "sentiment_score": -1.0, "sentiment_weight": 0.3},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertLess(payload["enhanced_rating"], 2)

    def test_assistant_retrieval(self) -> None:
        response = self.client.post(
            "/v1/assistant/ask",
            json={"question": "Why is macro F1 more useful than accuracy?", "max_evidence": 2},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["evidence"])
        self.assertIn("macro", payload["answer"].lower())


if __name__ == "__main__":
    unittest.main()
