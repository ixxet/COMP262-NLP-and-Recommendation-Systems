from __future__ import annotations

import unittest
from unittest.mock import patch

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
        self.assertEqual(payload["mode"], "retrieval_fallback")
        self.assertTrue(payload["citations"])

    def test_assistant_low_signal_question(self) -> None:
        response = self.client.post(
            "/v1/assistant/ask",
            json={"question": "hi", "max_evidence": 2},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["evidence"], [])
        self.assertIn("project-specific", payload["answer"])
        self.assertEqual(payload["citations"], [])

    @patch("giftcard_sentiment.api.main.maybe_generate_grounded_answer")
    def test_assistant_grounded_llm_path(self, grounded_mock) -> None:
        grounded_mock.return_value = type(
            "AssistantResultStub",
            (),
            {
                "answer": "TF-IDF was chosen for speed and interpretability.",
                "citations": ["tfidf-choice"],
                "mode": "vllm_grounded",
                "assistant_model": "mistralai/Mistral-7B-Instruct-v0.3",
            },
        )()
        response = self.client.post(
            "/v1/assistant/ask",
            json={"question": "Why TF-IDF?", "max_evidence": 2},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "vllm_grounded")
        self.assertEqual(payload["assistant_model"], "mistralai/Mistral-7B-Instruct-v0.3")
        self.assertEqual(payload["citations"], ["tfidf-choice"])
        grounded_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
