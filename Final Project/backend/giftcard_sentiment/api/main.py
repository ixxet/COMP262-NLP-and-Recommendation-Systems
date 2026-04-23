"""FastAPI entrypoint for the Gift Cards sentiment lab."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from giftcard_sentiment.artifacts import load_evidence, load_llm_examples, load_project_summary
from giftcard_sentiment.assistant import answer, retrieve
from giftcard_sentiment.schemas import (
    AskRequest,
    AskResponse,
    EnhanceRequest,
    EnhanceResponse,
    HealthResponse,
    LlmExamplesResponse,
    ProjectSummary,
    SentimentRequest,
    SentimentResponse,
)
from giftcard_sentiment.sentiment import enhanced_rating, rating_to_label, score_text
from giftcard_sentiment.settings import Settings, get_settings

PREDICTIONS = Counter("giftcard_sentiment_predictions_total", "Live demo predictions", ["label"])
ASKS = Counter("giftcard_project_questions_total", "Project assistant questions")

app = FastAPI(
    title="Gift Cards Sentiment Lab API",
    version="0.1.0",
    description=(
        "Presentation API for Amazon Gift Cards sentiment, model comparison, "
        "and review-enhanced recommendation."
    ),
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)


def current_settings() -> Settings:
    return get_settings()


SettingsDep = Depends(current_settings)


@app.get("/healthz", response_model=HealthResponse)
async def healthz(settings: Settings = SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok", service=settings.service_name)


@app.get("/readyz", response_model=dict[str, Any])
async def readyz(settings: Settings = SettingsDep) -> dict[str, Any]:
    summary = load_project_summary(str(settings.artifacts_dir))
    return {
        "ready": True,
        "artifact_dir": str(settings.artifacts_dir),
        "datasets": len(summary.datasets),
        "models": len(summary.model_metrics),
    }


@app.get("/v1/summary", response_model=ProjectSummary)
async def summary(settings: Settings = SettingsDep) -> ProjectSummary:
    return load_project_summary(str(settings.artifacts_dir))


@app.get("/v1/examples/summaries", response_model=LlmExamplesResponse)
async def examples(settings: Settings = SettingsDep) -> LlmExamplesResponse:
    return load_llm_examples(str(settings.artifacts_dir))


@app.post("/v1/sentiment/predict", response_model=SentimentResponse)
async def predict(payload: SentimentRequest) -> SentimentResponse:
    prediction = score_text(payload.text)
    PREDICTIONS.labels(label=prediction.label).inc()
    return SentimentResponse(
        model="live_lexicon_demo",
        label=prediction.label,
        score=prediction.score,
        confidence=prediction.confidence,
        rating_label=rating_to_label(payload.rating),
        rationale=prediction.rationale,
    )


@app.post("/v1/recommend/enhance", response_model=EnhanceResponse)
async def enhance(payload: EnhanceRequest) -> EnhanceResponse:
    return EnhanceResponse.model_validate(
        enhanced_rating(
            actual_rating=payload.actual_rating,
            sentiment_score=payload.sentiment_score,
            sentiment_weight=payload.sentiment_weight,
        )
    )


@app.post("/v1/assistant/ask", response_model=AskResponse)
async def ask(payload: AskRequest, settings: Settings = SettingsDep) -> AskResponse:
    ASKS.inc()
    hits = retrieve(
        payload.question,
        load_evidence(str(settings.artifacts_dir)),
        payload.max_evidence,
    )
    return AskResponse(answer=answer(payload.question, hits), evidence=hits)
