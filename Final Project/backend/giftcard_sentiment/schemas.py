"""Pydantic request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SentimentLabel = Literal["negative", "neutral", "positive"]


class HealthResponse(BaseModel):
    status: str
    service: str


class DatasetSummary(BaseModel):
    name: str
    raw_rows: int
    clean_rows: int
    unique_users: int
    unique_products: int
    missing_review_text_raw: int
    duplicate_clean_key_rows: int


class ModelMetric(BaseModel):
    model: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float


class RecommenderMetric(BaseModel):
    model: str
    mae: float
    rmse: float


class ProjectSummary(BaseModel):
    project: dict[str, str]
    datasets: list[DatasetSummary]
    phase1_sample: dict[str, int]
    phase2_modeling: dict[str, int]
    model_metrics: list[ModelMetric]
    recommender_metrics: list[RecommenderMetric]
    findings: list[str]


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    rating: float | None = Field(default=None, ge=1, le=5)


class SentimentResponse(BaseModel):
    model: str
    label: SentimentLabel
    score: float
    confidence: float
    rating_label: SentimentLabel | None = None
    rationale: list[str]


class EnhanceRequest(BaseModel):
    actual_rating: float = Field(ge=1, le=5)
    sentiment_score: float = Field(ge=-1, le=1)
    sentiment_weight: float = Field(default=0.3, ge=0, le=1)


class EnhanceResponse(BaseModel):
    actual_rating: float
    sentiment_score: float
    sentiment_virtual_rating: float
    sentiment_weight: float
    enhanced_rating: float
    interpretation: str


class SummaryExample(BaseModel):
    review_number: int
    rating: float
    label: SentimentLabel
    original_word_count: int
    summary_word_count: int
    summary: str


class LlmExamplesResponse(BaseModel):
    summarization_model: str
    response_model: str
    summaries: list[SummaryExample]
    service_response: dict[str, str]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    max_evidence: int = Field(default=3, ge=1, le=5)


class EvidenceHit(BaseModel):
    id: str
    title: str
    body: str
    score: float


class AskResponse(BaseModel):
    answer: str
    evidence: list[EvidenceHit]
