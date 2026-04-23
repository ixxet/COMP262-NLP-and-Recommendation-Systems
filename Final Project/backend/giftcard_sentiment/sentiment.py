"""Lightweight live sentiment and rating-enhancement helpers.

The notebook remains the authoritative model training artifact. The API uses this
small deterministic scorer for interactive demos so the service can start quickly
without retraining or loading raw Amazon data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from giftcard_sentiment.schemas import SentimentLabel

POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "best",
    "convenient",
    "easy",
    "excellent",
    "fast",
    "favorite",
    "flexible",
    "good",
    "great",
    "happy",
    "love",
    "nice",
    "perfect",
    "quick",
    "reliable",
    "simple",
    "useful",
    "wonderful",
}

NEGATIVE_WORDS = {
    "angry",
    "bad",
    "broken",
    "confusing",
    "delay",
    "delayed",
    "disappointed",
    "failed",
    "horrible",
    "issue",
    "late",
    "missing",
    "never",
    "problem",
    "refund",
    "scam",
    "terrible",
    "unhappy",
    "useless",
    "wrong",
}

NEGATORS = {"not", "never", "no", "hardly", "barely", "without"}
TOKEN_RE = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class LiveSentiment:
    label: SentimentLabel
    score: float
    confidence: float
    rationale: list[str]


def rating_to_label(rating: float | None) -> SentimentLabel | None:
    if rating is None:
        return None
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def score_text(text: str) -> LiveSentiment:
    tokens = [token.lower().strip("'") for token in TOKEN_RE.findall(text)]
    pos_hits: list[str] = []
    neg_hits: list[str] = []
    raw_score = 0.0

    for index, token in enumerate(tokens):
        negated = any(prev in NEGATORS for prev in tokens[max(0, index - 3) : index])
        if token in POSITIVE_WORDS:
            raw_score += -1.0 if negated else 1.0
            (neg_hits if negated else pos_hits).append(token)
        elif token in NEGATIVE_WORDS:
            raw_score += 1.0 if negated else -1.0
            (pos_hits if negated else neg_hits).append(token)

    exclamation_boost = min(text.count("!"), 3) * 0.08
    if raw_score > 0:
        raw_score += exclamation_boost
    elif raw_score < 0:
        raw_score -= exclamation_boost

    length_norm = math.sqrt(max(len(tokens), 1))
    score = max(-1.0, min(1.0, raw_score / max(length_norm, 2.5)))
    if score >= 0.08:
        label: SentimentLabel = "positive"
    elif score <= -0.08:
        label = "negative"
    else:
        label = "neutral"

    confidence = min(0.98, 0.5 + abs(score) * 0.48)
    rationale = []
    if pos_hits:
        rationale.append(f"positive cues: {', '.join(sorted(set(pos_hits))[:6])}")
    if neg_hits:
        rationale.append(f"negative cues: {', '.join(sorted(set(neg_hits))[:6])}")
    if not rationale:
        rationale.append("no strong lexicon cues found; prediction falls near neutral")
    if exclamation_boost:
        rationale.append("punctuation intensity adjusted the score")
    return LiveSentiment(
        label=label,
        score=round(score, 4),
        confidence=round(confidence, 4),
        rationale=rationale,
    )


def enhanced_rating(
    actual_rating: float,
    sentiment_score: float,
    sentiment_weight: float = 0.3,
) -> dict[str, float | str]:
    virtual_rating = max(1.0, min(5.0, 3 + 2 * sentiment_score))
    enhanced = (1 - sentiment_weight) * actual_rating + sentiment_weight * virtual_rating
    interpretation = (
        "Sentiment raises the rating signal."
        if enhanced > actual_rating
        else "Sentiment lowers the rating signal."
        if enhanced < actual_rating
        else "Sentiment leaves the rating signal unchanged."
    )
    return {
        "actual_rating": round(actual_rating, 4),
        "sentiment_score": round(sentiment_score, 4),
        "sentiment_virtual_rating": round(virtual_rating, 4),
        "sentiment_weight": round(sentiment_weight, 4),
        "enhanced_rating": round(enhanced, 4),
        "interpretation": interpretation,
    }
