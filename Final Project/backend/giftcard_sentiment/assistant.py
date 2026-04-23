"""Small retrieval assistant over project evidence."""

from __future__ import annotations

import re
from collections import Counter

from giftcard_sentiment.schemas import EvidenceHit

WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "why",
    "with",
}


def _tokens(text: str) -> Counter[str]:
    return Counter(tok for tok in WORD_RE.findall(text.lower()) if tok not in STOPWORDS)


def retrieve(question: str, evidence: list[dict[str, str]], limit: int) -> list[EvidenceHit]:
    query = _tokens(question)
    if not query:
        return []
    hits: list[EvidenceHit] = []
    for item in evidence:
        doc = _tokens(f"{item['title']} {item['body']}")
        if not doc:
            continue
        overlap = sum(min(query[token], doc[token]) for token in query)
        score = overlap / max(sum(query.values()), 1)
        if score > 0:
            hits.append(EvidenceHit(**item, score=round(score, 4)))
    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]


def answer(question: str, hits: list[EvidenceHit]) -> str:
    if not hits:
        return (
            "I could not find project evidence for that question. Ask a project-specific "
            "question about the dataset, lexicon models, TF-IDF, Logistic Regression, "
            "Naive Bayes, recommender results, or the LLM task outputs."
        )
    lead = hits[0]
    if "macro" in question.lower() or "accuracy" in question.lower():
        return (
            "Accuracy is not enough for this project because the Gift Cards reviews are heavily "
            "skewed positive. The better professional answer is to discuss macro-F1 and confusion "
            f"matrices. Evidence: {lead.body}"
        )
    if "recommend" in question.lower() or "enhance" in question.lower():
        return (
            "The recommender enhancement was implemented and tested, but it did not improve "
            "MAE/RMSE. The clean interpretation is that sentiment added noise for this very "
            f"positive dataset. Evidence: {lead.body}"
        )
    if "tfidf" in question.lower() or "tf-idf" in question.lower():
        return (
            "TF-IDF was chosen because it is fast, interpretable, and strong for classical "
            f"review classification. Evidence: {lead.body}"
        )
    return f"{lead.body} This is the strongest matching project evidence for: {question}"
