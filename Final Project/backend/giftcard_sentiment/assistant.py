"""Grounded project assistant over committed evidence with optional vLLM generation."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from giftcard_sentiment.schemas import EvidenceHit
from giftcard_sentiment.settings import Settings

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


@dataclass(slots=True)
class AssistantResult:
    answer: str
    citations: list[str]
    mode: str
    assistant_model: str | None


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


def fallback_answer(question: str, hits: list[EvidenceHit]) -> str:
    if not hits:
        return (
            "I could not find project evidence for that question. Ask a project-specific "
            "question about the dataset, lexicon models, TF-IDF, Logistic Regression, "
            "Naive Bayes, recommender results, or the LLM task outputs."
        )
    lead = hits[0]
    question_lower = question.lower()
    if "macro" in question_lower or "accuracy" in question_lower:
        return (
            "Accuracy is not enough for this project because the Gift Cards reviews are heavily "
            "skewed positive. The better professional answer is to discuss macro-F1 and confusion "
            f"matrices. Evidence: {lead.body}"
        )
    if "recommend" in question_lower or "enhance" in question_lower:
        return (
            "The recommender enhancement was implemented and tested, but it did not improve "
            "MAE/RMSE. The clean interpretation is that sentiment added noise for this very "
            f"positive dataset. Evidence: {lead.body}"
        )
    if "tfidf" in question_lower or "tf-idf" in question_lower:
        return (
            "TF-IDF was chosen because it is fast, interpretable, and strong for classical "
            f"review classification. Evidence: {lead.body}"
        )
    if "logistic" in question_lower or "naive bayes" in question_lower:
        return (
            "Logistic Regression was the better balanced model in this project because it handled "
            "minority-class performance better on macro-F1, while Naive Bayes mainly won on raw "
            f"accuracy. Evidence: {lead.body}"
        )
    if "vader" in question_lower or "sentiwordnet" in question_lower or "lexicon" in question_lower:
        return (
            "VADER is the more practical lexicon for short customer reviews, while SentiWordNet is "
            "useful for the academic comparison because it shows a more linguistic pipeline but is "
            f"more brittle. Evidence: {lead.body}"
        )
    if "limit" in question_lower or "presentation" in question_lower:
        return (
            "The professional presentation angle is to be honest about the limitations: "
            "strong class imbalance, short repetitive review text, and a recommender "
            f"enhancement that did not beat the baseline. Evidence: {lead.body}"
        )
    return f"{lead.body} This is the strongest matching project evidence for: {question}"


def answer(question: str, hits: list[EvidenceHit]) -> AssistantResult:
    citations = [hit.id for hit in hits[:2]]
    return AssistantResult(
        answer=fallback_answer(question, hits),
        citations=citations,
        mode="retrieval_fallback",
        assistant_model=None,
    )


def maybe_generate_grounded_answer(
    question: str, hits: list[EvidenceHit], settings: Settings
) -> AssistantResult | None:
    if not settings.llm_enabled or not hits:
        return None

    prompt = _build_prompt(question, hits)
    request_body = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a grounded assistant for a university NLP project. "
                    "Use only the supplied evidence. If the evidence is incomplete, "
                    "say what is missing instead of guessing. End with a final line in "
                    "this exact format: Citations: id1, id2"
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    payload = json.dumps(request_body).encode("utf-8")
    headers = {"content-type": "application/json"}
    if settings.llm_api_key:
        headers["authorization"] = f"Bearer {settings.llm_api_key}"
    request = Request(
        urljoin(settings.llm_base_url.rstrip("/") + "/", "chat/completions"),
        data=payload,
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=settings.llm_timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

    parsed = _parse_model_output(content, hits)
    if parsed is None:
        return None

    return AssistantResult(
        answer=parsed["answer"],
        citations=parsed["citations"],
        mode="vllm_grounded",
        assistant_model=settings.llm_model,
    )


def _build_prompt(question: str, hits: Iterable[EvidenceHit]) -> str:
    evidence_block = "\n\n".join(
        f"[{hit.id}] {hit.title}\n{hit.body}" for hit in hits
    )
    return (
        f"Question: {question}\n\n"
        "Use only the evidence below. Keep the answer direct and professional. "
        "Explain tradeoffs when relevant. If the evidence does not fully answer the "
        "question, say that explicitly.\n\n"
        f"Evidence:\n{evidence_block}"
    )


def _parse_model_output(content: str, hits: list[EvidenceHit]) -> dict[str, list[str] | str] | None:
    valid_ids = {hit.id for hit in hits}
    answer_text = content.strip()
    match = re.search(r"Citations:\s*(.+)$", answer_text, flags=re.IGNORECASE | re.MULTILINE)
    citations: list[str] = []
    if match:
        raw = [part.strip().strip("[]().;") for part in match.group(1).split(",")]
        citations = [item for item in raw if item in valid_ids]
        answer_text = answer_text[: match.start()].strip()
    if not answer_text:
        return None
    if not citations:
        return None
    return {"answer": answer_text, "citations": citations}
