"""Load committed notebook-derived artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from giftcard_sentiment.schemas import LlmExamplesResponse, ProjectSummary


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache
def load_project_summary(artifacts_dir: str) -> ProjectSummary:
    data = _read_json(Path(artifacts_dir) / "metrics" / "project_summary.json")
    return ProjectSummary.model_validate(data)


@lru_cache
def load_llm_examples(artifacts_dir: str) -> LlmExamplesResponse:
    data = _read_json(Path(artifacts_dir) / "examples" / "llm_examples.json")
    return LlmExamplesResponse.model_validate(data)


@lru_cache
def load_evidence(artifacts_dir: str) -> list[dict[str, str]]:
    data = _read_json(Path(artifacts_dir) / "evidence" / "project_evidence.json")
    return list(data["evidence"])
