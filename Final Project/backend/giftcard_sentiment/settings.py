"""Application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the API service."""

    model_config = SettingsConfigDict(env_prefix="GIFT_CARD_", env_file=".env", extra="ignore")

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        description="Final Project directory.",
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    service_name: str = "giftcard-sentiment-lab"
    llm_base_url: str | None = Field(
        default=None, description="OpenAI-compatible vLLM base URL."
    )
    llm_api_key: str | None = Field(
        default=None, description="Optional API key for the vLLM gateway."
    )
    llm_model: str | None = Field(
        default=None, description="Model name exposed by the vLLM server."
    )
    llm_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=320, ge=64, le=2048)

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
