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

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"


@lru_cache
def get_settings() -> Settings:
    return Settings()
