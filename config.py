from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings loaded from the environment and .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    qdrant_url: str
    qdrant_api_key: str | None = None
    collection_name: str = "developers"

    auth_key: str = "7t2m-jXdfb5AGQD8Hen-7j3k5hPZCUYAZAtXh-O_q6g"

    embedding_model_name: str = "all-MiniLM-L6-v2"
    vector_size: int = 384


@lru_cache
def get_settings() -> Settings:
    """Load and cache settings once per process."""
    return Settings()