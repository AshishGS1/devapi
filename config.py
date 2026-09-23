from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables / a .env file.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Qdrant Cloud connection (required — no defaults, must come from .env)
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "developers"

    # Embedding model — local sentence-transformers model by default.
    embedding_model_name: str = "all-MiniLM-L6-v2"
    vector_size: int = 384


@lru_cache
def get_settings() -> Settings:
    """
    Cached so .env is only read/validated once per process, not on every
    request. Import and call this function wherever settings are needed:
    """
    return Settings()