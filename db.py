from functools import lru_cache

from qdrant_client import QdrantClient

from config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Reuse a single Qdrant client across the app."""
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )