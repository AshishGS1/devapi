from functools import lru_cache

from qdrant_client import QdrantClient

from config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Single shared QdrantClient instance, reused across requests instead
    of opening a new connection per call."""
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )