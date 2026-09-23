from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client.models import Distance, VectorParams

from config import get_settings
from db import get_qdrant_client
from routers.developers import router as developers_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup: make sure the Qdrant collection exists before
    any request tries to read from or write to it.
    """
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.collection_name not in existing:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.vector_size,
                distance=Distance.COSINE,
            ),
        )
    yield
    # no teardown needed — QdrantClient over HTTPS doesn't hold a
    # connection that needs explicit closing


app = FastAPI(
    title="Developer Search API",
    description="Semantic search over a developer directory, backed by Qdrant.",
    lifespan=lifespan,
)

app.include_router(developers_router, prefix="/developers", tags=["developers"])


@app.get("/health")
def health_check():
    """Confirms the API is up AND can actually reach Qdrant (not just that .env loaded)."""
    client = get_qdrant_client()
    client.get_collections()  # raises if the connection or API key is broken
    return {"status": "ok", "collection": settings.collection_name}