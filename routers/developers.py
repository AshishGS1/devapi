from fastapi import APIRouter, HTTPException, status
from qdrant_client.models import PointStruct

from config import get_settings
from db import get_qdrant_client
from embeddings import build_embedding_text, embed_text
from models import Developer, SearchResult

router = APIRouter()
settings = get_settings()


@router.post("", response_model=Developer, status_code=status.HTTP_201_CREATED)
def create_developer(developer: Developer):
    """
    Embeds role + skills + description, then upserts a point into Qdrant
    using `id` as the point ID and the full record as payload.
    """
    client = get_qdrant_client()
    text = build_embedding_text(developer.role, developer.skills, developer.description)
    vector = embed_text(text)

    client.upsert(
        collection_name=settings.collection_name,
        points=[
            PointStruct(
                id=developer.id,
                vector=vector,
                payload=developer.model_dump(),
            )
        ],
    )
    return Developer(**developer.model_dump())


# NOTE: this is registered before /{developer_id} on purpose. FastAPI's
# int converter on developer_id would still correctly skip "search" and
# fall through to this route either way, but keeping specific paths
# ahead of parameterized ones avoids relying on that.
@router.get("/search", response_model=list[SearchResult])
def search(q: str, limit: int = 3):
    """
    Embeds the query string with the SAME function used at insert time
    (build_embedding_text isn't used here directly since a search query
    is free text, not three separate fields — embed_text alone is
    applied to whatever the user typed), then runs a cosine-similarity
    search against Qdrant.
    """
    client = get_qdrant_client()
    query_vector = embed_text(q)

    hits = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=limit,
    ).points

    return [
        SearchResult(developer=Developer(**hit.payload), score=hit.score)
        for hit in hits
    ]


@router.get("/{developer_id}", response_model=Developer)
def get_by_id(developer_id: int):
    """Direct point lookup by ID — no vector math involved."""
    client = get_qdrant_client()
    points = client.retrieve(
        collection_name=settings.collection_name,
        ids=[developer_id],
    )
    if not points:
        raise HTTPException(status_code=404, detail=f"Developer {developer_id} not found")
    return Developer(**points[0].payload)


@router.get("", response_model=list[Developer])
def get_all(limit: int = 100):
    """
    Pages through all points via scroll(). `limit` caps how many come
    back in one call
    """
    client = get_qdrant_client()
    points, _next_offset = client.scroll(
        collection_name=settings.collection_name,
        limit=limit,
    )
    return [Developer(**p.payload) for p in points]