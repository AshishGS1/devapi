from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client.models import PointStruct

from config import get_settings
from db import get_qdrant_client
from embeddings import build_embedding_text, embed_text
from models import Developer, SearchResult
from auth import verify_auth_key

router = APIRouter(dependencies=[Depends(verify_auth_key)])
settings = get_settings()


@router.post("", response_model=Developer, status_code=status.HTTP_201_CREATED)
def create_developer(dev: Developer):
    """Embed a developer profile and store it in Qdrant."""
    client = get_qdrant_client()
    text = build_embedding_text(dev.role, dev.skills, dev.description)
    vector = embed_text(text)

    client.upsert(
        collection_name=settings.collection_name,
        pts=[
            PointStruct(
                id=dev.id,
                vector=vector,
                payload=dev.model_dump(),
            )
        ],
    )
    return Developer(**dev.model_dump())


# Keep this route before /{developer_id} to avoid path conflicts.
@router.get("/search", response_model=list[SearchResult])
def search(q: str, limit: int = 3):
    """Search for developers using a text query."""
    client = get_qdrant_client()
    query_vector = embed_text(q)

    hits = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=limit,
    ).pts

    return [
        SearchResult(dev=Developer(**hit.payload), score=hit.score)
        for hit in hits
    ]


@router.get("/{developer_id}", response_model=Developer)
def get_by_id(developer_id: int):
    """Fetch a developer by ID."""
    client = get_qdrant_client()
    pts = client.retrieve(
        collection_name=settings.collection_name,
        ids=[developer_id],
    )
    if not pts:
        raise HTTPException(status_code=404, detail=f"Developer {developer_id} not found")
    return Developer(**pts[0].payload)


@router.get("", response_model=list[Developer])
def get_all(limit: int = 100):
    """Return a paged list of developers."""
    client = get_qdrant_client()
    pts, _ = client.scroll(
        collection_name=settings.collection_name,
        limit=limit,
    )
    return [Developer(**p.payload) for p in pts]