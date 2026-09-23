from pydantic import BaseModel, Field


class Developer(BaseModel):
    """Payload for creating a developer record."""

    id: int = Field(..., description="Unique developer ID — also used as the Qdrant point ID")
    name: str
    role: str
    skills: list[str]
    description: str


class SearchResult(BaseModel):
    """A developer record plus its cosine-similarity score against a query."""

    developer: Developer
    score: float