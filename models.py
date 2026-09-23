from pydantic import BaseModel, Field


class Developer(BaseModel):
    """Developer record used for create and read operations"""
    id: int = Field(..., description="Unique developer ID used as the Qdrant point ID")
    name: str
    role: str
    skills: list[str]
    description: str


class SearchResult(BaseModel):
    """A developer match with its similarity score"""
    dev: Developer
    score: float