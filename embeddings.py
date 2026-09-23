from functools import lru_cache
from sentence_transformers import SentenceTransformer
from config import get_settings


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model_name)


def build_embedding_text(role: str, skills: list[str], description: str) -> str:
    """Combine the searchable fields into one string."""
    skills_text = ", ".join(skills)
    return f"Role: {role}\nSkills: {skills_text}\nDescription: {description}"


def embed_text(text: str) -> list[float]:
    """Turn text into an embedding vector."""
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()