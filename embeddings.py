from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import get_settings


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """
    Loads the embedding model once per process and reuses it. The first
    call after startup downloads the model will be slower
    than every one after it.
    """
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model_name)


def build_embedding_text(role: str, skills: list[str], description: str) -> str:
    """
    Combines the three fields that should be searchable into one string.
    """
    skills_text = ", ".join(skills)
    return f"Role: {role}\nSkills: {skills_text}\nDescription: {description}"


def embed_text(text: str) -> list[float]:
    """Turns a string into a vector using the configured embedding model."""
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()