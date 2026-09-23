import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config import get_settings

# Let us raise our own clear 401 response instead of FastAPI's default one.
_auth_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_auth_key(auth_key: str | None = Security(_auth_key_header)) -> None:
    """Check the X-API-Key header and block unauthorised requests."""
    settings = get_settings()
    if auth_key is None or not secrets.compare_digest(auth_key, settings.auth_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )