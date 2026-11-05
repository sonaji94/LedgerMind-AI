"""
LedgerMind AI — Security & JWT Authentication Module

Provides:
- Token generation with HMAC-SHA256 signature and expiration
- Token validation and payload decoding
- FastAPI dependency for extracting and validating authenticated actor
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.core.config import get_settings

settings = get_settings()
security_scheme = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"


class TokenPayload(BaseModel):
    """Authenticated user context extracted from validated JWT."""

    sub: str
    role: str = "analyst"
    exp: datetime


class TokenResponse(BaseModel):
    """Response returned upon successful authentication."""

    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int


def create_access_token(
    subject: str | UUID,
    role: str = "analyst",
    expires_delta: timedelta | None = None,
) -> str:
    """Generate a signed JWT access token."""
    now = datetime.now(UTC)
    expire = now + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))

    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "iat": now,
        "exp": expire,
        "iss": "ledgermind-auth",
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)
    return str(token)


def decode_access_token(token: str) -> TokenPayload:
    """Validate and decode a signed JWT."""
    try:
        decoded = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[ALGORITHM],
            issuer="ledgermind-auth",
        )
        return TokenPayload(
            sub=decoded["sub"],
            role=decoded.get("role", "analyst"),
            exp=datetime.fromtimestamp(decoded["exp"], tz=UTC),
        )
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_current_actor(
    credentials: HTTPAuthorizationCredentials | None = Security(security_scheme),
) -> TokenPayload:
    """
    FastAPI dependency that enforces JWT authentication.
    In development mode, if no Authorization header is passed, returns a default system actor.
    In production mode, strictly raises 401 Unauthorized.
    """
    if credentials is None or not credentials.credentials:
        if settings.app_env in ("development", "testing"):
            # Permissive fallback for seamless local testing
            return TokenPayload(
                sub="00000000-0000-0000-0000-000000000000",
                role="system_admin",
                exp=datetime.now(UTC) + timedelta(hours=24),
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decode_access_token(credentials.credentials)
