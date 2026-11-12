"""
LedgerMind AI — Security, JWT Authentication & Role-Based Access Control (RBAC).

Provides:
- Token generation with HMAC-SHA256 signature and expiration
- Token validation and payload decoding
- Role-Based Access Control (Analyst, Compliance_Officer, Admin)
- FastAPI dependencies for extracting authenticated actors and enforcing permissions
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.core.config import get_settings

settings = get_settings()
security_scheme = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"


class UserRole(StrEnum):
    """Supported Role-Based Access Control tiers."""

    ANALYST = "Analyst"
    COMPLIANCE_OFFICER = "Compliance_Officer"
    ADMIN = "Admin"


class TokenPayload(BaseModel):
    """Authenticated user context extracted from validated JWT."""

    sub: str
    role: str = UserRole.ANALYST.value
    exp: datetime


class TokenResponse(BaseModel):
    """Response returned upon successful authentication."""

    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    role: str = UserRole.ANALYST.value


def create_access_token(
    subject: str | UUID,
    role: str = UserRole.ANALYST.value,
    expires_delta: timedelta | None = None,
) -> str:
    """Generate a signed JWT access token with role assertion."""
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
            role=decoded.get("role", UserRole.ANALYST.value),
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
    In development mode, if no Authorization header is passed, returns a default Admin actor.
    In production mode, strictly raises 401 Unauthorized.
    """
    if credentials is None or not credentials.credentials:
        if settings.app_env in ("development", "testing"):
            return TokenPayload(
                sub="00000000-0000-0000-0000-000000000000",
                role=UserRole.ADMIN.value,
                exp=datetime.now(UTC) + timedelta(hours=24),
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decode_access_token(credentials.credentials)


def require_role(allowed_roles: list[str]) -> Callable[..., Coroutine[Any, Any, TokenPayload]]:
    """
    FastAPI dependency factory enforcing Role-Based Access Control.
    Usage:
        @router.post("/freeze", dependencies=[Depends(require_role(["Compliance_Officer", "Admin"]))])
    """

    async def _role_checker(
        actor: TokenPayload = Depends(get_current_actor),
    ) -> TokenPayload:
        normalized_allowed = [r.lower() for r in allowed_roles]
        if actor.role.lower() not in normalized_allowed and actor.role.lower() != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires one of permissions: {allowed_roles}. Current role: '{actor.role}'",
            )
        return actor

    return _role_checker
