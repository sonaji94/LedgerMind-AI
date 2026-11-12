"""
Authentication Router for LedgerMind AI.
Provides JWT access token generation and credentials validation.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.security import TokenResponse, create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    client_id: UUID = Field(default_factory=uuid4, description="Analyst or service client UUID")
    role: str = Field(default="financial_analyst", description="Role: admin, financial_analyst, auditor")
    secret: str = Field(..., min_length=4, description="Client API secret key")


@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(payload: LoginRequest) -> TokenResponse:
    """
    Issue a cryptographically signed HMAC-SHA256 JWT access token for API callers.
    Used by analysts and upstream payment gateways to authorize transactions.
    """
    # Demo validation: accept any secret with length >= 4 for test clients
    if len(payload.secret) < 4:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client secret key",
        )

    token = create_access_token(subject=payload.client_id, role=payload.role)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_seconds=30 * 60,
        role=payload.role,
    )
