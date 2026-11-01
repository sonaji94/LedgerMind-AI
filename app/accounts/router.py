"""
Account Management Router for LedgerMind AI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.database import get_db
from app.core.schemas import AccountCreateRequest, AccountResponse
from app.ledger.service import LedgerService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/accounts", tags=["Accounts"])


@router.post("", response_model=AccountResponse, status_code=status.HTTP_201_CREATED)
async def create_account(
    payload: AccountCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> AccountResponse:
    """Create a new financial account and initialize its balance projection."""
    service = LedgerService(db)
    account = await service.create_account(
        owner_id=payload.owner_id,
        currency=payload.currency,
        initial_deposit_minor_units=payload.initial_deposit_minor_units,
    )

    balance = await service.get_balance(account.id)
    return AccountResponse(
        id=account.id,
        owner_id=account.owner_id,
        currency=account.currency,
        status=account.status,
        balance_minor_units=balance.minor_units,
        balance_major=float(balance.to_major()),
        created_at=account.created_at,
    )


@router.get("/{account_id}", response_model=AccountResponse)
async def get_account(
    account_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AccountResponse:
    """Retrieve an account by ID along with its live balance projection."""
    service = LedgerService(db)
    try:
        balance = await service.get_balance(account_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    from sqlalchemy import select

    from app.ledger.models import AccountModel

    result = await db.execute(select(AccountModel).where(AccountModel.id == account_id))
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")

    return AccountResponse(
        id=account.id,
        owner_id=account.owner_id,
        currency=account.currency,
        status=account.status,
        balance_minor_units=balance.minor_units,
        balance_major=float(balance.to_major()),
        created_at=account.created_at,
    )
