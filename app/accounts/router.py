"""
Account Management Router for LedgerMind AI.

Endpoints:
  POST   /accounts                             — Create account
  GET    /accounts/{id}                        — Get account with balance
  GET    /accounts/{id}/transactions           — Paginated journal history
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select

from app.core.database import get_db
from app.core.schemas import (
    AccountCreateRequest,
    AccountResponse,
    JournalEntryItem,
    TransactionHistoryResponse,
)
from app.ledger.models import AccountModel, JournalEntryModel, JournalTransactionModel
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


@router.get("/{account_id}/transactions", response_model=TransactionHistoryResponse)
async def get_account_transactions(
    account_id: UUID,
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page (max 100)"),
) -> TransactionHistoryResponse:
    """
    Return a paginated list of all journal entries (debits and credits)
    for the specified account, ordered newest-first.
    """
    # Verify the account exists
    acc_result = await db.execute(select(AccountModel).where(AccountModel.id == account_id))
    if not acc_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")

    # Count total entries for this account
    count_q = select(func.count(JournalEntryModel.id)).where(
        JournalEntryModel.account_id == account_id
    )
    total: int = (await db.execute(count_q)).scalar() or 0

    # Fetch paginated entries joined with their parent transaction (for posted_at)
    offset = (page - 1) * page_size
    rows_q = (
        select(JournalEntryModel, JournalTransactionModel.posted_at)
        .join(
            JournalTransactionModel,
            JournalEntryModel.transaction_id == JournalTransactionModel.id,
        )
        .where(JournalEntryModel.account_id == account_id)
        .order_by(JournalTransactionModel.posted_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    rows = (await db.execute(rows_q)).all()

    items = [
        JournalEntryItem(
            id=entry.id,
            transaction_id=entry.transaction_id,
            entry_type=entry.entry_type,
            amount_minor_units=entry.amount_minor_units,
            amount_major=float(entry.amount_minor_units) / 100.0,
            posted_at=posted_at,
        )
        for entry, posted_at in rows
    ]

    return TransactionHistoryResponse(
        account_id=account_id,
        page=page,
        page_size=page_size,
        total=total,
        items=items,
    )
