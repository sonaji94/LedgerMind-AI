"""
Financial Transfers & Transaction Router for LedgerMind AI.
Enforces double-entry atomic postings and scoped idempotency.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select

from app.core.database import get_db
from app.core.schemas import TransferRequest, TransferResponse
from app.ledger.domain import (
    AccountStatusError,
    CurrencyMismatchError,
    InsufficientFundsError,
    Money,
)
from app.ledger.models import IdempotencyRecordModel
from app.ledger.service import LedgerService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/transfers", tags=["Transfers"])


@router.post("", response_model=TransferResponse, status_code=status.HTTP_201_CREATED)
async def create_transfer(
    payload: TransferRequest,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    db: AsyncSession = Depends(get_db),
) -> TransferResponse:
    """
    Execute an atomic, balanced transfer between accounts.
    Guarantees idempotency when an Idempotency-Key header is supplied.
    """
    # 1. Idempotency Check
    if idempotency_key:
        query = select(IdempotencyRecordModel).where(
            IdempotencyRecordModel.idempotency_key == idempotency_key
        )
        existing = (await db.execute(query)).scalar_one_or_none()
        if existing:
            # Safe replay: return previously recorded response
            data = json.loads(existing.response_payload)
            return TransferResponse(**data)

    service = LedgerService(db)
    amount = Money(minor_units=payload.amount_minor_units, currency=payload.currency)

    try:
        domain_txn = await service.execute_transfer(
            source_account_id=payload.source_account_id,
            destination_account_id=payload.destination_account_id,
            amount=amount,
            idempotency_key=idempotency_key,
            description=payload.description,
        )
    except InsufficientFundsError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)) from e
    except (CurrencyMismatchError, AccountStatusError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    response = TransferResponse(
        transaction_id=domain_txn.transaction_id,
        source_account_id=payload.source_account_id,
        destination_account_id=payload.destination_account_id,
        amount_minor_units=payload.amount_minor_units,
        amount_major=float(amount.to_major()),
        currency=payload.currency,
        description=payload.description,
        posted_at=domain_txn.posted_at,
        idempotency_key=idempotency_key,
    )

    # Record idempotency outcome
    if idempotency_key:
        idemp_record = IdempotencyRecordModel(
            idempotency_key=idempotency_key,
            endpoint="/transfers",
            status_code=status.HTTP_201_CREATED,
            response_payload=response.model_dump_json(),
        )
        db.add(idemp_record)
        await db.flush()

    return response
