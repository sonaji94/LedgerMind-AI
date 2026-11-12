"""
LedgerMind AI — Real-Time Transaction Ingestion & Automated Fraud Triage Router.

Exposes:
- POST /api/v1/transactions/ingest: High-throughput webhook for card processors & payment gateways.
  - Idempotency-Key enforcement (prevents double charging)
  - Sub-millisecond Redis velocity recording
  - Real-time TreeSHAP risk scoring
  - Conditional atomic double-entry settlement or triage hold
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select

from app.core.database import get_db
from app.core.redis import RedisVelocityEngine
from app.fraud.features import IngestionTransaction
from app.fraud.service import FraudInferenceService
from app.ledger.domain import (
    AccountStatusError,
    CurrencyMismatchError,
    InsufficientFundsError,
    Money,
)
from app.ledger.models import AccountModel, IdempotencyRecordModel
from app.ledger.service import LedgerService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/transactions", tags=["Transaction Ingestion"])


@router.post(
    "/ingest",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest and evaluate live transaction stream",
)
async def ingest_transaction(
    payload: IngestionTransaction,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    High-throughput transaction ingestion endpoint:
    1. Idempotency Check (Idempotency-Key header or transaction_id)
    2. Velocity Store Record in Redis / In-memory
    3. Real-Time Machine Learning & TreeSHAP Anomaly Scoring
    4. If Suspicious: Quarantine funds, hold transfer, return risk factors
    5. If Safe: Atomically commit dual-legged ledger transfer
    """
    effective_idemp_key = idempotency_key or f"ingest-{payload.transaction_id}"

    # 1. Idempotency Deduplication Check
    q = select(IdempotencyRecordModel).where(
        IdempotencyRecordModel.idempotency_key == effective_idemp_key
    )
    existing = (await db.execute(q)).scalar_one_or_none()
    if existing:
        cached_data = json.loads(existing.response_payload)
        cached_data["cached_replay"] = True
        return cached_data

    # 2. Record Transaction in Redis Velocity Engine
    await RedisVelocityEngine.record_transaction_event(
        account_id=payload.source_account_id,
        amount_minor=payload.amount_minor_units,
        timestamp=payload.timestamp.timestamp(),
    )

    # 3. Score Transaction with Champion Model + TreeSHAP Explainability
    eval_result = await FraudInferenceService.predict_ingestion_transaction(payload)
    is_suspicious = eval_result["is_suspicious"]
    risk_score = eval_result["risk_score"]

    # 4. Handle Suspicious Outlier (Interception & Hold)
    if is_suspicious:
        response_payload = {
            "transaction_id": payload.transaction_id,
            "status": "FLAGGED_FOR_TRIAGE",
            "decision": "FLAGGED_INTERCEPT",
            "risk_score": risk_score,
            "threshold_used": eval_result["threshold_used"],
            "latency_ms": eval_result["latency_ms"],
            "risk_factors": eval_result["risk_factors"],
            "shap_attributions": eval_result["shap_attributions"],
            "behavioral_signals": eval_result["behavioral_signals"],
            "ledger_settled": False,
            "message": "Transaction intercepted due to elevated fraud risk. Quarantined in compliance queue.",
        }

        # Save Idempotency Record
        idemp = IdempotencyRecordModel(
            idempotency_key=effective_idemp_key,
            endpoint="/transactions/ingest",
            status_code=status.HTTP_200_OK,
            response_payload=json.dumps(response_payload),
        )
        db.add(idemp)
        await db.flush()
        return response_payload

    # 5. Handle Legitimate Transaction (Execute Atomic Double-Entry Settlement)
    service = LedgerService(db)

    # Resolve source account (or create if not present)
    try:
        src_uuid = UUID(payload.source_account_id)
    except ValueError:
        src_uuid = None

    src_query = select(AccountModel).where(AccountModel.id == src_uuid) if src_uuid else select(AccountModel).where(AccountModel.id == payload.source_account_id)
    src_row = (await db.execute(src_query)).scalar_one_or_none()
    if not src_row:
        src_acc = await service.create_account(
            owner_id=uuid4(),
            currency=payload.currency,
            initial_deposit_minor_units=payload.amount_minor_units * 3,
        )
        src_uuid = src_acc.id
    else:
        src_uuid = src_row.id

    # Resolve destination account (or create if not present)
    try:
        dst_uuid = UUID(payload.destination_account_id)
    except ValueError:
        dst_uuid = None

    dst_query = select(AccountModel).where(AccountModel.id == dst_uuid) if dst_uuid else select(AccountModel).where(AccountModel.id == payload.destination_account_id)
    dst_row = (await db.execute(dst_query)).scalar_one_or_none()
    if not dst_row:
        dst_acc = await service.create_account(
            owner_id=uuid4(),
            currency=payload.currency,
            initial_deposit_minor_units=50000,
        )
        dst_uuid = dst_acc.id
    else:
        dst_uuid = dst_row.id

    money = Money(minor_units=payload.amount_minor_units, currency=payload.currency)

    try:
        domain_txn = await service.execute_transfer(
            source_account_id=src_uuid,
            destination_account_id=dst_uuid,
            amount=money,
            idempotency_key=effective_idemp_key,
            description=f"Ingested Webhook Transfer {payload.transaction_id} via {payload.channel}",
        )
        settled_txn_id = str(domain_txn.transaction_id)
    except (InsufficientFundsError, AccountStatusError, CurrencyMismatchError) as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)) from e

    response_payload = {
        "transaction_id": payload.transaction_id,
        "status": "SETTLED_POSTED",
        "decision": "AUTOMATED_APPROVE",
        "risk_score": risk_score,
        "threshold_used": eval_result["threshold_used"],
        "latency_ms": eval_result["latency_ms"],
        "risk_factors": eval_result["risk_factors"],
        "ledger_settled": True,
        "ledger_transaction_id": settled_txn_id,
        "posted_at": domain_txn.posted_at.isoformat(),
        "amount_major": float(money.to_major()),
        "currency": money.currency,
        "message": "Transaction verified and settled atomically with zero ledger drift.",
    }

    # Save Idempotency Record
    idemp = IdempotencyRecordModel(
        idempotency_key=effective_idemp_key,
        endpoint="/transactions/ingest",
        status_code=status.HTTP_201_CREATED,
        response_payload=json.dumps(response_payload),
    )
    db.add(idemp)
    await db.flush()

    return response_payload
