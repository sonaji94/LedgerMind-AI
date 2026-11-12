"""
LedgerMind AI — Razorpay Webhook Ingestion, Reconciliation & ML Defense Router.

Endpoints:
- POST /api/v1/webhooks/razorpay:
    - HMAC-SHA256 signature verification on raw bytes (X-Razorpay-Signature)
    - Strict idempotency deduplication via webhook_events table
    - Sub-200ms ACK (HTTP 200) with async background processing
    - Multi-leg balanced double-entry reconciliation (paise precision)
    - Real-time feature extraction (Redis sliding windows + Postgres)
- GET /api/v1/webhooks/razorpay/events: Query recent webhook events
- POST /api/v1/webhooks/razorpay/feedback: Human-in-the-loop labeling
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from app.core.audit import record_audit_log
from app.core.config import get_settings
from app.core.database import async_session_factory, get_db
from app.core.security import TokenPayload, UserRole, get_current_actor, require_role
from app.fraud.feature_extractor import FeatureExtractor, PaymentEventContext
from app.fraud.service import FraudInferenceService
from app.ledger.models import WebhookEventModel
from app.ledger.service import LedgerService
from app.webhooks.security import verify_razorpay_signature

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/webhooks/razorpay", tags=["Razorpay Webhooks & Reconciliation"])


class LabelFeedbackRequest(BaseModel):
    """Analyst ground-truth label feedback payload."""

    entity_id: str = Field(..., description="Razorpay payment_id (pay_...) or payout_id (pout_...)")
    ground_truth_label: int = Field(..., ge=0, le=1, description="1 = Confirmed Fraud/Chargeback, 0 = Confirmed Legitimate")
    justification_reason: str = Field(..., min_length=4, description="Analyst reason for verified classification")


async def process_razorpay_event_background(
    event_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """
    Asynchronous background worker:
    1. Extracts real-world features from payload + Redis + PostgreSQL
    2. Runs champion ML model scoring
    3. Executes atomic double-entry ledger reconciliation
    4. Persists execution outcome and risk attributions
    """
    async with async_session_factory() as session:
        try:
            # 1. Retrieve the existing received event
            result = await session.execute(
                select(WebhookEventModel).where(WebhookEventModel.event_id == event_id)
            )
            event_row = result.scalar_one_or_none()
            if not event_row:
                logger.error("webhook_event_not_found_background", event_id=event_id)
                return

            event_row.status = "PROCESSING"
            await session.flush()

            # 2. Extract Entity Context from Razorpay payload
            payload_data = payload.get("payload", {})
            entity_id = "unknown"
            amount_minor = 0
            fee_minor = 0
            tax_minor = 0
            currency = "INR"
            method = "card"
            card_bin = None
            card_last4 = None
            card_intl = False
            vpa = None
            email = None
            contact = None
            account_id = None
            hours_since_bank = 720.0

            if "payment" in payload_data:
                pay_entity = payload_data["payment"].get("entity", {})
                entity_id = pay_entity.get("id", "unknown_pay")
                amount_minor = int(pay_entity.get("amount", 0))
                fee_minor = int(pay_entity.get("fee", 0))
                tax_minor = int(pay_entity.get("tax", 0))
                currency = pay_entity.get("currency", "INR")
                method = pay_entity.get("method", "card")
                email = pay_entity.get("email")
                contact = pay_entity.get("contact")
                vpa = pay_entity.get("vpa")
                card_info = pay_entity.get("card", {})
                if isinstance(card_info, dict):
                    card_bin = card_info.get("network") or card_info.get("issuer")
                    card_last4 = card_info.get("last4")
                    card_intl = bool(card_info.get("international", False))
                # Extra metadata
                notes = pay_entity.get("notes", {})
                if isinstance(notes, dict):
                    account_id = notes.get("account_id")

            elif "payout" in payload_data:
                pout_entity = payload_data["payout"].get("entity", {})
                entity_id = pout_entity.get("id", "unknown_pout")
                amount_minor = int(pout_entity.get("amount", 0))
                currency = pout_entity.get("currency", "INR")
                notes = pout_entity.get("notes", {})
                if isinstance(notes, dict):
                    account_id = notes.get("account_id")
                    if "hours_since_bank_change" in notes:
                        with contextlib.suppress(ValueError):
                            hours_since_bank = float(notes["hours_since_bank_change"])

            elif "dispute" in payload_data:
                disp_entity = payload_data["dispute"].get("entity", {})
                entity_id = disp_entity.get("id", "unknown_disp")
                amount_minor = int(disp_entity.get("amount", 0))
                currency = disp_entity.get("currency", "INR")

            ctx = PaymentEventContext(
                event_id=event_id,
                event_type=event_type,
                entity_id=entity_id,
                amount_minor_units=amount_minor,
                currency=currency,
                email=email,
                contact=contact,
                ip_address=payload.get("ip_address"),
                card_bin=card_bin,
                card_last4=card_last4,
                card_international=card_intl,
                vpa=vpa,
                method=method,
                account_id=account_id,
                hours_since_bank_change=hours_since_bank,
            )

            # 3. Extract Real-World Behavioral Features
            fvec, telemetry = await FeatureExtractor.extract_features(ctx, session)

            # 4. Score Transaction with ML Champion Model
            if not FraudInferenceService.is_ready():
                FraudInferenceService.load()

            # Adapt features to model input array
            feat_arr = fvec.to_array()
            risk_score = 0.05
            is_suspicious = False
            threshold = 0.50

            if FraudInferenceService._model is not None:
                try:
                    import numpy as np
                    probs = FraudInferenceService._model.predict_proba(np.array([feat_arr]))
                    risk_score = float(probs[0, 1])
                    threshold = float(getattr(FraudInferenceService._model, "optimal_threshold", 0.50))
                    is_suspicious = risk_score >= threshold
                except Exception:
                    # Fallback scoring
                    if fvec.distinct_cards_ip_10m >= 3 or fvec.payout_to_balance_ratio >= 0.85:
                        risk_score = 0.88
                        is_suspicious = True

            top_factors = FeatureExtractor.synthesize_top_risk_factors(fvec, risk_score)

            # 5. Automated Double-Entry Ledger Reconciliation
            ledger_service = LedgerService(session)
            journal_txn_id: UUID | None = None

            if event_type == "payment.captured":
                journal_txn_id = await ledger_service.reconcile_captured_payment(
                    payment_id=entity_id,
                    amount_minor=amount_minor,
                    fee_minor=fee_minor,
                    tax_minor=tax_minor,
                    currency=currency,
                    idempotency_key=f"rzp-capture-{entity_id}",
                )

            elif event_type == "payout.initiated":
                journal_txn_id = await ledger_service.reconcile_payout_hold(
                    payout_id=entity_id,
                    amount_minor=amount_minor,
                    currency=currency,
                    idempotency_key=f"rzp-payout-{entity_id}",
                )

            elif event_type == "payout.reversed":
                journal_txn_id = await ledger_service.reconcile_payout_reversed(
                    payout_id=entity_id,
                    amount_minor=amount_minor,
                    currency=currency,
                    idempotency_key=f"rzp-payout-rev-{entity_id}",
                )

            elif event_type == "payment.dispute.created":
                pay_id = payload_data.get("dispute", {}).get("entity", {}).get("payment_id", entity_id)
                journal_txn_id = await ledger_service.reconcile_dispute_created(
                    dispute_id=entity_id,
                    payment_id=pay_id,
                    amount_minor=amount_minor,
                    currency=currency,
                    idempotency_key=f"rzp-dispute-{entity_id}",
                )
                # Automatically mark dispute as confirmed fraud for active retraining
                event_row.ground_truth_label = 1
                event_row.labeled_by = "RAZORPAY_DISPUTE_WEBHOOK"
                event_row.labeled_at = datetime.now(UTC)

            # 6. Update Webhook Event Record
            event_row.status = "FLAGGED_FRAUD" if is_suspicious else "PROCESSED"
            event_row.risk_score = risk_score
            event_row.risk_decision = "FLAGGED_INTERCEPT" if is_suspicious else "CLEARED_APPROVE"
            event_row.journal_transaction_id = journal_txn_id
            event_row.processed_at = datetime.now(UTC)

            # Append telemetry into payload copy
            current_raw = dict(event_row.raw_payload)
            current_raw["_evaluation"] = {
                "risk_score": risk_score,
                "is_suspicious": is_suspicious,
                "threshold_applied": threshold,
                "top_factors": top_factors,
                "telemetry": telemetry,
            }
            event_row.raw_payload = current_raw

            await session.commit()
            logger.info(
                "razorpay_event_processed",
                event_id=event_id,
                event_type=event_type,
                risk_score=risk_score,
                journal_txn=str(journal_txn_id) if journal_txn_id else None,
            )

        except Exception as exc:
            await session.rollback()
            logger.error("razorpay_processing_failed", event_id=event_id, error=str(exc))
            try:
                result = await session.execute(
                    select(WebhookEventModel).where(WebhookEventModel.event_id == event_id)
                )
                err_row = result.scalar_one_or_none()
                if err_row:
                    err_row.status = "FAILED"
                    err_row.error_message = str(exc)
                    await session.commit()
            except Exception:
                pass


@router.post(
    "",
    status_code=status.HTTP_200_OK,
    summary="Secure Razorpay Webhook Ingestion",
)
async def receive_razorpay_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_razorpay_signature: str | None = Header(None, alias="X-Razorpay-Signature"),
    x_razorpay_event_id: str | None = Header(None, alias="X-Razorpay-Event-Id"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Enterprise Razorpay Webhook Receiver:
    1. Constant-time HMAC-SHA256 signature verification on raw request bytes
    2. Strict idempotency deduplication (prevents dual ledger posting)
    3. Fast acknowledgment (< 200ms) with background task execution
    """
    # 1. Read Raw Bytes before JSON parsing
    raw_body = await request.body()

    # 2. Verify HMAC-SHA256 Signature
    is_valid = verify_razorpay_signature(
        raw_body=raw_body,
        signature=x_razorpay_signature,
        secret=settings.razorpay_webhook_secret,
    )
    if not is_valid:
        logger.warning("razorpay_webhook_signature_rejected", header_present=bool(x_razorpay_signature))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Razorpay-Signature HMAC digest.",
        )

    # 3. Parse JSON Body
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed JSON payload: {exc}",
        ) from exc

    # 4. Resolve Event ID
    event_id = x_razorpay_event_id or payload.get("event_id") or payload.get("id")
    if not event_id:
        import uuid
        event_id = f"rzp_ev_{uuid.uuid4().hex[:16]}"

    event_type = payload.get("event", "payment.captured")

    # 5. Idempotency Deduplication Check
    existing = (
        await db.execute(select(WebhookEventModel).where(WebhookEventModel.event_id == event_id))
    ).scalar_one_or_none()

    if existing:
        return {
            "status": "already_processed",
            "event_id": event_id,
            "cached": True,
            "current_status": existing.status,
            "message": "Duplicate webhook retry intercepted. Existing ledger record preserved.",
        }

    # Extract high-level amounts for index record
    pdata = payload.get("payload", {})
    amt = 0
    fee = 0
    tax = 0
    curr = "INR"
    ent_id = event_id

    if "payment" in pdata:
        p = pdata["payment"].get("entity", {})
        ent_id = p.get("id", ent_id)
        amt = int(p.get("amount", 0))
        fee = int(p.get("fee", 0))
        tax = int(p.get("tax", 0))
        curr = p.get("currency", "INR")
    elif "payout" in pdata:
        p = pdata["payout"].get("entity", {})
        ent_id = p.get("id", ent_id)
        amt = int(p.get("amount", 0))
        curr = p.get("currency", "INR")
    elif "dispute" in pdata:
        p = pdata["dispute"].get("entity", {})
        ent_id = p.get("id", ent_id)
        amt = int(p.get("amount", 0))
        curr = p.get("currency", "INR")

    # 6. Insert Initial RECEIVED State Record
    event_record = WebhookEventModel(
        event_id=event_id,
        entity_id=ent_id,
        event_type=event_type,
        amount_minor_units=amt,
        fee_minor_units=fee,
        tax_minor_units=tax,
        currency=curr,
        status="RECEIVED",
        raw_payload=payload,
    )
    db.add(event_record)
    await db.commit()

    # 7. Offload ML Scoring & Ledger Posting to Background Worker
    background_tasks.add_task(
        process_razorpay_event_background,
        event_id=event_id,
        event_type=event_type,
        payload=payload,
    )

    return {
        "status": "acknowledged",
        "event_id": event_id,
        "event": event_type,
        "acknowledged_at": datetime.now(UTC).isoformat(),
        "message": "Webhook authenticated and queued for background reconciliation.",
    }


@router.get(
    "/events",
    summary="List Recent Razorpay Webhook Events",
)
async def list_webhook_events(
    limit: int = Query(25, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user_actor: TokenPayload = Depends(get_current_actor),
) -> list[dict[str, Any]]:
    """Retrieve the recent Razorpay webhook events and reconciliation states."""
    stmt = (
        select(WebhookEventModel)
        .order_by(desc(WebhookEventModel.created_at))
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()

    return [
        {
            "id": str(r.id),
            "event_id": r.event_id,
            "entity_id": r.entity_id,
            "event_type": r.event_type,
            "amount_minor": r.amount_minor_units,
            "amount_major": float(r.amount_minor_units) / 100.0,
            "fee_major": float(r.fee_minor_units) / 100.0,
            "tax_major": float(r.tax_minor_units) / 100.0,
            "currency": r.currency,
            "status": r.status,
            "risk_score": r.risk_score,
            "risk_decision": r.risk_decision,
            "ground_truth_label": r.ground_truth_label,
            "labeled_by": r.labeled_by,
            "journal_transaction_id": str(r.journal_transaction_id) if r.journal_transaction_id else None,
            "created_at": r.created_at.isoformat(),
            "processed_at": r.processed_at.isoformat() if r.processed_at else None,
            "evaluation": r.raw_payload.get("_evaluation") if isinstance(r.raw_payload, dict) else None,
        }
        for r in rows
    ]


@router.post(
    "/feedback",
    summary="Submit Analyst Ground-Truth Feedback",
)
async def submit_analyst_feedback(
    payload: LabelFeedbackRequest,
    db: AsyncSession = Depends(get_db),
    user_actor: TokenPayload = Depends(require_role([UserRole.ANALYST, UserRole.COMPLIANCE_OFFICER, UserRole.ADMIN])),
) -> dict[str, Any]:
    """
    Human-in-the-Loop Feedback:
    Analysts label transactions as confirmed fraud (1) or safe false positive (0).
    Writes to webhook_events and records an immutable compliance audit log.
    """
    stmt = select(WebhookEventModel).where(
        (WebhookEventModel.entity_id == payload.entity_id) | (WebhookEventModel.event_id == payload.entity_id)
    )
    event_row = (await db.execute(stmt)).scalars().first()
    if not event_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook event with entity ID {payload.entity_id} not found.",
        )

    role_name = user_actor.role
    event_row.ground_truth_label = payload.ground_truth_label
    event_row.labeled_by = f"{role_name}_analyst"
    event_row.labeled_at = datetime.now(UTC)

    # Record Audit Log
    label_str = "CONFIRMED_FRAUD" if payload.ground_truth_label == 1 else "MARKED_SAFE_FALSE_POSITIVE"
    await record_audit_log(
        session=db,
        operator_id=f"analyst-{role_name}",
        operator_role=role_name,
        action=f"LABEL_{label_str}",
        target_type="TRANSACTION_FEEDBACK",
        target_id=payload.entity_id,
        justification_reason=payload.justification_reason,
        metadata_json={
            "ground_truth_label": payload.ground_truth_label,
            "event_type": event_row.event_type,
            "amount_minor": event_row.amount_minor_units,
        },
    )
    await db.commit()

    return {
        "status": "success",
        "entity_id": payload.entity_id,
        "ground_truth_label": payload.ground_truth_label,
        "classification": label_str,
        "message": "Ground truth label committed to PostgreSQL. Available for active model retraining.",
    }
