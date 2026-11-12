"""
LedgerMind AI — Immutable Compliance Audit Logging Engine.

Records every operator intervention (account freezes, 2FA step-up challenges,
alert dismissals, SAR report submissions) into the append-only audit_logs table.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.ledger.models import AuditLogModel

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


async def record_audit_log(
    session: AsyncSession,
    operator_id: str,
    operator_role: str,
    action: str,
    target_type: str,
    target_id: str,
    justification_reason: str,
    ip_address: str | None = None,
    metadata_json: dict[str, Any] | None = None,
) -> AuditLogModel:
    """
    Persist an immutable audit record of a compliance action.
    Never updates or deletes past audit events.
    """
    log_entry = AuditLogModel(
        id=uuid4(),
        operator_id=operator_id,
        operator_role=operator_role,
        action=action,
        target_type=target_type,
        target_id=target_id,
        ip_address=ip_address,
        justification_reason=justification_reason,
        metadata_json=metadata_json,
        created_at=datetime.now(UTC),
    )
    session.add(log_entry)
    await session.flush()

    logger.info(
        "compliance_audit_logged",
        action=action,
        operator_id=operator_id,
        operator_role=operator_role,
        target_type=target_type,
        target_id=target_id,
        ip=ip_address,
    )
    return log_entry
