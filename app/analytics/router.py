"""
Financial Analytics & SQL Aggregation Router for LedgerMind AI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from app.core.database import get_db
from app.core.schemas import AnalyticsSummaryResponse
from app.ledger.models import AccountModel, JournalTransactionModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary", response_model=AnalyticsSummaryResponse)
async def get_analytics_summary(
    db: AsyncSession = Depends(get_db),
) -> AnalyticsSummaryResponse:
    """Calculate platform-wide financial summary metrics via SQL aggregations."""
    # Count total accounts
    acc_count = (await db.execute(select(func.count(AccountModel.id)))).scalar() or 0

    # Count total journal transactions
    txn_count = (await db.execute(select(func.count(JournalTransactionModel.id)))).scalar() or 0

    # Compute total transfer volume from JournalEntryModel (CREDITS)
    from app.ledger.models import JournalEntryModel

    vol_query = select(func.sum(JournalEntryModel.amount_minor_units)).where(
        JournalEntryModel.entry_type == "CREDIT"
    )
    total_volume_minor = (await db.execute(vol_query)).scalar() or 0

    total_volume_major = float(total_volume_minor) / 100.0
    avg_txn_major = (total_volume_major / txn_count) if txn_count > 0 else 0.0

    return AnalyticsSummaryResponse(
        total_accounts=acc_count,
        total_transactions=txn_count,
        total_volume_minor_units=total_volume_minor,
        total_volume_major=round(total_volume_major, 2),
        average_transaction_major=round(avg_txn_major, 2),
        currency="EUR",
    )
