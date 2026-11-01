"""
AI Financial Assistant Router for LedgerMind AI.
Provides grounded natural language insights based strictly on verified SQL data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from app.core.database import get_db
from app.core.schemas import AIAssistantRequest, AIAssistantResponse
from app.ledger.models import AccountModel, BalanceProjectionModel, JournalTransactionModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/ai", tags=["AI Assistant"])


@router.post("/ask", response_model=AIAssistantResponse)
async def ask_financial_assistant(
    payload: AIAssistantRequest,
    db: AsyncSession = Depends(get_db),
) -> AIAssistantResponse:
    """
    Answer financial questions grounded strictly in database records.
    Uses parameterized, read-only analytical queries to prevent hallucinations.
    """
    question_lower = payload.question.lower()
    grounded_data: dict[str, object] = {}

    # Tool 1: Account-specific balance inquiry
    if payload.account_id:
        proj_res = await db.execute(
            select(BalanceProjectionModel, AccountModel.currency)
            .join(AccountModel, BalanceProjectionModel.account_id == AccountModel.id)
            .where(BalanceProjectionModel.account_id == payload.account_id)
        )
        row = proj_res.first()
        if row:
            proj, currency = row
            grounded_data["account_id"] = str(payload.account_id)
            grounded_data["balance_minor_units"] = proj.balance_minor_units
            grounded_data["balance_major"] = float(proj.balance_minor_units) / 100.0
            grounded_data["currency"] = currency

            answer = (
                f"According to verified ledger records, Account {payload.account_id} "
                f"currently holds a balance of {grounded_data['balance_major']:,.2f} {currency} "
                f"({proj.balance_minor_units:,} minor units) as of {proj.updated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}."
            )
            return AIAssistantResponse(
                question=payload.question,
                answer=answer,
                grounded_data=grounded_data,
                timestamp=datetime.now(UTC),
            )

    # Tool 2: Global platform transaction volume
    if any(k in question_lower for k in ["total", "volume", "summary", "how many", "overview"]):
        acc_count = (await db.execute(select(func.count(AccountModel.id)))).scalar() or 0
        txn_count = (await db.execute(select(func.count(JournalTransactionModel.id)))).scalar() or 0

        from app.ledger.models import JournalEntryModel

        vol_res = await db.execute(
            select(func.sum(JournalEntryModel.amount_minor_units)).where(
                JournalEntryModel.entry_type == "CREDIT"
            )
        )
        total_minor = vol_res.scalar() or 0
        total_major = float(total_minor) / 100.0

        grounded_data["total_accounts"] = acc_count
        grounded_data["total_transactions"] = txn_count
        grounded_data["total_volume_major"] = total_major
        grounded_data["currency"] = "EUR"

        answer = (
            f"Based on authoritative ledger records: There are {acc_count:,} registered accounts "
            f"and {txn_count:,} posted double-entry journal transactions, representing a total settled "
            f"volume of €{total_major:,.2f}."
        )
        return AIAssistantResponse(
            question=payload.question,
            answer=answer,
            grounded_data=grounded_data,
            timestamp=datetime.now(UTC),
        )

    # Tool 3: Fraud model inquiry
    if any(k in question_lower for k in ["fraud", "model", "risk", "detection", "threshold"]):
        from app.fraud.service import FraudInferenceService

        info = FraudInferenceService.get_info()
        grounded_data["model_metadata"] = info
        answer = (
            f"The fraud detection engine is operating with active model: {info['model_type']}. "
            f"It uses 31 engineered features and an optimal cost-calibrated decision threshold "
            f"of {info['optimal_threshold']:.3f}. Predictions strictly advise analysts and cannot "
            f"alter ledger transactions."
        )
        return AIAssistantResponse(
            question=payload.question,
            answer=answer,
            grounded_data=grounded_data,
            timestamp=datetime.now(UTC),
        )

    # Default: State insufficient context to prevent hallucination
    answer = (
        "I can answer questions regarding account balances, transaction volumes, and fraud model metadata. "
        "The available database evidence was insufficient to verify your query with mathematical certainty."
    )
    return AIAssistantResponse(
        question=payload.question,
        answer=answer,
        grounded_data=grounded_data,
        timestamp=datetime.now(UTC),
    )
