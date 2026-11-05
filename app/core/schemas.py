"""
Pydantic v2 Validation Schemas for LedgerMind AI REST APIs.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ── Account Schemas ───────────────────────────────────────────────
class AccountCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_id: UUID
    currency: str = Field(default="EUR", min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")
    initial_deposit_minor_units: int = Field(default=0, ge=0)


class AccountResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    owner_id: UUID
    currency: str
    status: str
    balance_minor_units: int
    balance_major: float
    created_at: datetime


# ── Journal Entry / Transaction History Schemas ───────────────────
class JournalEntryItem(BaseModel):
    """A single debit or credit leg of a journal transaction."""

    id: UUID
    transaction_id: UUID
    entry_type: str          # "DEBIT" or "CREDIT"
    amount_minor_units: int
    amount_major: float
    posted_at: datetime


class TransactionHistoryResponse(BaseModel):
    """Paginated list of journal entries for a given account."""

    account_id: UUID
    page: int
    page_size: int
    total: int
    items: list[JournalEntryItem]


# ── Transfer Schemas ──────────────────────────────────────────────
class TransferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_account_id: UUID
    destination_account_id: UUID
    amount_minor_units: int = Field(gt=0, description="Amount in minor units (e.g. 1050 = €10.50)")
    currency: str = Field(default="EUR", min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")
    description: str = Field(default="Transfer", max_length=255)


class TransferResponse(BaseModel):
    transaction_id: UUID
    source_account_id: UUID
    destination_account_id: UUID
    amount_minor_units: int
    amount_major: float
    currency: str
    description: str
    posted_at: datetime
    idempotency_key: str | None = None


# ── Fraud ML Schemas ──────────────────────────────────────────────
class TransactionFeatures(BaseModel):
    model_config = ConfigDict(extra="allow")

    Time: float = Field(ge=0.0, description="Elapsed seconds from epoch or day start")
    Amount: float = Field(ge=0.0, description="Transaction amount in currency units")
    # PCA features V1..V28 (optional in request, default to 0.0 if not provided)


class FraudPredictionRequest(BaseModel):
    transaction_id: str | None = None
    features: dict[str, float]


class FraudPredictionResponse(BaseModel):
    transaction_id: str | None
    risk_score: float = Field(ge=0.0, le=1.0)
    is_suspicious: bool
    threshold_applied: float
    model_type: str
    latency_ms: float


class ModelInfoResponse(BaseModel):
    model_type: str
    optimal_threshold: float
    features_count: int
    preprocessor_type: str
    status: str


# ── Analytics Schemas ─────────────────────────────────────────────
class AnalyticsSummaryResponse(BaseModel):
    total_accounts: int
    total_transactions: int
    total_volume_minor_units: int
    total_volume_major: float
    average_transaction_major: float
    currency: str


# ── AI Assistant Schemas ──────────────────────────────────────────
class AIAssistantRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    account_id: UUID | None = None


class AIAssistantResponse(BaseModel):
    question: str
    answer: str
    grounded_data: dict[str, object]
    timestamp: datetime
