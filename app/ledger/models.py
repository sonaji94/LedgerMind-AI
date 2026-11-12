"""
SQLAlchemy ORM Models for LedgerMind AI Financial Engine.

Tables:
1. accounts: Account identity, currency, status, version with check constraints.
2. journal_transactions: Authoritative double-entry transaction record.
3. journal_entries: Individual debit/credit legs strictly stored as BIGINT minor units.
4. domain_events: Append-only event store for state reconstruction.
5. balance_projections: Rebuildable read-optimized account balances with solvency constraint.
6. idempotency_records: Deduplication store for safe HTTP retries.
7. audit_logs: Immutable compliance record of operator actions (freeze, 2FA, SAR, dismissals).
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.core.database import Base


class AccountModel(Base):
    __tablename__ = "accounts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(nullable=False, index=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="EUR")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="ACTIVE")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        CheckConstraint("status IN ('ACTIVE', 'FROZEN', 'CLOSED', 'SUSPENDED')", name="ck_account_status"),
    )

    # Relationships
    entries: Mapped[list[JournalEntryModel]] = relationship(
        "JournalEntryModel", back_populates="account"
    )
    projection: Mapped[BalanceProjectionModel | None] = relationship(
        "BalanceProjectionModel", back_populates="account", uselist=False
    )


class JournalTransactionModel(Base):
    __tablename__ = "journal_transactions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    idempotency_key: Mapped[str | None] = mapped_column(
        String(255), unique=True, nullable=True, index=True
    )
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    posted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    entries: Mapped[list[JournalEntryModel]] = relationship(
        "JournalEntryModel", back_populates="transaction", cascade="all, delete-orphan"
    )


class JournalEntryModel(Base):
    __tablename__ = "journal_entries"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    transaction_id: Mapped[UUID] = mapped_column(
        ForeignKey("journal_transactions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    account_id: Mapped[UUID] = mapped_column(
        ForeignKey("accounts.id"), nullable=False, index=True
    )
    entry_type: Mapped[str] = mapped_column(String(6), nullable=False)  # DEBIT or CREDIT
    amount_minor_units: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        CheckConstraint("amount_minor_units > 0", name="ck_journal_entry_positive_amount"),
        CheckConstraint("entry_type IN ('DEBIT', 'CREDIT')", name="ck_journal_entry_type"),
    )

    transaction: Mapped[JournalTransactionModel] = relationship(
        "JournalTransactionModel", back_populates="entries"
    )
    account: Mapped[AccountModel] = relationship(
        "AccountModel", back_populates="entries"
    )


class DomainEventModel(Base):
    __tablename__ = "domain_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    aggregate_id: Mapped[UUID] = mapped_column(nullable=False, index=True)
    aggregate_type: Mapped[str] = mapped_column(String(50), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    payload: Mapped[dict[str, object]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("ix_domain_events_aggregate_version", "aggregate_id", "version", unique=True),
    )


class BalanceProjectionModel(Base):
    __tablename__ = "balance_projections"

    account_id: Mapped[UUID] = mapped_column(
        ForeignKey("accounts.id"), primary_key=True
    )
    balance_minor_units: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_transaction_id: Mapped[UUID | None] = mapped_column(nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        CheckConstraint("balance_minor_units >= 0", name="ck_balance_non_negative"),
    )

    account: Mapped[AccountModel] = relationship(
        "AccountModel", back_populates="projection"
    )


class IdempotencyRecordModel(Base):
    __tablename__ = "idempotency_records"

    idempotency_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(100), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    response_payload: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class AuditLogModel(Base):
    """
    Immutable compliance audit trail recording every operator action
    (e.g., account freezes, 2FA step-up triggers, SAR reports, alert dismissals).
    """
    __tablename__ = "audit_logs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    operator_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    operator_role: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    target_type: Mapped[str] = mapped_column(String(50), nullable=False)  # ACCOUNT, TRANSACTION, INCIDENT
    target_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    ip_address: Mapped[str | None] = mapped_column(String(50), nullable=True)
    justification_reason: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, object] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )
