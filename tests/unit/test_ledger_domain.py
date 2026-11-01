"""
Unit tests for core financial domain models and accounting invariants.
"""

from decimal import Decimal
from uuid import uuid4

import pytest

from app.ledger.domain import (
    AccountingInvariantError,
    CurrencyMismatchError,
    DomainEvent,
    EntryType,
    JournalEntry,
    JournalTransaction,
    Money,
)


@pytest.mark.unit
def test_money_representation_and_math() -> None:
    """Verify exact minor unit math and conversions without floating point error."""
    m1 = Money.from_major("10.50", currency="EUR")
    m2 = Money.from_major(4.25, currency="EUR")

    assert m1.minor_units == 1050
    assert m2.minor_units == 425
    assert m1.to_major() == Decimal("10.50")

    # Exact addition & subtraction
    added = m1.add(m2)
    assert added.minor_units == 1475
    assert added.to_major() == Decimal("14.75")

    subtracted = m1.subtract(m2)
    assert subtracted.minor_units == 625
    assert subtracted.to_major() == Decimal("6.25")


@pytest.mark.unit
def test_money_prevents_currency_mismatch() -> None:
    """Operations across different currencies must be rejected."""
    eur = Money(minor_units=1000, currency="EUR")
    usd = Money(minor_units=1000, currency="USD")

    with pytest.raises(CurrencyMismatchError):
        eur.add(usd)

    with pytest.raises(CurrencyMismatchError):
        _ = eur < usd


@pytest.mark.unit
def test_journal_entry_rejects_zero_or_negative() -> None:
    """Journal entries must strictly have positive amounts."""
    acc_id = uuid4()
    with pytest.raises(AccountingInvariantError, match="strictly positive"):
        JournalEntry(
            account_id=acc_id,
            entry_type=EntryType.DEBIT,
            amount=Money(minor_units=0, currency="EUR"),
        )

    with pytest.raises(AccountingInvariantError, match="strictly positive"):
        JournalEntry(
            account_id=acc_id,
            entry_type=EntryType.CREDIT,
            amount=Money(minor_units=-500, currency="EUR"),
        )


@pytest.mark.unit
def test_balanced_journal_transaction_succeeds() -> None:
    """A transaction where Debits == Credits must post successfully."""
    src_account = uuid4()
    dst_account = uuid4()
    amount = Money(minor_units=5000, currency="EUR")  # €50.00

    entries = [
        JournalEntry(account_id=src_account, entry_type=EntryType.CREDIT, amount=amount),
        JournalEntry(account_id=dst_account, entry_type=EntryType.DEBIT, amount=amount),
    ]

    txn = JournalTransaction(
        entries=entries,
        description="Transfer from Alice to Bob",
        idempotency_key="idemp_123",
    )

    assert txn.total_amount == amount
    assert len(txn.entries) == 2


@pytest.mark.unit
def test_unbalanced_journal_transaction_raises_invariant_error() -> None:
    """A transaction where Debits != Credits must be rejected with AccountingInvariantError."""
    src_account = uuid4()
    dst_account = uuid4()

    entries = [
        JournalEntry(account_id=src_account, entry_type=EntryType.CREDIT, amount=Money(5000, "EUR")),
        JournalEntry(account_id=dst_account, entry_type=EntryType.DEBIT, amount=Money(4500, "EUR")),
    ]

    with pytest.raises(AccountingInvariantError, match="Unbalanced journal transaction"):
        JournalTransaction(
            entries=entries,
            description="Unbalanced transfer attempt",
        )


@pytest.mark.unit
def test_journal_transaction_rejects_single_entry() -> None:
    """A transaction with only 1 entry cannot form a double-entry posting."""
    with pytest.raises(AccountingInvariantError, match="at least 2 entries"):
        JournalTransaction(
            entries=[
                JournalEntry(account_id=uuid4(), entry_type=EntryType.DEBIT, amount=Money(100, "EUR"))
            ],
            description="Single entry attempt",
        )


@pytest.mark.unit
def test_domain_event_creation() -> None:
    """Verify domain events record immutable metadata."""
    agg_id = uuid4()
    event = DomainEvent(
        aggregate_id=agg_id,
        aggregate_type="Account",
        event_type="AccountOpened",
        payload={"currency": "EUR", "initial_balance": 0},
        version=1,
    )

    assert event.aggregate_id == agg_id
    assert event.event_type == "AccountOpened"
    assert event.version == 1
