"""
Financial Domain Model & Accounting Invariants for LedgerMind AI.

Enforces:
1. Strict integer minor units for money (no floating-point rounding errors)
2. Double-entry accounting rule (sum(DEBITS) == sum(CREDITS))
3. Immutable domain events and journal transactions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from uuid import UUID, uuid4


class AccountingInvariantError(Exception):
    """Raised when a core double-entry accounting rule is violated."""


class InsufficientFundsError(AccountingInvariantError):
    """Raised when an account cannot satisfy a withdrawal or debit without violating policy."""


class CurrencyMismatchError(AccountingInvariantError):
    """Raised when attempting financial operations across differing currencies."""


class AccountStatusError(AccountingInvariantError):
    """Raised when attempting operations on a frozen or closed account."""


class AccountType(StrEnum):
    ASSET = "ASSET"
    LIABILITY = "LIABILITY"  # Customer deposits are bank liabilities
    EQUITY = "EQUITY"
    REVENUE = "REVENUE"
    EXPENSE = "EXPENSE"


class AccountStatus(StrEnum):
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"
    CLOSED = "CLOSED"


class EntryType(StrEnum):
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"


@dataclass(frozen=True)
class Money:
    """
    Immutable monetary value represented strictly in integer minor units (e.g. cents, paise).
    Floating-point representation is strictly prohibited to avoid IEEE-754 drift.
    """

    minor_units: int
    currency: str = "EUR"

    def __post_init__(self) -> None:
        if not isinstance(self.minor_units, int):
            raise TypeError(f"minor_units must be an integer, got {type(self.minor_units).__name__}")
        if len(self.currency) != 3 or not self.currency.isupper():
            raise ValueError(f"currency must be a 3-letter ISO code, got '{self.currency}'")

    @classmethod
    def from_major(cls, amount: float | str | Decimal, currency: str = "EUR") -> Money:
        """Convert major monetary amount (e.g. 10.50 EUR) to integer minor units (1050)."""
        dec = Decimal(str(amount))
        minor = int((dec * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        return cls(minor_units=minor, currency=currency)

    def to_major(self) -> Decimal:
        """Convert minor units back to Decimal for human-readable display."""
        return Decimal(self.minor_units) / Decimal(100)

    def add(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise CurrencyMismatchError(f"Cannot add {self.currency} and {other.currency}")
        return Money(minor_units=self.minor_units + other.minor_units, currency=self.currency)

    def subtract(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise CurrencyMismatchError(f"Cannot subtract {other.currency} from {self.currency}")
        return Money(minor_units=self.minor_units - other.minor_units, currency=self.currency)

    def __lt__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise CurrencyMismatchError(f"Cannot compare {self.currency} and {other.currency}")
        return self.minor_units < other.minor_units

    def __le__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise CurrencyMismatchError(f"Cannot compare {self.currency} and {other.currency}")
        return self.minor_units <= other.minor_units

    def __gt__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise CurrencyMismatchError(f"Cannot compare {self.currency} and {other.currency}")
        return self.minor_units > other.minor_units


@dataclass(frozen=True)
class JournalEntry:
    """An individual debit or credit entry within a journal transaction."""

    account_id: UUID
    entry_type: EntryType
    amount: Money
    entry_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.amount.minor_units <= 0:
            raise AccountingInvariantError("Journal entry amount must be strictly positive.")


@dataclass(frozen=True)
class JournalTransaction:
    """
    Authoritative double-entry financial transaction.
    Invariant: Sum of all DEBITS must equal Sum of all CREDITS for the transaction to post.
    """

    entries: list[JournalEntry]
    description: str
    idempotency_key: str | None = None
    transaction_id: UUID = field(default_factory=uuid4)
    posted_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if len(self.entries) < 2:
            raise AccountingInvariantError("A journal transaction must have at least 2 entries (one debit, one credit).")

        # Check currency uniformity
        currencies = {e.amount.currency for e in self.entries}
        if len(currencies) > 1:
            raise CurrencyMismatchError("All entries in a single transaction must share the same currency.")

        # Invariant check: total debits == total credits
        total_debits = sum(e.amount.minor_units for e in self.entries if e.entry_type == EntryType.DEBIT)
        total_credits = sum(e.amount.minor_units for e in self.entries if e.entry_type == EntryType.CREDIT)

        if total_debits != total_credits:
            raise AccountingInvariantError(
                f"Unbalanced journal transaction: Total Debits ({total_debits}) != Total Credits ({total_credits})"
            )

    @property
    def total_amount(self) -> Money:
        """The balanced transaction volume."""
        debits = sum(e.amount.minor_units for e in self.entries if e.entry_type == EntryType.DEBIT)
        currency = self.entries[0].amount.currency
        return Money(minor_units=debits, currency=currency)


@dataclass(frozen=True)
class DomainEvent:
    """Immutable domain event capturing a state transition."""

    aggregate_id: UUID
    aggregate_type: str
    event_type: str
    payload: dict[str, object]
    version: int
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
