"""
Integration tests for LedgerService: double-entry transfers, invariants, and projection rebuilding.
"""

from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base
from app.ledger.domain import (
    AccountStatusError,
    CurrencyMismatchError,
    InsufficientFundsError,
    Money,
)
from app.ledger.service import LedgerService


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """In-memory async database session fixture for integration tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_maker() as session:
        yield session

    await engine.dispose()


@pytest.mark.integration
async def test_create_account_and_balance(db_session: AsyncSession) -> None:
    """Verify account initialization and initial projection."""
    service = LedgerService(db_session)
    owner = uuid4()
    account = await service.create_account(owner_id=owner, currency="EUR", initial_deposit_minor_units=5000)

    balance = await service.get_balance(account.id)
    assert balance.minor_units == 5000
    assert balance.currency == "EUR"


@pytest.mark.integration
async def test_atomic_transfer_updates_projections_and_posts_journal(db_session: AsyncSession) -> None:
    """A valid transfer must debit recipient, credit sender, and update projections."""
    service = LedgerService(db_session)

    # Alice has €100.00 (10000 minor units)
    alice = await service.create_account(owner_id=uuid4(), currency="EUR", initial_deposit_minor_units=10000)
    # Bob has €20.00 (2000 minor units)
    bob = await service.create_account(owner_id=uuid4(), currency="EUR", initial_deposit_minor_units=2000)

    # Transfer €35.50 (3550 minor units) from Alice to Bob
    amount = Money.from_major("35.50", currency="EUR")
    txn = await service.execute_transfer(
        source_account_id=alice.id,
        destination_account_id=bob.id,
        amount=amount,
        idempotency_key="transfer_abc_1",
        description="Payment for dinner",
    )

    assert txn.total_amount == amount

    # Check updated projections
    alice_bal = await service.get_balance(alice.id)
    bob_bal = await service.get_balance(bob.id)

    assert alice_bal.minor_units == 6450  # 10000 - 3550
    assert bob_bal.minor_units == 5550    # 2000 + 3550


@pytest.mark.integration
async def test_transfer_fails_on_insufficient_funds(db_session: AsyncSession) -> None:
    """Transfer exceeding sender's balance must raise InsufficientFundsError."""
    service = LedgerService(db_session)

    alice = await service.create_account(owner_id=uuid4(), currency="EUR", initial_deposit_minor_units=1000)
    bob = await service.create_account(owner_id=uuid4(), currency="EUR", initial_deposit_minor_units=0)

    with pytest.raises(InsufficientFundsError):
        await service.execute_transfer(
            source_account_id=alice.id,
            destination_account_id=bob.id,
            amount=Money.from_major("50.00", currency="EUR"),  # €50 > €10
        )

    # Balances must remain unchanged
    alice_bal = await service.get_balance(alice.id)
    assert alice_bal.minor_units == 1000


@pytest.mark.integration
async def test_transfer_fails_on_currency_mismatch(db_session: AsyncSession) -> None:
    """Transfer across differing currencies must be rejected."""
    service = LedgerService(db_session)

    eur_acc = await service.create_account(owner_id=uuid4(), currency="EUR", initial_deposit_minor_units=5000)
    usd_acc = await service.create_account(owner_id=uuid4(), currency="USD", initial_deposit_minor_units=5000)

    with pytest.raises(CurrencyMismatchError):
        await service.execute_transfer(
            source_account_id=eur_acc.id,
            destination_account_id=usd_acc.id,
            amount=Money.from_major("10.00", currency="EUR"),
        )


@pytest.mark.integration
async def test_transfer_fails_to_same_account(db_session: AsyncSession) -> None:
    """Cannot transfer money to oneself."""
    service = LedgerService(db_session)
    acc = await service.create_account(owner_id=uuid4(), currency="EUR", initial_deposit_minor_units=5000)

    with pytest.raises(AccountStatusError, match="same account"):
        await service.execute_transfer(
            source_account_id=acc.id,
            destination_account_id=acc.id,
            amount=Money(100, "EUR"),
        )


@pytest.mark.integration
async def test_rebuild_projection_from_journal_entries(db_session: AsyncSession) -> None:
    """Verify that balance projections can be deterministically reconstructed from journal history."""
    service = LedgerService(db_session)

    acc1 = await service.create_account(owner_id=uuid4(), currency="EUR", initial_deposit_minor_units=10000)
    acc2 = await service.create_account(owner_id=uuid4(), currency="EUR", initial_deposit_minor_units=0)

    # Perform two transfers
    await service.execute_transfer(acc1.id, acc2.id, Money(2000, "EUR"))
    await service.execute_transfer(acc1.id, acc2.id, Money(1500, "EUR"))

    # acc2 received 2000 + 1500 = 3500 debits
    rebuilt_acc2 = await service.rebuild_projection_from_journal(acc2.id)
    assert rebuilt_acc2 == 3500

    bal_acc2 = await service.get_balance(acc2.id)
    assert bal_acc2.minor_units == 3500
