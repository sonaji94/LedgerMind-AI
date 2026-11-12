"""
Ledger Engine Service for LedgerMind AI.

Manages:
1. Account lifecycle and balance projections
2. Atomic double-entry financial transfers
3. Append-only domain event logging
4. Projection rebuilding and ledger reconciliation
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from app.ledger.domain import (
    AccountStatus,
    AccountStatusError,
    CurrencyMismatchError,
    EntryType,
    InsufficientFundsError,
    JournalEntry,
    JournalTransaction,
    Money,
)
from app.ledger.models import (
    AccountModel,
    BalanceProjectionModel,
    DomainEventModel,
    JournalEntryModel,
    JournalTransactionModel,
)


class LedgerService:
    """Core transactional service for ledger operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_account(
        self,
        owner_id: UUID,
        currency: str = "EUR",
        initial_deposit_minor_units: int = 0,
    ) -> AccountModel:
        """Create a new financial account and its corresponding balance projection."""
        account_id = uuid4()
        account = AccountModel(
            id=account_id,
            owner_id=owner_id,
            currency=currency,
            status=AccountStatus.ACTIVE.value,
            version=1,
        )
        self.session.add(account)

        projection = BalanceProjectionModel(
            account_id=account_id,
            balance_minor_units=initial_deposit_minor_units,
            version=1,
        )
        self.session.add(projection)

        event = DomainEventModel(
            aggregate_id=account_id,
            aggregate_type="Account",
            event_type="AccountCreated",
            payload={
                "owner_id": str(owner_id),
                "currency": currency,
                "initial_deposit": initial_deposit_minor_units,
            },
            version=1,
        )
        self.session.add(event)

        if initial_deposit_minor_units > 0:
            init_txn = JournalTransactionModel(
                id=uuid4(),
                idempotency_key=f"initial-deposit-{account_id}",
                description="Initial Account Deposit",
                posted_at=datetime.now(UTC),
            )
            self.session.add(init_txn)
            entry = JournalEntryModel(
                id=uuid4(),
                transaction_id=init_txn.id,
                account_id=account_id,
                entry_type=EntryType.DEBIT.value,
                amount_minor_units=initial_deposit_minor_units,
            )
            self.session.add(entry)

        await self.session.flush()

        return account

    async def get_balance(self, account_id: UUID) -> Money:
        """Retrieve the materialized balance for an account."""
        query = (
            select(BalanceProjectionModel, AccountModel.currency)
            .join(AccountModel, BalanceProjectionModel.account_id == AccountModel.id)
            .where(BalanceProjectionModel.account_id == account_id)
        )
        result = await self.session.execute(query)
        row = result.first()
        if not row:
            raise AccountStatusError(f"Account {account_id} not found.")

        proj, currency = row
        return Money(minor_units=proj.balance_minor_units, currency=currency)

    async def execute_transfer(
        self,
        source_account_id: UUID,
        destination_account_id: UUID,
        amount: Money,
        idempotency_key: str | None = None,
        description: str = "Internal Transfer",
    ) -> JournalTransaction:
        """
        Execute an atomic, balanced financial transfer between two accounts.

        Steps:
        1. Verify source and destination accounts exist, are ACTIVE, and share the currency.
        2. Verify source account has sufficient funds.
        3. Create balanced JournalTransaction domain entity.
        4. Persist journal transaction and journal entries.
        5. Update source and destination balance projections.
        6. Append TransferPosted domain event.
        """
        if source_account_id == destination_account_id:
            raise AccountStatusError("Cannot transfer funds to the same account.")

        # 1. Fetch both accounts & projections
        src_res = await self.session.execute(
            select(AccountModel, BalanceProjectionModel)
            .join(BalanceProjectionModel, AccountModel.id == BalanceProjectionModel.account_id)
            .where(AccountModel.id == source_account_id)
        )
        src_row = src_res.first()
        if not src_row:
            raise AccountStatusError(f"Source account {source_account_id} not found.")
        src_acc, src_proj = src_row

        dst_res = await self.session.execute(
            select(AccountModel, BalanceProjectionModel)
            .join(BalanceProjectionModel, AccountModel.id == BalanceProjectionModel.account_id)
            .where(AccountModel.id == destination_account_id)
        )
        dst_row = dst_res.first()
        if not dst_row:
            raise AccountStatusError(f"Destination account {destination_account_id} not found.")
        dst_acc, dst_proj = dst_row

        # 2. Status & currency checks
        if src_acc.status != AccountStatus.ACTIVE.value:
            raise AccountStatusError(f"Source account is {src_acc.status}.")
        if dst_acc.status != AccountStatus.ACTIVE.value:
            raise AccountStatusError(f"Destination account is {dst_acc.status}.")

        if src_acc.currency != amount.currency or dst_acc.currency != amount.currency:
            raise CurrencyMismatchError("Transaction currency must match account currencies.")

        # 3. Solvency check (no overdraft)
        if src_proj.balance_minor_units < amount.minor_units:
            raise InsufficientFundsError(
                f"Insufficient balance. Required: {amount.minor_units}, Available: {src_proj.balance_minor_units}"
            )

        # 4. Construct domain journal transaction
        domain_txn = JournalTransaction(
            entries=[
                JournalEntry(account_id=source_account_id, entry_type=EntryType.CREDIT, amount=amount),
                JournalEntry(account_id=destination_account_id, entry_type=EntryType.DEBIT, amount=amount),
            ],
            description=description,
            idempotency_key=idempotency_key,
        )

        # 5. Persist to database
        db_txn = JournalTransactionModel(
            id=domain_txn.transaction_id,
            idempotency_key=idempotency_key,
            description=description,
            posted_at=domain_txn.posted_at,
        )
        self.session.add(db_txn)

        entry_src = JournalEntryModel(
            id=domain_txn.entries[0].entry_id,
            transaction_id=domain_txn.transaction_id,
            account_id=source_account_id,
            entry_type=EntryType.CREDIT.value,
            amount_minor_units=amount.minor_units,
        )
        entry_dst = JournalEntryModel(
            id=domain_txn.entries[1].entry_id,
            transaction_id=domain_txn.transaction_id,
            account_id=destination_account_id,
            entry_type=EntryType.DEBIT.value,
            amount_minor_units=amount.minor_units,
        )
        self.session.add_all([entry_src, entry_dst])

        # 6. Update projections
        src_proj.balance_minor_units -= amount.minor_units
        src_proj.version += 1
        src_proj.last_transaction_id = domain_txn.transaction_id
        src_proj.updated_at = datetime.now(UTC)

        dst_proj.balance_minor_units += amount.minor_units
        dst_proj.version += 1
        dst_proj.last_transaction_id = domain_txn.transaction_id
        dst_proj.updated_at = datetime.now(UTC)

        # 7. Record Domain Event
        event = DomainEventModel(
            aggregate_id=domain_txn.transaction_id,
            aggregate_type="Transfer",
            event_type="TransferPosted",
            payload={
                "source_account_id": str(source_account_id),
                "destination_account_id": str(destination_account_id),
                "amount_minor_units": amount.minor_units,
                "currency": amount.currency,
                "idempotency_key": idempotency_key,
            },
            version=1,
        )
        self.session.add(event)
        await self.session.flush()

        return domain_txn

    async def rebuild_projection_from_journal(self, account_id: UUID) -> int:
        """
        Reconstruct account balance directly from the immutable journal entry history.
        Demonstrates that balance projections are fully derived and rebuildable.
        """
        query = select(JournalEntryModel).where(JournalEntryModel.account_id == account_id)
        result = await self.session.execute(query)
        entries = result.scalars().all()

        recalculated_balance = 0
        for entry in entries:
            if entry.entry_type == EntryType.DEBIT.value:
                recalculated_balance += entry.amount_minor_units
            elif entry.entry_type == EntryType.CREDIT.value:
                recalculated_balance -= entry.amount_minor_units

        # Update projection
        proj = await self.session.get(BalanceProjectionModel, account_id)
        if proj:
            proj.balance_minor_units = recalculated_balance
            proj.updated_at = datetime.now(UTC)
            await self.session.flush()

        return recalculated_balance
