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

# ── Deterministic Institutional System Accounts ───────────────────────
GATEWAY_CLEARING_ACCOUNT_ID = UUID("00000000-0000-0000-0002-000000000001")
MERCHANT_ESCROW_ACCOUNT_ID = UUID("00000000-0000-0000-0002-000000000002")
GATEWAY_FEE_ACCOUNT_ID = UUID("00000000-0000-0000-0002-000000000003")
GST_TAX_ACCOUNT_ID = UUID("00000000-0000-0000-0002-000000000004")
PENDING_PAYOUT_HOLD_ID = UUID("00000000-0000-0000-0002-000000000005")
FRAUD_LOSS_RESERVE_ID = UUID("00000000-0000-0000-0002-000000000006")


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
            genesis_id = UUID("00000000-0000-0000-0000-000000000000")
            genesis = await self.session.get(AccountModel, genesis_id)
            if not genesis:
                genesis = AccountModel(
                    id=genesis_id,
                    owner_id=genesis_id,
                    currency=currency,
                    status=AccountStatus.ACTIVE.value,
                    version=1,
                )
                self.session.add(genesis)
                genesis_proj = BalanceProjectionModel(
                    account_id=genesis_id,
                    balance_minor_units=0,
                    version=1,
                )
                self.session.add(genesis_proj)

            init_txn = JournalTransactionModel(
                id=uuid4(),
                idempotency_key=f"initial-deposit-{account_id}",
                description="Capital Inflow / Initial Deposit",
                posted_at=datetime.now(UTC),
            )
            self.session.add(init_txn)

            entry_dst = JournalEntryModel(
                id=uuid4(),
                transaction_id=init_txn.id,
                account_id=account_id,
                entry_type=EntryType.DEBIT.value,
                amount_minor_units=initial_deposit_minor_units,
            )
            entry_src = JournalEntryModel(
                id=uuid4(),
                transaction_id=init_txn.id,
                account_id=genesis_id,
                entry_type=EntryType.CREDIT.value,
                amount_minor_units=initial_deposit_minor_units,
            )
            self.session.add_all([entry_dst, entry_src])

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

        # 1. Fetch both accounts & projections with row-level locking (SELECT FOR UPDATE)
        src_res = await self.session.execute(
            select(AccountModel, BalanceProjectionModel)
            .join(BalanceProjectionModel, AccountModel.id == BalanceProjectionModel.account_id)
            .where(AccountModel.id == source_account_id)
            .with_for_update()
        )
        src_row = src_res.first()
        if not src_row:
            raise AccountStatusError(f"Source account {source_account_id} not found.")
        src_acc, src_proj = src_row

        dst_res = await self.session.execute(
            select(AccountModel, BalanceProjectionModel)
            .join(BalanceProjectionModel, AccountModel.id == BalanceProjectionModel.account_id)
            .where(AccountModel.id == destination_account_id)
            .with_for_update()
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

    async def freeze_account(
        self,
        account_id: UUID,
        reason: str,
        operator_id: str = "system",
    ) -> None:
        """
        Freeze an account by emitting an append-only domain event and transitioning status.
        Never deletes or updates financial records.
        """
        acc = await self.session.get(AccountModel, account_id)
        if not acc:
            raise AccountStatusError(f"Account {account_id} not found.")

        acc.status = AccountStatus.FROZEN.value
        acc.version += 1

        event = DomainEventModel(
            aggregate_id=account_id,
            aggregate_type="Account",
            event_type="AccountFrozen",
            payload={
                "account_id": str(account_id),
                "reason": reason,
                "operator_id": operator_id,
                "status": "FROZEN",
            },
            version=acc.version,
        )
        self.session.add(event)
        await self.session.flush()

    async def execute_reversal(
        self,
        original_transaction_id: UUID,
        reason: str,
        idempotency_key: str | None = None,
    ) -> JournalTransaction:
        """
        Execute an immutable compensating transaction that reverses an existing transfer.
        Never updates or deletes historical records; creates an inverse dual-legged journal transaction.
        """
        query = select(JournalEntryModel).where(JournalEntryModel.transaction_id == original_transaction_id)
        result = await self.session.execute(query)
        entries = result.scalars().all()

        if len(entries) < 2:
            raise AccountStatusError(f"Original transaction {original_transaction_id} not found or incomplete.")

        # Identify original credit and debit legs
        orig_credit = next((e for e in entries if e.entry_type == EntryType.CREDIT.value), None)
        orig_debit = next((e for e in entries if e.entry_type == EntryType.DEBIT.value), None)

        if not orig_credit or not orig_debit:
            raise AccountStatusError("Invalid original transaction structure for reversal.")

        # Retrieve currency from account
        src_acc = await self.session.get(AccountModel, orig_debit.account_id)
        currency = src_acc.currency if src_acc else "EUR"

        # Inverse transfer: original debit account now pays back original credit account
        return await self.execute_transfer(
            source_account_id=orig_debit.account_id,
            destination_account_id=orig_credit.account_id,
            amount=Money(minor_units=orig_credit.amount_minor_units, currency=currency),
            idempotency_key=idempotency_key or f"reversal-{original_transaction_id}",
            description=f"Compensating Reversal for TX {original_transaction_id}: {reason}",
        )

    async def ensure_system_account(
        self,
        account_id: UUID,
        name: str,
        currency: str = "INR",
        initial_balance_minor: int = 10_000_000_00,  # 10M INR buffer for clearing/escrow
    ) -> AccountModel:
        """Retrieve or create an institutional system account with initial solvency."""
        acc = await self.session.get(AccountModel, account_id)
        if acc:
            return acc

        acc = AccountModel(
            id=account_id,
            owner_id=UUID("00000000-0000-0000-0000-000000000000"),
            currency=currency,
            status=AccountStatus.ACTIVE.value,
            version=1,
        )
        self.session.add(acc)

        proj = BalanceProjectionModel(
            account_id=account_id,
            balance_minor_units=initial_balance_minor,
            version=1,
        )
        self.session.add(proj)
        await self.session.flush()
        return acc

    async def reconcile_captured_payment(
        self,
        payment_id: str,
        amount_minor: int,
        fee_minor: int,
        tax_minor: int,
        currency: str = "INR",
        merchant_account_id: UUID | None = None,
        idempotency_key: str | None = None,
    ) -> UUID:
        """
        Post balanced double-entry journal transaction for captured payment:
        - Debit Gateway_Clearing_Account: gross captured amount
        - Credit Merchant_Escrow_Liability: net amount (amount - fee - tax)
        - Credit Gateway_Fee_Expense: fee
        - Credit GST_Tax_Payable: tax
        Invariant: SUM(debits) == SUM(credits) exactly in minor units (paise).
        """
        net_merchant = amount_minor - fee_minor - tax_minor
        if net_merchant < 0:
            raise AccountStatusError(f"Fee and tax ({fee_minor + tax_minor}) exceed gross amount ({amount_minor})")

        # 1. Ensure system accounts exist
        gateway_acc = await self.ensure_system_account(GATEWAY_CLEARING_ACCOUNT_ID, "Gateway Clearing", currency)
        merchant_target = merchant_account_id or MERCHANT_ESCROW_ACCOUNT_ID
        await self.ensure_system_account(merchant_target, "Merchant Escrow Liability", currency)
        await self.ensure_system_account(GATEWAY_FEE_ACCOUNT_ID, "Gateway Fee Expense", currency)
        await self.ensure_system_account(GST_TAX_ACCOUNT_ID, "GST Tax Payable", currency)

        # 2. Lock accounts with SELECT FOR UPDATE
        account_ids = [gateway_acc.id, merchant_target, GATEWAY_FEE_ACCOUNT_ID, GST_TAX_ACCOUNT_ID]
        res = await self.session.execute(
            select(BalanceProjectionModel)
            .where(BalanceProjectionModel.account_id.in_(account_ids))
            .with_for_update()
        )
        projections = {p.account_id: p for p in res.scalars().all()}

        # 3. Create Balanced Journal Transaction
        txn_id = uuid4()
        now = datetime.now(UTC)
        effective_idemp = idempotency_key or f"reconcile-pay-{payment_id}"

        db_txn = JournalTransactionModel(
            id=txn_id,
            idempotency_key=effective_idemp,
            description=f"Razorpay Capture Reconciliation {payment_id}",
            posted_at=now,
        )
        self.session.add(db_txn)

        # Legs:
        # 1 Debit: Gross amount to Clearing
        entry_clearing = JournalEntryModel(
            id=uuid4(),
            transaction_id=txn_id,
            account_id=gateway_acc.id,
            entry_type=EntryType.DEBIT.value,
            amount_minor_units=amount_minor,
        )
        # 3 Credits: Net to Merchant, Fee to Expense, Tax to GST
        entry_net = JournalEntryModel(
            id=uuid4(),
            transaction_id=txn_id,
            account_id=merchant_target,
            entry_type=EntryType.CREDIT.value,
            amount_minor_units=net_merchant,
        )
        entry_fee = JournalEntryModel(
            id=uuid4(),
            transaction_id=txn_id,
            account_id=GATEWAY_FEE_ACCOUNT_ID,
            entry_type=EntryType.CREDIT.value,
            amount_minor_units=fee_minor if fee_minor > 0 else 1,  # check constraint positive
        )
        entry_tax = JournalEntryModel(
            id=uuid4(),
            transaction_id=txn_id,
            account_id=GST_TAX_ACCOUNT_ID,
            entry_type=EntryType.CREDIT.value,
            amount_minor_units=tax_minor if tax_minor > 0 else 1,
        )

        entries_to_add = [entry_clearing, entry_net]
        if fee_minor > 0:
            entries_to_add.append(entry_fee)
        if tax_minor > 0:
            entries_to_add.append(entry_tax)

        self.session.add_all(entries_to_add)

        # 4. Update Balance Projections
        if gateway_acc.id in projections:
            projections[gateway_acc.id].balance_minor_units += amount_minor
            projections[gateway_acc.id].last_transaction_id = txn_id
            projections[gateway_acc.id].updated_at = now
        if merchant_target in projections:
            projections[merchant_target].balance_minor_units += net_merchant
            projections[merchant_target].last_transaction_id = txn_id
            projections[merchant_target].updated_at = now
        if fee_minor > 0 and GATEWAY_FEE_ACCOUNT_ID in projections:
            projections[GATEWAY_FEE_ACCOUNT_ID].balance_minor_units += fee_minor
            projections[GATEWAY_FEE_ACCOUNT_ID].last_transaction_id = txn_id
            projections[GATEWAY_FEE_ACCOUNT_ID].updated_at = now
        if tax_minor > 0 and GST_TAX_ACCOUNT_ID in projections:
            projections[GST_TAX_ACCOUNT_ID].balance_minor_units += tax_minor
            projections[GST_TAX_ACCOUNT_ID].last_transaction_id = txn_id
            projections[GST_TAX_ACCOUNT_ID].updated_at = now

        # 5. Emit Domain Event
        event = DomainEventModel(
            aggregate_id=txn_id,
            aggregate_type="PaymentCaptureReconciliation",
            event_type="PaymentCapturedReconciled",
            payload={
                "payment_id": payment_id,
                "amount_minor": amount_minor,
                "fee_minor": fee_minor,
                "tax_minor": tax_minor,
                "net_merchant": net_merchant,
                "currency": currency,
            },
            version=1,
        )
        self.session.add(event)
        await self.session.flush()
        return txn_id

    async def reconcile_payout_hold(
        self,
        payout_id: str,
        amount_minor: int,
        currency: str = "INR",
        merchant_account_id: UUID | None = None,
        idempotency_key: str | None = None,
    ) -> UUID:
        """
        On payout.initiated: move funds from Merchant Escrow to Pending_Payout_Hold.
        Debit Merchant_Escrow_Liability, Credit Pending_Payout_Hold.
        """
        merchant_src = merchant_account_id or MERCHANT_ESCROW_ACCOUNT_ID
        await self.ensure_system_account(merchant_src, "Merchant Escrow Liability", currency)
        await self.ensure_system_account(PENDING_PAYOUT_HOLD_ID, "Pending Payout Hold", currency)

        txn = await self.execute_transfer(
            source_account_id=merchant_src,
            destination_account_id=PENDING_PAYOUT_HOLD_ID,
            amount=Money(minor_units=amount_minor, currency=currency),
            idempotency_key=idempotency_key or f"payout-hold-{payout_id}",
            description=f"Ledger Hold for Initiated Payout {payout_id}",
        )
        return txn.transaction_id

    async def reconcile_payout_reversed(
        self,
        payout_id: str,
        amount_minor: int,
        currency: str = "INR",
        merchant_account_id: UUID | None = None,
        idempotency_key: str | None = None,
    ) -> UUID:
        """
        On payout.reversed: release hold and return funds back to Merchant Escrow.
        Compensating entry: Debit Pending_Payout_Hold, Credit Merchant_Escrow_Liability.
        """
        merchant_dst = merchant_account_id or MERCHANT_ESCROW_ACCOUNT_ID
        await self.ensure_system_account(PENDING_PAYOUT_HOLD_ID, "Pending Payout Hold", currency)
        await self.ensure_system_account(merchant_dst, "Merchant Escrow Liability", currency)

        txn = await self.execute_transfer(
            source_account_id=PENDING_PAYOUT_HOLD_ID,
            destination_account_id=merchant_dst,
            amount=Money(minor_units=amount_minor, currency=currency),
            idempotency_key=idempotency_key or f"payout-reverse-{payout_id}",
            description=f"Compensating Release of Reversed Payout {payout_id}",
        )
        return txn.transaction_id

    async def reconcile_dispute_created(
        self,
        dispute_id: str,
        payment_id: str,
        amount_minor: int,
        currency: str = "INR",
        idempotency_key: str | None = None,
    ) -> UUID:
        """
        On payment.dispute.created: move contested amount to Fraud_Loss_Reserve.
        Debit Fraud_Loss_Reserve, Credit Gateway_Clearing_Account.
        """
        await self.ensure_system_account(FRAUD_LOSS_RESERVE_ID, "Fraud Loss Reserve", currency)
        await self.ensure_system_account(GATEWAY_CLEARING_ACCOUNT_ID, "Gateway Clearing", currency)

        txn = await self.execute_transfer(
            source_account_id=FRAUD_LOSS_RESERVE_ID,
            destination_account_id=GATEWAY_CLEARING_ACCOUNT_ID,
            amount=Money(minor_units=amount_minor, currency=currency),
            idempotency_key=idempotency_key or f"dispute-reserve-{dispute_id}",
            description=f"Clawback Reserve for Payment Dispute {dispute_id} on {payment_id}",
        )
        return txn.transaction_id
