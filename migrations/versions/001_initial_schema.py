"""initial schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-11-01 21:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Accounts
    op.create_table(
        "accounts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_accounts_owner_id"), "accounts", ["owner_id"], unique=False)

    # 2. Journal Transactions
    op.create_table(
        "journal_transactions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("description", sa.String(length=255), nullable=False),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_journal_transactions_idempotency_key"),
        "journal_transactions",
        ["idempotency_key"],
        unique=True,
    )

    # 3. Journal Entries
    op.create_table(
        "journal_entries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("transaction_id", sa.Uuid(), nullable=False),
        sa.Column("account_id", sa.Uuid(), nullable=False),
        sa.Column("entry_type", sa.String(length=6), nullable=False),
        sa.Column("amount_minor_units", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["account_id"],
            ["accounts.id"],
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["journal_transactions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_journal_entries_account_id"),
        "journal_entries",
        ["account_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_journal_entries_transaction_id"),
        "journal_entries",
        ["transaction_id"],
        unique=False,
    )

    # 4. Domain Events
    op.create_table(
        "domain_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("aggregate_id", sa.Uuid(), nullable=False),
        sa.Column("aggregate_type", sa.String(length=50), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column(
            "payload",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_domain_events_aggregate_id"),
        "domain_events",
        ["aggregate_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_domain_events_event_type"),
        "domain_events",
        ["event_type"],
        unique=False,
    )
    op.create_index(
        "ix_domain_events_aggregate_version",
        "domain_events",
        ["aggregate_id", "version"],
        unique=True,
    )

    # 5. Balance Projections
    op.create_table(
        "balance_projections",
        sa.Column("account_id", sa.Uuid(), nullable=False),
        sa.Column("balance_minor_units", sa.BigInteger(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("last_transaction_id", sa.Uuid(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["account_id"],
            ["accounts.id"],
        ),
        sa.PrimaryKeyConstraint("account_id"),
    )

    # 6. Idempotency Records
    op.create_table(
        "idempotency_records",
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("endpoint", sa.String(length=100), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("response_payload", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("idempotency_key"),
    )


def down_override() -> None:
    pass


def downgrade() -> None:
    op.drop_table("idempotency_records")
    op.drop_table("balance_projections")
    op.drop_table("domain_events")
    op.drop_table("journal_entries")
    op.drop_table("journal_transactions")
    op.drop_table("accounts")
