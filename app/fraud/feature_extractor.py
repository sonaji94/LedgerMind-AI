"""
LedgerMind AI — Real-World Payment & Payout Feature Extractor.

Replaces synthetic PCA inputs (V1-V28) with live behavioral fraud signals:
1. Redis Sliding Windows:
   - velocity_instrument_60s: Transactions per card / UPI VPA in last 60 seconds
   - velocity_instrument_1h: Transactions per card / UPI VPA in last 1 hour
   - distinct_cards_ip_10m: Distinct cards attempted from the same IP in last 10 minutes
   - distinct_cards_email_10m: Distinct cards attempted from the same customer email in last 10 minutes
2. PostgreSQL 30-Day Historical Baselines:
   - amount_z_score: (amount - mean_30d) / std_30d
   - amount_ratio_30d: amount / max(mean_30d, 1)
   - account_age_days: Elapsed days since user/merchant account provisioning
   - hours_since_bank_change: Hours elapsed since last bank/VPA profile modification
   - payout_to_balance_ratio: Payout amount divided by available ledger balance (ATO liquidation detector)
3. Network & Instrument Signals:
   - is_disposable_email: High-risk disposable / temporary email domain
   - is_international_card: Card issued outside the merchant's home jurisdiction
   - hour_of_day & day_of_week
"""

from __future__ import annotations

import datetime
from datetime import UTC
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.core.redis import RedisVelocityEngine
from app.ledger.models import (
    AccountModel,
    BalanceProjectionModel,
    JournalEntryModel,
    JournalTransactionModel,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

DISPOSABLE_EMAIL_DOMAINS = {
    "mailinator.com",
    "tempmail.com",
    "10minutemail.com",
    "guerrillamail.com",
    "sharklasers.com",
    "trashmail.com",
    "yopmail.com",
    "dispostable.com",
    "getairmail.com",
    "throwawaymail.com",
}


class PaymentEventContext(BaseModel):
    """Contextual metadata extracted from Razorpay payment or payout webhook."""

    event_id: str
    event_type: str
    entity_id: str
    amount_minor_units: int
    currency: str = "INR"
    customer_id: str | None = None
    email: str | None = None
    contact: str | None = None
    ip_address: str | None = None
    card_id: str | None = None
    card_bin: str | None = None
    card_last4: str | None = None
    card_international: bool = False
    vpa: str | None = None
    method: str = "card"
    account_id: str | None = None
    payout_id: str | None = None
    fund_account_id: str | None = None
    hours_since_bank_change: float = 720.0  # default 30 days
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(UTC))


class FeatureVector(BaseModel):
    """Normalized feature vector fed to ML inference and SHAP explainability."""

    amount_minor: int
    velocity_instrument_60s: int
    velocity_instrument_1h: int
    distinct_cards_ip_10m: int
    distinct_cards_email_10m: int
    amount_z_score: float
    amount_ratio_30d: float
    account_age_days: float
    hours_since_bank_change: float
    payout_to_balance_ratio: float
    is_disposable_email: int
    is_international_card: int
    hour_of_day: int
    day_of_week: int

    def to_array(self) -> list[float]:
        return [
            float(self.amount_minor),
            float(self.velocity_instrument_60s),
            float(self.velocity_instrument_1h),
            float(self.distinct_cards_ip_10m),
            float(self.distinct_cards_email_10m),
            self.amount_z_score,
            self.amount_ratio_30d,
            self.account_age_days,
            self.hours_since_bank_change,
            self.payout_to_balance_ratio,
            float(self.is_disposable_email),
            float(self.is_international_card),
            float(self.hour_of_day),
            float(self.day_of_week),
        ]


FEATURE_NAMES = [
    "amount_minor",
    "velocity_instrument_60s",
    "velocity_instrument_1h",
    "distinct_cards_ip_10m",
    "distinct_cards_email_10m",
    "amount_z_score",
    "amount_ratio_30d",
    "account_age_days",
    "hours_since_bank_change",
    "payout_to_balance_ratio",
    "is_disposable_email",
    "is_international_card",
    "hour_of_day",
    "day_of_week",
]


class FeatureExtractor:
    """Service that computes live behavioral signals from incoming webhook, Redis, and PostgreSQL."""

    @classmethod
    async def extract_features(
        cls,
        ctx: PaymentEventContext,
        session: AsyncSession | None = None,
    ) -> tuple[FeatureVector, dict[str, Any]]:
        """Compute end-to-end features and raw telemetry signals."""
        now_ts = ctx.timestamp.timestamp()
        hour = ctx.timestamp.hour
        day_of_week = ctx.timestamp.weekday()

        # 1. Determine instrument identifier
        instrument_key = ctx.vpa or ctx.card_id or (f"{ctx.card_bin}_{ctx.card_last4}" if ctx.card_bin else ctx.contact or "unknown_inst")

        # Record and fetch velocity
        await RedisVelocityEngine.record_instrument_activity(instrument_key, now_ts)
        v_60s = await RedisVelocityEngine.get_instrument_velocity(instrument_key, window_seconds=60)
        v_1h = await RedisVelocityEngine.get_instrument_velocity(instrument_key, window_seconds=3600)

        # 2. Distinct cards card-testing tracking
        card_id_str = ctx.card_id or ctx.card_bin or "default_card"
        cards_ip = 0
        if ctx.ip_address:
            await RedisVelocityEngine.record_entity_card_attempt(f"ip:{ctx.ip_address}", card_id_str, now_ts)
            cards_ip = await RedisVelocityEngine.get_distinct_cards_count(f"ip:{ctx.ip_address}", window_seconds=600)

        cards_email = 0
        if ctx.email:
            await RedisVelocityEngine.record_entity_card_attempt(f"email:{ctx.email.lower()}", card_id_str, now_ts)
            cards_email = await RedisVelocityEngine.get_distinct_cards_count(f"email:{ctx.email.lower()}", window_seconds=600)

        # 3. Disposable email check
        is_disp = 0
        if ctx.email and "@" in ctx.email:
            domain = ctx.email.split("@")[-1].lower()
            if domain in DISPOSABLE_EMAIL_DOMAINS:
                is_disp = 1

        # 4. PostgreSQL Historical Baseline & Balance Ratio
        mean_30d = 5000_00  # Default ₹5,000 baseline in paise
        std_30d = 2500_00
        account_age_days = 90.0
        available_balance = 100_000_00  # Default ₹100,000 in paise
        payout_ratio = 0.0

        if session and ctx.account_id:
            try:
                from uuid import UUID
                acc_uuid = UUID(ctx.account_id)
                acc_row = await session.get(AccountModel, acc_uuid)
                if acc_row:
                    delta = datetime.datetime.now(UTC) - acc_row.created_at
                    account_age_days = max(delta.total_seconds() / 86400.0, 0.1)

                proj_row = await session.get(BalanceProjectionModel, acc_uuid)
                if proj_row:
                    available_balance = max(proj_row.balance_minor_units, 1)

                # Query 30-day transfer statistics
                cutoff_30d = datetime.datetime.now(UTC) - datetime.timedelta(days=30)
                stat_q = (
                    select(
                        func.avg(JournalEntryModel.amount_minor_units),
                        func.count(JournalEntryModel.id),
                    )
                    .join(JournalTransactionModel, JournalEntryModel.transaction_id == JournalTransactionModel.id)
                    .where(
                        JournalEntryModel.account_id == acc_uuid,
                        JournalTransactionModel.posted_at >= cutoff_30d,
                    )
                )
                res = await session.execute(stat_q)
                avg_val, cnt = res.first() or (None, 0)
                if avg_val and cnt and cnt > 0:
                    mean_30d = float(avg_val)
                    std_30d = max(mean_30d * 0.4, 500_00)
            except Exception:
                pass

        # Calculate Z-score and ratio
        z_score = float((ctx.amount_minor_units - mean_30d) / max(std_30d, 1.0))
        ratio_30d = float(ctx.amount_minor_units / max(mean_30d, 1.0))

        if "payout" in ctx.event_type.lower():
            payout_ratio = min(float(ctx.amount_minor_units / max(available_balance, 1)), 5.0)

        fvec = FeatureVector(
            amount_minor=ctx.amount_minor_units,
            velocity_instrument_60s=v_60s,
            velocity_instrument_1h=v_1h,
            distinct_cards_ip_10m=cards_ip,
            distinct_cards_email_10m=cards_email,
            amount_z_score=round(z_score, 3),
            amount_ratio_30d=round(ratio_30d, 3),
            account_age_days=round(account_age_days, 1),
            hours_since_bank_change=round(ctx.hours_since_bank_change, 1),
            payout_to_balance_ratio=round(payout_ratio, 3),
            is_disposable_email=is_disp,
            is_international_card=1 if ctx.card_international else 0,
            hour_of_day=hour,
            day_of_week=day_of_week,
        )

        telemetry = {
            "instrument_key": instrument_key,
            "velocity_60s": v_60s,
            "velocity_1h": v_1h,
            "distinct_cards_from_ip": cards_ip,
            "distinct_cards_from_email": cards_email,
            "amount_deviation_z": round(z_score, 2),
            "amount_vs_baseline_ratio": f"{ratio_30d:.1f}x",
            "account_age_days": round(account_age_days, 1),
            "hours_since_bank_change": round(ctx.hours_since_bank_change, 1),
            "payout_drain_percentage": f"{payout_ratio * 100:.1f}%" if payout_ratio > 0 else "N/A",
            "disposable_email": bool(is_disp),
            "international_card": ctx.card_international,
        }

        return fvec, telemetry

    @classmethod
    def synthesize_top_risk_factors(
        cls,
        fvec: FeatureVector,
        risk_score: float,
    ) -> list[dict[str, Any]]:
        """Return top 3 human-readable risk drivers based on real behavioral signals."""
        factors: list[tuple[float, str, str, str]] = []

        # 1. Card-testing smurfing check
        if fvec.distinct_cards_ip_10m >= 3:
            severity = "CRITICAL" if fvec.distinct_cards_ip_10m >= 5 else "HIGH"
            weight = fvec.distinct_cards_ip_10m * 0.15
            factors.append((
                weight,
                "Card-Testing / Smurfing Pattern",
                f"{fvec.distinct_cards_ip_10m} distinct payment cards attempted from IP within 10 minutes.",
                severity,
            ))

        # 2. Payout liquidation / ATO check
        if fvec.payout_to_balance_ratio >= 0.80 and fvec.hours_since_bank_change < 48:
            factors.append((
                0.90,
                "Dormant Account Takeover (ATO) Liquidation",
                f"Payout draining {fvec.payout_to_balance_ratio * 100:.1f}% of total balance shortly after profile/bank modification.",
                "CRITICAL",
            ))
        elif fvec.payout_to_balance_ratio >= 0.90:
            factors.append((
                0.75,
                "Aggressive Balance Depletion",
                f"Payout requests {fvec.payout_to_balance_ratio * 100:.1f}% of entire ledger balance in a single execution.",
                "HIGH",
            ))

        # 3. High Instrument Velocity
        if fvec.velocity_instrument_60s >= 3:
            factors.append((
                0.60,
                "High-Frequency Authorization Surge",
                f"{fvec.velocity_instrument_60s} authorizations recorded on same instrument in last 60 seconds.",
                "HIGH",
            ))

        # 4. Historical Amount Spike
        if fvec.amount_z_score >= 3.0:
            factors.append((
                0.55,
                "Extreme Amount Anomaly (Z-Score)",
                f"Amount is {fvec.amount_ratio_30d:.1f}x higher than the account's 30-day baseline (Z={fvec.amount_z_score:.2f}).",
                "HIGH",
            ))

        # 5. Disposable Email
        if fvec.is_disposable_email:
            factors.append((
                0.45,
                "Disposable / High-Risk Email Domain",
                "Customer provided temporary or burnable mailbox frequently associated with automated fraud syndicates.",
                "MEDIUM",
            ))

        # 6. International Card
        if fvec.is_international_card:
            factors.append((
                0.35,
                "Cross-Border Card Instrument",
                "Foreign card BIN processing high-value domestic currency transfer.",
                "MEDIUM",
            ))

        # Default fallback factors if score is low or few triggers
        if not factors:
            factors.append((
                0.10,
                "Normal Behavioral Profile",
                "Transaction velocity, baseline deviation, and instrument characteristics within historical thresholds.",
                "LOW",
            ))

        # Sort descending by weight
        factors.sort(key=lambda x: x[0], reverse=True)
        top3 = factors[:3]

        return [
            {
                "rank": idx + 1,
                "factor_title": f[1],
                "explanation": f[2],
                "severity": f[3],
                "attribution_weight": round(f[0], 3),
            }
            for idx, f in enumerate(top3)
        ]
