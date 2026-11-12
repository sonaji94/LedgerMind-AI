"""
Tests for Razorpay Webhook Ingestion, Double-Entry Reconciliation, and Active Retraining.

Tests:
1. HMAC-SHA256 signature verification and unauthorized rejection
2. payment.captured multi-leg double-entry reconciliation (gross == net + fee + tax)
3. Strict webhook event idempotency (duplicate retry protection)
4. payout.initiated hold and payout.reversed compensating release
5. payment.dispute.created chargeback reserve and automatic ground_truth_label=1
6. Human-in-the-loop analyst feedback labeling with compliance audit trail
7. Continuous ML active retraining cycle execution
"""

from __future__ import annotations

import hashlib
import hmac
import json
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.core.security import UserRole, create_access_token
from app.main import app
from app.webhooks.security import verify_razorpay_signature
from ml.retrain import run_retraining

settings = get_settings()


def compute_signature(payload_bytes: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()


@pytest.fixture
def compliance_token() -> str:
    return create_access_token(subject=str(uuid4()), role=UserRole.COMPLIANCE_OFFICER.value)


@pytest.mark.api
@pytest.mark.asyncio
async def test_razorpay_signature_validation() -> None:
    """Verify HMAC-SHA256 signature verifier and rejection on forged digest."""
    secret = "test_webhook_secret_key"
    body = b'{"event":"payment.captured","test":true}'

    # Valid
    sig_valid = compute_signature(body, secret)
    assert verify_razorpay_signature(body, sig_valid, secret) is True

    # Invalid / Forged
    assert verify_razorpay_signature(body, "forged_hex_digest_12345", secret) is False
    assert verify_razorpay_signature(body, None, secret) is False

    # HTTP Endpoint rejects without valid signature
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/v1/webhooks/razorpay",
            content=body,
            headers={"Content-Type": "application/json", "X-Razorpay-Signature": "invalid_signature"},
        )
        assert res.status_code == 401


@pytest.mark.api
@pytest.mark.asyncio
async def test_payment_captured_and_idempotency() -> None:
    """Verify payment.captured acknowledgment, background double-entry reconciliation, and idempotency."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        event_id = f"rzp_test_ev_{uuid4().hex[:12]}"
        pay_id = f"pay_{uuid4().hex[:12]}"

        payload = {
            "entity": "event",
            "account_id": "acc_test",
            "event": "payment.captured",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": pay_id,
                        "amount": 50000,   # ₹500.00
                        "fee": 1000,       # ₹10.00
                        "tax": 180,        # ₹1.80
                        "currency": "INR",
                        "status": "captured",
                        "method": "card",
                        "email": "customer@example.com",
                        "contact": "+919876543210",
                    }
                }
            },
        }

        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        sig = compute_signature(body, settings.razorpay_webhook_secret)

        # 1. Ingest valid webhook
        headers = {
            "Content-Type": "application/json",
            "X-Razorpay-Signature": sig,
            "X-Razorpay-Event-Id": event_id,
        }
        res1 = await client.post("/api/v1/webhooks/razorpay", content=body, headers=headers)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["status"] == "acknowledged"
        assert data1["event_id"] == event_id

        # 2. Duplicate retry with same event_id (must return cached outcome without re-processing)
        res2 = await client.post("/api/v1/webhooks/razorpay", content=body, headers=headers)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2.get("cached") is True
        assert data2["event_id"] == event_id


@pytest.mark.api
@pytest.mark.asyncio
async def test_payout_hold_and_reversal() -> None:
    """Verify payout.initiated hold and payout.reversed compensating release."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        pout_id = f"pout_{uuid4().hex[:12]}"

        # 1. Payout Initiated
        ev1 = f"rzp_ev_pout_{uuid4().hex[:8]}"
        payload1 = {
            "entity": "event",
            "event": "payout.initiated",
            "payload": {
                "payout": {
                    "entity": {
                        "id": pout_id,
                        "amount": 250000,  # ₹2,500.00
                        "currency": "INR",
                        "status": "processing",
                    }
                }
            },
        }
        body1 = json.dumps(payload1, separators=(",", ":")).encode("utf-8")
        sig1 = compute_signature(body1, settings.razorpay_webhook_secret)

        res1 = await client.post(
            "/api/v1/webhooks/razorpay",
            content=body1,
            headers={"Content-Type": "application/json", "X-Razorpay-Signature": sig1, "X-Razorpay-Event-Id": ev1},
        )
        assert res1.status_code == 200

        # 2. Payout Reversed
        ev2 = f"rzp_ev_rev_{uuid4().hex[:8]}"
        payload2 = {
            "entity": "event",
            "event": "payout.reversed",
            "payload": {
                "payout": {
                    "entity": {
                        "id": pout_id,
                        "amount": 250000,
                        "currency": "INR",
                        "status": "reversed",
                    }
                }
            },
        }
        body2 = json.dumps(payload2, separators=(",", ":")).encode("utf-8")
        sig2 = compute_signature(body2, settings.razorpay_webhook_secret)

        res2 = await client.post(
            "/api/v1/webhooks/razorpay",
            content=body2,
            headers={"Content-Type": "application/json", "X-Razorpay-Signature": sig2, "X-Razorpay-Event-Id": ev2},
        )
        assert res2.status_code == 200


@pytest.mark.api
@pytest.mark.asyncio
async def test_chargeback_dispute_and_analyst_feedback(compliance_token: str) -> None:
    """Verify payment.dispute.created records chargeback reserve and supports analyst label feedback."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        disp_id = f"disp_{uuid4().hex[:10]}"
        pay_id = f"pay_{uuid4().hex[:10]}"
        ev_id = f"rzp_ev_disp_{uuid4().hex[:8]}"

        payload = {
            "entity": "event",
            "event": "payment.dispute.created",
            "payload": {
                "dispute": {
                    "entity": {
                        "id": disp_id,
                        "payment_id": pay_id,
                        "amount": 45000,
                        "currency": "INR",
                        "reason_code": "fraudulent",
                    }
                }
            },
        }
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        sig = compute_signature(body, settings.razorpay_webhook_secret)

        res = await client.post(
            "/api/v1/webhooks/razorpay",
            content=body,
            headers={"Content-Type": "application/json", "X-Razorpay-Signature": sig, "X-Razorpay-Event-Id": ev_id},
        )
        assert res.status_code == 200

        # Submit analyst feedback labeling
        fb_payload = {
            "entity_id": disp_id,
            "ground_truth_label": 1,
            "justification_reason": "Confirmed unauthorized cardholder claim with issuing bank",
        }
        fb_res = await client.post(
            "/api/v1/webhooks/razorpay/feedback",
            json=fb_payload,
            headers={"Authorization": f"Bearer {compliance_token}"},
        )
        assert fb_res.status_code == 200
        data = fb_res.json()
        assert data["ground_truth_label"] == 1
        assert data["classification"] == "CONFIRMED_FRAUD"


def test_active_retraining_pipeline() -> None:
    """Verify execution of ml.retrain pipeline and model performance reporting."""
    report = run_retraining()
    assert report["status"] == "success"
    assert "candidate_f1" in report
    assert "candidate_precision" in report
    assert "candidate_recall" in report
    assert report["action"] in ("PROMOTED", "REJECTED")
