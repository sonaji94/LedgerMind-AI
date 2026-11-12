"""
LedgerMind AI — Razorpay Webhook Test & Simulation Client.

Sends cryptographically signed Razorpay webhook events to the LedgerMind AI
webhook receiver (POST /api/v1/webhooks/razorpay).

Supports:
1. Standard Events:
   - payment.authorized
   - payment.captured (triggers multi-leg double-entry reconciliation: clearing, escrow, fee, tax)
   - payout.initiated (triggers ledger hold: Pending_Payout_Hold)
   - payout.reversed (triggers compensating ledger release)
   - payment.dispute.created (triggers chargeback reserve + ground-truth label = 1)
2. Adversarial Fraud Scenarios:
   - --scenario card-testing: Rapid burst of 6 authorizations from single IP with distinct cards
   - --scenario account-drain: Payout draining 95% balance shortly after bank account modification

Usage:
    python scripts/send_razorpay_test_webhook.py --event payment.captured --amount 500
    python scripts/send_razorpay_test_webhook.py --scenario card-testing
    python scripts/send_razorpay_test_webhook.py --scenario account-drain
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import time
from uuid import uuid4

import httpx

DEFAULT_WEBHOOK_URL = "http://localhost:8000/api/v1/webhooks/razorpay"
DEFAULT_SECRET = "rzp_whsec_test_secret_key_12345"


def sign_payload(raw_bytes: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 hex digest for Razorpay webhook verification."""
    return hmac.new(
        key=secret.encode("utf-8"),
        msg=raw_bytes,
        digestmod=hashlib.sha256,
    ).hexdigest()


def build_payment_captured_payload(
    payment_id: str | None = None,
    amount_inr: float = 500.0,
    email: str = "customer.verma@gmail.com",
    contact: str = "+919876543210",
    vpa: str | None = None,
    card_last4: str = "4242",
) -> dict:
    """Construct realistic Razorpay payment.captured webhook JSON."""
    pid = payment_id or f"pay_{uuid4().hex[:14]}"
    amount_paise = int(amount_inr * 100)
    fee_paise = int(amount_paise * 0.02)  # 2% payment gateway fee
    tax_paise = int(fee_paise * 0.18)    # 18% GST on fee

    return {
        "entity": "event",
        "account_id": "acc_ledgermind_live_01",
        "event": "payment.captured",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": pid,
                    "entity": "payment",
                    "amount": amount_paise,
                    "currency": "INR",
                    "status": "captured",
                    "order_id": f"order_{uuid4().hex[:12]}",
                    "invoice_id": None,
                    "international": False,
                    "method": "upi" if vpa else "card",
                    "amount_refunded": 0,
                    "refund_status": None,
                    "captured": True,
                    "description": "LedgerMind Enterprise Settlement",
                    "card_id": f"card_{uuid4().hex[:10]}" if not vpa else None,
                    "card": {
                        "id": f"card_{uuid4().hex[:10]}",
                        "entity": "card",
                        "name": "Arun Verma",
                        "last4": card_last4,
                        "network": "Visa",
                        "type": "credit",
                        "issuer": "HDFC",
                        "international": False,
                    } if not vpa else None,
                    "bank": None if vpa else "HDFC",
                    "wallet": None,
                    "vpa": vpa,
                    "email": email,
                    "contact": contact,
                    "notes": {
                        "account_id": "00000000-0000-0000-0002-000000000002",
                        "merchant_order": "ORD-99182",
                    },
                    "fee": fee_paise,
                    "tax": tax_paise,
                    "error_code": None,
                    "error_description": None,
                    "created_at": int(time.time()),
                }
            }
        },
        "created_at": int(time.time()),
    }


def build_payout_initiated_payload(
    payout_id: str | None = None,
    amount_inr: float = 2500.0,
    hours_since_bank_change: float = 720.0,
) -> dict:
    """Construct realistic RazorpayX payout.initiated webhook JSON."""
    pout_id = payout_id or f"pout_{uuid4().hex[:14]}"
    amount_paise = int(amount_inr * 100)

    return {
        "entity": "event",
        "account_id": "acc_ledgermind_live_01",
        "event": "payout.initiated",
        "contains": ["payout"],
        "payload": {
            "payout": {
                "entity": {
                    "id": pout_id,
                    "entity": "payout",
                    "fund_account_id": f"fa_{uuid4().hex[:12]}",
                    "amount": amount_paise,
                    "currency": "INR",
                    "notes": {
                        "account_id": "00000000-0000-0000-0002-000000000002",
                        "hours_since_bank_change": str(hours_since_bank_change),
                        "purpose": "vendor_settlement",
                    },
                    "fees": 590,  # ₹5.90 IMPS fee
                    "tax": 90,
                    "status": "processing",
                    "utr": f"UTR{int(time.time())}",
                    "mode": "IMPS",
                    "created_at": int(time.time()),
                }
            }
        },
        "created_at": int(time.time()),
    }


def build_payout_reversed_payload(payout_id: str, amount_inr: float = 2500.0) -> dict:
    """Construct realistic RazorpayX payout.reversed webhook JSON."""
    amount_paise = int(amount_inr * 100)
    return {
        "entity": "event",
        "account_id": "acc_ledgermind_live_01",
        "event": "payout.reversed",
        "contains": ["payout"],
        "payload": {
            "payout": {
                "entity": {
                    "id": payout_id,
                    "entity": "payout",
                    "amount": amount_paise,
                    "currency": "INR",
                    "status": "reversed",
                    "notes": {"account_id": "00000000-0000-0000-0002-000000000002"},
                    "created_at": int(time.time()),
                }
            }
        },
        "created_at": int(time.time()),
    }


def build_dispute_created_payload(payment_id: str, amount_inr: float = 500.0) -> dict:
    """Construct realistic Razorpay payment.dispute.created webhook JSON."""
    disp_id = f"disp_{uuid4().hex[:12]}"
    amount_paise = int(amount_inr * 100)
    return {
        "entity": "event",
        "account_id": "acc_ledgermind_live_01",
        "event": "payment.dispute.created",
        "contains": ["dispute", "payment"],
        "payload": {
            "dispute": {
                "entity": {
                    "id": disp_id,
                    "entity": "dispute",
                    "payment_id": payment_id,
                    "amount": amount_paise,
                    "currency": "INR",
                    "reason_code": "fraudulent",
                    "status": "under_review",
                    "phase": "chargeback",
                    "created_at": int(time.time()),
                }
            }
        },
        "created_at": int(time.time()),
    }


def send_webhook(
    url: str,
    secret: str,
    payload: dict,
    event_id: str | None = None,
) -> tuple[int, dict]:
    """Sign and transmit webhook payload via HTTP POST."""
    ev_id = event_id or f"rzp_ev_{uuid4().hex[:16]}"
    raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = sign_payload(raw_body, secret)

    headers = {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": signature,
        "X-Razorpay-Event-Id": ev_id,
    }

    t0 = time.perf_counter()
    with httpx.Client(timeout=10.0) as client:
        res = client.post(url, content=raw_body, headers=headers)
        latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"📡 [{res.status_code}] {payload.get('event')} (ACK in {latency_ms:.1f}ms) -> Event ID: {ev_id}")
    try:
        data = res.json()
        print(f"   Response: {json.dumps(data, indent=2)}")
        return res.status_code, data
    except Exception:
        print(f"   Raw text: {res.text}")
        return res.status_code, {}


def run_card_testing_attack(url: str, secret: str) -> None:
    """Simulate rapid card-testing smurfing attack (6 micro-transactions with different cards from single IP)."""
    print("\n🚨 SIMULATING ADVERSARIAL CARD-TESTING ATTACK (SMURFING)")
    print("   Sending 6 rapid micro-authorizations from IP 185.220.101.44 using rotating cards...\n")
    test_ip = "185.220.101.44"

    for i in range(1, 7):
        last4 = f"{1000 + i}"
        payload = build_payment_captured_payload(
            amount_inr=10.00,  # ₹10 micro-transaction
            email=f"bot_{i}@tempmail.com",
            card_last4=last4,
        )
        payload["ip_address"] = test_ip
        send_webhook(url, secret, payload)
        time.sleep(0.3)

    print("\n✅ Card-testing burst completed. Check operations dashboard for flagged fraud alerts.\n")


def run_account_drain_attack(url: str, secret: str) -> None:
    """Simulate account takeover (ATO) liquidation: 95% balance payout after fresh bank change."""
    print("\n🚨 SIMULATING ATO DORMANT ACCOUNT DRAIN")
    print("   Sending payout.initiated requesting ₹95,000 (95% balance) with bank change 2.5 hours ago...\n")

    payload = build_payout_initiated_payload(
        amount_inr=95000.00,
        hours_since_bank_change=2.5,  # changed 2.5 hours ago
    )
    send_webhook(url, secret, payload)
    print("\n✅ ATO Payout initiated. Check operations dashboard for Pending_Payout_Hold & TreeSHAP triage.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Razorpay Webhook Test Transmitter")
    parser.add_argument("--url", default=DEFAULT_WEBHOOK_URL, help="Webhook receiver URL")
    parser.add_argument("--secret", default=DEFAULT_SECRET, help="Razorpay webhook secret")
    parser.add_argument("--event", choices=["payment.captured", "payout.initiated", "payout.reversed", "payment.dispute.created"], default="payment.captured")
    parser.add_argument("--amount", type=float, default=750.0, help="Transaction amount in INR")
    parser.add_argument("--scenario", choices=["card-testing", "account-drain"], help="Run attack scenario")

    args = parser.parse_args()

    if args.scenario == "card-testing":
        run_card_testing_attack(args.url, args.secret)
    elif args.scenario == "account-drain":
        run_account_drain_attack(args.url, args.secret)
    else:
        print(f"\n🚀 Transmitting {args.event} for ₹{args.amount:.2f}...")
        if args.event == "payment.captured":
            payload = build_payment_captured_payload(amount_inr=args.amount)
        elif args.event == "payout.initiated":
            payload = build_payout_initiated_payload(amount_inr=args.amount)
        elif args.event == "payout.reversed":
            payload = build_payout_reversed_payload(payout_id=f"pout_{uuid4().hex[:12]}", amount_inr=args.amount)
        elif args.event == "payment.dispute.created":
            payload = build_dispute_created_payload(payment_id=f"pay_{uuid4().hex[:12]}", amount_inr=args.amount)
        else:
            payload = build_payment_captured_payload(amount_inr=args.amount)

        send_webhook(args.url, args.secret, payload)


if __name__ == "__main__":
    main()
