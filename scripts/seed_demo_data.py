"""
LedgerMind AI — Demo Data Seeder

Creates a realistic demo dataset so reviewers and interviewers can
explore the full platform immediately after docker compose up.

What this creates:
  - 5 demo accounts (1 treasury + 4 customer accounts)
  - 1 initial treasury funding event (5,000,000 EUR)
  - 10 realistic customer transfers with idempotency keys
  - 5 fraud prediction calls (mix of normal and suspicious)

Usage:
    python scripts/seed_demo_data.py

Requires the FastAPI server to be running at http://localhost:8000
"""

from __future__ import annotations

import sys
import uuid

import httpx

BASE_URL = "http://localhost:8000/api/v1"
TIMEOUT = 10.0


def post(client: httpx.Client, path: str, **kwargs: object) -> httpx.Response:
    resp = client.post(f"{BASE_URL}{path}", **kwargs)  # type: ignore[arg-type]
    if resp.status_code not in (200, 201):
        print(f"  ⚠️  {path} → {resp.status_code}: {resp.text[:200]}")
    return resp


def get(client: httpx.Client, path: str, **kwargs: object) -> httpx.Response:
    return client.get(f"{BASE_URL}{path}", **kwargs)  # type: ignore[arg-type]


def main() -> None:
    print("🌱 LedgerMind AI — Seeding Demo Data")
    print("=" * 50)

    with httpx.Client(timeout=TIMEOUT) as client:
        # ── 1. Create Accounts ────────────────────────────────────
        print("\n📋 Creating accounts...")

        treasury_owner = uuid.UUID("00000000-0000-0000-0000-000000000001")
        customer_owners = [
            uuid.UUID(f"00000000-0000-0000-0000-00000000000{i}") for i in range(2, 7)
        ]

        # Treasury account with 5,000,000 EUR initial deposit
        resp = post(
            client,
            "/accounts",
            json={
                "owner_id": str(treasury_owner),
                "currency": "EUR",
                "initial_deposit_minor_units": 500_000_000,  # €5,000,000.00
            },
        )
        if resp.status_code != 201:
            print("  ❌ Failed to create treasury account. Is the server running?")
            sys.exit(1)
        treasury_id = resp.json()["id"]
        print(f"  ✅ Treasury account: {treasury_id}  (€5,000,000.00)")

        # Customer accounts
        customer_ids: list[str] = []
        names = ["Alice Chen", "Bob Kumar", "Priya Shah", "Carlos Rivera", "Emily Zhang"]
        for _i, (owner, name) in enumerate(zip(customer_owners, names, strict=False)):
            resp = post(
                client,
                "/accounts",
                json={
                    "owner_id": str(owner),
                    "currency": "EUR",
                    "initial_deposit_minor_units": 0,
                },
            )
            if resp.status_code == 201:
                cid = resp.json()["id"]
                customer_ids.append(cid)
                print(f"  ✅ {name}: {cid}")

        # ── 2. Fund Customer Accounts from Treasury ───────────────
        print("\n💸 Funding customer accounts from treasury...")
        funding_amounts = [
            (500_000, "€5,000.00 — Initial funding for Alice Chen"),
            (1_250_000, "€12,500.00 — Initial funding for Bob Kumar"),
            (750_000, "€7,500.00 — Initial funding for Priya Shah"),
            (2_000_000, "€20,000.00 — Initial funding for Carlos Rivera"),
            (300_000, "€3,000.00 — Initial funding for Emily Zhang"),
        ]
        for cid, (amount, desc) in zip(customer_ids, funding_amounts, strict=False):
            ikey = f"seed-fund-{cid}"
            resp = post(
                client,
                "/transfers",
                headers={"Idempotency-Key": ikey},
                json={
                    "source_account_id": treasury_id,
                    "destination_account_id": cid,
                    "amount_minor_units": amount,
                    "currency": "EUR",
                    "description": desc,
                },
            )
            if resp.status_code == 201:
                print(f"  ✅ {desc}")

        # ── 3. Customer-to-Customer Transfers ─────────────────────
        print("\n🔄 Simulating customer-to-customer transfers...")
        transfers = [
            (customer_ids[0], customer_ids[1], 50_000, "Invoice #1001 — Consulting services"),
            (customer_ids[1], customer_ids[2], 25_000, "Invoice #1002 — Design work"),
            (customer_ids[2], customer_ids[3], 10_000, "Personal transfer — Birthday gift"),
            (customer_ids[3], customer_ids[4], 15_000, "Split bill — Dinner"),
            (customer_ids[4], customer_ids[0], 8_500, "Reimbursement — Travel"),
        ]
        for src, dst, amt, desc in transfers:
            ikey = f"seed-transfer-{src[:8]}-{dst[:8]}"
            resp = post(
                client,
                "/transfers",
                headers={"Idempotency-Key": ikey},
                json={
                    "source_account_id": src,
                    "destination_account_id": dst,
                    "amount_minor_units": amt,
                    "currency": "EUR",
                    "description": desc,
                },
            )
            if resp.status_code == 201:
                major = amt / 100
                print(f"  ✅ €{major:,.2f} — {desc}")

        # ── 4. Fraud Predictions ──────────────────────────────────
        print("\n🛡️ Running fraud prediction samples...")
        fraud_cases = [
            # Normal transactions
            {"Time": 3600.0, "Amount": 48.50, "label": "Normal — Low amount daytime"},
            {"Time": 43200.0, "Amount": 220.00, "label": "Normal — Mid amount afternoon"},
            # Suspicious transactions
            {"Time": 7800.0, "Amount": 48500.0, "V1": -4.5, "V2": 3.8, "label": "SUSPICIOUS — Very high amount"},
            {"Time": 2.0, "Amount": 9999.99, "V1": -6.0, "V3": 4.1, "label": "SUSPICIOUS — 2AM large transaction"},
            {"Time": 90000.0, "Amount": 0.01, "V1": 5.0, "label": "SUSPICIOUS — Probe transaction pattern"},
        ]
        for case in fraud_cases:
            label = str(case.pop("label"))
            features = {k: float(str(v)) for k, v in case.items()}
            resp = post(
                client,
                "/fraud/predict",
                json={"transaction_id": str(uuid.uuid4()), "features": features},
            )
            if resp.status_code == 200:
                r = resp.json()
                flag = "🚨" if r["is_suspicious"] else "✅"
                print(f"  {flag} [{label}] score={r['risk_score']:.3f}")

        # ── 5. Final Summary ──────────────────────────────────────
        print("\n📊 Platform Analytics Summary:")
        resp = get(client, "/analytics/summary")
        if resp.status_code == 200:
            s = resp.json()
            print(f"  Accounts:     {s['total_accounts']:,}")
            print(f"  Transactions: {s['total_transactions']:,}")
            print(f"  Volume:       €{s['total_volume_major']:,.2f}")

        print("\n✅ Demo data seeded successfully!")
        print("   API Docs: http://localhost:8000/docs")
        print("   Dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
