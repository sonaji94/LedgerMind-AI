"""
LedgerMind AI — Live Background Transaction Simulator & Attack Streamer.

Continuously streams realistic legitimate commerce and adversarial cyber attacks
(Account Takeovers, Smurfing Waves, Botnet Card-Testing, Dormant Account Drains)
to the FastAPI /api/v1/transactions/ingest endpoint.

Usage:
    python scripts/live_traffic_generator.py --rate 1.0 --count 100
    python scripts/live_traffic_generator.py --rate 0.5   # Continuous stream until Ctrl+C
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx

# Color formatting for terminal observability
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

ACCOUNT_POOL = [
    "acc-treasury-reserve-01",
    "acc-corporate-payroll-02",
    "acc-retail-merchant-hub-03",
    "acc-fintech-clearing-04",
    "acc-global-settlement-05",
    "acc-customer-wallet-101",
    "acc-customer-wallet-102",
    "acc-customer-wallet-103",
]


def generate_normal_commerce() -> dict[str, Any]:
    """Generate typical retail / grocery / e-commerce transaction."""
    amt = round(random.choice([
        random.uniform(3.50, 25.00),     # Coffee / snacks
        random.uniform(25.00, 85.00),    # Groceries
        random.uniform(90.00, 280.00),   # Retail shopping
    ]), 2)
    src = random.choice(ACCOUNT_POOL[:4])
    dst = random.choice(ACCOUNT_POOL[4:])

    return {
        "transaction_id": f"TX-NORM-{uuid4().hex[:8].upper()}",
        "source_account_id": src,
        "destination_account_id": dst,
        "amount_minor_units": int(round(amt * 100)),
        "currency": "EUR",
        "channel": random.choice(["POS_CHIP", "ONLINE_CNP"]),
        "device_id": f"dev-known-{src[-4:]}",
        "ip_address": f"192.168.1.{random.randint(10, 99)}",
        "latitude": 50.1109,
        "longitude": 8.6821,
        "prev_latitude": 50.1115,
        "prev_longitude": 8.6830,
        "prev_timestamp": (datetime.now(UTC) - timedelta(hours=random.uniform(2, 6))).isoformat(),
        "is_new_device": False,
        "user_historical_avg_minor": 12000,
    }


def generate_ato_attack() -> dict[str, Any]:
    """Account Takeover (ATO): Sudden midnight large transfer from a new device."""
    amt = round(random.uniform(3500.00, 8500.00), 2)
    src = "acc-corporate-payroll-02"
    dst = "acc-offshore-mule-99"

    return {
        "transaction_id": f"TX-ATO-{uuid4().hex[:8].upper()}",
        "source_account_id": src,
        "destination_account_id": dst,
        "amount_minor_units": int(round(amt * 100)),
        "currency": "EUR",
        "channel": "WIRE_TRANSFER",
        "device_id": f"dev-tor-unknown-{uuid4().hex[:6]}",
        "ip_address": f"185.220.{random.randint(100, 255)}.{random.randint(1, 99)}",
        "latitude": 1.3521,    # Singapore
        "longitude": 103.8198,
        "prev_latitude": 50.1109,  # Frankfurt
        "prev_longitude": 8.6821,
        "prev_timestamp": (datetime.now(UTC) - timedelta(minutes=45)).isoformat(),  # 9,000 km in 45m!
        "is_new_device": True,
        "user_historical_avg_minor": 15000,
    }


def generate_card_testing_attack() -> dict[str, Any]:
    """Micro card-testing botnet burst."""
    amt = round(random.uniform(0.50, 2.50), 2)
    src = "acc-retail-merchant-hub-03"
    dst = "acc-botnet-drop-12"

    return {
        "transaction_id": f"TX-TEST-{uuid4().hex[:8].upper()}",
        "source_account_id": src,
        "destination_account_id": dst,
        "amount_minor_units": int(round(amt * 100)),
        "currency": "EUR",
        "channel": "ONLINE_CNP",
        "device_id": f"botnet-worker-{random.randint(1, 20)}",
        "ip_address": f"103.45.{random.randint(10, 50)}.{random.randint(1, 200)}",
        "latitude": 13.7563,
        "longitude": 100.5018,
        "is_new_device": True,
        "user_historical_avg_minor": 8000,
    }


def generate_smurfing_wave() -> dict[str, Any]:
    """AML Structuring (Smurfing): Just below the statutory €10k threshold."""
    amt = round(random.uniform(9750.00, 9950.00), 2)
    src = "acc-treasury-reserve-01"
    dst = f"acc-mule-node-{random.randint(1, 6)}"

    return {
        "transaction_id": f"TX-SMURF-{uuid4().hex[:8].upper()}",
        "source_account_id": src,
        "destination_account_id": dst,
        "amount_minor_units": int(round(amt * 100)),
        "currency": "EUR",
        "channel": "WIRE_TRANSFER",
        "device_id": f"dev-mule-{random.randint(1, 5)}",
        "ip_address": f"45.142.{random.randint(1, 99)}.{random.randint(1, 99)}",
        "latitude": 47.3769,  # Zurich
        "longitude": 8.5417,
        "is_new_device": True,
        "user_historical_avg_minor": 12000,
    }


def stream_traffic(endpoint: str, rate_sec: float, total_count: int | None, attack_ratio: float) -> None:
    print(f"\n{CYAN}{BOLD}========================================================================{RESET}")
    print(f"{CYAN}{BOLD}   LedgerMind AI — Live Transaction Streamer & Attack Simulator        {RESET}")
    print(f"{CYAN}{BOLD}========================================================================{RESET}")
    print(f"Target Ingestion Endpoint : {BOLD}{endpoint}{RESET}")
    print(f"Streaming Frequency       : 1 transaction every {rate_sec:.1f}s")
    print(f"Adversarial Attack Ratio  : {attack_ratio * 100:.0f}%")
    print(f"Total Transactions Target : {'Continuous (Ctrl+C to stop)' if total_count is None else total_count}")
    print(f"{CYAN}------------------------------------------------------------------------{RESET}\n")

    sent = 0
    approved_vol = 0.0
    prevented_vol = 0.0
    flagged_count = 0
    approved_count = 0

    client = httpx.Client(timeout=10.0)

    try:
        while True:
            if total_count is not None and sent >= total_count:
                break

            # Pick archetype
            is_attack = random.random() < attack_ratio
            if is_attack:
                attack_type = random.choice(["ATO", "CARD_TESTING", "SMURFING"])
                if attack_type == "ATO":
                    payload = generate_ato_attack()
                elif attack_type == "CARD_TESTING":
                    payload = generate_card_testing_attack()
                else:
                    payload = generate_smurfing_wave()
            else:
                payload = generate_normal_commerce()

            amt = payload["amount_minor_units"] / 100.0
            tx_id = payload["transaction_id"]

            t0 = time.perf_counter()
            try:
                res = client.post(
                    endpoint,
                    json=payload,
                    headers={"Idempotency-Key": tx_id},
                )
                dt = (time.perf_counter() - t0) * 1000.0

                if res.status_code in (200, 201):
                    data = res.json()
                    decision = data.get("decision", "UNKNOWN")
                    risk_pct = data.get("risk_score", 0.0) * 100.0

                    if decision == "FLAGGED_INTERCEPT":
                        flagged_count += 1
                        prevented_vol += amt
                        reasons = data.get("risk_factors", [])
                        top_reason = reasons[0] if reasons else "High anomaly score"
                        print(
                            f"{RED}{BOLD}[INTERCEPT {res.status_code}]{RESET} {tx_id:<18} | "
                            f"€{amt:>9,2f} | Risk: {risk_pct:>5.1f}% | "
                            f"{dt:>5.1f}ms | {RED}FLAGGED QUARANTINED{RESET} ({top_reason})"
                        )
                    else:
                        approved_count += 1
                        approved_vol += amt
                        print(
                            f"{GREEN}[SETTLED   {res.status_code}]{RESET} {tx_id:<18} | "
                            f"€{amt:>9,2f} | Risk: {risk_pct:>5.1f}% | "
                            f"{dt:>5.1f}ms | {GREEN}APPROVED SETTLED{RESET}"
                        )
                else:
                    print(f"{YELLOW}[ERROR {res.status_code}]{RESET} {tx_id}: {res.text}")

            except httpx.ConnectError:
                print(f"{RED}❌ Could not connect to {endpoint}. Is the FastAPI backend running on port 8000?{RESET}")
                time.sleep(3.0)
            except Exception as e:
                print(f"{YELLOW}⚠️ Ingestion Exception: {e}{RESET}")

            sent += 1
            time.sleep(rate_sec)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}🛑 Live traffic simulation halted by operator.{RESET}")

    finally:
        client.close()
        print(f"\n{CYAN}{BOLD}========================================================================{RESET}")
        print(f"{CYAN}{BOLD}                    SIMULATION STREAM SUMMARY                           {RESET}")
        print(f"{CYAN}{BOLD}========================================================================{RESET}")
        print(f"Total Transactions Processed : {sent}")
        print(f"Approved Normal Settlements  : {GREEN}{approved_count} (€{approved_vol:,.2f}){RESET}")
        print(f"Intercepted Cyber Attacks    : {RED}{flagged_count} (€{prevented_vol:,.2f} Preserved){RESET}")
        if sent > 0:
            print(f"Fraud Detection Catch Rate   : {flagged_count / max(sent * attack_ratio, 1.0) * 100:.1f}%")
        print(f"{CYAN}========================================================================{RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LedgerMind AI Live Transaction Streamer")
    parser.add_argument("--endpoint", default="http://localhost:8000/api/v1/transactions/ingest", help="Target API endpoint")
    parser.add_argument("--rate", type=float, default=0.8, help="Delay between transactions in seconds")
    parser.add_argument("--count", type=int, default=None, help="Total transactions to send (default: unlimited)")
    parser.add_argument("--attack-ratio", type=float, default=0.25, help="Ratio of adversarial attacks (0.0 to 1.0)")
    args = parser.parse_args()

    stream_traffic(args.endpoint, args.rate, args.count, args.attack_ratio)
