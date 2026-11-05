"""
End-to-end API tests for LedgerMind AI REST endpoints.

All routes are mounted under /api/v1/ in production mode.
"""

from uuid import uuid4

import pytest
from httpx import AsyncClient

V1 = "/api/v1"


@pytest.mark.api
async def test_health_check_deep(client: AsyncClient) -> None:
    """Deep health check must return expected fields including DB and ML status."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert "database" in data
    assert "ml_model" in data
    assert data["version"] == "1.0.0"


@pytest.mark.api
async def test_request_id_header_injected(client: AsyncClient) -> None:
    """Every response must carry an X-Request-ID correlation header."""
    resp = await client.get("/health")
    assert "x-request-id" in resp.headers


@pytest.mark.api
async def test_create_and_get_account_flow(client: AsyncClient) -> None:
    """Verify account creation and retrieval via REST API."""
    owner_id = str(uuid4())
    create_payload = {
        "owner_id": owner_id,
        "currency": "EUR",
        "initial_deposit_minor_units": 15000,
    }

    create_resp = await client.post(f"{V1}/accounts", json=create_payload)
    assert create_resp.status_code == 201
    data = create_resp.json()

    assert data["currency"] == "EUR"
    assert data["balance_minor_units"] == 15000
    assert data["balance_major"] == 150.0
    account_id = data["id"]

    get_resp = await client.get(f"{V1}/accounts/{account_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == account_id


@pytest.mark.api
async def test_account_not_found_returns_404(client: AsyncClient) -> None:
    """Retrieving a non-existent account must return HTTP 404."""
    fake_id = str(uuid4())
    resp = await client.get(f"{V1}/accounts/{fake_id}")
    assert resp.status_code == 404


@pytest.mark.api
async def test_transfer_with_idempotency_replay(client: AsyncClient) -> None:
    """Verify atomic transfer and idempotent response deduplication."""
    acc1 = (await client.post(f"{V1}/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 10000})).json()
    acc2 = (await client.post(f"{V1}/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 0})).json()

    transfer_payload = {
        "source_account_id": acc1["id"],
        "destination_account_id": acc2["id"],
        "amount_minor_units": 4000,
        "currency": "EUR",
        "description": "Invoice #101",
    }
    headers = {"Idempotency-Key": f"idemp_test_{uuid4().hex}"}

    # 1. First execution
    resp1 = await client.post(f"{V1}/transfers", json=transfer_payload, headers=headers)
    assert resp1.status_code == 201
    data1 = resp1.json()
    assert data1["amount_minor_units"] == 4000

    # Verify sender balance updated
    acc1_updated = (await client.get(f"{V1}/accounts/{acc1['id']}")).json()
    assert acc1_updated["balance_minor_units"] == 6000

    # 2. Second execution with identical idempotency key (simulating network retry)
    resp2 = await client.post(f"{V1}/transfers", json=transfer_payload, headers=headers)
    assert resp2.status_code == 201
    data2 = resp2.json()

    # Must return exact same transaction ID and outcome
    assert data2["transaction_id"] == data1["transaction_id"]

    # Verify sender was NOT double-debited!
    acc1_check = (await client.get(f"{V1}/accounts/{acc1['id']}")).json()
    assert acc1_check["balance_minor_units"] == 6000


@pytest.mark.api
async def test_transfer_insufficient_funds_returns_422(client: AsyncClient) -> None:
    """Transfer exceeding balance must return HTTP 422 with clear message."""
    acc1 = (await client.post(f"{V1}/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 500})).json()
    acc2 = (await client.post(f"{V1}/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 0})).json()

    payload = {
        "source_account_id": acc1["id"],
        "destination_account_id": acc2["id"],
        "amount_minor_units": 2000,
        "currency": "EUR",
    }
    resp = await client.post(f"{V1}/transfers", json=payload)
    assert resp.status_code == 422
    # Custom exception handler returns {"error": ..., "detail": ...}
    body = resp.json()
    assert "detail" in body or "Insufficient" in str(body)


@pytest.mark.api
async def test_account_transaction_history_pagination(client: AsyncClient) -> None:
    """Transaction history endpoint must return paginated journal entries."""
    # Create two accounts
    acc1 = (await client.post(f"{V1}/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 50000})).json()
    acc2 = (await client.post(f"{V1}/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 0})).json()

    # Execute 3 transfers so acc1 has 3 DEBIT entries
    for i in range(3):
        await client.post(
            f"{V1}/transfers",
            json={
                "source_account_id": acc1["id"],
                "destination_account_id": acc2["id"],
                "amount_minor_units": 1000,
                "currency": "EUR",
                "description": f"Transfer #{i + 1}",
            },
            headers={"Idempotency-Key": f"hist-test-{uuid4().hex}"},
        )

    # Fetch transaction history (page 1)
    resp = await client.get(f"{V1}/accounts/{acc1['id']}/transactions?page=1&page_size=10")
    assert resp.status_code == 200
    data = resp.json()

    assert data["account_id"] == acc1["id"]
    assert data["total"] >= 3
    assert len(data["items"]) >= 3
    # Initial deposit entry + 3 debits = at least 4 entries
    for item in data["items"]:
        assert item["entry_type"] in ("DEBIT", "CREDIT")
        assert item["amount_minor_units"] > 0
        assert item["amount_major"] > 0.0


@pytest.mark.api
async def test_account_transaction_history_404_for_unknown(client: AsyncClient) -> None:
    """Transaction history for a non-existent account returns 404."""
    resp = await client.get(f"{V1}/accounts/{uuid4()}/transactions")
    assert resp.status_code == 404


@pytest.mark.api
async def test_fraud_prediction_endpoint(client: AsyncClient) -> None:
    """Verify live fraud prediction scoring endpoint."""
    features = {
        "Time": 45000.0,
        "Amount": 149.99,
        "V1": -1.2,
        "V2": 0.8,
        "V3": -0.5,
    }
    resp = await client.post(f"{V1}/fraud/predict", json={"transaction_id": "txn_001", "features": features})
    assert resp.status_code == 200
    data = resp.json()

    assert data["transaction_id"] == "txn_001"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert isinstance(data["is_suspicious"], bool)
    assert data["latency_ms"] >= 0.0


@pytest.mark.api
async def test_fraud_model_info_endpoint(client: AsyncClient) -> None:
    """Verify model metadata endpoint."""
    resp = await client.get(f"{V1}/fraud/model-info")
    assert resp.status_code == 200
    data = resp.json()

    assert data["features_count"] == 31
    assert isinstance(data["optimal_threshold"], float)


@pytest.mark.api
async def test_analytics_summary_endpoint(client: AsyncClient) -> None:
    """Verify SQL aggregation summary endpoint."""
    resp = await client.get(f"{V1}/analytics/summary")
    assert resp.status_code == 200
    data = resp.json()

    assert "total_accounts" in data
    assert "total_transactions" in data
    assert "total_volume_major" in data


@pytest.mark.api
async def test_ai_assistant_ask_endpoint(client: AsyncClient) -> None:
    """Verify AI Assistant answers grounded in database records."""
    # Create an account to ask about
    acc = (await client.post(f"{V1}/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 8500})).json()

    # Ask about account balance
    resp = await client.post(f"{V1}/ai/ask", json={"question": "What is my account balance?", "account_id": acc["id"]})
    assert resp.status_code == 200
    data = resp.json()

    assert "85.00 EUR" in data["answer"]
    assert data["grounded_data"]["balance_minor_units"] == 8500

    # Ask about platform overview
    summary_resp = await client.post(f"{V1}/ai/ask", json={"question": "What is the total transaction volume?"})
    assert summary_resp.status_code == 200
    assert "authoritative ledger records" in summary_resp.json()["answer"]
