"""
End-to-end API tests for LedgerMind AI REST endpoints.
"""

from uuid import uuid4

import pytest
from httpx import AsyncClient


@pytest.mark.api
async def test_create_and_get_account_flow(client: AsyncClient) -> None:
    """Verify account creation and retrieval via REST API."""
    owner_id = str(uuid4())
    create_payload = {
        "owner_id": owner_id,
        "currency": "EUR",
        "initial_deposit_minor_units": 15000,
    }

    create_resp = await client.post("/accounts", json=create_payload)
    assert create_resp.status_code == 201
    data = create_resp.json()

    assert data["currency"] == "EUR"
    assert data["balance_minor_units"] == 15000
    assert data["balance_major"] == 150.0
    account_id = data["id"]

    get_resp = await client.get(f"/accounts/{account_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == account_id


@pytest.mark.api
async def test_transfer_with_idempotency_replay(client: AsyncClient) -> None:
    """Verify atomic transfer and idempotent response deduplication."""
    acc1 = (await client.post("/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 10000})).json()
    acc2 = (await client.post("/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 0})).json()

    transfer_payload = {
        "source_account_id": acc1["id"],
        "destination_account_id": acc2["id"],
        "amount_minor_units": 4000,
        "currency": "EUR",
        "description": "Invoice #101",
    }
    headers = {"Idempotency-Key": "idemp_test_key_999"}

    # 1. First execution
    resp1 = await client.post("/transfers", json=transfer_payload, headers=headers)
    assert resp1.status_code == 201
    data1 = resp1.json()
    assert data1["amount_minor_units"] == 4000

    # Verify sender balance updated
    acc1_updated = (await client.get(f"/accounts/{acc1['id']}")).json()
    assert acc1_updated["balance_minor_units"] == 6000

    # 2. Second execution with identical idempotency key (simulating network retry)
    resp2 = await client.post("/transfers", json=transfer_payload, headers=headers)
    assert resp2.status_code == 201
    data2 = resp2.json()

    # Must return exact same transaction ID and outcome
    assert data2["transaction_id"] == data1["transaction_id"]

    # Verify sender was NOT double-debited!
    acc1_check = (await client.get(f"/accounts/{acc1['id']}")).json()
    assert acc1_check["balance_minor_units"] == 6000


@pytest.mark.api
async def test_transfer_insufficient_funds_returns_422(client: AsyncClient) -> None:
    """Transfer exceeding balance must return HTTP 422 with clear message."""
    acc1 = (await client.post("/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 500})).json()
    acc2 = (await client.post("/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 0})).json()

    payload = {
        "source_account_id": acc1["id"],
        "destination_account_id": acc2["id"],
        "amount_minor_units": 2000,
        "currency": "EUR",
    }
    resp = await client.post("/transfers", json=payload)
    assert resp.status_code == 422
    assert "Insufficient" in resp.json()["detail"]


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
    resp = await client.post("/fraud/predict", json={"transaction_id": "txn_001", "features": features})
    assert resp.status_code == 200
    data = resp.json()

    assert data["transaction_id"] == "txn_001"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert isinstance(data["is_suspicious"], bool)
    assert data["latency_ms"] >= 0.0


@pytest.mark.api
async def test_fraud_model_info_endpoint(client: AsyncClient) -> None:
    """Verify model metadata endpoint."""
    resp = await client.get("/fraud/model-info")
    assert resp.status_code == 200
    data = resp.json()

    assert data["features_count"] == 31
    assert "threshold" in data["optimal_threshold"].__class__.__name__.lower() or isinstance(data["optimal_threshold"], float)


@pytest.mark.api
async def test_analytics_summary_endpoint(client: AsyncClient) -> None:
    """Verify SQL aggregation summary endpoint."""
    resp = await client.get("/analytics/summary")
    assert resp.status_code == 200
    data = resp.json()

    assert "total_accounts" in data
    assert "total_transactions" in data
    assert "total_volume_major" in data


@pytest.mark.api
async def test_ai_assistant_ask_endpoint(client: AsyncClient) -> None:
    """Verify AI Assistant answers grounded in database records."""
    # Create an account to ask about
    acc = (await client.post("/accounts", json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 8500})).json()

    # Ask about account balance
    resp = await client.post("/ai/ask", json={"question": "What is my account balance?", "account_id": acc["id"]})
    assert resp.status_code == 200
    data = resp.json()

    assert "85.00 EUR" in data["answer"]
    assert data["grounded_data"]["balance_minor_units"] == 8500

    # Ask about platform overview
    summary_resp = await client.post("/ai/ask", json={"question": "What is the total transaction volume?"})
    assert summary_resp.status_code == 200
    assert "authoritative ledger records" in summary_resp.json()["answer"]
