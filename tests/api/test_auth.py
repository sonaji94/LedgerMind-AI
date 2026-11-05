"""
API tests for LedgerMind AI JWT Authentication and Token Validation.
"""

from uuid import uuid4

import pytest
from httpx import AsyncClient

V1 = "/api/v1"


@pytest.mark.api
async def test_auth_token_issuance(client: AsyncClient) -> None:
    """Verify that clients can obtain a valid HMAC-SHA256 JWT access token."""
    client_id = str(uuid4())
    payload = {
        "client_id": client_id,
        "role": "financial_analyst",
        "secret": "valid-secret-key",
    }
    resp = await client.post(f"{V1}/auth/token", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in_seconds"] == 1800
    assert len(data["access_token"].split(".")) == 3  # Standard 3-part JWT header.payload.signature


@pytest.mark.api
async def test_auth_token_rejected_on_short_secret(client: AsyncClient) -> None:
    """Verify that inadequate secret lengths are rejected with 422/401."""
    payload = {
        "client_id": str(uuid4()),
        "role": "analyst",
        "secret": "abc",  # less than min_length 4
    }
    resp = await client.post(f"{V1}/auth/token", json=payload)
    assert resp.status_code == 422


@pytest.mark.api
async def test_fraud_prediction_with_shap_explainability(client: AsyncClient) -> None:
    """Verify that fraud prediction with explain=True returns top SHAP feature contributions."""
    features = {
        "Time": 3600.0,
        "Amount": 48500.0,
        "V1": -3.5,
        "V2": 2.8,
        "V14": -5.2,
    }
    resp = await client.post(
        f"{V1}/fraud/predict",
        json={"transaction_id": "txn_shap_001", "features": features, "explain": True},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["transaction_id"] == "txn_shap_001"
    assert "top_shap_contributions" in data
    if data["top_shap_contributions"] is not None:
        assert isinstance(data["top_shap_contributions"], dict)
        assert len(data["top_shap_contributions"]) <= 5
