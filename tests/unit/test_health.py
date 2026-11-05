"""Smoke test — verify the application starts and the health endpoint works."""

import pytest
from httpx import AsyncClient


@pytest.mark.unit
async def test_health_endpoint(client: AsyncClient) -> None:
    """GET /health should return 200 with deep health metrics."""
    response = await client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["app"] == "LedgerMind AI"
    assert data["version"] == "1.0.0"
    assert "database" in data
    assert "ml_model" in data

@pytest.mark.unit
async def test_health_environment_field(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["environment"] in ("development", "testing")
