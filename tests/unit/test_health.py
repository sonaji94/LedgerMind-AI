"""Smoke test — verify the application starts and the health endpoint works."""

import pytest
from httpx import AsyncClient


@pytest.mark.unit
async def test_health_endpoint(client: AsyncClient) -> None:
    """GET /health should return 200 with status=healthy."""
    response = await client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "LedgerMind AI"
    assert data["version"] == "0.1.0"

@pytest.mark.unit
async def test_health_environment_field(client: AsyncClient):
    response = await client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["environment"] == "development"
