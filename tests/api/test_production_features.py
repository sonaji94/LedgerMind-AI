"""
Production Feature Tests for LedgerMind AI.

Tests:
1. Kubernetes Liveness (/healthz) & Readiness (/readyz) probes
2. Real-time transaction ingestion with Idempotency-Key deduplication
3. Compliance actions (Account Freeze, Step-up 2FA, SAR generation) with RBAC
4. Immutable compliance audit log recording
5. PII masking and tokenization engine
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.compliance.pii_masker import PIIMasker
from app.core.security import UserRole, create_access_token
from app.main import app


@pytest.fixture
def admin_token() -> str:
    return create_access_token(subject=str(uuid4()), role=UserRole.ADMIN.value)


@pytest.fixture
def compliance_token() -> str:
    return create_access_token(subject=str(uuid4()), role=UserRole.COMPLIANCE_OFFICER.value)


@pytest.fixture
def analyst_token() -> str:
    return create_access_token(subject=str(uuid4()), role=UserRole.ANALYST.value)


@pytest.mark.api
@pytest.mark.asyncio
async def test_kubernetes_probes() -> None:
    """Verify /healthz liveness and /readyz deep readiness endpoints."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Liveness
        res_live = await client.get("/healthz")
        assert res_live.status_code == 200
        assert res_live.json()["status"] == "ok"

        # Readiness
        res_ready = await client.get("/readyz")
        assert res_ready.status_code == 200
        data = res_ready.json()
        assert data["status"] == "ready"
        assert data["database"]["connected"] is True
        assert data["ml_model"]["status"] == "ready"


@pytest.mark.api
@pytest.mark.asyncio
async def test_transaction_ingestion_and_idempotency() -> None:
    """Verify live transaction ingestion, risk scoring, and idempotency replay."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        tx_id = f"test-ingest-{uuid4().hex[:8]}"
        payload = {
            "transaction_id": tx_id,
            "source_account_id": str(uuid4()),
            "destination_account_id": str(uuid4()),
            "amount_minor_units": 2500,
            "currency": "EUR",
            "channel": "POS_CHIP",
            "is_new_device": False,
        }

        # 1. First Ingestion
        res1 = await client.post(
            "/api/v1/transactions/ingest",
            json=payload,
            headers={"Idempotency-Key": tx_id},
        )
        assert res1.status_code in (200, 201)
        data1 = res1.json()
        assert data1["transaction_id"] == tx_id
        assert "risk_score" in data1
        assert "decision" in data1

        # 2. Idempotent Retry (Must return exact cached outcome)
        res2 = await client.post(
            "/api/v1/transactions/ingest",
            json=payload,
            headers={"Idempotency-Key": tx_id},
        )
        assert res2.status_code in (200, 201)
        data2 = res2.json()
        assert data2["transaction_id"] == tx_id
        assert data2.get("cached_replay") is True


@pytest.mark.api
@pytest.mark.asyncio
async def test_compliance_freeze_and_audit_log(compliance_token: str, analyst_token: str) -> None:
    """Verify RBAC on account freeze and audit log trail creation."""
    target_account = str(uuid4())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Analyst cannot freeze account (Forbidden 403)
        res_forbidden = await client.post(
            "/api/v1/compliance/freeze",
            json={"account_id": target_account, "justification_reason": "Suspicious activity detected"},
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert res_forbidden.status_code == 403

        # 2. Provision account first via accounts API
        create_acc_res = await client.post(
            "/api/v1/accounts",
            json={"owner_id": str(uuid4()), "currency": "EUR", "initial_deposit_minor_units": 10000},
        )
        acc_id = create_acc_res.json()["id"]

        # 3. Compliance Officer can freeze account
        res_ok = await client.post(
            "/api/v1/compliance/freeze",
            json={"account_id": acc_id, "justification_reason": "Confirmed mule account pattern"},
            headers={"Authorization": f"Bearer {compliance_token}"},
        )
        assert res_ok.status_code == 200
        assert res_ok.json()["action"] == "ACCOUNT_FREEZE"

        # 4. Check Audit Logs
        res_logs = await client.get(
            "/api/v1/compliance/audit-logs",
            headers={"Authorization": f"Bearer {compliance_token}"},
        )
        assert res_logs.status_code == 200
        logs = res_logs.json()
        assert len(logs) > 0
        latest = logs[0]
        assert latest["action"] == "ACCOUNT_FREEZE"
        assert latest["target_id"] == acc_id


@pytest.mark.api
@pytest.mark.asyncio
async def test_sar_generator_with_pii_masking(compliance_token: str) -> None:
    """Verify FinCEN SAR generation with PII redaction."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        sar_payload = {
            "incident_id": "INC-9901",
            "account_id": "acc-9b12-treasury-secret",
            "amount_eur": 45000.0,
            "fraud_probability": 0.985,
            "detected_typology": "Coordinated Structuring Anomaly",
            "red_flag_indicators": [
                "Transferred from IP 192.168.1.100 to PAN 4111-2222-3333-4444",
                "Contact email user@suspiciousdomain.com",
            ],
            "apply_pii_masking": True,
            "justification_reason": "Statutory SAR threshold breach",
        }

        res = await client.post(
            "/api/v1/compliance/sar-generate",
            json=sar_payload,
            headers={"Authorization": f"Bearer {compliance_token}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "SAR-" in data["document_control_number"]
        report = data["report_markdown"]
        # Assert PII was scrubbed
        assert "192.168.1.100" not in report
        assert "4111-2222-3333-4444" not in report
        assert "user@suspiciousdomain.com" not in report
        assert "acc-9b12-treasury-secret" not in report


@pytest.mark.unit
def test_pii_masker_standalone() -> None:
    """Unit test for PIIMasker regex and tokenization rules."""
    sample = (
        "Account acc-corp-7721 sent €500 to 192.168.4.50 using card 4532 1122 3344 5566. "
        "Contact: admin@bank.com or +1 (555) 234-5678."
    )
    scrubbed, token_map = PIIMasker.scrub_text(sample)

    assert "192.168.4.50" not in scrubbed
    assert "4532 1122 3344 5566" not in scrubbed
    assert "admin@bank.com" not in scrubbed
    assert "acc-corp-7721" not in scrubbed
    assert "****-****-****-5566" in scrubbed
    assert "192.***.***.50" in scrubbed
    assert len(token_map) > 0
