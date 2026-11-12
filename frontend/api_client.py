"""
LedgerMind AI — Frontend API Client.

Decouples the Streamlit presentation layer from backend operations.
Communicates strictly via REST endpoints with JWT authorization.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")
BASE_HOST = DEFAULT_BACKEND_URL.replace("/api/v1", "")


class BackendAPIClient:
    """Synchronous & async HTTP client for FastAPI backend services."""

    def __init__(self, base_url: str = DEFAULT_BACKEND_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.token: str | None = None
        self._client = httpx.Client(base_url=self.base_url, timeout=12.0)

    def set_token(self, token: str) -> None:
        self.token = token
        self._client.headers["Authorization"] = f"Bearer {token}"

    def check_health(self) -> dict[str, Any]:
        """Check backend readiness."""
        try:
            res = httpx.get(f"{BASE_HOST}/readyz", timeout=4.0)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass

        try:
            res = httpx.get(f"{BASE_HOST}/health", timeout=4.0)
            return res.json() if res.status_code == 200 else {"status": "degraded"}
        except Exception as e:
            return {"status": "offline", "error": str(e)}

    def ingest_transaction(self, payload: dict[str, Any], idempotency_key: str | None = None) -> dict[str, Any]:
        """Submit live transaction for fraud evaluation and atomic settlement."""
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
        res = self._client.post("/transactions/ingest", json=payload, headers=headers)
        return res.json()

    def execute_transfer(
        self,
        source_account_id: str,
        destination_account_id: str,
        amount_minor_units: int,
        description: str,
    ) -> dict[str, Any]:
        """Execute double-entry ledger transfer."""
        payload = {
            "source_account_id": source_account_id,
            "destination_account_id": destination_account_id,
            "amount_minor_units": amount_minor_units,
            "currency": "EUR",
            "description": description,
        }
        res = self._client.post("/transfers", json=payload)
        return res.json()

    def freeze_account(self, account_id: str, justification_reason: str) -> dict[str, Any]:
        """Request compliance account freeze with audit trail."""
        payload = {"account_id": account_id, "justification_reason": justification_reason}
        res = self._client.post("/compliance/freeze", json=payload)
        return res.json()

    def dispatch_2fa(self, account_id: str, incident_id: str, reason: str) -> dict[str, Any]:
        """Trigger step-up 2FA challenge."""
        payload = {"account_id": account_id, "incident_id": incident_id, "justification_reason": reason}
        res = self._client.post("/compliance/step-up-2fa", json=payload)
        return res.json()

    def generate_sar(
        self,
        incident_id: str,
        account_id: str,
        amount_eur: float,
        fraud_probability: float,
        detected_typology: str,
        red_flag_indicators: list[str],
        apply_pii_masking: bool = True,
    ) -> dict[str, Any]:
        """Generate FinCEN SAR report with PII masking."""
        payload = {
            "incident_id": incident_id,
            "account_id": account_id,
            "amount_eur": amount_eur,
            "fraud_probability": fraud_probability,
            "detected_typology": detected_typology,
            "red_flag_indicators": red_flag_indicators,
            "apply_pii_masking": apply_pii_masking,
            "justification_reason": "Statutory reporting of high-risk outlier pattern",
        }
        res = self._client.post("/compliance/sar-generate", json=payload)
        return res.json()

    def get_audit_logs(self, limit: int = 25) -> list[dict[str, Any]]:
        """Retrieve operator compliance audit trail."""
        try:
            res = self._client.get(f"/compliance/audit-logs?limit={limit}")
            return res.json() if res.status_code == 200 else []
        except Exception:
            return []


# Global singleton client
api_client = BackendAPIClient()
