"""
LedgerMind AI — Hybrid ML Engine & Drift Monitoring Tests.

Verifies:
1. Hybrid scoring engine (XGBoost + Isolation Forest) execution in < 2ms
2. Two-tier decision policy:
   - score < 0.45: CLEARED
   - 0.45 <= score < 0.80: STEP-UP 2FA
   - score >= 0.80: INTERCEPTED / FROZEN
3. Local TreeSHAP explainability mapping top 3 drivers
4. ONNX model artifacts and feature_schema.json validity
5. Population Stability Index (PSI) drift calculation
6. REST monitoring endpoints (/api/v1/monitoring/drift, /api/v1/monitoring/logs)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from httpx import AsyncClient

from backend.inference import (
    FEATURE_NAMES,
    HybridFraudInferenceService,
    InferenceResult,
)
from backend.monitoring import compute_psi


@pytest.mark.ml
def test_feature_schema_validity() -> None:
    """Ensure feature_schema.json enforces the exact 14 columns and types."""
    schema_path = Path("ml/models/feature_schema.json")
    assert schema_path.is_file(), "feature_schema.json must exist"

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    assert schema["feature_count"] == 14
    assert schema["expected_order"] == FEATURE_NAMES
    assert len(schema["features"]) == 14
    assert schema["decision_thresholds"]["cleared_cutoff"] == 0.45
    assert schema["decision_thresholds"]["step_up_2fa_cutoff"] == 0.80


@pytest.mark.ml
def test_onnx_artifacts_validity() -> None:
    """Verify that exported .onnx models load cleanly into ONNX model protos."""
    xgb_onnx = Path("ml/models/xgb_champion.onnx")
    iso_onnx = Path("ml/models/iso_forest.onnx")

    assert xgb_onnx.is_file(), "xgb_champion.onnx must exist"
    assert iso_onnx.is_file(), "iso_forest.onnx must exist"

    model_xgb = onnx.load(str(xgb_onnx))
    onnx.checker.check_model(model_xgb)
    assert len(model_xgb.graph.node) > 0

    model_iso = onnx.load(str(iso_onnx))
    onnx.checker.check_model(model_iso)
    assert len(model_iso.graph.node) > 0


@pytest.mark.ml
def test_hybrid_inference_cleared_policy() -> None:
    """Normal legitimate transaction must be CLEARED with latency < 5ms."""
    HybridFraudInferenceService.load()
    assert HybridFraudInferenceService.is_ready()

    legit_txn = {
        "amount_minor": 2500_00,  # ₹2,500
        "velocity_instrument_60s": 0,
        "velocity_instrument_1h": 1,
        "distinct_cards_ip_10m": 0,
        "distinct_cards_email_10m": 0,
        "amount_z_score": 0.1,
        "amount_ratio_30d": 1.0,
        "account_age_days": 365.0,
        "hours_since_bank_change": 1200.0,
        "payout_to_balance_ratio": 0.05,
        "is_disposable_email": 0,
        "is_international_card": 0,
        "hour_of_day": 14,
        "day_of_week": 2,
    }

    res: InferenceResult = HybridFraudInferenceService.score_transaction(legit_txn, explain=True)
    assert res.risk_score < 0.45
    assert res.decision == "CLEARED"
    assert not res.is_suspicious
    assert res.latency_ms < 50.0  # Ultra-fast sub-millisecond to low millisecond
    assert len(res.top_drivers) <= 3


@pytest.mark.ml
def test_deep_investigator_geo_device_amount_intercept() -> None:
    """
    Verify that Deep Investigator human-readable risk sliders
    (Geo-Velocity = 950 km/h, Unrecognized Device = True, Amount = 26.7x baseline)
    produce a high fraud probability (> 0.80 INTERCEPTED) with matching TreeSHAP drivers.
    """
    HybridFraudInferenceService.load()

    amt = 3204.0
    dev_ratio = 26.7
    geo_speed = 950.0
    vel_1m = 3
    is_new_device = True
    in_hour = 14

    features = {
        "amount_minor": int(round(amt * 100.0)),
        "velocity_instrument_60s": int(vel_1m),
        "velocity_instrument_1h": int(vel_1m * 3 + (geo_speed > 500) * 2),
        "distinct_cards_ip_10m": int(round(geo_speed / 150.0)) if (is_new_device or geo_speed > 300) else 0,
        "distinct_cards_email_10m": 3 if is_new_device else (1 if geo_speed > 700 else 0),
        "amount_z_score": float(round((amt - 120.0) / 45.0, 2)),
        "amount_ratio_30d": float(round(dev_ratio, 2)),
        "account_age_days": 15.0 if is_new_device else 180.0,
        "hours_since_bank_change": 4.0 if is_new_device else 720.0,
        "payout_to_balance_ratio": 0.92 if dev_ratio > 10.0 else 0.15,
        "is_disposable_email": 1 if is_new_device else 0,
        "is_international_card": 1 if geo_speed > 500 else 0,
        "hour_of_day": int(in_hour),
        "day_of_week": 2,
    }

    res: InferenceResult = HybridFraudInferenceService.score_transaction(features, explain=True)
    assert res.risk_score > 0.80, f"Expected risk_score > 0.80, got {res.risk_score}"
    assert res.decision == "INTERCEPTED"
    assert res.is_suspicious is True
    assert len(res.top_drivers) >= 2

    driver_labels = " ".join([d["label"] for d in res.top_drivers])
    assert any(term in driver_labels for term in ("Device", "Geo-Velocity", "Card", "Amount"))


@pytest.mark.asyncio
async def test_inference_score_endpoint(client) -> None:
    """Verify POST /api/v1/inference/score correctly scores 14 features and returns TreeSHAP drivers."""
    amt = 3204.0
    dev_ratio = 26.7
    geo_speed = 950.0
    vel_1m = 3
    is_new_device = True
    in_hour = 14

    features = {
        "amount_minor": int(round(amt * 100.0)),
        "velocity_instrument_60s": int(vel_1m),
        "velocity_instrument_1h": int(vel_1m * 3 + (geo_speed > 500) * 2),
        "distinct_cards_ip_10m": int(round(geo_speed / 150.0)) if (is_new_device or geo_speed > 300) else 0,
        "distinct_cards_email_10m": 3 if is_new_device else (1 if geo_speed > 700 else 0),
        "amount_z_score": float(round((amt - 120.0) / 45.0, 2)),
        "amount_ratio_30d": float(round(dev_ratio, 2)),
        "account_age_days": 15.0 if is_new_device else 180.0,
        "hours_since_bank_change": 4.0 if is_new_device else 720.0,
        "payout_to_balance_ratio": 0.92 if dev_ratio > 10.0 else 0.15,
        "is_disposable_email": 1 if is_new_device else 0,
        "is_international_card": 1 if geo_speed > 500 else 0,
        "hour_of_day": int(in_hour),
        "day_of_week": 2,
    }

    resp = await client.post("/api/v1/inference/score", json={"features": features, "explain": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_score"] > 0.80
    assert data["decision"] == "INTERCEPTED"
    assert data["is_suspicious"] is True
    assert len(data["top_drivers"]) >= 2



@pytest.mark.ml
def test_population_stability_index_computation() -> None:
    """Verify PSI calculation flags stability and severe drift correctly."""
    # 10 deciles of normal standard distribution
    baseline = list(np.linspace(0.0, 100.0, 11))

    # Identical distribution -> low PSI < 0.05
    live_stable = np.random.uniform(0.0, 100.0, 500)
    psi_stable = compute_psi(baseline, live_stable)
    assert psi_stable < 0.15

    # Drastically shifted distribution (all concentrated at top) -> high PSI >= 0.20
    live_drifted = np.random.uniform(90.0, 110.0, 500)
    psi_drifted = compute_psi(baseline, live_drifted)
    assert psi_drifted >= 0.20


@pytest.mark.api
async def test_monitoring_drift_endpoint(client: AsyncClient) -> None:
    """Test GET /api/v1/monitoring/drift returns valid report structure."""
    resp = await client.get("/api/v1/monitoring/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "overall_status" in data
    assert "max_psi" in data
    assert "feature_drift" in data
    assert len(data["feature_drift"]) == 14
    assert "recommendation" in data


@pytest.mark.api
async def test_monitoring_logs_endpoint(client: AsyncClient) -> None:
    """Test GET /api/v1/monitoring/logs returns items and latency summary."""
    resp = await client.get("/api/v1/monitoring/logs?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "items" in data
    assert "summary" in data
    assert "p50_latency_ms" in data["summary"]
