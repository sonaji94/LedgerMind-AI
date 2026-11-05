"""
Machine Learning Fraud Detection API Router for LedgerMind AI.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.schemas import (
    FraudPredictionRequest,
    FraudPredictionResponse,
    ModelInfoResponse,
)
from app.fraud.service import FraudInferenceService

router = APIRouter(prefix="/fraud", tags=["Fraud Detection"])


@router.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(payload: FraudPredictionRequest) -> FraudPredictionResponse:
    """Score a transaction's features in real-time and return fraud risk probability with SHAP attribution."""
    risk_score, is_suspicious, latency_ms, shap_contribs = FraudInferenceService.predict(
        payload.features,
        explain=payload.explain,
    )
    info = FraudInferenceService.get_info()

    return FraudPredictionResponse(
        transaction_id=payload.transaction_id,
        risk_score=round(risk_score, 4),
        is_suspicious=is_suspicious,
        threshold_applied=info["optimal_threshold"],
        model_type=info["model_type"],
        latency_ms=round(latency_ms, 2),
        top_shap_contributions=shap_contribs,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    """Return active model metadata and optimal calibrated decision threshold."""
    info = FraudInferenceService.get_info()
    return ModelInfoResponse(**info)
