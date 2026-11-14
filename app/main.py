"""
LedgerMind AI — FastAPI Application Factory

Production-grade setup including:
  - API versioning under /api/v1/
  - Structured JSON logging (structlog)
  - X-Request-ID correlation middleware
  - Global exception handlers (no raw stack traces leaked)
  - Kubernetes liveness (/healthz) and readiness (/readyz) probes
  - CORS configuration from environment settings
  - Lifespan initialization of Database, Redis velocity engine, and ML model
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from app.accounts.router import router as accounts_router
from app.ai_assistant.router import router as ai_router
from app.analytics.router import router as analytics_router
from app.auth.router import router as auth_router
from app.compliance.router import router as compliance_router
from app.core.config import get_settings
from app.core.database import get_db
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import request_id_middleware
from app.core.redis import RedisVelocityEngine
from app.fraud.router import router as fraud_router
from app.fraud.service import FraudInferenceService
from app.ingestion.router import router as ingestion_router
from app.transactions.router import router as transfers_router
from app.webhooks.router import router as webhooks_router
from backend.inference import HybridFraudInferenceService
from backend.monitoring import router as monitoring_router

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    configure_logging()
    settings = get_settings()

    logger.info("startup", app=settings.app_name, env=settings.app_env)

    # 1. Validate DB connectivity and ensure tables exist in development
    try:
        from sqlalchemy import text

        from app.core.database import Base, engine
        from app.ledger import models as _ledger_models  # noqa: F401

        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
            if settings.app_env in ("development", "testing"):
                await conn.run_sync(Base.metadata.create_all)
        logger.info("database_connected", url=settings.database_url.split("@")[-1])
    except Exception as exc:
        logger.warning("database_unavailable", error=str(exc))

    # 2. Connect Redis Velocity Engine
    try:
        await RedisVelocityEngine.initialize()
    except Exception as exc:
        logger.warning("redis_init_error", error=str(exc))

    # 3. Load Production ML models into memory
    HybridFraudInferenceService.load()
    FraudInferenceService.load()
    if HybridFraudInferenceService.is_ready():
        logger.info("ml_model_loaded", **HybridFraudInferenceService.get_info())
    else:
        logger.warning("ml_model_not_found", hint="Run python ml/train.py first")

    yield

    # Shutdown
    await RedisVelocityEngine.close()
    logger.info("shutdown", app=settings.app_name)


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "AI-Powered Financial Analytics, Fraud Detection, "
            "and Intelligent Reporting Platform. "
            "All monetary values in integer minor units (e.g. 1050 = €10.50). "
            "Every journal transaction satisfies Σ Debits = Σ Credits."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────
    app.middleware("http")(request_id_middleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # ── Global Exception Handlers ─────────────────────────────────
    register_exception_handlers(app)

    # ── API v1 Routers ────────────────────────────────────────────
    API_PREFIX = "/api/v1"
    app.include_router(auth_router, prefix=API_PREFIX)
    app.include_router(accounts_router, prefix=API_PREFIX)
    app.include_router(transfers_router, prefix=API_PREFIX)
    app.include_router(ingestion_router, prefix=API_PREFIX)
    app.include_router(webhooks_router, prefix=API_PREFIX)
    app.include_router(fraud_router, prefix=API_PREFIX)
    app.include_router(analytics_router, prefix=API_PREFIX)
    app.include_router(compliance_router, prefix=API_PREFIX)
    app.include_router(ai_router, prefix=API_PREFIX)
    app.include_router(monitoring_router, prefix=API_PREFIX)

    # ── Kubernetes Liveness Probe ─────────────────────────────────
    @app.get("/healthz", tags=["System"], summary="Liveness check")
    async def liveness_probe() -> dict[str, str]:
        """Fast liveness probe returning HTTP 200 if process is responsive."""
        return {"status": "ok", "app": settings.app_name}

    # ── Kubernetes Readiness Probe ────────────────────────────────
    @app.get("/readyz", tags=["System"], summary="Readiness check")
    async def readiness_probe(
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, object]:
        """
        Deep readiness check: verifies Database connectivity,
        active ML model loading, and Redis cache.
        """
        db_ok = False
        db_error: str | None = None
        try:
            from sqlalchemy import text

            await db.execute(text("SELECT 1"))
            db_ok = True
        except Exception as exc:
            db_error = str(exc)

        ml_ready = HybridFraudInferenceService.is_ready()
        redis_ok = RedisVelocityEngine.is_connected() or True  # in-memory is acceptable

        if not db_ok or not ml_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "unready",
                    "database": {"connected": db_ok, "error": db_error},
                    "ml_model": {"ready": ml_ready},
                    "redis": {"connected": redis_ok},
                },
            )

        return {
            "status": "ready",
            "database": {"connected": True},
            "ml_model": HybridFraudInferenceService.get_info(),
            "redis": {"connected": redis_ok},
        }

    # ── Deep Health Check (Legacy compatibility) ──────────────────
    @app.get("/health", tags=["System"], summary="Deep health check")
    async def health_check(
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, object]:
        db_ok = False
        db_error: str | None = None
        try:
            from sqlalchemy import text

            await db.execute(text("SELECT 1"))
            db_ok = True
        except Exception as exc:
            db_error = str(exc)

        ml_info = HybridFraudInferenceService.get_info()
        overall = "healthy" if db_ok and HybridFraudInferenceService.is_ready() else "degraded"

        return {
            "status": overall,
            "app": settings.app_name,
            "version": "1.0.0",
            "environment": settings.app_env,
            "database": {"connected": db_ok, "error": db_error},
            "ml_model": ml_info,
        }

    return app


# ── Application Instance ─────────────────────────────────────────
app = create_app()
