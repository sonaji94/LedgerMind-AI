"""
LedgerMind AI — FastAPI Application Factory

Production-grade setup including:
  - API versioning under /api/v1/
  - Structured JSON logging (structlog)
  - X-Request-ID correlation middleware
  - Global exception handlers (no raw stack traces leaked)
  - Deep /health endpoint (DB ping + ML model status)
  - CORS configuration from environment settings
  - ML model preloading in lifespan
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.accounts.router import router as accounts_router
from app.ai_assistant.router import router as ai_router
from app.analytics.router import router as analytics_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import request_id_middleware
from app.fraud.router import router as fraud_router
from app.fraud.service import FraudInferenceService
from app.transactions.router import router as transfers_router

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    configure_logging()
    settings = get_settings()

    logger.info("startup", app=settings.app_name, env=settings.app_env)

    # Validate DB connectivity at startup
    try:
        from sqlalchemy import text

        from app.core.database import engine
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("database_connected", host=settings.postgres_host, db=settings.postgres_db)
    except Exception as exc:
        # In development, the DB may not be running — log but don't crash
        logger.warning("database_unavailable", error=str(exc))

    # Load ML models into memory
    FraudInferenceService.load()
    if FraudInferenceService.is_ready():
        logger.info("ml_model_loaded", model=FraudInferenceService.get_info()["model_type"])
    else:
        logger.warning("ml_model_not_found", hint="Run scripts/train_baseline.py first")

    yield

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
    app.include_router(accounts_router, prefix=API_PREFIX)
    app.include_router(transfers_router, prefix=API_PREFIX)
    app.include_router(fraud_router, prefix=API_PREFIX)
    app.include_router(analytics_router, prefix=API_PREFIX)
    app.include_router(ai_router, prefix=API_PREFIX)

    # ── Deep Health Check ─────────────────────────────────────────
    @app.get("/health", tags=["System"], summary="Deep health check")
    async def health_check() -> dict[str, object]:
        """
        Returns the operational status of the API, database connection,
        and ML fraud model. Safe for load-balancer probes.
        """
        db_ok = False
        db_error: str | None = None
        try:
            from sqlalchemy import text

            from app.core.database import engine
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            db_ok = True
        except Exception as exc:
            db_error = str(exc)

        ml_info = FraudInferenceService.get_info()
        overall = "healthy" if db_ok and FraudInferenceService.is_ready() else "degraded"

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
