"""
LedgerMind AI — FastAPI Application Factory

Creates and configures the FastAPI application instance,
registers routers, middleware, and lifecycle events.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.accounts.router import router as accounts_router
from app.ai_assistant.router import router as ai_router
from app.analytics.router import router as analytics_router
from app.core.config import get_settings
from app.fraud.router import router as fraud_router
from app.fraud.service import FraudInferenceService
from app.transactions.router import router as transfers_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    # ── Startup ──────────────────────────────────────────────
    settings = get_settings()
    print(f"🚀 Starting {settings.app_name} [{settings.app_env}]")
    FraudInferenceService.load()
    if FraudInferenceService.is_ready():
        print("🧠 Loaded Fraud Detection Model & Preprocessor into memory.")
    yield
    # ── Shutdown ─────────────────────────────────────────────
    print(f"🛑 Shutting down {settings.app_name}")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "AI-Powered Financial Analytics, Fraud Detection, "
            "and Intelligent Reporting Platform"
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────
    app.include_router(accounts_router)
    app.include_router(transfers_router)
    app.include_router(fraud_router)
    app.include_router(analytics_router)
    app.include_router(ai_router)

    # ── Health Check ─────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, str]:
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": "0.1.0",
            "environment": settings.app_env,
        }

    return app


# ── Application Instance ─────────────────────────────────────────
app = create_app()
