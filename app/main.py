"""
LedgerMind AI — FastAPI Application Factory

Creates and configures the FastAPI application instance,
registers routers, middleware, and lifecycle events.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    # ── Startup ──────────────────────────────────────────────
    settings = get_settings()
    print(f"🚀 Starting {settings.app_name} [{settings.app_env}]")
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
    # Registered progressively as modules are built:
    #   app.include_router(accounts.router, prefix="/accounts", tags=["Accounts"])
    #   app.include_router(transfers.router, prefix="/transfers", tags=["Transfers"])
    #   app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
    #   app.include_router(fraud.router, prefix="/fraud", tags=["Fraud Detection"])
    #   app.include_router(ai_assistant.router, prefix="/ai", tags=["AI Assistant"])

    # ── Health Check ─────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check() -> dict:
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": "0.1.0",
            "environment": settings.app_env,
        }

    return app


# ── Application Instance ─────────────────────────────────────────
app = create_app()
