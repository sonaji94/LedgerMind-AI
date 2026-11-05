"""
LedgerMind AI — Global Exception Handlers

Registers FastAPI exception handlers that:
  1. Return clean JSON error envelopes (never raw stack traces).
  2. Log the full error context via structlog.
  3. Map domain errors to the correct HTTP status codes.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.ledger.domain import (
    AccountingInvariantError,
    AccountStatusError,
    CurrencyMismatchError,
    InsufficientFundsError,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def _error_envelope(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"status": status, "error": error, "detail": detail},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all global exception handlers to the FastAPI application."""

    @app.exception_handler(InsufficientFundsError)
    async def handle_insufficient_funds(
        request: Request, exc: InsufficientFundsError
    ) -> JSONResponse:
        logger.warning("insufficient_funds", detail=str(exc), path=request.url.path)
        return _error_envelope(422, "InsufficientFunds", str(exc))

    @app.exception_handler(CurrencyMismatchError)
    async def handle_currency_mismatch(
        request: Request, exc: CurrencyMismatchError
    ) -> JSONResponse:
        logger.warning("currency_mismatch", detail=str(exc), path=request.url.path)
        return _error_envelope(400, "CurrencyMismatch", str(exc))

    @app.exception_handler(AccountStatusError)
    async def handle_account_status(
        request: Request, exc: AccountStatusError
    ) -> JSONResponse:
        logger.warning("account_status_error", detail=str(exc), path=request.url.path)
        return _error_envelope(400, "AccountStatusError", str(exc))

    @app.exception_handler(AccountingInvariantError)
    async def handle_invariant(
        request: Request, exc: AccountingInvariantError
    ) -> JSONResponse:
        logger.error("ledger_invariant_violated", detail=str(exc), path=request.url.path)
        return _error_envelope(500, "LedgerInvariantViolation", str(exc))

    @app.exception_handler(404)
    async def handle_404(request: Request, exc: Exception) -> JSONResponse:
        return _error_envelope(404, "NotFound", f"Resource not found: {request.url.path}")

    @app.exception_handler(405)
    async def handle_405(request: Request, exc: Exception) -> JSONResponse:
        return _error_envelope(405, "MethodNotAllowed", f"Method {request.method} not allowed")

    @app.exception_handler(500)
    async def handle_500(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_server_error", path=request.url.path)
        return _error_envelope(500, "InternalServerError", "An unexpected error occurred.")
