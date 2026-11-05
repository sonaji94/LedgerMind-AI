"""
LedgerMind AI — Request Middleware

Provides:
  1. X-Request-ID correlation header (injected on every response).
  2. Structured access log per request (method, path, status, duration).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Assign a unique X-Request-ID to every inbound request.

    If the client already supplies one, it is preserved; otherwise a UUID4 is
    generated. The ID is attached to the structlog context so every log line
    emitted during that request carries the same correlation key.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    # Bind to structlog context-vars so all loggers in this request inherit it
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request_handled",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response
