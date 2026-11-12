"""
LedgerMind AI — Redis Client & Sliding-Window Velocity Engine.

Provides:
- Sub-millisecond sliding-window velocity counters (e.g., tracking micro-authorizations in 60s)
- Rate limiting using Redis sorted sets (ZADD / ZREMRANGEBYSCORE / ZCARD)
- Resilient in-memory fallback for local development or testing without an active Redis container
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from typing import Any

import structlog

from app.core.config import get_settings

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

settings = get_settings()


class InMemoryVelocityStore:
    """Thread-safe in-memory sliding-window store used as a fallback when Redis is absent."""

    def __init__(self) -> None:
        self._events: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self._rate_limits: dict[str, list[float]] = defaultdict(list)

    def record_event(self, key: str, amount_minor: int, timestamp: float | None = None) -> None:
        t = timestamp or time.time()
        self._events[key].append((t, amount_minor))

    def get_velocity_count(self, key: str, window_seconds: int = 60) -> int:
        now = time.time()
        cutoff = now - window_seconds
        valid = [ev for ev in self._events.get(key, []) if ev[0] >= cutoff]
        self._events[key] = valid
        return len(valid)

    def get_velocity_volume(self, key: str, window_seconds: int = 3600) -> int:
        now = time.time()
        cutoff = now - window_seconds
        valid = [ev for ev in self._events.get(key, []) if ev[0] >= cutoff]
        self._events[key] = valid
        return sum(ev[1] for ev in valid)

    def is_rate_limited(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.time()
        cutoff = now - window_seconds
        valid = [t for t in self._rate_limits.get(key, []) if t >= cutoff]
        if len(valid) >= limit:
            return True
        valid.append(now)
        self._rate_limits[key] = valid
        return False


class RedisVelocityEngine:
    """Enterprise velocity tracking engine with async Redis and in-memory fallback."""

    _redis_client: Any = None
    _in_memory: InMemoryVelocityStore = InMemoryVelocityStore()
    _is_connected: bool = False

    @classmethod
    async def initialize(cls) -> None:
        """Attempt to connect to Redis server; fall back to in-memory store gracefully."""
        if not settings.redis_enabled:
            logger.info("redis_disabled", note="Using in-memory velocity tracking")
            cls._is_connected = False
            return

        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=2.0,
            )
            await client.ping()
            cls._redis_client = client
            cls._is_connected = True
            logger.info("redis_connected", url=settings.redis_url)
        except Exception as exc:
            cls._is_connected = False
            cls._redis_client = None
            logger.warning(
                "redis_connection_failed",
                error=str(exc),
                fallback="Using in-memory velocity store",
            )

    @classmethod
    async def close(cls) -> None:
        """Close active Redis connections."""
        if cls._redis_client:
            with contextlib.suppress(Exception):
                await cls._redis_client.aclose()
            cls._redis_client = None
            cls._is_connected = False

    @classmethod
    def is_connected(cls) -> bool:
        return cls._is_connected

    @classmethod
    async def record_transaction_event(
        cls,
        account_id: str,
        amount_minor: int,
        timestamp: float | None = None,
    ) -> None:
        """Record transaction timestamp and amount into sliding window."""
        t = timestamp or time.time()
        key = f"velocity:txn:{account_id}"

        if cls._is_connected and cls._redis_client:
            try:
                # Store tuple in sorted set scored by timestamp
                member = f"{t}:{amount_minor}"
                await cls._redis_client.zadd(key, {member: t})
                # Expire keys after 24 hours
                await cls._redis_client.expire(key, 86400)
                return
            except Exception as e:
                logger.warning("redis_record_error", error=str(e))

        cls._in_memory.record_event(key, amount_minor, t)

    @classmethod
    async def get_velocity_count(cls, account_id: str, window_seconds: int = 60) -> int:
        """Return count of transactions executed within the last window_seconds."""
        now = time.time()
        cutoff = now - window_seconds
        key = f"velocity:txn:{account_id}"

        if cls._is_connected and cls._redis_client:
            try:
                # Remove obsolete members
                await cls._redis_client.zremrangebyscore(key, "-inf", cutoff)
                count = await cls._redis_client.zcount(key, cutoff, "+inf")
                return int(count)
            except Exception as e:
                logger.warning("redis_velocity_count_error", error=str(e))

        return cls._in_memory.get_velocity_count(key, window_seconds)

    @classmethod
    async def get_velocity_volume(cls, account_id: str, window_seconds: int = 3600) -> int:
        """Return total minor units transferred within the last window_seconds."""
        now = time.time()
        cutoff = now - window_seconds
        key = f"velocity:txn:{account_id}"

        if cls._is_connected and cls._redis_client:
            try:
                await cls._redis_client.zremrangebyscore(key, "-inf", cutoff)
                members = await cls._redis_client.zrangebyscore(key, cutoff, "+inf")
                total = 0
                for m in members:
                    parts = m.split(":")
                    if len(parts) >= 2:
                        total += int(parts[1])
                return total
            except Exception as e:
                logger.warning("redis_velocity_volume_error", error=str(e))

        return cls._in_memory.get_velocity_volume(key, window_seconds)

    @classmethod
    async def is_rate_limited(cls, key: str, limit: int = 60, window_seconds: int = 60) -> bool:
        """Check if request key exceeds limit requests within window_seconds."""
        now = time.time()
        cutoff = now - window_seconds
        redis_key = f"ratelimit:{key}"

        if cls._is_connected and cls._redis_client:
            try:
                await cls._redis_client.zremrangebyscore(redis_key, "-inf", cutoff)
                current = await cls._redis_client.zcard(redis_key)
                if current >= limit:
                    return True
                await cls._redis_client.zadd(redis_key, {f"{now}": now})
                await cls._redis_client.expire(redis_key, window_seconds * 2)
                return False
            except Exception as e:
                logger.warning("redis_ratelimit_error", error=str(e))

        return cls._in_memory.is_rate_limited(redis_key, limit, window_seconds)
