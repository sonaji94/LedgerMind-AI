"""
LedgerMind AI — Core Configuration

Loads settings from environment variables with Pydantic Settings.
All secrets and environment-specific values live in .env (never committed).

Production safety rules enforced at startup:
  - SECRET_KEY must be changed from the default placeholder.
  - DEBUG must be False in production.
  - CORS origins must be explicitly set in production.
"""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_DEFAULTS = {
    "change-this-to-a-random-secret-key",
    "secret",
    "changeme",
    "",
}


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────
    app_name: str = "LedgerMind AI"
    app_env: Literal["development", "testing", "staging", "production"] = "development"
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Server ────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── CORS ──────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.
    # Defaults to permissive for local development only.
    cors_origins: str = "*"

    # ── PostgreSQL ────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "ledgermind"
    postgres_password: str = "changeme"
    postgres_db: str = "ledgermind_db"

    @property
    def database_url(self) -> str:
        """Async PostgreSQL connection string for SQLAlchemy."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        """Sync PostgreSQL connection string (used by Alembic migrations)."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def allowed_origins(self) -> list[str]:
        """Parse CORS_ORIGINS into a list."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ── Security ──────────────────────────────────────────────────
    secret_key: str = "change-this-to-a-random-secret-key"
    access_token_expire_minutes: int = 30

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Block insecure placeholder keys in non-development environments."""
        # Validation at the field level — environment check done in model_validator
        return v

    @model_validator(mode="after")
    def enforce_production_security(self) -> "Settings":
        """Raise at startup if production is configured with unsafe defaults."""
        if self.app_env == "production":
            if self.secret_key in _INSECURE_DEFAULTS or len(self.secret_key) < 32:
                raise ValueError(
                    "SECRET_KEY must be a securely generated random string of "
                    "at least 32 characters in production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            if self.debug:
                raise ValueError("DEBUG must be False in production.")
            if self.cors_origins.strip() == "*":
                raise ValueError(
                    "CORS_ORIGINS must list explicit allowed origins in production, "
                    "not the wildcard '*'."
                )
        return self

    # ── ML ────────────────────────────────────────────────────────
    model_path: str = "ml/models/fraud_model.joblib"
    prediction_threshold: float = 0.5


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env once)."""
    return Settings()
