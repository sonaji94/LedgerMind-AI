"""
LedgerMind AI — Backend Application Entry Point.
Delegates to app.main:app
"""
from app.main import app, create_app

__all__ = ["app", "create_app"]
