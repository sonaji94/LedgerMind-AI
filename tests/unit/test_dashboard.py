"""
Unit tests for Streamlit dashboard helper functions and ledger state reader.
"""

from pathlib import Path

import pytest

from app.dashboard import get_live_ledger_data


@pytest.mark.unit
def test_get_live_ledger_data_structure() -> None:
    """Verify that get_live_ledger_data returns a valid dictionary structure."""
    stats = get_live_ledger_data()

    assert isinstance(stats, dict)
    assert "has_data" in stats
    assert "total_accounts" in stats
    assert "total_transactions" in stats
    assert "settled_volume" in stats
    assert "avg_transaction" in stats
    assert "debits_sum" in stats
    assert "credits_sum" in stats
    assert "is_balanced" in stats
    assert "recent_txns" in stats
    assert "account_projections" in stats

    # Invariant assertion
    assert stats["debits_sum"] >= 0.0
    assert stats["credits_sum"] >= 0.0
    if stats["is_balanced"]:
        assert stats["debits_sum"] == stats["credits_sum"]


@pytest.mark.unit
def test_get_live_ledger_data_with_missing_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fallback behavior when database file is absent."""
    import app.dashboard as db_mod

    fake_db = tmp_path / "nonexistent.db"
    monkeypatch.setattr(db_mod, "DB_PATH", fake_db)

    stats = db_mod.get_live_ledger_data()
    assert stats["has_data"] is False
    assert stats["total_accounts"] > 0
    assert stats["is_balanced"] is True
