"""Lightweight JSON persistence for saved multi-factor strategies.

Strategies are stored in .cache/strategies.json.  Each strategy captures:
  - name        : display name chosen by the user
  - factors     : {factor_name: raw_weight} mapping
  - rebal_freq  : rebalance frequency string (ME / W-FRI / QE)
  - created_at  : ISO-8601 timestamp string
"""
from __future__ import annotations

import json
import datetime
from pathlib import Path

_STORE_PATH = Path(__file__).parent.parent.parent / ".cache" / "strategies.json"


def _load_raw() -> list[dict]:
    if not _STORE_PATH.exists():
        return []
    try:
        return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_raw(strategies: list[dict]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(
        json.dumps(strategies, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_strategies() -> list[dict]:
    """Return all saved strategies, newest first."""
    return sorted(_load_raw(), key=lambda s: s.get("created_at", ""), reverse=True)


def save_strategy(
    name: str,
    factors: dict[str, float],   # factor_name → raw weight
    rebal_freq: str,
) -> dict:
    """Upsert a strategy by name (replaces existing with the same name).

    Returns the saved strategy dict.
    """
    strategies = _load_raw()
    # Remove any existing strategy with the same name
    strategies = [s for s in strategies if s.get("name") != name]
    entry = {
        "name": name,
        "factors": factors,
        "rebal_freq": rebal_freq,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    strategies.append(entry)
    _save_raw(strategies)
    return entry


def delete_strategy(name: str) -> None:
    """Remove a strategy by name (no-op if not found)."""
    strategies = [s for s in _load_raw() if s.get("name") != name]
    _save_raw(strategies)
