"""Persistent custom ticker list — user-added tickers stored in .cache/custom_tickers.json."""
from __future__ import annotations

import json
from pathlib import Path

_STORE = Path(".cache/custom_tickers.json")

ALL_SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary",
    "Consumer Staples", "Communication Services", "Industrials",
    "Energy", "Materials", "Utilities", "Real Estate", "Custom",
]


def load_custom() -> dict[str, str]:
    """Return {ticker: sector} for all user-added tickers."""
    if not _STORE.exists():
        return {}
    try:
        return json.loads(_STORE.read_text())
    except Exception:
        return {}


def save_custom(d: dict[str, str]) -> None:
    _STORE.parent.mkdir(exist_ok=True)
    _STORE.write_text(json.dumps(d, indent=2))


def add_ticker(ticker: str, sector: str = "Custom") -> None:
    d = load_custom()
    d[ticker.upper().strip()] = sector
    save_custom(d)


def remove_ticker(ticker: str) -> None:
    d = load_custom()
    d.pop(ticker.upper().strip(), None)
    save_custom(d)
