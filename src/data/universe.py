"""Static large-cap universe with sector classification."""

# 75 S&P 500 large-caps across all 11 GICS sectors.
# SPY is included as market benchmark for beta computation but excluded
# from the cross-sectional factor universe.

SECTOR_MAP: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "AVGO", "AMD", "QCOM", "ADBE", "CRM",
        "CSCO", "IBM", "TXN", "INTU", "NOW",
    ],
    "Healthcare": [
        "JNJ", "UNH", "LLY", "ABBV", "MRK",
        "TMO", "ABT", "DHR", "AMGN", "ISRG",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS",
        "V", "MA", "BRK-B", "AXP", "BLK",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE",
        "SBUX", "LOW", "TJX", "BKNG", "CMG",
    ],
    "Consumer Staples": [
        "WMT", "KO", "PEP", "PG", "COST",
        "PM", "MO",
    ],
    "Communication Services": [
        "NFLX", "DIS", "T", "VZ", "CMCSA",
    ],
    "Industrials": [
        "CAT", "HON", "RTX", "UPS", "DE",
        "GE", "LMT",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB",
    ],
    "Materials": [
        "LIN", "APD", "FCX",
    ],
    "Utilities": [
        "NEE", "DUK", "SO",
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX",
    ],
}

# Flat ticker list (no SPY)
UNIVERSE: list[str] = [t for tickers in SECTOR_MAP.values() for t in tickers]

# Reverse map: ticker -> sector
TICKER_SECTOR: dict[str, str] = {
    t: sector for sector, tickers in SECTOR_MAP.items() for t in tickers
}

# Benchmark always included in downloads but excluded from factor universe
BENCHMARK = "SPY"


def get_download_tickers(include_benchmark: bool = True) -> list[str]:
    tickers = list(UNIVERSE)
    if include_benchmark and BENCHMARK not in tickers:
        tickers.append(BENCHMARK)
    return sorted(set(tickers))


def filter_by_sectors(sectors: list[str]) -> list[str]:
    result = []
    for s in sectors:
        result.extend(SECTOR_MAP.get(s, []))
    return result


# ---------------------------------------------------------------------------
# Dynamic getters — merge base universe with user-added tickers
# ---------------------------------------------------------------------------

def get_universe() -> list[str]:
    """Base universe + any user-added tickers (deduplicated, order preserved)."""
    from src.data.custom_universe import load_custom
    custom = [t for t in load_custom() if t not in UNIVERSE]
    return UNIVERSE + custom


def get_ticker_sector() -> dict[str, str]:
    """Base ticker→sector map + user-added tickers (user entries win on conflict)."""
    from src.data.custom_universe import load_custom
    result = dict(TICKER_SECTOR)
    result.update(load_custom())
    return result


def get_sector_map() -> dict[str, list[str]]:
    """Base sector→tickers map extended with user-added tickers."""
    from src.data.custom_universe import load_custom
    result: dict[str, list[str]] = {k: list(v) for k, v in SECTOR_MAP.items()}
    for ticker, sector in load_custom().items():
        result.setdefault(sector, [])
        if ticker not in result[sector]:
            result[sector].append(ticker)
    return result
