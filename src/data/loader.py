"""Data loading layer: prices and fundamentals via yfinance with pickle cache."""
from __future__ import annotations

import warnings
import pandas as pd
import yfinance as yf
from loguru import logger

from src.data.cache import cached


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def _download_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Raw yfinance download — not cached directly; called by the cached wrappers below."""
    logger.info(f"Downloading prices for {len(tickers)} tickers [{start} → {end}]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    if raw.empty:
        raise ValueError("yfinance returned empty DataFrame — check tickers / date range.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = list(tickers)

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna(axis=1, how="all")
    logger.info(f"Prices loaded: {prices.shape[0]} dates × {prices.shape[1]} tickers")
    return prices


@cached(ttl_days=1.0)
def get_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for the base universe.

    Args:
        tickers: sorted tuple of ticker symbols (tuple for cache-key stability)
        start: YYYY-MM-DD
        end:   YYYY-MM-DD

    Returns:
        DataFrame of adjusted close prices, columns = tickers, index = DatetimeIndex.
    """
    return _download_prices(tickers, start, end)


@cached(ttl_days=1.0)
def get_prices_custom(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for user-added tickers.

    Stored under a separate cache prefix (``get_prices_custom_*.pkl``) so that
    adding or removing custom tickers only busts this cache, leaving the base
    universe cache intact.
    """
    return _download_prices(tickers, start, end)


def load_prices(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load prices, routing base and custom tickers to separate cached calls.

    Use this instead of ``get_prices`` in pages so that custom-ticker additions
    never invalidate the base universe price cache.

    Args:
        tickers: full sorted tuple (base + custom + benchmark)
        start / end: YYYY-MM-DD date range
        force_refresh: bypass cache and re-download

    Returns:
        Merged DataFrame of adjusted close prices for all requested tickers.
    """
    from src.data.universe import UNIVERSE, BENCHMARK

    base_set = set(UNIVERSE) | {BENCHMARK}
    base    = tuple(t for t in tickers if t in base_set)
    custom  = tuple(t for t in tickers if t not in base_set)

    base_prices = get_prices(base, start, end, force_refresh=force_refresh) if base else pd.DataFrame()

    if not custom:
        return base_prices

    custom_prices = get_prices_custom(custom, start, end, force_refresh=force_refresh)
    merged = pd.concat([base_prices, custom_prices], axis=1)
    merged.index = pd.to_datetime(merged.index)
    return merged.sort_index()


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price DataFrame."""
    return prices.pct_change().dropna(how="all")


# ---------------------------------------------------------------------------
# Fundamental data (snapshot — not time-series)
# ---------------------------------------------------------------------------

@cached(ttl_days=1.0)
def get_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Fetch point-in-time fundamental data from yfinance.info.

    NOTE: This returns CURRENT values only — no historical fundamentals.
    Suitable for cross-sectional factor construction, not time-series IC.

    Returns:
        DataFrame indexed by ticker with columns:
        sector, industry, market_cap, pb, pe, forward_pe, roe,
        gross_margins, beta, peg, dividend_yield
    """
    logger.info(f"Fetching fundamentals for {len(tickers)} tickers")
    records = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            records.append(
                {
                    "ticker": ticker,
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "market_cap": info.get("marketCap"),
                    "pb": info.get("priceToBook"),
                    "pe": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "roe": info.get("returnOnEquity"),
                    "gross_margins": info.get("grossMargins"),
                    "beta": info.get("beta"),
                    "peg": info.get("pegRatio"),
                    "dividend_yield": info.get("dividendYield"),
                    "short_ratio": info.get("shortRatio"),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
            records.append({"ticker": ticker})

    df = pd.DataFrame(records).set_index("ticker")
    logger.info(f"Fundamentals loaded for {len(df)} tickers")
    return df
